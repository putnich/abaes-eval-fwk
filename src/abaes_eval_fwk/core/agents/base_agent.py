"""BaseAgent: shared prompt execution, retries, JSON parsing, and token tracking for agents."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TypedDict

import openai
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from abaes_eval_fwk.config.llm_config import Reasoning
from abaes_eval_fwk.core.agents.schemas.results import GradingResponse
from abaes_eval_fwk.core.llm.llm_loader import structured_output
from abaes_eval_fwk.core.models.assessment import HumanAssessment, LLMAssessment, TokenUsage
from abaes_eval_fwk.core.models.experiment import ResearchQuestion
from abaes_eval_fwk.core.models.rubric import Rubric
from abaes_eval_fwk.utils.backoff import agent_backoff
from abaes_eval_fwk.utils.conversation_logging import ConversationLogger, conversation_log_path
from abaes_eval_fwk.utils.prompt_formatters import format_examples, truncate_examples_block

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3  # any error (network, malformed JSON, domain validation) shares this budget

ALLOW_CONTEXT_RECOVERY = False

EXAMPLE_FRACTIONS = [1.0, 0.8, 0.6, 0.4, 0.2]  # fraction of each example's own length


class TruncatedResponseError(Exception):
    """Raised when a structured-output generation looks cut off mid-object (finish_reason == "length")."""


_LENGTH_LIMIT_ERRORS = (
    ContextOverflowError,
    TruncatedResponseError,
    openai.LengthFinishReasonError,
)

_BREVITY_INSTRUCTION = " Keep the justification to 2-3 sentences — be concise."


def _is_context_overflow(exc: openai.BadRequestError) -> bool:
    """vLLM/TGI backends flag context overflow via body["param"] == "input_tokens" rather than
    the OpenAI error codes langchain_openai's own detection recognizes."""
    return exc.param == "input_tokens"


class _StructuredResult(TypedDict):
    """Shape of `with_structured_output(GradingResponse, include_raw=True)`'s output."""

    raw: AIMessage
    parsed: BaseModel | None
    parsing_error: BaseException | None


class TokenUsageCallback(BaseCallbackHandler):
    """Captures token counts from OpenAI-compatible LLM responses."""

    def __init__(self) -> None:
        self.usage = TokenUsage()

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        token_usage = (response.llm_output or {}).get("token_usage", {})
        if token_usage:
            self.usage = self.usage + TokenUsage(
                prompt_tokens=token_usage.get("prompt_tokens", 0),
                completion_tokens=token_usage.get("completion_tokens", 0),
            )
            return
        # Streaming responses leave llm_output empty; LangChain attaches usage to the
        # AIMessage's usage_metadata instead.
        for generation in response.generations:
            for gen in generation:
                message = getattr(gen, "message", None)
                usage_metadata = getattr(message, "usage_metadata", None) if message else None
                if usage_metadata:
                    self.usage = self.usage + TokenUsage(
                        prompt_tokens=usage_metadata.get("input_tokens", 0),
                        completion_tokens=usage_metadata.get("output_tokens", 0),
                    )


class BaseAgent(ABC):
    """Provide prompt execution, retries, JSON parsing, and token tracking."""

    _conversation_logger: ConversationLogger | None = None
    _attempt_number: int = 0  # 0 = first attempt; set by _attempt_with_retry before each retry
    # Every response this grading call consumed, including attempts whose result was discarded; a
    # failed parse or an out-of-range score still cost tokens, so they are not dropped.
    _call_usage: TokenUsage = TokenUsage()
    # Per-callback total already folded in, so a callback reused across turns is not counted twice.
    _counted_usage: dict[TokenUsageCallback, TokenUsage] = {}

    def __init__(self, prompt_template: str | None = None) -> None:
        self._prompt_template = prompt_template

    def _record_usage(self, callback: TokenUsageCallback) -> None:
        """Fold a callback's newly reported tokens into this call's total.

        A callback accumulates across every turn it is passed to, so only its growth since the
        last fold is added; re-reading it whole would count each response once per turn.
        """
        previous = self._counted_usage.get(callback, TokenUsage())
        self._counted_usage[callback] = callback.usage
        self._call_usage = self._call_usage + TokenUsage(
            prompt_tokens=callback.usage.prompt_tokens - previous.prompt_tokens,
            completion_tokens=callback.usage.completion_tokens - previous.completion_tokens,
        )

    def _begin_call(self) -> None:
        """Start a grading call's token accounting, so a prior call's total cannot leak into it."""
        self._call_usage = TokenUsage()
        self._counted_usage = {}

    def _make_chain(
        self,
        llm: BaseChatModel,
        rq: ResearchQuestion | None,
        reasoning: Reasoning,
    ) -> Runnable:
        """Wrap an already-built LLM runnable in this agent's own prompt template."""
        if self._prompt_template is None:
            raise RuntimeError("subclass must supply a prompt_template")
        prompt = PromptTemplate.from_template(self._prompt_template) | (lambda p: p.text)
        structured_llm = structured_output(llm, GradingResponse)
        if reasoning == Reasoning.NO_THINK:
            # DeepSeek-R1-distill has no template-level thinking toggle, so an empty <think>
            # assistant turn suppresses reasoning generation instead (see llm_loader.py).
            prompt = prompt | (
                lambda text: [
                    HumanMessage(content=text),
                    AIMessage(content="<think>\n\n</think>\n\n"),
                ]
            )
        return prompt | structured_llm

    def _make_conversation_logger(
        self, rubric: Rubric, human_assessment: HumanAssessment
    ) -> ConversationLogger | None:
        """Build this grading call's conversation logger, or None if not configured for this
        process (see utils.conversation_logging.configure_conversation_logging)."""
        path = conversation_log_path(
            rubric.rubric_id, human_assessment.essay_id, human_assessment.criterion.name
        )
        return ConversationLogger(path) if path else None

    def _invoke_config(self, *callbacks: BaseCallbackHandler, turn_name: str) -> dict:
        """Build one LLM invoke() call's config: callbacks plus a turn_name identifying this
        turn in the conversation log (e.g. "Review: ideas", "Final synthesis")."""
        if self._attempt_number > 0:
            turn_name = f"{turn_name} (retry {self._attempt_number})"
        cbs = (
            [*callbacks, self._conversation_logger]
            if self._conversation_logger
            else list(callbacks)
        )
        return {"callbacks": cbs, "metadata": {"turn_name": turn_name}}

    def _invoke(self, chain, inputs: dict) -> tuple[_StructuredResult, TokenUsage]:
        """Invoke a chain once and return its structured-output result and token use."""
        token_usage_cb = TokenUsageCallback()
        config = self._invoke_config(token_usage_cb, turn_name=self.__class__.__name__)
        try:
            llm_response = chain.invoke(inputs, config=config)
        except openai.BadRequestError as exc:
            if _is_context_overflow(exc):
                raise TruncatedResponseError(str(exc)) from exc
            raise
        finally:
            # Counted whatever the outcome: a response whose parse or score is rejected was still
            # paid for, and the caller's total must be the cost of the assessment, not of the
            # attempt that happened to succeed.
            self._record_usage(token_usage_cb)
        return llm_response, token_usage_cb.usage

    def _attempt(self, inputs: dict, rubric: Rubric) -> LLMAssessment:
        """Run one invocation and result-parsing attempt against the structured-output chain."""
        structured, usage = self._invoke(self._chain, inputs)
        parsed = structured["parsed"]
        if parsed is None:
            if structured["raw"].response_metadata.get("finish_reason") == "length":
                raise TruncatedResponseError(
                    "structured-output generation was cut off (finish_reason=length); "
                    "likely hit max_tokens"
                )
            raise ValueError(f"structured output parsing failed: {structured['parsing_error']}")
        result_dict = parsed.model_dump()
        # ChatDeepSeek stashes native <think> content here when the serving stack exposes it;
        # it's the only source for LLMAssessment.reasoning (plain ChatOpenAI drops this field).
        native_reasoning = structured["raw"].additional_kwargs.get("reasoning_content")
        if native_reasoning:
            result_dict["reasoning"] = native_reasoning
        return self._parse_result(result_dict, usage, rubric)

    def _attempt_with_retry(self, inputs: dict, rubric: Rubric) -> LLMAssessment:
        """Run one attempt, retrying up to MAX_ATTEMPTS on ordinary errors; fail fast on a length limit."""
        last_error: str | None = None
        for i in range(MAX_ATTEMPTS):
            self._attempt_number = i
            try:
                result = self._attempt(inputs, rubric)
            except _LENGTH_LIMIT_ERRORS:
                raise
            except Exception as exc:
                if i == MAX_ATTEMPTS - 1:
                    logger.warning(
                        "%s: failed after %d attempts", self.__class__.__name__, MAX_ATTEMPTS
                    )
                    raise
                last_error = f"attempt {i + 1}/{MAX_ATTEMPTS}: {type(exc).__name__}: {exc}"
                time.sleep(agent_backoff(i))
            else:
                result.statistics.retry_error = last_error
                return result

    def _run_chain(self, inputs: dict, rubric: Rubric) -> LLMAssessment:
        """Run a grading chain with retry, recovering from a length-limit failure in two stages:
        first a brevity instruction at full examples, then a multi-step example-shrinking loop."""
        # (example_fraction, use_brevity_instruction) per attempt, in order.
        attempts = [(1.0, False)]
        if ALLOW_CONTEXT_RECOVERY:
            attempts.extend([(1.0, True), *((f, True) for f in EXAMPLE_FRACTIONS[1:])])
        original_examples = inputs.get("examples", "")
        start_time = datetime.now(UTC)
        # One grading call's total, spanning every attempt and recovery stage it goes through.
        self._begin_call()

        for level, (fraction, brevity) in enumerate(attempts):
            truncated_examples = truncate_examples_block(original_examples, fraction)
            attempt_inputs = {
                **inputs,
                "examples_block": (
                    f"\nReference examples:\n\n{truncated_examples}" if truncated_examples else ""
                ),
                "examples_comparison_step": (
                    " Also compare it against the reference examples provided to calibrate "
                    "your judgment."
                    if truncated_examples
                    else ""
                ),
                "brevity_instruction": _BREVITY_INSTRUCTION if brevity else "",
            }
            try:
                result = self._attempt_with_retry(attempt_inputs, rubric)
            except _LENGTH_LIMIT_ERRORS as exc:
                if level == len(attempts) - 1:
                    logger.warning(
                        "%s: still hitting a length limit after shrinking examples to %d%% "
                        "with a brevity instruction: %s",
                        self.__class__.__name__,
                        fraction * 100,
                        exc,
                    )
                    raise
                next_fraction, next_brevity = attempts[level + 1]
                next_step = (
                    "a brevity instruction"
                    if next_brevity and not brevity
                    else f"{next_fraction * 100:.0f}% examples (brevity instruction still on)"
                )
                logger.warning(
                    "%s: hit a length limit at %.0f%% examples (%s), trying %s",
                    self.__class__.__name__,
                    fraction * 100,
                    exc,
                    next_step,
                )
                continue

            result.statistics.start_time = start_time
            result.statistics.end_time = datetime.now(UTC)
            result.statistics.example_cap = fraction
            result.statistics.token_usage = self._call_usage
            return result

    @abstractmethod
    def _parse_result(
        self,
        result: dict,
        usage,
        rubric: Rubric,
    ) -> LLMAssessment:
        """Parse an LLM response into this agent's result."""
        raise NotImplementedError

    def _build_inputs(self, rubric: Rubric, human_assessment: HumanAssessment, **kwargs) -> dict:
        """Build this agent's prompt input values."""
        raise NotImplementedError

    def _build_common_inputs(
        self,
        rubric: Rubric,
        human_assessment: HumanAssessment,
        reference_examples: list[HumanAssessment],
    ) -> dict:
        """Build shared prompt inputs for fresh-submission grading."""
        return {
            "assignment_type": rubric.assignment_type,
            "assignment_prompt": rubric.prompt,
            "source_essay": (
                f'\n\nSource text the student read:\n"""\n{rubric.source_text}\n"""'
                if rubric.source_text
                else ""
            ),
            "examples": format_examples(reference_examples),
            "student_work": human_assessment.submission.text,
            # grade prompt uses a numbered step prefix for source verification
            "source_verification_step": (
                "\n2a. Verify that the student's evidence and examples are accurate and genuinely drawn from the source text provided above."
                if rubric.source_text
                else ""
            ),
        }

    def grade(
        self,
        rubric: Rubric,
        human_assessment: HumanAssessment,
        llm: BaseChatModel,
        rq: ResearchQuestion | None = None,
        reasoning: Reasoning = Reasoning.NONE,
        **kwargs,
    ) -> LLMAssessment:
        """Grade one submission with this agent's prompt and LLM. Extra kwargs (e.g."""
        self._chain = self._make_chain(llm, rq, reasoning)
        self._conversation_logger = self._make_conversation_logger(rubric, human_assessment)
        inputs = self._build_inputs(rubric, human_assessment, **kwargs)
        return self._run_chain(inputs, rubric)
