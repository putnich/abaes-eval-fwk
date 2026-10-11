"""Construct provider clients and bind the engine's typed output schemas."""

from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING

from abaes_eval_fwk.config.llm_config import LLMConfig

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import Runnable
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel


_CONFIG_ATTRIBUTE = "_abaes_eval_fwk_config"


def _default_llm_config() -> LLMConfig:
    """An LLMConfig for callers that never attached one, built from the same YAML the
    full pipeline merges in — not a second, hardcoded copy of those values."""
    from abaes_eval_fwk.utils.resources import load_defaults

    return LLMConfig(**load_defaults()["model_defaults"])


def _configuration(llm) -> LLMConfig:
    found = getattr(llm, _CONFIG_ATTRIBUTE, None)
    return found if found is not None else _default_llm_config()


def bind_tools(llm: BaseChatModel, tool: StructuredTool) -> Runnable:
    """Bind a grading tool using the configured provider's tool-choice setting."""
    config = _configuration(llm)
    choice = tool.name if config.tool_choice == "named" else config.tool_choice
    options = {"tool_choice": choice} if choice is not None else {}
    bound = llm.bind_tools([tool], **options)
    object.__setattr__(bound, _CONFIG_ATTRIBUTE, config)
    return bound


def structured_output(llm, schema: type[BaseModel]) -> Runnable:
    """Build a typed response chain that preserves raw messages and parsing errors."""
    config = _configuration(llm)
    options = {"include_raw": True, "method": config.structured_output_method}
    if config.structured_output_method == "function_calling" and config.tool_choice != "named":
        options["tool_choice"] = config.tool_choice
    chain = llm.with_structured_output(schema, **options)
    if config.structured_output_method != "json_mode":
        return chain
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.runnables import RunnableLambda

    instructions = "Return only a JSON object matching this schema: " + json.dumps(
        schema.model_json_schema()
    )

    def add_schema(value):
        messages = value.to_messages() if hasattr(value, "to_messages") else value
        messages = [HumanMessage(content=messages)] if isinstance(messages, str) else list(messages)
        return [SystemMessage(content=instructions), *messages]

    return RunnableLambda(add_schema) | chain


class LLMLoader:
    """Build a shared client once per experiment without caching mutable request maps."""

    @staticmethod
    def load(llm_cfg: LLMConfig | None = None):
        """Load a built-in provider or a researcher-supplied module:function adapter."""
        config = llm_cfg or _default_llm_config()
        if not config.model_name.strip():
            raise ValueError("llm.model_name is required")
        if config.adapter_factory:
            module_name, factory_name = config.adapter_factory.split(":", 1)
            factory = getattr(importlib.import_module(module_name), factory_name)
            llm = factory(config)
        else:
            if config.provider == "openai_compatible" and not config.base_url:
                raise ValueError(
                    "openai_compatible requires base_url, e.g. http://localhost:8000/v1"
                )
            model_class: type
            if config.provider == "deepseek":
                try:
                    from langchain_deepseek import ChatDeepSeek
                except ImportError as exc:
                    raise ImportError(
                        "Install the deepseek extra: pip install '.[deepseek]'"
                    ) from exc
                model_class = ChatDeepSeek
            else:
                from langchain_openai import ChatOpenAI

                model_class = ChatOpenAI
            options = {
                "model": config.model_name,
                "api_key": config.resolve_api_key(),
                "temperature": config.temperature,
                "timeout": config.timeout,
                "streaming": config.streaming,
                "max_retries": 0,
            }
            if config.base_url:
                options["base_url"] = config.base_url
            if config.streaming:
                options["stream_usage"] = True
            if config.max_tokens is not None:
                token_field = (
                    "max_completion_tokens" if config.provider == "openai" else "max_tokens"
                )
                options[token_field] = config.max_tokens
            if config.use_responses_api is not None:
                options["use_responses_api"] = config.use_responses_api
            options.update(config.resolved_request_kwargs())
            llm = model_class(**options)
        if not all(
            callable(getattr(llm, method, None))
            for method in ("invoke", "bind_tools", "with_structured_output")
        ):
            raise TypeError(
                "LLM adapters must implement invoke, bind_tools, and with_structured_output"
            )
        object.__setattr__(llm, _CONFIG_ATTRIBUTE, config)
        return llm
