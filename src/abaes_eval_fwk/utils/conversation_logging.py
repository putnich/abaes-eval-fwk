"""Persists every LLM turn's full conversation (messages in, response out) to a readable
per-essay text file.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

_INVALID_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|]')

_run_dir: Path | None = None
_write_lock = threading.Lock()


def configure_conversation_logging(
    run_id: str | None, conversations_dir: str | Path | None = None
) -> Path | None:
    """Set (or clear, with run_id=None) this process's conversation-log run directory."""
    global _run_dir
    if conversations_dir is None or run_id is None:
        _run_dir = None
    else:
        _run_dir = Path(conversations_dir) / run_id
    return _run_dir


def _sanitize(part: str) -> str:
    return _INVALID_FILENAME_CHARS.sub("_", part)


def _slugify(label: str) -> str:
    """Match DatasetConfig.slug (label.lower().replace(" ", "_")), so conversation-log folders
    line up with the calibration-pool/index/rubric filenames already used for the same dataset."""
    return label.lower().replace(" ", "_")


def conversation_log_path(
    dataset_label: str, essay_id: str, criterion: str | None = None
) -> Path | None:
    """Return this essay's (optionally per-criterion) conversation-log file path, or None if
    conversation logging hasn't been configured for this process (see
    configure_conversation_logging)."""
    if _run_dir is None:
        return None
    name = _sanitize(essay_id)
    if criterion and criterion != "holistic":
        name = f"{name}__{_sanitize(criterion.replace('_', ' '))}"
    return _run_dir / _slugify(dataset_label) / f"{name}.txt"


class ConversationLogger(BaseCallbackHandler):
    """Appends each LLM turn's rendered messages and response to one essay's log file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._rendered_ids: set[int] = set()

    def _append(self, text: str) -> None:
        with _write_lock, open(self._path, "a", encoding="utf-8") as f:
            f.write(text)

    def _render_new(self, messages: list[BaseMessage]) -> list[str]:
        """Render only the messages not already written to this essay's file, and mark them written."""
        new = [m for m in messages if id(m) not in self._rendered_ids]
        self._rendered_ids.update(id(m) for m in new)
        return [f"--- {msg.type.upper()} ---\n{msg.content}\n" for msg in new]

    def mark_logged(self, *messages: BaseMessage) -> None:
        """Record that `messages` were already shown some other way (e.g. `on_llm_end`'s own
        formatting), so a later delta-render of a list containing them skips rather than reprints
        them raw."""
        self._rendered_ids.update(id(m) for m in messages)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        turn_name = (metadata or {}).get("turn_name", "Turn")
        lines = [f"{'=' * 80}\n{turn_name}\n{'=' * 80}\n", *self._render_new(messages[0])]
        self._append("\n".join(lines) + "\n")

    def log_adopted_turn(self, turn_name: str, messages: list[BaseMessage], result: dict) -> None:
        """Write a turn that adopted a prior assessment verbatim instead of calling the LLM, so this essay's transcript stays complete even though no chain.invoke() ever fired the usual callbacks."""
        lines = [f"{'=' * 80}\n{turn_name}\n{'=' * 80}\n", *self._render_new(messages)]
        args = "\n".join(f"{k}: {v}" for k, v in result.items())
        lines.append(f"--- ADOPTED ASSESSMENT ---\n{args}\n")
        self._append("\n".join(lines) + "\n")

    def on_llm_end(self, response, *, run_id: UUID, **kwargs: Any) -> None:
        generation = response.generations[0][0]
        text = generation.text
        message = getattr(generation, "message", None)
        # Native <think> content (ChatDeepSeek only — see llm_loader.py); not part of the
        # visible response text, so it'd otherwise be invisible in these transcripts.
        reasoning = (message.additional_kwargs or {}).get("reasoning_content") if message else None
        lines = []
        if reasoning:
            lines.append(f"--- REASONING ---\n{reasoning}\n")
        # with_structured_output/bind_tools force a tool call — the model's answer lands in
        # tool_calls, not content, so `text` is empty for every structured-output turn.
        tool_calls = getattr(message, "tool_calls", None) if message else None
        if not text and tool_calls:
            for call in tool_calls:
                args = "\n".join(f"{k}: {v}" for k, v in call["args"].items())
                lines.append(f"--- TOOL CALL: {call['name']} ---\n{args}\n")
        else:
            lines.append(f"--- RESPONSE ---\n{text}\n")
        self._append("\n".join(lines))
