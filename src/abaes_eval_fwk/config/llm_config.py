"""Provider settings and explicit model-specific reasoning request parameters."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from enum import Enum

JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
JSONDict = dict[str, JSONValue]


class Reasoning(str, Enum):
    """Reasoning labels recorded by the experiment engine."""

    REASONING = "reasoning"
    NON_REASONING = "non_reasoning"
    NO_THINK = "no_think"
    NONE = ""


def model_display_name(model_name: str) -> str:
    """Keep arbitrary provider model identifiers intact in metric output."""
    return model_name


def _merge(base: dict, overrides: dict) -> dict:
    """Recursively merge request options without mutating either input."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


@dataclass(frozen=True, kw_only=True)
class LLMConfig:
    """Serializable model settings; credentials are resolved from the environment at execution.

    `provider`, `max_tokens`, `timeout`, `streaming`, `tool_choice`, and
    `structured_output_method` have no default here: all six are already defaulted in
    config/config_files/defaults/models.yaml (model_defaults), which model_config() in
    configuration.py always merges in before construction for the YAML-driven path.
    Standalone construction (llm_loader.py's bare fallback; see model_defaults there)
    supplies the same YAML values explicitly instead of duplicating them as a second
    Python-side default.

    `temperature` keeps its own `None` default deliberately: unlike the six above, it
    doesn't duplicate model_defaults.yaml's value (0) — `None` means "omit the request
    parameter entirely", a distinct, meaningful choice for standalone use that YAML's
    sitewide `temperature: 0` doesn't express.
    """

    model_name: str = ""
    model_shorthand: str = ""
    provider: str
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float | None = None
    max_tokens: int | None
    timeout: float
    streaming: bool
    reasoning: Reasoning = Reasoning.NONE
    request_kwargs: JSONDict = field(default_factory=dict)
    extra_body: JSONDict = field(default_factory=dict)
    reasoning_on: JSONDict = field(default_factory=dict)
    reasoning_off: JSONDict = field(default_factory=dict)
    tool_choice: str | dict | None
    structured_output_method: str
    adapter_factory: str | None = None
    use_responses_api: bool | None = None

    def __post_init__(self) -> None:
        aliases = {
            "": Reasoning.NONE,
            "yes": Reasoning.REASONING,
            "no": Reasoning.NON_REASONING,
            "no_think": Reasoning.NO_THINK,
        }
        value = self.reasoning
        if isinstance(value, bool):
            value = Reasoning.REASONING if value else Reasoning.NON_REASONING
        object.__setattr__(self, "reasoning", Reasoning(aliases.get(value, value)))
        if not self.model_shorthand and self.model_name:
            object.__setattr__(self, "model_shorthand", self.model_name.rsplit("/", 1)[-1])
        self._validate_llm_config()

    def _validate_llm_config(self) -> None:
        """Validate all LLMConfig fields."""
        if not isinstance(self.model_name, str) or not isinstance(self.model_shorthand, str):
            raise TypeError("model_name and model_shorthand must be strings")
        if (
            self.provider not in {"openai", "openai_compatible", "deepseek"}
            and not self.adapter_factory
        ):
            raise ValueError("Unknown provider requires adapter_factory: module:function")
        if self.adapter_factory and (
            ":" not in self.adapter_factory or self.adapter_factory.endswith(":")
        ):
            raise ValueError("adapter_factory must be module:function")
        if self.api_key_env is not None and not isinstance(self.api_key_env, str):
            raise ValueError("api_key_env must be an environment variable name")
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer or null")
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number")
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
        ):
            raise ValueError("temperature must be a finite number or null")
        if (
            not isinstance(self.streaming, bool)
            or self.use_responses_api is not None
            and not isinstance(self.use_responses_api, bool)
        ):
            raise TypeError(
                "streaming must be boolean and use_responses_api must be boolean or null"
            )
        if self.structured_output_method not in {"function_calling", "json_schema", "json_mode"}:
            raise ValueError(
                "structured_output_method must be function_calling, json_schema, or json_mode"
            )
        for name in ("request_kwargs", "extra_body", "reasoning_on", "reasoning_off"):
            options = getattr(self, name)
            if not isinstance(options, dict):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, copy.deepcopy(options))
        if self.reasoning == Reasoning.REASONING and not self.reasoning_on:
            raise ValueError(
                "reasoning='reasoning' requires explicit reasoning_on request parameters"
            )
        if self.reasoning == Reasoning.NON_REASONING and not self.reasoning_off:
            raise ValueError(
                "reasoning='non_reasoning' requires explicit reasoning_off request parameters"
            )

    def resolve_api_key(self) -> str:
        """Resolve credentials only when a provider client is built."""
        default_env = {"openai": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}.get(
            self.provider, "PROVIDER_API_KEY"
        )
        env_name = self.api_key_env if self.api_key_env is not None else default_env
        key = os.environ.get(env_name, "") if env_name else ""
        if key:
            return key
        if self.provider == "openai_compatible" and self.api_key_env is None:
            return "not-required"
        if self.provider == "openai_compatible" and not env_name:
            return "not-required"
        raise ValueError(f"Set the {env_name or default_env} environment variable before running")

    def resolved_request_kwargs(self) -> JSONDict:
        """Merge explicit reasoning settings into provider options without assuming model capabilities."""
        options = _merge(
            self.request_kwargs, {"extra_body": self.extra_body} if self.extra_body else {}
        )
        if self.reasoning == Reasoning.REASONING:
            options = _merge(options, self.reasoning_on)
        elif self.reasoning == Reasoning.NON_REASONING:
            options = _merge(options, self.reasoning_off)
        elif self.reasoning == Reasoning.NO_THINK:
            options = _merge(
                options,
                {"extra_body": {"continue_final_message": True, "add_generation_prompt": False}},
            )
            options = _merge(options, self.reasoning_off)
        return options
