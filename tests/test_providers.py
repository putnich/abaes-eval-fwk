"""Provider configuration and request construction without remote inference."""

import dataclasses
import sys
import types

import pytest

from abaes_eval_fwk.config.llm_config import LLMConfig, Reasoning
from abaes_eval_fwk.core.llm.llm_loader import LLMLoader, bind_tools, structured_output

# Mirrors config/config_files/defaults/models.yaml's model_defaults: LLMConfig no longer
# defaults these itself (see llm_config.py), so tests that don't care about a particular
# field still need to supply all of them explicitly.
MODEL_DEFAULTS = {
    "provider": "openai_compatible",
    "max_tokens": 4096,
    "timeout": 120.0,
    "streaming": False,
    "tool_choice": "required",
    "structured_output_method": "function_calling",
}


def make_llm_config(**overrides) -> LLMConfig:
    return LLMConfig(**{**MODEL_DEFAULTS, **overrides})


class FakeChatModel:
    def __init__(self, **kwargs):
        self.options = kwargs

    def invoke(self, *args, **kwargs):
        return None

    def bind_tools(self, tools, **kwargs):
        return FakeChatModel(tools=tools, **kwargs)

    def with_structured_output(self, schema, **kwargs):
        return {"schema": schema, **kwargs}


@pytest.fixture
def fake_openai(monkeypatch):
    module = types.ModuleType("langchain_openai")
    module.ChatOpenAI = FakeChatModel
    monkeypatch.setitem(sys.modules, "langchain_openai", module)
    return module


def test_arbitrary_future_model_and_credentials_are_runtime_only(monkeypatch, fake_openai):
    monkeypatch.delenv("FUTURE_API_KEY", raising=False)
    config = make_llm_config(
        model_name="future-research-model", provider="openai", api_key_env="FUTURE_API_KEY"
    )
    with pytest.raises(ValueError, match="FUTURE_API_KEY"):
        LLMLoader.load(config)
    monkeypatch.setenv("FUTURE_API_KEY", "private-value")
    llm = LLMLoader.load(config)
    assert llm.options["model"] == "future-research-model"
    assert llm.options["api_key"] == "private-value"
    assert llm.options["temperature"] is None
    assert "base_url" not in llm.options
    assert "private-value" not in repr(config)
    assert "private-value" not in str(dataclasses.asdict(config))


def test_reasoning_maps_preserve_false_and_merge_nested_options(fake_openai, monkeypatch):
    monkeypatch.delenv("PROVIDER_API_KEY", raising=False)
    config = make_llm_config(
        model_name="researcher/local-model",
        base_url="http://localhost:8000/v1",
        reasoning="no",
        extra_body={"chat_template_kwargs": {"style": "concise"}},
        reasoning_off={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    )
    llm = LLMLoader.load(config)
    assert llm.options["extra_body"] == {
        "chat_template_kwargs": {"style": "concise", "enable_thinking": False}
    }
    assert config.extra_body == {"chat_template_kwargs": {"style": "concise"}}
    assert config.reasoning == Reasoning.NON_REASONING
    assert llm.options["api_key"] == "not-required"


def test_reasoning_is_explicit_and_responses_api_is_configurable(fake_openai, monkeypatch):
    with pytest.raises(ValueError, match="reasoning_on"):
        make_llm_config(model_name="future-model", reasoning="reasoning")
    with pytest.raises(ValueError, match="reasoning_off"):
        make_llm_config(model_name="future-model", reasoning="no")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = make_llm_config(
        model_name="future-model",
        provider="openai",
        use_responses_api=True,
        reasoning="reasoning",
        reasoning_on={"reasoning": {"effort": "high"}},
        request_kwargs={"service_tier": "default"},
        max_tokens=None,
    )
    llm = LLMLoader.load(config)
    assert llm.options["use_responses_api"] is True
    assert llm.options["reasoning"] == {"effort": "high"}
    assert "max_completion_tokens" not in llm.options


def test_configurable_structured_output_survives_binding(fake_openai):
    config = make_llm_config(
        model_name="local-model",
        base_url="http://localhost:8000/v1",
        structured_output_method="json_schema",
        tool_choice="auto",
    )
    llm = LLMLoader.load(config)
    bound = bind_tools(llm, types.SimpleNamespace(name="record_score"))
    assert bound.options["tool_choice"] == "auto"
    assert structured_output(bound, dict) == {
        "schema": dict,
        "include_raw": True,
        "method": "json_schema",
    }


def test_function_calling_uses_explicit_tool_choice(fake_openai):
    llm = LLMLoader.load(
        make_llm_config(model_name="model", base_url="http://localhost:8000/v1", tool_choice="auto")
    )
    assert structured_output(llm, dict)["tool_choice"] == "auto"


def test_custom_provider_factory_receives_public_configuration(monkeypatch):
    module = types.ModuleType("research_provider")
    captured = []
    module.create = lambda config: (captured.append(config), FakeChatModel())[1]
    monkeypatch.setitem(sys.modules, "research_provider", module)
    config = make_llm_config(
        model_name="vendor-model",
        provider="custom-vendor",
        adapter_factory="research_provider:create",
    )
    LLMLoader.load(config)
    assert captured == [config]


def test_deepseek_uses_selected_provider_independent_of_reasoning(monkeypatch):
    module = types.ModuleType("langchain_deepseek")
    module.ChatDeepSeek = FakeChatModel
    monkeypatch.setitem(sys.modules, "langchain_deepseek", module)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    llm = LLMLoader.load(make_llm_config(model_name="research-model", provider="deepseek"))
    assert llm.options["model"] == "research-model"
    assert llm.options["max_tokens"] == 4096


def test_local_backend_requires_endpoint_and_explicit_auth_env(fake_openai, monkeypatch):
    with pytest.raises(ValueError, match="base_url"):
        LLMLoader.load(make_llm_config(model_name="local-model"))
    monkeypatch.delenv("LOCAL_SERVER_KEY", raising=False)
    with pytest.raises(ValueError, match="LOCAL_SERVER_KEY"):
        LLMLoader.load(
            make_llm_config(
                model_name="local-model",
                base_url="http://localhost:8000/v1",
                api_key_env="LOCAL_SERVER_KEY",
            )
        )
