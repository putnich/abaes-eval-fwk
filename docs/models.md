# Models

How to connect the framework to a model: local servers, hosted APIs, reasoning settings, and
custom adapters.

## Contents

- [Profiles](#profiles)
- [Providers](#providers)
- [Credentials](#credentials)
- [Profile settings](#profile-settings)
- [Reasoning modes](#reasoning-modes)
- [Structured output](#structured-output)
- [Custom adapters](#custom-adapters)
- [Hosting and hardware](#hosting-and-hardware)
- [Checking a profile](#checking-a-profile)

## Profiles

`models` maps a profile name to its settings. Experiments refer to models only by profile name,
so the same experiment matrix can be pointed at different models by editing this one file.

```yaml
model_defaults:            # optional; merged under every profile
  temperature: 0
  timeout: 120
models:
  model_a:
    provider: openai_compatible
    model_name: neuralmagic-ent/Llama-3.3-70B-Instruct-FP8-dynamic
    base_url: http://localhost:8000/v1
  model_final:
    provider: openai_compatible
    model_name: google/gemma-4-31B-it
    base_url: http://localhost:8003/v1
```

Each profile is merged over `model_defaults`, which itself sits over the packaged defaults:
`provider: openai_compatible`, `temperature: 0`, `max_tokens: 4096`, `timeout: 120`,
`streaming: false`, `tool_choice: required`, `structured_output_method: function_calling`.

The framework keeps no list of supported models and assumes nothing about a model's
capabilities. Every model-specific request option is written explicitly in the profile.

## Providers

| `provider` | Client | Needs |
|---|---|---|
| `openai_compatible` | `langchain_openai.ChatOpenAI` | `base_url` (vLLM, SGLang, TGI, Ollama's `/v1`, LM Studio, any OpenAI-compatible gateway) |
| `openai` | `ChatOpenAI` | `OPENAI_API_KEY`; sends `max_completion_tokens` instead of `max_tokens` |
| `deepseek` | `langchain_deepseek.ChatDeepSeek` | `DEEPSEEK_API_KEY`; `pip install -e '.[deepseek]'` |
| anything else | your `adapter_factory` | see [custom adapters](#custom-adapters) |

The framework handles retries itself (`runtime.max_attempts`), so the client's own retries are
disabled to keep attempt counts accurate.

## Credentials

Keys are read from the environment only when a client is built, and never written into a
manifest or config.

| Provider | Default variable | Override |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | `api_key_env: MY_VAR` |
| `deepseek` | `DEEPSEEK_API_KEY` | `api_key_env: MY_VAR` |
| `openai_compatible` | `PROVIDER_API_KEY`, optional | `api_key_env: MY_VAR` |

For `openai_compatible`, a missing key is fine: local servers that don't check keys receive a
placeholder. If you set `api_key_env` explicitly, that variable must exist. To keep keys in a
file, copy [`.env.example`](../.env.example) to `.env` (git-ignored) and load it with
`set -a; source .env; set +a`, so the keys are exported to the framework's processes.

## Profile settings

| Key | Default | Meaning |
|---|---|---|
| `model_name` | required | The identifier the server expects; recorded in results |
| `model_shorthand` | profile name | Short label used in run names and file names |
| `provider` | `openai_compatible` | See [providers](#providers) |
| `base_url` | none | API root, e.g. `http://localhost:8000/v1` |
| `api_key_env` | per provider | Environment variable holding the key |
| `temperature` | `0` | `null` omits the parameter entirely |
| `max_tokens` | `4096` | Output cap; `null` omits it |
| `timeout` | `120` | Seconds per request |
| `streaming` | `false` | Stream responses (token usage is still collected) |
| `request_kwargs` | `{}` | Extra client arguments, merged into the request |
| `extra_body` | `{}` | Extra JSON body fields, for server-specific options |
| `reasoning_on` / `reasoning_off` | `{}` | Options merged in for reasoning modes |
| `tool_choice` | `required` | Tool-choice value; `named` forces the grading tool by name; `null` omits it |
| `structured_output_method` | `function_calling` | `function_calling`, `json_schema`, or `json_mode` |
| `use_responses_api` | none | Force OpenAI's Responses API on or off |
| `adapter_factory` | none | `module:function` that builds the client |

## Reasoning modes

An experiment's `reasoning` selects which of the profile's options are merged into each request.
The profile has to say how *its* model turns thinking on or off, because servers and model
families differ. For a vLLM-served model whose chat template has a thinking switch, as in the
`study` template:

```yaml
models:
  qwen:
    provider: openai_compatible
    model_name: Qwen/Qwen3.8-27B-FP8
    base_url: http://localhost:8000/v1
    reasoning_on:
      extra_body: {chat_template_kwargs: {enable_thinking: true}}
    reasoning_off:
      extra_body: {chat_template_kwargs: {enable_thinking: false}}
```

| Experiment `reasoning` | Merged into the request |
|---|---|
| `""` (default) | nothing; the model's native behavior |
| `'yes'` | `reasoning_on` (validation fails if it's empty) |
| `'no'` | `reasoning_off` (validation fails if it's empty) |
| `no_think` | `reasoning_off`, plus `continue_final_message: true` and `add_generation_prompt: false` in the body, so a vLLM-style server continues a prefilled turn without a thinking block |

Quote `'yes'` and `'no'` in YAML, which otherwise reads them as booleans. Any reasoning text the
model returns is kept in the results CSV next to the score.

## Structured output

Every agent asks for a typed response (score, justification, confidence), and the model profile
controls how that is requested:

- `function_calling` (default): the response schema is bound as a tool, and `tool_choice` decides
  whether the model must call it. Works with most OpenAI-compatible servers that support tools.
- `json_schema`: uses the server's native JSON-schema mode. Prefer it where the server enforces
  the schema during decoding.
- `json_mode`: plain JSON mode, with the schema stated in a system message. Use it for servers
  that support neither of the above.

If a model keeps returning malformed or truncated output, try another method or raise
`max_tokens` before changing prompts. Truncated responses are detected and retried.

## Custom adapters

For a provider without an OpenAI-compatible endpoint, or for a test double, name a factory:

```yaml
models:
  local:
    provider: my_provider          # any value works when adapter_factory is set
    model_name: my-model
    adapter_factory: extensions.adapters:create_model
```

```python
# extensions/adapters.py, beside benchmark.yaml
def create_model(config):  # config: the resolved LLMConfig
    return MyChatModel(model=config.model_name, temperature=config.temperature)
```

The returned object must implement `invoke`, `bind_tools`, and `with_structured_output`, as any
LangChain chat model does. The playground's profiles use
`abaes_eval_fwk.playground:create_model`, which answers offline. Because the adapter's source is
hashed into the run manifest, editing it starts a new experiment.

## Hosting and hardware

The framework only sends HTTP requests to models. GPUs matter for the model server, not for the
framework itself. The packaged study template expects one OpenAI-compatible server per profile
(ports 8000–8003), but profiles can share a server. For example, with vLLM:

```bash
vllm serve neuralmagic-ent/Llama-3.3-70B-Instruct-FP8-dynamic --port 8000 \
  --enable-auto-tool-choice --tool-call-parser llama3_json
```

Tool calling has to be enabled on the server for the default `function_calling` method; flag
names vary by server and model family. Throughput is governed by `runtime.max_workers`: raise it
until the server saturates, then stop.

Embeddings for `semantic`/`hybrid` retrieval run inside the framework process. They default to
CPU, which is adequate for calibration pools of a few hundred essays. Set
`retrieval.embedding_device` to `cuda:0` or `mps` for large pools.

Record hardware and serving details in `metadata` so they travel with each run's manifest. Remote
server hardware can't be detected from the client.

## Checking a profile

```bash
abaes-eval-fwk validate benchmark.yaml                       # profile fields and reasoning options
abaes-eval-fwk run benchmark.yaml --experiment X --dry-run   # plan, no requests
abaes-eval-fwk run benchmark.yaml --experiment X \
  --set sampling.sample_size=2 --set runtime.max_workers=1   # smallest real call
```

If the small run fails, its `logs/run.log` shows the provider error. Typical causes are a wrong
`base_url` (it must end at `/v1` for most servers), tool calling disabled on the server, or
`max_tokens` too small for a reasoning model.
