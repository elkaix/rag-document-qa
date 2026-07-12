# CONTEXT — domain glossary

The vocabulary of this codebase, so names stay consistent across modules, tests,
and ADRs. New module names enter here as the architecture-deepening spec
([issue #16](https://github.com/elkaix/rag-document-qa/issues/16)) lands them.

Design terms use the `/codebase-design` vocabulary: **module** (a unit with an
interface hiding behaviour), **interface** (the public surface), **depth** (much
behaviour behind a small interface), **seam** (a boundary you can substitute at),
**adapter** (a module presenting one interface over another), **leverage**,
**locality**.

## Modules & types

- **ProviderAdapter** — the interface (Protocol) one LLM provider hides behind:
  `generate(messages) -> GenerationResult` and `stream(messages)` yielding text
  chunks then a terminal `Usage`. Implementations: `OpenAICompatibleAdapter`
  (OpenAI + GLM), `AnthropicAdapter`, `OllamaAdapter`, `DummyAdapter`. Each is
  constructed with an injected `client_factory` so it is testable with a fake.
  See [ADR 0002](docs/adr/0002-provider-adapters.md).
- **Usage** — a value object of `(prompt_tokens, completion_tokens)` for one
  generation call. Provider-reported where the SDK returns it, adapter-counted
  otherwise.
- **GenerationResult** — the `(text, usage)` pair returned by a non-streaming
  generation, so callers never rebuild a prompt to estimate what was billed.
- **LLMHandler** — owns provider *selection* (from the model-name prefix), the
  single-prompt → messages translation, and the unconfigured-provider fallback
  to `DummyAdapter`. Delegates all SDK work to one selected `ProviderAdapter`.
- **telemetry** (`src/telemetry/`) — core token counting (`count_tokens`) and
  cost pricing (`cost_usd`, `MODEL_PRICES`). Owned by the core so both production
  telemetry assembly and the eval harness use one source of truth; the eval
  package imports from here, never the reverse. See
  [ADR 0003](docs/adr/0003-telemetry-ownership.md).
