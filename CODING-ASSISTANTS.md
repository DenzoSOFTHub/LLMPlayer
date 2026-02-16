# Using LLMPlayer with Coding Assistants

LLMPlayer exposes both OpenAI-compatible and Anthropic-compatible APIs, making it work with most AI coding assistants out of the box.

## Starting the Server

```bash
# Compile
mvn clean compile

# Start with a model
./run.sh --web --model gguf/your-model.gguf

# Server runs on http://localhost:8080 by default
# Custom port: ./run.sh --web --port 9000 --model gguf/your-model.gguf
```

## Claude Code

Claude Code uses the Anthropic Messages API (`/v1/messages`), which LLMPlayer implements natively.

```bash
# Set environment variables
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_API_KEY=any-value

# Launch Claude Code with local model
claude --model local-model
```

The `ANTHROPIC_API_KEY` is required by the client but LLMPlayer accepts any value.

## Continue.dev (VS Code / JetBrains)

Add to your `~/.continue/config.json`:

```json
{
  "models": [
    {
      "provider": "openai",
      "title": "LLMPlayer Local",
      "model": "local-model",
      "apiBase": "http://localhost:8080/v1",
      "apiKey": "any-value"
    }
  ]
}
```

## Cursor

In Cursor Settings > Models, add a custom OpenAI-compatible model:

- **API Base URL:** `http://localhost:8080/v1`
- **API Key:** `any-value`
- **Model name:** `local-model`

## aider

```bash
# Using OpenAI-compatible API
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=any-value
aider --model openai/local-model
```

## Open WebUI

In the admin settings, add an OpenAI-compatible connection:

- **URL:** `http://localhost:8080/v1`
- **API Key:** `any-value`

## API Endpoints

| Endpoint | Protocol | Used By |
|----------|----------|---------|
| `POST /v1/messages` | Anthropic Messages API | Claude Code |
| `POST /v1/messages/count_tokens` | Anthropic Messages API | Claude Code |
| `POST /v1/chat/completions` | OpenAI Chat Completions | Continue.dev, Cursor, aider, Open WebUI |
| `GET /v1/models` | OpenAI Models | All OpenAI-compatible clients |
| `POST /v1/embeddings` | OpenAI Embeddings | Embedding clients |

## Notes

- The `model` field in requests is accepted but ignored — LLMPlayer always uses the currently loaded model.
- Bearer tokens / API keys are accepted and ignored (any value works).
- LLMPlayer supports streaming for both APIs (SSE format).
- Tool calling is supported via system prompt injection — the model must be capable enough to follow tool-use instructions.
- Only one generation can run at a time. Concurrent requests return HTTP 429.
