# LLMPlayer REST API

LLMPlayer exposes three groups of REST APIs when started with `--web` (default port 8080):

- **`/v1/*`** — OpenAI Chat Completions compatible API. Works with standard OpenAI clients (Open WebUI, LangChain, LiteLLM, Cursor, Continue.dev, etc.).
- **`/api/*`** — LLMPlayer-specific management API for model loading, GPU configuration, and hardware diagnostics.
- **`/api/chats/*`** — Chat persistence API with conversation branching. Used by the chat UI at `/chat`.

All endpoints support CORS (`Access-Control-Allow-Origin: *`).

---

## Starting the Server

```bash
# Start with web UI + API
./run.sh --web

# Custom port
./run.sh --web --port 9090
```

The server serves the model config UI at `http://localhost:8080/`, the chat UI at `http://localhost:8080/chat`, and the APIs at `/v1/*`, `/api/*`, and `/api/chats/*`. GGUF model files should be placed in the `gguf/` directory relative to the working directory.

---

## OpenAI-Compatible API (`/v1/*`)

These endpoints follow the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat) specification. The `Authorization: Bearer <token>` header is accepted and ignored (no authentication required).

### POST `/v1/chat/completions`

Generates a model response from a multi-turn conversation.

#### Request

```json
{
  "model": "any-string",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
  ],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 256,
  "top_p": 0.9,
  "stop": ["\n\n"],
  "frequency_penalty": 0.0
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | — | Accepted for compatibility, ignored (uses the loaded model) |
| `messages` | array | **required** | Array of `{role, content}` messages. Roles: `system`, `user`, `assistant` |
| `stream` | boolean | `false` | If `true`, response is streamed via SSE |
| `temperature` | float | `0.7` | Sampling temperature (0.0 = greedy, 2.0 = max) |
| `max_tokens` | int | `256` | Maximum number of tokens to generate |
| `max_completion_tokens` | int | `256` | Alias for `max_tokens` (OpenAI v2 compatibility) |
| `top_p` | float | `0.9` | Nucleus sampling |
| `top_k` | int | `40` | Top-K sampling (extension, not standard OpenAI) |
| `stop` | string or array | `null` | Stop sequences. Generation stops when the output text contains any of these strings |
| `frequency_penalty` | float | `0.0` | OpenAI frequency penalty. Mapped internally to `repetition_penalty = 1.0 + frequency_penalty * 0.5` |
| `repetition_penalty` | float | `1.1` | Direct repetition penalty (extension, takes precedence over `frequency_penalty`) |

#### Non-streaming Response (`stream: false`)

```
HTTP/1.1 200 OK
Content-Type: application/json
```

```json
{
  "id": "chatcmpl-abc123def456789012345678901",
  "object": "chat.completion",
  "created": 1739654400,
  "model": "Qwen2.5-3B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 9,
    "total_tokens": 34
  }
}
```

| `finish_reason` value | Meaning |
|------------------------|---------|
| `"stop"` | Generation completed (EOS token or stop sequence reached) |
| `"length"` | Generation truncated at `max_tokens` |

#### Streaming Response (`stream: true`)

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
```

The stream sends a series of SSE events. Each event is a `data: <json>` line followed by an empty line.

**1. Initial chunk** (role):
```
data: {"id":"chatcmpl-abc123...","object":"chat.completion.chunk","created":1739654400,"model":"Qwen2.5-3B-Instruct","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

```

**2. Content chunk** (one per generated token):
```
data: {"id":"chatcmpl-abc123...","object":"chat.completion.chunk","created":1739654400,"model":"Qwen2.5-3B-Instruct","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

```

**3. Final chunk** (finish reason):
```
data: {"id":"chatcmpl-abc123...","object":"chat.completion.chunk","created":1739654400,"model":"Qwen2.5-3B-Instruct","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

```

**4. Usage chunk** (token statistics):
```
data: {"id":"chatcmpl-abc123...","object":"chat.completion.chunk","created":1739654400,"model":"Qwen2.5-3B-Instruct","choices":[],"usage":{"prompt_tokens":25,"completion_tokens":9,"total_tokens":34}}

```

**5. Stream terminator**:
```
data: [DONE]

```

To interrupt generation during streaming, the client can close the SSE connection.

#### Errors

OpenAI error format:

```json
{
  "error": {
    "message": "No model loaded. Load a model first.",
    "type": "invalid_request_error",
    "code": null
  }
}
```

| HTTP Status | Error Type | Cause |
|-------------|------------|-------|
| 400 | `invalid_request_error` | Missing or invalid parameters |
| 405 | `invalid_request_error` | Unsupported HTTP method |
| 429 | `rate_limit_error` | Generation already in progress |
| 503 | `invalid_request_error` | No model loaded |
| 500 | `invalid_request_error` | Internal error |

---

### GET `/v1/models`

Lists available models.

#### Response

If a model is loaded, returns only that model:

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen2.5-3B-Instruct",
      "object": "model",
      "created": 1739654400,
      "owned_by": "local"
    }
  ]
}
```

If no model is loaded, lists available GGUF files in the `gguf/` directory:

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
      "object": "model",
      "created": 1739000000,
      "owned_by": "local"
    },
    {
      "id": "Llama-3.2-3B-Q3_K_L.gguf",
      "object": "model",
      "created": 1738900000,
      "owned_by": "local"
    }
  ]
}
```

---

## Management API (`/api/*`)

LLMPlayer-specific endpoints for model loading/unloading, GPU configuration, and diagnostics. Used by the integrated web UI.

### GET `/api/models`

Lists available GGUF files in the `gguf/` directory with path and size.

#### Response

```json
[
  {
    "name": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    "path": "gguf/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    "size": 2134567890
  },
  {
    "name": "Llama-3.2-3B-Q3_K_L.gguf",
    "path": "gguf/Llama-3.2-3B-Q3_K_L.gguf",
    "size": 1876543210
  }
]
```

---

### POST `/api/models/load`

Loads a GGUF model into memory. Automatically unloads the previous model if one is loaded.

#### Request

```json
{
  "path": "gguf/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
  "contextLength": 2048,
  "gpu": true,
  "gpuDevice": 0,
  "gpuLayers": -1
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | **required** | Path to the GGUF file (relative to working directory or absolute) |
| `contextLength` | int | `2048` | Maximum context length (tokens) |
| `gpu` | boolean | `false` | Enable GPU offloading via OpenCL |
| `gpuDevice` | int | `0` | OpenCL device index |
| `gpuLayers` | int | `-1` | Number of layers on GPU. `-1` = auto (computed from VRAM), `0` = all |

#### Response

```json
{
  "status": "loaded",
  "loadTimeMs": 3450,
  "model": {
    "name": "Qwen2.5-3B-Instruct",
    "architecture": "qwen2",
    "embeddingLength": 2048,
    "blockCount": 36,
    "headCount": 16,
    "headCountKV": 2,
    "contextLength": 2048,
    "vocabSize": 151936,
    "intermediateSize": 11008
  },
  "gpuDevice": "NVIDIA RTX 4050 Laptop GPU",
  "gpuLayers": 36,
  "totalLayers": 36
}
```

The `gpuDevice`, `gpuLayers`, and `totalLayers` fields are only present when GPU is active.

---

### POST `/api/models/unload`

Unloads the current model from memory.

#### Request

Empty body or `{}`.

#### Response

```json
{
  "status": "unloaded"
}
```

---

### GET `/api/models/info`

Returns detailed information about the loaded model, including GPU details.

#### Response

```json
{
  "name": "Qwen2.5-3B-Instruct",
  "architecture": "qwen2",
  "embeddingLength": 2048,
  "blockCount": 36,
  "headCount": 16,
  "headCountKV": 2,
  "contextLength": 2048,
  "vocabSize": 151936,
  "intermediateSize": 11008,
  "gpuLayers": 36,
  "gpuDeviceName": "NVIDIA RTX 4050 Laptop GPU",
  "moeOptimizedGpu": false
}
```

| Field | Description |
|-------|-------------|
| `gpuLayers` | Number of layers offloaded to GPU (`-1` if GPU is not active) |
| `gpuDeviceName` | GPU device name, or `null` if CPU-only |
| `moeOptimizedGpu` | `true` if MoE-optimized strategy is active (attention on GPU, experts on CPU) |

Returns HTTP 400 if no model is loaded.

---

### POST `/api/chat`

Generates a streaming response (legacy LLMPlayer format).

> **Note:** The web UI uses `/v1/chat/completions` instead of this endpoint. Kept for backward compatibility.

#### Request

```json
{
  "prompt": "Hello, how are you?",
  "systemMessage": "You are a helpful assistant.",
  "temperature": 0.7,
  "maxTokens": 256,
  "topK": 40,
  "topP": 0.9,
  "repPenalty": 1.1
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | `""` | User message |
| `systemMessage` | string | `null` | System message (optional) |
| `temperature` | float | `0.7` | Sampling temperature |
| `maxTokens` | int | `256` | Maximum number of tokens |
| `topK` | int | `40` | Top-K sampling |
| `topP` | float | `0.9` | Top-P (nucleus) sampling |
| `repPenalty` | float | `1.1` | Repetition penalty |

#### Response (SSE stream)

Token chunk:
```
data: {"token":"Hello","done":false}
```

Final event with statistics:
```
data: {"done":true,"stats":{"tokenCount":15,"promptTokenCount":28,"tokensPerSecond":5.2,"timeMs":2880}}
```

If generation is interrupted:
```
data: {"done":true,"stats":{...},"stopped":true}
```

Error during generation:
```
data: {"error":"Generation error message","done":true}
```

Returns HTTP 400 if no model is loaded, HTTP 409 if a generation is already in progress.

---

### POST `/api/chat/stop`

Stops the current generation (both `/api/chat` and `/v1/chat/completions`).

#### Request

Empty body or `{}`.

#### Response

```json
{
  "status": "stopping"
}
```

---

### GET `/api/gpu/devices`

Enumerates available OpenCL devices in the system.

#### Response

```json
[
  {
    "index": 0,
    "name": "pthread-Intel(R) Core(TM) Ultra 7 155H",
    "vendor": "GenuineIntel",
    "globalMemory": 16697147392,
    "computeUnits": 22,
    "deviceType": "CPU"
  },
  {
    "index": 1,
    "name": "NVIDIA RTX 4050 Laptop GPU",
    "vendor": "NVIDIA Corporation",
    "globalMemory": 6437404672,
    "computeUnits": 24,
    "deviceType": "GPU"
  }
]
```

Returns an empty array `[]` if OpenCL is not available (Java < 21 or no driver installed).

---

### POST `/api/memory/check`

Checks whether the system has sufficient RAM to load a model.

#### Request

```json
{
  "path": "gguf/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
  "contextLength": 2048
}
```

#### Response

```json
{
  "estimatedRam": 2500000000,
  "availableRam": 16000000000,
  "safe": true,
  "message": "Memory OK: ~2384 MB needed, 15258 MB available"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `estimatedRam` | long | Estimated bytes needed (model + KV cache) |
| `availableRam` | long | Available RAM in bytes |
| `safe` | boolean | `true` if the estimate is < 90% of available RAM |
| `message` | string | Human-readable message |

---

### POST `/api/hardware/plan`

Builds an optimal hardware configuration plan for a model, considering available RAM and GPU.

#### Request

```json
{
  "path": "gguf/Qwen3-Coder-30B-A3B-Q4_K_M.gguf",
  "contextLength": 2048
}
```

#### Response

```json
{
  "modelName": "Qwen3-Coder-30B-A3B",
  "gpuAvailable": true,
  "gpuDeviceName": "NVIDIA RTX 4050 Laptop GPU",
  "gpuVram": 6437404672,
  "gpuLayers": 48,
  "totalLayers": 48,
  "recommended": true,
  "summary": "Model: Qwen3-Coder-30B-A3B (16.5 GB)\nLayers: 48\nRAM: Memory OK...\nGPU: NVIDIA RTX 4050...\nPlan: MoE-optimized...",
  "memorySafe": true,
  "estimatedRam": 17800000000,
  "availableRam": 32000000000
}
```

| Field | Type | Description |
|-------|------|-------------|
| `modelName` | string | Model name from GGUF metadata |
| `gpuAvailable` | boolean | Whether at least one OpenCL device was found |
| `gpuDeviceName` | string | Selected GPU device name (the one with the most VRAM) |
| `gpuVram` | long | VRAM in bytes of the selected device |
| `gpuLayers` | int | Layers placed on GPU by the plan |
| `totalLayers` | int | Total model layers |
| `recommended` | boolean | `true` if the plan is safe (enough RAM) |
| `summary` | string | Human-readable plan summary |
| `memorySafe` | boolean | `true` if RAM is sufficient |
| `estimatedRam` | long | Estimated RAM needed in bytes |
| `availableRam` | long | Available RAM in bytes |

---

## OpenAI Client Configuration

### Open WebUI

```
Settings > Connections > OpenAI API
  URL: http://localhost:8080/v1
  API Key: sk-fake (any value)
```

### Python (openai SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-fake"
)

# Non-streaming
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=256,
    temperature=0.7
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Write a haiku"}],
    max_tokens=50,
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### curl

```bash
# Non-streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'

# Streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50,
    "stream": true
  }'

# With Authorization header (accepted and ignored)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-any-value" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'

# List models
curl http://localhost:8080/v1/models

# Load a model (management API)
curl -X POST http://localhost:8080/api/models/load \
  -H "Content-Type: application/json" \
  -d '{"path": "gguf/model.gguf", "contextLength": 2048}'
```

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-fake",
    model="default",
    temperature=0.7,
    max_tokens=256
)

response = llm.invoke("Hello, how are you?")
print(response.content)
```

---

## Chat Persistence API (`/api/chats/*`)

Server-side conversation persistence with tree-based branching. Conversations are stored as JSON files in the `chats/` directory (created automatically). Used by the chat UI at `/chat`.

### Data Model

Each conversation is a JSON file (`chats/conv_{timestamp}.json`) with a flat message map forming a tree:

```json
{
  "id": "conv_1708012345678",
  "title": "Write a Java factorial class",
  "created": 1708012345678,
  "updated": 1708012400000,
  "settings": {
    "temperature": 0.7,
    "maxTokens": 256,
    "topK": 40,
    "topP": 0.9,
    "repetitionPenalty": 1.1,
    "systemMessage": ""
  },
  "messages": {
    "msg_1": {
      "id": "msg_1",
      "role": "user",
      "content": "Hello",
      "parentId": null,
      "children": ["msg_2"],
      "timestamp": 1708012345680
    },
    "msg_2": {
      "id": "msg_2",
      "role": "assistant",
      "content": "Hi there!",
      "parentId": "msg_1",
      "children": [],
      "timestamp": 1708012345700,
      "stats": {"tokenCount": 45, "promptTokenCount": 12, "tokensPerSecond": 5.2, "timeMs": 8650}
    }
  },
  "rootChildren": ["msg_1"],
  "activeLeafId": "msg_2"
}
```

**Branching:** Editing a user message or regenerating an assistant response creates a new message with the same `parentId` as the original — a sibling in the tree. The chat UI navigates between branches with arrow controls.

---

### GET `/api/chats`

Lists all conversations, sorted by most recently updated.

#### Response

```json
[
  {
    "id": "conv_1708012345678",
    "title": "Write a Java factorial class",
    "created": 1708012345678,
    "updated": 1708012400000,
    "messageCount": 4
  }
]
```

---

### POST `/api/chats`

Creates a new conversation.

#### Request

Empty body or:

```json
{
  "title": "My Chat",
  "settings": {
    "temperature": 0.8,
    "maxTokens": 512
  }
}
```

#### Response (201)

The full conversation object (see Data Model above).

---

### GET `/api/chats/{id}`

Returns the full conversation including the message tree.

#### Response

The full conversation object (see Data Model above). Returns 404 if not found.

---

### DELETE `/api/chats/{id}`

Deletes a conversation.

#### Response

```json
{
  "status": "deleted"
}
```

---

### PUT `/api/chats/{id}/title`

Renames a conversation.

#### Request

```json
{
  "title": "New title"
}
```

#### Response

The updated conversation object.

---

### POST `/api/chats/{id}/messages`

Adds a message to a conversation.

#### Request

```json
{
  "role": "user",
  "content": "Hello, how are you?",
  "parentId": "msg_1",
  "stats": {
    "tokenCount": 45,
    "promptTokenCount": 12,
    "tokensPerSecond": 5.2,
    "timeMs": 8650
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `role` | string | `"user"` or `"assistant"` (required) |
| `content` | string | Message text |
| `parentId` | string or null | Parent message ID. `null` for root messages |
| `stats` | object | Generation statistics (optional, for assistant messages) |

The first user message auto-generates the conversation title (truncated to 50 chars).

#### Response (201)

```json
{
  "message": { "id": "msg_3", "role": "user", "content": "...", ... },
  "conversationId": "conv_1708012345678"
}
```

---

### PUT `/api/chats/{id}/messages/{msgId}`

Edits a message by creating a sibling branch. The original message is preserved.

#### Request

```json
{
  "content": "Edited message text"
}
```

#### Response

```json
{
  "message": { "id": "msg_4", "role": "user", "content": "Edited message text", "parentId": "msg_1", ... },
  "conversationId": "conv_1708012345678"
}
```

The new message has the same `parentId` as the original, making it a sibling in the tree.

---

### PUT `/api/chats/{id}/active-leaf`

Updates the active branch leaf pointer.

#### Request

```json
{
  "activeLeafId": "msg_4"
}
```

#### Response

```json
{
  "activeLeafId": "msg_4"
}
```

---

### PUT `/api/chats/{id}/settings`

Updates per-conversation settings.

#### Request

```json
{
  "temperature": 0.8,
  "maxTokens": 512,
  "systemMessage": "You are a coding assistant."
}
```

#### Response

```json
{
  "settings": { "temperature": 0.8, "maxTokens": 512, ... }
}
```

---

### GET `/api/chats/export/{id}`

Exports a conversation as a downloadable JSON file.

#### Response

Returns the full conversation JSON with `Content-Disposition: attachment` header.

---

## Notes

- **Single model:** LLMPlayer loads one model at a time. The `model` field in requests is ignored; the server always uses the currently loaded model.
- **Single generation:** Only one generation at a time is supported. Concurrent requests receive HTTP 429.
- **Multi-turn conversation:** The `/v1/chat/completions` API is stateless — the client must send the entire message history with each request. The chat UI at `/chat` handles this automatically using the `/api/chats/*` persistence API.
- **BOS token:** Automatically prepended by the server for all chat template formats. Do not include it in messages.
- **Stop sequences:** Stop sequence post-processing occurs server-side. Text is truncated when it contains a stop sequence.
- **Supported architectures:** Llama 3, Qwen2, Qwen3, Qwen3MoE, DeepSeek2, GLM4, Phi-3/4, Mistral3/Devstral. Each architecture uses its own chat template for formatting messages.
