# Tool Calling

LLMPlayer supports function/tool calling via the OpenAI-compatible API (`/v1/chat/completions`). When tools are provided in the request, the model can choose to call one or more functions instead of generating a text response.

Tool calling is **architecture-aware**: SmolLM3 uses its native Hermes-style XML format (`<tool_call>` tags), while other models use a generic JSON-based prompt injection. This happens transparently — the API interface is identical regardless of the model.

## Supported Models

| Model | Tool Format | Notes |
|-------|------------|-------|
| SmolLM3 | Native Hermes XML (`<tool_call>` tags) | Best tool calling support — purpose-built agentic model |
| Qwen3 | Generic JSON | Works via system prompt injection |
| Qwen2 | Generic JSON | Works via system prompt injection |
| Llama 3 | Generic JSON | Works via system prompt injection |
| Phi-4 | Generic JSON | Works via system prompt injection |
| Any other | Generic JSON | Works via system prompt injection |

SmolLM3 is recommended for tool calling tasks — it was specifically trained for agentic function calling and produces the most reliable tool call outputs.

## Quick Start

### 1. Start the server

```bash
./run.sh --web --port 8080
```

Then load a model via the web UI at `http://localhost:8080` or via API:

```bash
curl -X POST http://localhost:8080/api/models/load \
  -H "Content-Type: application/json" \
  -d '{"path": "gguf/SmolLM3-Q4_K_M.gguf", "contextLength": 2048}'
```

### 2. Make a tool calling request

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SmolLM3",
    "messages": [
      {"role": "user", "content": "What is the weather in Rome?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a city",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {
                "type": "string",
                "description": "The city name"
              }
            },
            "required": ["city"]
          }
        }
      }
    ]
  }'
```

### 3. Response with tool call

The model responds with `tool_calls` instead of text content:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_xyz789",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"Rome\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 12,
    "total_tokens": 57
  }
}
```

### 4. Send the tool result back

After executing the function, send the result back:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SmolLM3",
    "messages": [
      {"role": "user", "content": "What is the weather in Rome?"},
      {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_xyz789",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"Rome\"}"
            }
          }
        ]
      },
      {
        "role": "tool",
        "tool_call_id": "call_xyz789",
        "content": "{\"temperature\": 24, \"condition\": \"sunny\", \"humidity\": 45}"
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a city",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"}
            },
            "required": ["city"]
          }
        }
      }
    ]
  }'
```

The model then generates a natural language response using the tool result:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "The weather in Rome is sunny with a temperature of 24°C and 45% humidity."
      },
      "finish_reason": "stop"
    }
  ]
}
```

## Complete Examples

### Example 1: Calculator

A simple calculator tool that the model can use for math operations.

**Define the tool:**

```json
{
  "type": "function",
  "function": {
    "name": "calculate",
    "description": "Perform a mathematical calculation",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string",
          "description": "The math expression to evaluate, e.g. '2 + 3 * 4'"
        }
      },
      "required": ["expression"]
    }
  }
}
```

**Full conversation flow with `curl`:**

```bash
# Step 1: User asks a math question
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SmolLM3",
    "messages": [
      {"role": "user", "content": "What is 15% of 340?"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {
          "type": "object",
          "properties": {
            "expression": {"type": "string", "description": "Math expression"}
          },
          "required": ["expression"]
        }
      }
    }]
  }' | python3 -m json.tool
```

The model responds with:
```json
{
  "choices": [{
    "message": {
      "tool_calls": [{
        "id": "call_abc",
        "function": {"name": "calculate", "arguments": "{\"expression\": \"340 * 0.15\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

```bash
# Step 2: Send the calculation result back
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SmolLM3",
    "messages": [
      {"role": "user", "content": "What is 15% of 340?"},
      {"role": "assistant", "content": null, "tool_calls": [{"id": "call_abc", "type": "function", "function": {"name": "calculate", "arguments": "{\"expression\": \"340 * 0.15\"}"}}]},
      {"role": "tool", "tool_call_id": "call_abc", "content": "51.0"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
      }
    }]
  }' | python3 -m json.tool
```

Response: `"15% of 340 is 51."`

### Example 2: Multiple Tools

Define multiple tools and let the model choose which one to call.

```bash
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SmolLM3",
    "messages": [
      {"role": "user", "content": "Search for recent news about Java 25"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "web_search",
          "description": "Search the web for information",
          "parameters": {
            "type": "object",
            "properties": {
              "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather for a city",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"}
            },
            "required": ["city"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "send_email",
          "description": "Send an email",
          "parameters": {
            "type": "object",
            "properties": {
              "to": {"type": "string", "description": "Recipient email"},
              "subject": {"type": "string"},
              "body": {"type": "string"}
            },
            "required": ["to", "subject", "body"]
          }
        }
      }
    ]
  }' | python3 -m json.tool
```

The model will choose `web_search` with `{"query": "Java 25 news"}` because it's the most relevant tool for the request.

### Example 3: Streaming with Tool Calls

Tool calling works with streaming too. The `finish_reason` in the final chunk indicates whether the model made a tool call.

```bash
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SmolLM3",
    "messages": [
      {"role": "user", "content": "What time is it in Tokyo?"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_time",
        "description": "Get the current time in a timezone",
        "parameters": {
          "type": "object",
          "properties": {
            "timezone": {"type": "string", "description": "IANA timezone, e.g. Asia/Tokyo"}
          },
          "required": ["timezone"]
        }
      }
    }],
    "stream": true
  }'
```

Streaming output:
```
data: {"choices":[{"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"choices":[{"delta":{"content":"<tool_call>"},"finish_reason":null}]}

data: {"choices":[{"delta":{"content":"{\"name\": \"get_time\", ...}"},"finish_reason":null}]}

data: {"choices":[{"delta":{"content":"</tool_call>"},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]
```

When `finish_reason` is `"tool_calls"`, parse the accumulated text to extract the tool call, execute the function, and send the result back.

### Example 4: Tool Calling with Thinking Mode

Combine thinking/reasoning with tool calling for more reliable function selection:

```bash
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SmolLM3",
    "thinking": true,
    "messages": [
      {"role": "user", "content": "I need to convert 100 USD to EUR. What is the current exchange rate?"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_exchange_rate",
        "description": "Get the exchange rate between two currencies",
        "parameters": {
          "type": "object",
          "properties": {
            "from": {"type": "string", "description": "Source currency code"},
            "to": {"type": "string", "description": "Target currency code"}
          },
          "required": ["from", "to"]
        }
      }
    }]
  }' | python3 -m json.tool
```

The model will first reason about which tool to use, then make the call.

## Python Client Example

Using the standard `openai` Python library:

```python
from openai import OpenAI
import json

# Point to LLMPlayer
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # LLMPlayer accepts any key
)

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# Simulated tool implementation
def get_weather(city: str) -> dict:
    # In a real app, call a weather API
    return {"temperature": 22, "condition": "cloudy", "city": city}

# Start conversation
messages = [{"role": "user", "content": "What's the weather like in Milan?"}]

# First call: model may request a tool call
response = client.chat.completions.create(
    model="SmolLM3",
    messages=messages,
    tools=tools
)

choice = response.choices[0]

# Check if the model wants to call a tool
if choice.finish_reason == "tool_calls":
    # Execute each tool call
    for tool_call in choice.message.tool_calls:
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)

        print(f"Model called: {fn_name}({fn_args})")

        # Execute the function
        if fn_name == "get_weather":
            result = get_weather(**fn_args)
        else:
            result = {"error": f"Unknown function: {fn_name}"}

        # Add assistant message with tool calls
        messages.append(choice.message)

        # Add tool result
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

    # Second call: model generates response using tool result
    response = client.chat.completions.create(
        model="SmolLM3",
        messages=messages,
        tools=tools
    )
    print(response.choices[0].message.content)
else:
    # Model responded directly without calling a tool
    print(choice.message.content)
```

Output:
```
Model called: get_weather({'city': 'Milan'})
The weather in Milan is cloudy with a temperature of 22°C.
```

## Java Client Example

Using `java.net.http` (no external dependencies):

```java
import java.net.URI;
import java.net.http.*;
import java.net.http.HttpResponse.BodyHandlers;

public class ToolCallingExample {

    static final String BASE_URL = "http://localhost:8080";
    static final HttpClient client = HttpClient.newHttpClient();

    public static void main(String[] args) throws Exception {
        // Step 1: Ask a question with tools available
        String request1 = """
            {
              "model": "SmolLM3",
              "messages": [
                {"role": "user", "content": "What is the weather in Rome?"}
              ],
              "tools": [{
                "type": "function",
                "function": {
                  "name": "get_weather",
                  "description": "Get current weather for a city",
                  "parameters": {
                    "type": "object",
                    "properties": {
                      "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                  }
                }
              }]
            }
            """;

        String response1 = post("/v1/chat/completions", request1);
        System.out.println("Response 1: " + response1);

        // Check if finish_reason is "tool_calls"
        if (response1.contains("\"tool_calls\"")) {
            // Step 2: Execute the function and send result back
            // (Parse tool_call from response1, execute get_weather("Rome"))
            String weatherResult = "{\"temperature\": 28, \"condition\": \"sunny\"}";

            String request2 = """
                {
                  "model": "SmolLM3",
                  "messages": [
                    {"role": "user", "content": "What is the weather in Rome?"},
                    {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\\"city\\": \\"Rome\\"}"}}]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "%s"}
                  ],
                  "tools": [{
                    "type": "function",
                    "function": {
                      "name": "get_weather",
                      "description": "Get current weather for a city",
                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
                    }
                  }]
                }
                """.formatted(weatherResult);

            String response2 = post("/v1/chat/completions", request2);
            System.out.println("Response 2: " + response2);
        }
    }

    static String post(String path, String body) throws Exception {
        HttpRequest req = HttpRequest.newBuilder()
            .uri(URI.create(BASE_URL + path))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(body))
            .build();
        return client.send(req, BodyHandlers.ofString()).body();
    }
}
```

## JavaScript/Node.js Example

```javascript
const BASE_URL = "http://localhost:8080";

// Tool implementations
const tools = {
  get_weather: async (args) => {
    // In a real app, call a weather API
    return { temperature: 18, condition: "rainy", city: args.city };
  },
  calculate: async (args) => {
    return { result: eval(args.expression) };  // simplified
  }
};

// Tool definitions for the API
const toolDefs = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Get current weather for a city",
      parameters: {
        type: "object",
        properties: { city: { type: "string" } },
        required: ["city"]
      }
    }
  },
  {
    type: "function",
    function: {
      name: "calculate",
      description: "Evaluate a math expression",
      parameters: {
        type: "object",
        properties: { expression: { type: "string" } },
        required: ["expression"]
      }
    }
  }
];

async function chat(userMessage) {
  let messages = [{ role: "user", content: userMessage }];

  // Loop to handle multi-step tool calling
  while (true) {
    const response = await fetch(`${BASE_URL}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "SmolLM3",
        messages,
        tools: toolDefs
      })
    });

    const data = await response.json();
    const choice = data.choices[0];

    if (choice.finish_reason === "tool_calls") {
      // Add assistant message with tool calls
      messages.push(choice.message);

      // Execute each tool call
      for (const tc of choice.message.tool_calls) {
        const args = JSON.parse(tc.function.arguments);
        console.log(`Calling ${tc.function.name}(${JSON.stringify(args)})`);

        const result = await tools[tc.function.name](args);

        messages.push({
          role: "tool",
          tool_call_id: tc.id,
          content: JSON.stringify(result)
        });
      }
      // Continue the loop — model will process tool results
    } else {
      // Model gave a text response — done
      console.log(choice.message.content);
      return choice.message.content;
    }
  }
}

// Usage
chat("What is the weather in Berlin and what is 25 * 4?");
```

## How It Works Internally

### Tool prompt injection

When tools are provided in the request, LLMPlayer injects a system prompt describing the available tools. The format depends on the model architecture:

**SmolLM3 (Hermes XML format):**
```
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools:
<tools>
{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

**Other models (generic JSON format):**
```
You have access to the following tools:

Function: get_weather
Description: Get current weather
Parameters: {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}

When you need to call a tool, respond ONLY with a JSON object in this exact format:
{"name": "function_name", "arguments": {"arg1": "value1"}}
Do not include any other text when making a tool call.
```

### Response parsing

LLMPlayer's response parser detects tool calls in the model output using two strategies:

1. **XML tag detection** (SmolLM3/Hermes): looks for `<tool_call>...</tool_call>` tags, extracts the JSON inside each tag. Supports multiple tool calls in a single response.

2. **Bare JSON detection** (fallback for all models): looks for a JSON object with a `"name"` field matching one of the defined tool names.

If tool calls are detected, the response uses `finish_reason: "tool_calls"` and the `message.tool_calls` array instead of `message.content`.

### Multi-turn tool result formatting

Tool results sent back to the model are formatted differently per architecture:

**SmolLM3:**
```
<|im_start|>tool
<tool_response>
{"temperature": 24, "condition": "sunny"}
</tool_response><|im_end|>
```

**Other models:**
```
<|start_header_id|>user<|end_header_id|>

[Tool result for call_xyz]: {"temperature": 24, "condition": "sunny"}<|eot_id|>
```

## API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tools` | Array | — | List of tool definitions (OpenAI format) |
| `tool_choice` | String/Object | `"auto"` | `"auto"` = model decides, `"none"` = no tools |
| `thinking` | Boolean | false | Enable thinking/reasoning before tool selection |
| `stream` | Boolean | false | Stream the response (tool calls visible in stream) |

## Tips

- **Use SmolLM3** for the best tool calling experience — it was purpose-built for agentic tasks
- **Provide clear descriptions** in your tool definitions — the model uses these to decide which tool to call
- **Keep parameter schemas simple** — use basic types (string, number, boolean) with clear descriptions
- **Handle the loop** — after sending a tool result, the model might call another tool. Implement a loop (see JavaScript example above)
- **Set reasonable `max_tokens`** — tool call outputs are short (typically 20-50 tokens), so 100-200 is usually enough for the first call
- **Combine with thinking mode** — for complex tool selection decisions, enable `"thinking": true` to improve accuracy
