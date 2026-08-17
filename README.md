# ai-agent

An autonomous coding agent in Node 24 + TypeScript: a manual ReAct loop, strict
Zod-validated tools, sliding-window context compaction, and an OpenAI-compatible
client you can point at a local open-weight model.

No build step is required to run it — Node 24 executes the TypeScript sources
directly via type stripping.

## Quick start

```bash
npm install
cp .env.example .env          # point OPENAI_BASE_URL at your model server
npm run dev -- "Add a health endpoint to src/server.ts and run the tests"
```

Or build and run the compiled CLI:

```bash
npm run build && node dist/index.js --help
```

Useful flags: `--model`, `--base-url`, `--workspace`, `--max-iterations`,
`--context-tokens`, `--task-file`, `--log-level debug`.

Works with any OpenAI-compatible server:

| Server    | `OPENAI_BASE_URL`            |
| --------- | ---------------------------- |
| Ollama    | `http://localhost:11434/v1`  |
| vLLM      | `http://localhost:8000/v1`   |
| llama.cpp | `http://localhost:8080/v1`   |
| LM Studio | `http://localhost:1234/v1`   |
| OpenAI    | `https://api.openai.com/v1`  |

## How the loop works

```
user task ──▶ [compact if ≥85% of context] ──▶ LLM ──┬─ tool_calls? ─▶ validate (Zod) ─▶ execute ─▶ append tool results ─┐
                     ▲                                │                                                                  │
                     └──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                      └─ plain text ──▶ done
```

Exit conditions, all returned as an `AgentResult.status`:

| status             | meaning                                                        |
| ------------------ | -------------------------------------------------------------- |
| `completed`        | model replied with text and no tool calls                        |
| `max_iterations`   | hit the iteration safeguard (default 20)                         |
| `context_overflow` | provider rejected the prompt even after emergency compaction     |
| `aborted`          | `SIGINT`/`SIGTERM` or a caller-supplied `AbortSignal`            |
| `failed`           | unrecoverable transport or provider error                        |

## Context management

`src/context.ts` estimates tokens (~3.6 chars/token, rounded up) and compacts as
soon as the transcript reaches `AGENT_COMPACTION_THRESHOLD` (85%) of
`AGENT_CONTEXT_TOKENS` (32k by default):

1. keep the system prompt and the last `AGENT_KEEP_RECENT_TURNS` turns verbatim;
2. replace everything older with one synthesized summary — written by the model
   itself, with a deterministic digest as fallback;
3. truncate bulky tool output in the older kept turns;
4. if still over budget, evict the oldest kept turns, then hard-truncate.

Tool results always stay attached to the assistant message that requested them,
so the compacted transcript remains a valid chat-completions payload.

Because token counting is an estimate, the provider stays the source of truth:
if the API returns a context-length `400`, the agent compacts hard against a
halved budget and retries once. If that still overflows it stops and prints

```
Error: Context limit exceeded. Pruning failed or task is too complex.
```

## Tools

| tool              | description                                                                    |
| ----------------- | ------------------------------------------------------------------------------ |
| `execute_command` | shell command in the workspace; process-group kill on timeout, output truncated |
| `read_file`       | UTF-8 read with optional line window                                             |
| `write_file`      | create/overwrite/append, parent directories created                              |
| `mcp_call`        | placeholder MCP bridge (JSON-RPC `tools/call` over `MCP_SERVER_URL`)             |

Every tool declares a Zod schema; the JSON Schema sent to the model is derived
from it (`z.toJSONSchema`), so the contract can never drift. Invalid arguments
and runtime failures come back as tool results, letting the model self-correct.
All paths are confined to the workspace root.

## Embedding the agent

```ts
import { Agent } from './src/agent.ts';
import { loadConfig } from './src/config.ts';
import { ToolRegistry, builtinTools } from './src/tools.ts';

const registry = new ToolRegistry([...builtinTools, myTool]);
const agent = new Agent({
  config: loadConfig(process.env, { maxIterations: 40 }),
  registry,
  onEvent: (event) => console.error(event.type),
});

const result = await agent.run('Refactor the parser and keep the tests green');
```

## Development

```bash
npm run lint && npm run typecheck && npm test && npm run build
```

See [`.windsurfrules`](./.windsurfrules) for the invariants any change must
preserve.
