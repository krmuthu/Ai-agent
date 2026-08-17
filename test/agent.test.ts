import assert from 'node:assert/strict';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import OpenAI from 'openai';
import { Agent, CONTEXT_LIMIT_MESSAGE } from '../src/agent.ts';
import { loadConfig, type AgentConfig } from '../src/config.ts';
import type { LlmClient } from '../src/llm.ts';
import { ToolRegistry } from '../src/tools.ts';
import type { AgentEvent } from '../src/types.ts';

type Reply = Awaited<ReturnType<LlmClient['complete']>>;

function assistant(content: string | null, toolCalls: Reply['tool_calls'] = undefined): Reply {
  return { role: 'assistant', content, refusal: null, ...(toolCalls ? { tool_calls: toolCalls } : {}) } as Reply;
}

function scriptedLlm(replies: readonly (Reply | Error)[]): LlmClient & { calls: number } {
  let calls = 0;
  const client = {
    model: 'test-model',
    get calls(): number {
      return calls;
    },
    async complete(): Promise<Reply> {
      const reply = replies[Math.min(calls, replies.length - 1)];
      calls += 1;
      if (reply instanceof Error) throw reply;
      if (reply === undefined) throw new Error('no scripted reply');
      return reply;
    },
  };
  return client as LlmClient & { calls: number };
}

async function testConfig(overrides: Partial<AgentConfig> = {}): Promise<AgentConfig> {
  const workspace = await mkdtemp(join(tmpdir(), 'agent-test-'));
  return loadConfig({}, { workspace, maxIterations: 5, ...overrides });
}

test('the loop stops when the model answers without tool calls', async () => {
  const config = await testConfig();
  const llm = scriptedLlm([assistant('all done')]);
  const result = await new Agent({ config, llm }).run('say hi');

  assert.equal(result.status, 'completed');
  assert.equal(result.output, 'all done');
  assert.equal(result.iterations, 1);
});

test('tool calls are executed and their results appended to the history', async () => {
  const config = await testConfig();
  const llm = scriptedLlm([
    assistant(null, [
      {
        id: 'call_1',
        type: 'function',
        function: { name: 'write_file', arguments: JSON.stringify({ path: 'a.txt', content: 'hello' }) },
      },
    ]),
    assistant('wrote the file'),
  ]);

  const events: AgentEvent[] = [];
  const agent = new Agent({ config, llm, onEvent: (event) => events.push(event) });
  const result = await agent.run('create a.txt');

  assert.equal(result.status, 'completed');
  const toolMessage = result.messages.find((message) => message.role === 'tool');
  assert.ok(toolMessage, 'expected a tool result message');
  assert.match(String(toolMessage.content), /Wrote "a\.txt"/);
  assert.ok(events.some((event) => event.type === 'tool_result' && event.ok));
});

test('invalid tool arguments come back as a recoverable error, not a crash', async () => {
  const registry = new ToolRegistry();
  const context = {
    workspace: await mkdtemp(join(tmpdir(), 'agent-test-')),
    commandTimeoutMs: 5_000,
    maxOutputChars: 1_000,
    mcpServerUrl: undefined,
    signal: AbortSignal.timeout(10_000),
  };

  const missingField = await registry.execute('read_file', '{}', context);
  assert.equal(missingField.ok, false);
  assert.match(missingField.content, /Invalid arguments for "read_file"/);

  const escape = await registry.execute('read_file', JSON.stringify({ path: '../../etc/passwd' }), context);
  assert.equal(escape.ok, false);
  assert.match(escape.content, /outside the agent workspace/);
});

test('MAX_ITERATIONS stops a model that never stops calling tools', async () => {
  const config = await testConfig({ maxIterations: 3 });
  const llm = scriptedLlm([
    assistant(null, [
      { id: 'call_loop', type: 'function', function: { name: 'execute_command', arguments: '{"command":"true"}' } },
    ]),
  ]);

  const result = await new Agent({ config, llm }).run('loop forever');
  assert.equal(result.status, 'max_iterations');
  assert.equal(result.iterations, 3);
});

test('an unrecoverable context overflow refuses gracefully', async () => {
  const config = await testConfig();
  const overflow = new OpenAI.APIError(
    400,
    { error: { code: 'context_length_exceeded', message: "This model's maximum context length is 32768 tokens" } },
    'context_length_exceeded',
    undefined,
  );
  const llm = scriptedLlm([overflow]);

  const events: AgentEvent[] = [];
  const result = await new Agent({ config, llm, onEvent: (event) => events.push(event) }).run('big task');

  assert.equal(result.status, 'context_overflow');
  assert.equal(result.output, CONTEXT_LIMIT_MESSAGE);
  assert.ok(events.some((event) => event.type === 'error' && event.message === CONTEXT_LIMIT_MESSAGE));
});

test('execute_command captures output and enforces its timeout', async () => {
  const registry = new ToolRegistry();
  const context = {
    workspace: await mkdtemp(join(tmpdir(), 'agent-test-')),
    commandTimeoutMs: 1_000,
    maxOutputChars: 10_000,
    mcpServerUrl: undefined,
    signal: AbortSignal.timeout(30_000),
  };

  const ok = await registry.execute('execute_command', JSON.stringify({ command: 'echo hello' }), context);
  assert.equal(ok.ok, true);
  assert.match(ok.content, /hello/);

  const timedOut = await registry.execute('execute_command', JSON.stringify({ command: 'sleep 5' }), context);
  assert.equal(timedOut.ok, false);
  assert.match(timedOut.content, /timed out after 1000ms/);
});
