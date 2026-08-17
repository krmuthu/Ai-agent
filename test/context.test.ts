import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  compactMessages,
  countTokens,
  estimateTokens,
  needsCompaction,
  type CompactionOptions,
} from '../src/context.ts';
import type { ChatMessage } from '../src/types.ts';

const options: CompactionOptions = {
  contextTokens: 1_000,
  compactionThreshold: 0.85,
  keepRecentTurns: 2,
};
const budget = options.contextTokens * options.compactionThreshold;

function turn(index: number, payloadChars: number): ChatMessage[] {
  return [
    { role: 'user', content: `task ${index}` },
    {
      role: 'assistant',
      content: null,
      tool_calls: [
        {
          id: `call_${index}`,
          type: 'function',
          function: { name: 'execute_command', arguments: JSON.stringify({ command: `echo ${index}` }) },
        },
      ],
    },
    { role: 'tool', tool_call_id: `call_${index}`, content: 'x'.repeat(payloadChars) },
  ];
}

test('estimateTokens grows with text length and is zero for empty input', () => {
  assert.equal(estimateTokens(''), 0);
  assert.ok(estimateTokens('a'.repeat(360)) > estimateTokens('a'.repeat(36)));
});

test('needsCompaction triggers at the configured threshold', () => {
  const small: ChatMessage[] = [{ role: 'user', content: 'hi' }];
  assert.equal(needsCompaction(small, options), false);
  const big: ChatMessage[] = [{ role: 'user', content: 'x'.repeat(10_000) }];
  assert.equal(needsCompaction(big, options), true);
});

test('compaction keeps the system prompt and the last N turns intact', async () => {
  const messages: ChatMessage[] = [
    { role: 'system', content: 'SYSTEM PROMPT' },
    ...turn(1, 4_000),
    ...turn(2, 4_000),
    ...turn(3, 200),
    ...turn(4, 200),
  ];

  const result = await compactMessages(messages, options);

  assert.equal(result.messages[0]?.content, 'SYSTEM PROMPT');
  assert.match(String(result.messages[1]?.content), /^\[compacted-history]/);
  assert.ok(result.tokensAfter < result.tokensBefore);
  assert.ok(result.droppedMessages > 0);

  const users = result.messages.filter((message) => message.role === 'user');
  assert.deepEqual(users.map((message) => message.content), ['task 3', 'task 4']);
});

test('compaction never orphans a tool message from its assistant call', async () => {
  const messages: ChatMessage[] = [{ role: 'system', content: 'S' }, ...turn(1, 9_000), ...turn(2, 9_000)];
  const result = await compactMessages(messages, { ...options, keepRecentTurns: 1 });

  const ids = new Set<string>();
  for (const message of result.messages) {
    if (message.role === 'assistant') {
      for (const call of message.tool_calls ?? []) ids.add(call.id);
    }
    if (message.role === 'tool') assert.ok(ids.has(message.tool_call_id));
  }
});

test('an LLM summariser replaces the deterministic digest', async () => {
  const messages: ChatMessage[] = [{ role: 'system', content: 'S' }, ...turn(1, 5_000), ...turn(2, 100)];
  const result = await compactMessages(messages, {
    ...options,
    keepRecentTurns: 1,
    summarize: async () => 'model-written summary',
  });
  assert.equal(result.messages[1]?.content, '[compacted-history] model-written summary');
});

test('compaction drives the transcript under the token budget', async () => {
  const messages: ChatMessage[] = [
    { role: 'system', content: 'S' },
    ...turn(1, 20_000),
    ...turn(2, 20_000),
    ...turn(3, 20_000),
  ];
  const result = await compactMessages(messages, options);
  assert.ok(countTokens(result.messages) <= budget);
  assert.equal(result.withinBudget, true);
});

test('a long history whose digest alone would overflow still fits', async () => {
  const messages: ChatMessage[] = [{ role: 'system', content: 'S' }];
  for (let index = 0; index < 200; index += 1) messages.push(...turn(index, 300));

  const result = await compactMessages(messages, options);
  assert.ok(countTokens(result.messages) <= budget);
  assert.equal(result.withinBudget, true);
});

test('an oversized model-written summary is clipped to fit', async () => {
  const messages: ChatMessage[] = [{ role: 'system', content: 'S' }, ...turn(1, 5_000), ...turn(2, 100)];
  const result = await compactMessages(messages, {
    ...options,
    keepRecentTurns: 1,
    summarize: async () => 'S'.repeat(100_000),
  });

  assert.match(String(result.messages[1]?.content), /^\[compacted-history]/);
  assert.ok(countTokens(result.messages) <= budget);
});

test('a single message larger than the whole window is clamped, not passed through', async () => {
  const hugeTask: ChatMessage[] = [{ role: 'system', content: 'S' }, { role: 'user', content: 'z'.repeat(50_000) }];
  const task = await compactMessages(hugeTask, options);
  assert.ok(countTokens(task.messages) <= budget);
  assert.match(String(task.messages.at(-1)?.content), /characters pruned by compaction]$/);

  const hugeAnswer: ChatMessage[] = [
    { role: 'system', content: 'S' },
    { role: 'user', content: 'go' },
    { role: 'assistant', content: 'y'.repeat(50_000) },
  ];
  const answer = await compactMessages(hugeAnswer, options);
  assert.ok(countTokens(answer.messages) <= budget);
});

test('oversized tool-call arguments are pruned to valid JSON without orphaning results', async () => {
  const messages: ChatMessage[] = [
    { role: 'system', content: 'S' },
    { role: 'user', content: 'write it' },
    {
      role: 'assistant',
      content: null,
      tool_calls: [
        {
          id: 'call_big',
          type: 'function',
          function: { name: 'write_file', arguments: JSON.stringify({ path: 'a.txt', content: 'q'.repeat(60_000) }) },
        },
      ],
    },
    { role: 'tool', tool_call_id: 'call_big', content: 'ok' },
  ];

  const result = await compactMessages(messages, options);
  assert.ok(countTokens(result.messages) <= budget);

  const assistantMessage = result.messages.find((message) => message.role === 'assistant');
  assert.ok(assistantMessage?.role === 'assistant');
  const call = assistantMessage.tool_calls?.[0];
  assert.ok(call !== undefined && call.type === 'function');
  assert.equal(call.id, 'call_big');
  assert.deepEqual(Object.keys(JSON.parse(call.function.arguments) as object), ['pruned_by_compaction']);
  assert.ok(result.messages.some((message) => message.role === 'tool' && message.tool_call_id === 'call_big'));
});

test('repeated compactions keep exactly one history summary', async () => {
  let messages: ChatMessage[] = [{ role: 'system', content: 'SYSTEM PROMPT' }];

  for (let round = 0; round < 5; round += 1) {
    for (let index = 0; index < 4; index += 1) messages.push(...turn(round * 4 + index, 3_000));
    const result = await compactMessages(messages, { ...options, summarize: async () => 'S'.repeat(5_000) });
    messages = result.messages;

    assert.equal(messages[0]?.content, 'SYSTEM PROMPT');
    const summaries = messages.filter(
      (message) => message.role === 'system' && String(message.content).startsWith('[compacted-history]'),
    );
    assert.equal(summaries.length, 1);
    assert.ok(countTokens(messages) <= budget);
    assert.equal(result.withinBudget, true);
  }
});

test('a system prompt bigger than the window reports withinBudget false and stays intact', async () => {
  const messages: ChatMessage[] = [
    { role: 'system', content: 'S'.repeat(20_000) },
    { role: 'user', content: 'go' },
  ];
  const result = await compactMessages(messages, options);

  assert.equal(result.messages[0]?.content, 'S'.repeat(20_000));
  assert.equal(result.withinBudget, false);
});
