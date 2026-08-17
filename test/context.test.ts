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
  assert.ok(countTokens(result.messages) <= options.contextTokens * options.compactionThreshold);
});
