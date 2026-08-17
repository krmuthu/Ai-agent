import type { AssistantMessage, ChatMessage } from './types.ts';

/** Per-message framing overhead charged by chat APIs (role, delimiters). */
const MESSAGE_OVERHEAD_TOKENS = 4;
/** Average characters per token for code-heavy English text. */
const CHARS_PER_TOKEN = 3.6;
const SUMMARY_MARKER = '[compacted-history]';
/** Share of the token budget the history summary may occupy. */
const SUMMARY_BUDGET_FRACTION = 0.25;
/** Smallest body a message is clamped to before the next, harsher pass. */
const MIN_MESSAGE_CHARS = 200;
/** First body size the final clamp tries. */
const CLAMP_START_CHARS = 2_000;

export interface CompactionOptions {
  /** Hard capacity of the model's context window, in tokens. */
  readonly contextTokens: number;
  /** Fraction of `contextTokens` that triggers compaction (e.g. 0.85). */
  readonly compactionThreshold: number;
  /** Number of trailing user-initiated turns kept verbatim. */
  readonly keepRecentTurns: number;
  /** Characters retained per tool result inside older kept turns. */
  readonly maxToolResultChars?: number;
  /** Optional LLM summariser; falls back to a deterministic digest on failure. */
  readonly summarize?: (messages: readonly ChatMessage[]) => Promise<string>;
}

export interface CompactionResult {
  readonly messages: ChatMessage[];
  readonly tokensBefore: number;
  readonly tokensAfter: number;
  /** Number of messages removed or replaced by the summary. */
  readonly droppedMessages: number;
  /** False when even the harshest pruning could not reach the budget. */
  readonly withinBudget: boolean;
}

/**
 * Cheap, dependency-free token estimate. Deliberately conservative: it rounds
 * up so the agent compacts slightly early rather than overflowing the window.
 */
export function estimateTokens(text: string): number {
  if (text.length === 0) return 0;
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

export function countMessageTokens(message: ChatMessage): number {
  let tokens = MESSAGE_OVERHEAD_TOKENS + estimateTokens(message.role);
  tokens += estimateTokens(extractText(message));
  if (message.role === 'assistant' && message.tool_calls) {
    for (const call of message.tool_calls) {
      if (call.type !== 'function') continue;
      tokens += estimateTokens(`${call.function.name}${call.function.arguments}`);
    }
  }
  return tokens;
}

export function countTokens(messages: readonly ChatMessage[]): number {
  return messages.reduce((total, message) => total + countMessageTokens(message), 0);
}

export function compactionBudget(options: CompactionOptions): number {
  return Math.floor(options.contextTokens * options.compactionThreshold);
}

export function needsCompaction(
  messages: readonly ChatMessage[],
  options: CompactionOptions,
): boolean {
  return countTokens(messages) >= compactionBudget(options);
}

/**
 * Sliding-window compaction.
 *
 * Keeps the system prompt and the last `keepRecentTurns` turns intact, replaces
 * everything older with a single synthesized summary, trims bulky tool output
 * from the older kept turns, and — only if the budget is still blown — evicts
 * the oldest kept turns one by one and finally clamps the bodies of whatever
 * survives. Tool results always stay attached to the assistant message that
 * requested them, so the transcript stays API-valid.
 */
export async function compactMessages(
  messages: readonly ChatMessage[],
  options: CompactionOptions,
): Promise<CompactionResult> {
  const tokensBefore = countTokens(messages);
  const budget = compactionBudget(options);
  const { prefix, rest } = splitSystemPrefix(messages);
  const turns = groupIntoTurns(rest);

  const keep = Math.max(1, options.keepRecentTurns);
  const recent = turns.slice(-keep);
  const stale = turns.slice(0, Math.max(0, turns.length - keep));

  const summaryMaxChars = Math.max(
    MIN_MESSAGE_CHARS,
    Math.floor(budget * SUMMARY_BUDGET_FRACTION * CHARS_PER_TOKEN),
  );
  const summary =
    stale.length > 0
      ? clip(await summarizeTurns(stale.flat(), options), summaryMaxChars)
      : undefined;

  let kept = trimHeavyToolResults(recent, options.maxToolResultChars ?? 2_000);
  let assembled = assemble(prefix, summary, kept);

  // Still over budget: evict whole turns from the oldest end.
  while (countTokens(assembled) > budget && kept.length > 1) {
    kept = kept.slice(1);
    assembled = assemble(prefix, summary, kept);
  }

  // Aggressively truncate tool output in the surviving turn too.
  if (countTokens(assembled) > budget) {
    kept = trimHeavyToolResults(kept, 500, true);
    assembled = assemble(prefix, summary, kept);
  }

  // Last resort: a single oversized body (a huge task, a long answer, bulky
  // tool-call arguments) can still blow the window on its own. Clamp bodies
  // outside the system prefix until they fit.
  if (countTokens(assembled) > budget) {
    assembled = clampToBudget(assembled, budget, prefix.length);
  }

  const tokensAfter = countTokens(assembled);
  return {
    messages: assembled,
    tokensBefore,
    tokensAfter,
    droppedMessages: Math.max(0, messages.length - assembled.length),
    withinBudget: tokensAfter <= budget,
  };
}

/**
 * Shrinks message bodies, largest pass first, until the transcript fits. The
 * system prefix is never touched; message identity, roles and tool-call ids are
 * preserved so the transcript stays API-valid.
 */
function clampToBudget(
  messages: readonly ChatMessage[],
  budget: number,
  prefixLength: number,
): ChatMessage[] {
  const clamped = [...messages];
  for (let maxChars = CLAMP_START_CHARS; maxChars >= MIN_MESSAGE_CHARS; maxChars = Math.floor(maxChars / 2)) {
    for (let index = prefixLength; index < clamped.length; index += 1) {
      if (countTokens(clamped) <= budget) return clamped;
      const message = clamped[index];
      if (message === undefined) continue;
      clamped[index] = shrinkMessage(message, maxChars);
    }
  }
  return clamped;
}

function shrinkMessage(message: ChatMessage, maxChars: number): ChatMessage {
  if (message.role === 'tool') return truncateToolMessage(message, maxChars);

  const text = extractText(message);
  const content = text.length > maxChars ? prune(text, maxChars) : undefined;

  if (message.role === 'assistant') {
    const calls = message.tool_calls;
    const shrunkCalls = calls?.map((call) =>
      call.type === 'function' && call.function.arguments.length > maxChars
        ? {
            ...call,
            function: {
              ...call.function,
              arguments: JSON.stringify({
                pruned_by_compaction: `${call.function.arguments.length} characters of arguments omitted`,
              }),
            },
          }
        : call,
    );
    if (content === undefined && shrunkCalls === undefined) return message;
    return {
      ...message,
      ...(content !== undefined ? { content } : {}),
      ...(shrunkCalls !== undefined ? { tool_calls: shrunkCalls } : {}),
    };
  }

  if (content === undefined) return message;
  return { ...message, content };
}

function prune(text: string, maxChars: number): string {
  return `${text.slice(0, maxChars)}\n… [${text.length - maxChars} characters pruned by compaction]`;
}

function assemble(
  prefix: readonly ChatMessage[],
  summary: string | undefined,
  turns: readonly ChatMessage[][],
): ChatMessage[] {
  const head: ChatMessage[] = [...prefix];
  if (summary !== undefined) {
    head.push({ role: 'system', content: `${SUMMARY_MARKER} ${summary}` });
  }
  return [...head, ...turns.flat()];
}

function splitSystemPrefix(messages: readonly ChatMessage[]): {
  prefix: ChatMessage[];
  rest: ChatMessage[];
} {
  const prefix: ChatMessage[] = [];
  let index = 0;
  while (index < messages.length) {
    const message = messages[index];
    if (message === undefined) break;
    const isPrefix =
      message.role === 'system' ||
      message.role === 'developer' ||
      (message.role === 'user' && index === 0 && messages.length === 1);
    if (!isPrefix) break;
    prefix.push(message);
    index += 1;
  }
  return { prefix, rest: messages.slice(index) };
}

/**
 * A turn is a user message plus every assistant/tool message it produced.
 * Messages preceding the first user message form their own leading turn.
 */
function groupIntoTurns(messages: readonly ChatMessage[]): ChatMessage[][] {
  const turns: ChatMessage[][] = [];
  let current: ChatMessage[] = [];
  for (const message of messages) {
    if (message.role === 'user' && current.length > 0) {
      turns.push(current);
      current = [];
    }
    current.push(message);
  }
  if (current.length > 0) turns.push(current);
  return turns;
}

/** Truncates tool output, by default sparing the most recent turn. */
function trimHeavyToolResults(
  turns: readonly ChatMessage[][],
  maxChars: number,
  includeLatest = false,
): ChatMessage[][] {
  const lastIndex = turns.length - 1;
  return turns.map((turn, turnIndex) =>
    turnIndex === lastIndex && !includeLatest
      ? [...turn]
      : turn.map((message) =>
          message.role === 'tool' ? truncateToolMessage(message, maxChars) : message,
        ),
  );
}

function truncateToolMessage(
  message: Extract<ChatMessage, { role: 'tool' }>,
  maxChars: number,
): ChatMessage {
  const text = extractText(message);
  if (text.length <= maxChars) return message;
  return { ...message, content: prune(text, maxChars) };
}

async function summarizeTurns(
  stale: readonly ChatMessage[],
  options: CompactionOptions,
): Promise<string> {
  const deterministic = digest(stale);
  if (options.summarize === undefined) return deterministic;
  try {
    const summary = (await options.summarize(stale)).trim();
    return summary.length > 0 ? summary : deterministic;
  } catch {
    return deterministic;
  }
}

/**
 * Deterministic fallback digest: preserves intent, tool names and outcomes
 * while discarding the bodies of tool results.
 */
export function digest(messages: readonly ChatMessage[]): string {
  const lines: string[] = [];
  for (const message of messages) {
    const text = collapse(extractText(message));
    switch (message.role) {
      case 'user':
        lines.push(`user asked: ${clip(text, 300)}`);
        break;
      case 'assistant': {
        if (text.length > 0) lines.push(`assistant said: ${clip(text, 300)}`);
        for (const call of toolCallsOf(message)) {
          lines.push(`called ${call.function.name}(${clip(collapse(call.function.arguments), 160)})`);
        }
        break;
      }
      case 'tool':
        lines.push(`tool result: ${clip(text, 200)}`);
        break;
      default:
        break;
    }
  }
  return [
    `Summary of ${messages.length} earlier messages (verbatim content pruned to fit the context window):`,
    ...lines.map((line) => `- ${line}`),
  ].join('\n');
}

function toolCallsOf(message: AssistantMessage) {
  return (message.tool_calls ?? []).filter((call) => call.type === 'function');
}

export function extractText(message: ChatMessage): string {
  const { content } = message;
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .map((part) => {
      if (typeof part === 'string') return part;
      return 'text' in part && typeof part.text === 'string' ? part.text : '';
    })
    .join('');
}

function collapse(text: string): string {
  return text.replace(/\s+/g, ' ').trim();
}

function clip(text: string, maxChars: number): string {
  return text.length <= maxChars ? text : `${text.slice(0, maxChars)}…`;
}
