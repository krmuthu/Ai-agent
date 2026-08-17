import type { AgentConfig } from './config.ts';
import {
  compactMessages,
  countTokens,
  extractText,
  needsCompaction,
  type CompactionOptions,
} from './context.ts';
import { ContextOverflowError, createLlmClient, isContextOverflow, type LlmClient } from './llm.ts';
import { COMPACTION_SUMMARY_PROMPT, buildSystemPrompt } from './prompt.ts';
import { ToolRegistry, errorMessage } from './tools.ts';
import type {
  AgentEventHandler,
  AgentResult,
  AssistantMessage,
  ChatMessage,
  ToolContext,
  ToolMessage,
} from './types.ts';

export const CONTEXT_LIMIT_MESSAGE =
  'Error: Context limit exceeded. Pruning failed or task is too complex.';

export interface AgentOptions {
  readonly config: AgentConfig;
  readonly registry?: ToolRegistry;
  readonly llm?: LlmClient;
  readonly onEvent?: AgentEventHandler;
  readonly systemPrompt?: string;
}

export class Agent {
  readonly #config: AgentConfig;
  readonly #registry: ToolRegistry;
  readonly #llm: LlmClient;
  readonly #onEvent: AgentEventHandler;
  #messages: ChatMessage[];

  constructor(options: AgentOptions) {
    this.#config = options.config;
    this.#registry = options.registry ?? new ToolRegistry();
    this.#llm = options.llm ?? createLlmClient(options.config);
    this.#onEvent = options.onEvent ?? ((): void => {});
    this.#messages = [
      {
        role: 'system',
        content:
          options.systemPrompt ??
          buildSystemPrompt({
            workspace: options.config.workspace,
            toolNames: this.#registry.names(),
            maxIterations: options.config.maxIterations,
          }),
      },
    ];
  }

  get messages(): readonly ChatMessage[] {
    return this.#messages;
  }

  /**
   * Runs the ReAct loop until the model answers without tool calls, the
   * iteration budget is spent, the run is aborted, or the context overflows.
   */
  async run(task: string, signal: AbortSignal = new AbortController().signal): Promise<AgentResult> {
    this.#messages.push({ role: 'user', content: task });

    const toolContext: ToolContext = {
      workspace: this.#config.workspace,
      commandTimeoutMs: this.#config.commandTimeoutMs,
      maxOutputChars: this.#config.maxOutputChars,
      mcpServerUrl: this.#config.mcpServerUrl,
      signal,
    };
    const specs = this.#registry.specs();
    let iteration = 0;

    while (true) {
      if (signal.aborted) return this.#finish('aborted', 'Run aborted.', iteration);

      iteration += 1;
      if (iteration > this.#config.maxIterations) {
        return this.#finish(
          'max_iterations',
          `Stopped after the ${this.#config.maxIterations}-iteration safeguard without a final answer.`,
          iteration - 1,
        );
      }

      await this.#maybeCompact(signal);
      this.#onEvent({ type: 'iteration', iteration, tokens: countTokens(this.#messages) });

      let assistant: AssistantMessage;
      try {
        assistant = await this.#llm.complete({
          messages: this.#messages,
          tools: specs,
          signal,
          onRetry: (attempt, delayMs, reason) =>
            this.#onEvent({ type: 'retry', attempt, delayMs, reason }),
        });
      } catch (error) {
        // The provider is the source of truth on context size: our estimator can
        // be wrong, so an overflow may still surface after compaction.
        if (isContextOverflow(error)) {
          const recovered = await this.#recoverFromOverflow(signal, specs);
          if (recovered === undefined) {
            this.#onEvent({ type: 'error', message: CONTEXT_LIMIT_MESSAGE });
            return this.#finish('context_overflow', CONTEXT_LIMIT_MESSAGE, iteration);
          }
          assistant = recovered;
        } else if (signal.aborted) {
          return this.#finish('aborted', 'Run aborted.', iteration);
        } else {
          const message = `Agent failed: ${errorMessage(error)}`;
          this.#onEvent({ type: 'error', message });
          return this.#finish('failed', message, iteration);
        }
      }

      this.#messages.push(normalizeAssistant(assistant));
      const text = extractText(assistant);
      if (text.trim().length > 0) this.#onEvent({ type: 'assistant', content: text });

      const toolCalls = (assistant.tool_calls ?? []).filter((call) => call.type === 'function');
      if (toolCalls.length === 0) {
        return this.#finish('completed', text, iteration);
      }

      for (const call of toolCalls) {
        this.#onEvent({ type: 'tool_call', name: call.function.name, args: call.function.arguments });
        const result = await this.#registry.execute(call.function.name, call.function.arguments, toolContext);
        this.#onEvent({
          type: 'tool_result',
          name: call.function.name,
          ok: result.ok,
          content: result.content,
        });
        const toolMessage: ToolMessage = {
          role: 'tool',
          tool_call_id: call.id,
          content: result.ok ? result.content : `ERROR: ${result.content}`,
        };
        this.#messages.push(toolMessage);
      }
    }
  }

  /** Compacts the transcript when it crosses the configured threshold. */
  async #maybeCompact(signal: AbortSignal): Promise<void> {
    const options = this.#compactionOptions(signal);
    if (!needsCompaction(this.#messages, options)) return;
    const result = await compactMessages(this.#messages, options);
    this.#messages = result.messages;
    this.#onEvent({
      type: 'compaction',
      before: result.tokensBefore,
      after: result.tokensAfter,
      dropped: result.droppedMessages,
    });
  }

  /**
   * Last-ditch recovery after a provider-side overflow: compact hard against a
   * halved budget and retry once. Returns `undefined` when that also fails.
   */
  async #recoverFromOverflow(
    signal: AbortSignal,
    specs: ReturnType<ToolRegistry['specs']>,
  ): Promise<AssistantMessage | undefined> {
    const before = countTokens(this.#messages);
    // Target half of what we actually hold: the provider rejected the current
    // transcript, so compacting against the configured window again — which our
    // estimator already believes we fit into — would be a no-op.
    const emergency: CompactionOptions = {
      ...this.#compactionOptions(signal),
      contextTokens: Math.max(1, Math.floor(Math.min(this.#config.contextTokens, before) / 2)),
      compactionThreshold: 1,
      keepRecentTurns: 1,
      maxToolResultChars: 500,
    };
    const compacted = await compactMessages(this.#messages, emergency);
    if (compacted.tokensAfter >= before) return undefined;

    this.#messages = compacted.messages;
    this.#onEvent({
      type: 'compaction',
      before: compacted.tokensBefore,
      after: compacted.tokensAfter,
      dropped: compacted.droppedMessages,
    });

    try {
      return await this.#llm.complete({ messages: this.#messages, tools: specs, signal });
    } catch (error) {
      if (isContextOverflow(error) || error instanceof ContextOverflowError) return undefined;
      throw error;
    }
  }

  #compactionOptions(signal: AbortSignal): CompactionOptions {
    return {
      contextTokens: this.#config.contextTokens,
      compactionThreshold: this.#config.compactionThreshold,
      keepRecentTurns: this.#config.keepRecentTurns,
      summarize: (messages) => this.#summarize(messages, signal),
    };
  }

  /** Uses the model itself to summarise pruned history; falls back on error. */
  async #summarize(messages: readonly ChatMessage[], signal: AbortSignal): Promise<string> {
    const transcript = messages
      .map((message) => `${message.role}: ${extractText(message).slice(0, 1_500)}`)
      .join('\n')
      .slice(0, 24_000);
    const summary = await this.#llm.complete({
      messages: [
        { role: 'system', content: COMPACTION_SUMMARY_PROMPT },
        { role: 'user', content: transcript },
      ],
      signal,
    });
    return extractText(summary);
  }

  #finish(status: AgentResult['status'], output: string, iterations: number): AgentResult {
    this.#onEvent({ type: 'done', status, iterations });
    return { status, output, iterations, messages: this.#messages };
  }
}

/** Strips provider-specific extras so the message can be replayed verbatim. */
function normalizeAssistant(message: AssistantMessage): AssistantMessage {
  const toolCalls = message.tool_calls?.filter((call) => call.type === 'function');
  return {
    role: 'assistant',
    content: typeof message.content === 'string' ? message.content : (message.content ?? null),
    ...(toolCalls !== undefined && toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
  };
}
