import OpenAI, { APIError } from 'openai';
import type { AgentConfig } from './config.ts';
import type { ChatMessage, ToolSpec } from './types.ts';

const CONTEXT_OVERFLOW_PATTERNS = [
  /context[_ ]length[_ ]exceeded/i,
  /maximum context length/i,
  /context window/i,
  /too many tokens/i,
  /reduce the length of the messages/i,
  /prompt is too long/i,
  /input length .* exceeds/i,
];

/** Raised when the provider rejects the request because the prompt is too big. */
export class ContextOverflowError extends Error {
  constructor(message: string, cause?: unknown) {
    super(message, { cause });
    this.name = 'ContextOverflowError';
  }
}

export function isContextOverflow(error: unknown): boolean {
  if (error instanceof ContextOverflowError) return true;
  if (!(error instanceof APIError)) return false;
  if (error.status !== 400 && error.status !== 413) return false;
  const haystack = `${error.message} ${JSON.stringify(error.error ?? {})}`;
  return CONTEXT_OVERFLOW_PATTERNS.some((pattern) => pattern.test(haystack));
}

function isRetryable(error: unknown): boolean {
  if (isContextOverflow(error)) return false;
  if (error instanceof APIError) {
    return error.status === undefined || error.status === 408 || error.status === 409 || error.status === 429 || error.status >= 500;
  }
  return error instanceof Error && /timeout|ECONNRESET|ECONNREFUSED|EAI_AGAIN|fetch failed/i.test(error.message);
}

export interface CompletionRequest {
  readonly messages: readonly ChatMessage[];
  readonly tools?: readonly ToolSpec[];
  readonly signal: AbortSignal;
  readonly onRetry?: (attempt: number, delayMs: number, reason: string) => void;
}

export interface LlmClient {
  readonly model: string;
  complete(request: CompletionRequest): Promise<OpenAI.Chat.Completions.ChatCompletionMessage>;
}

/**
 * Thin wrapper over the OpenAI SDK. `baseUrl` makes it usable against any
 * OpenAI-compatible server (vLLM, llama.cpp, Ollama, LM Studio, ...).
 */
export function createLlmClient(config: AgentConfig): LlmClient {
  const client = new OpenAI({
    apiKey: config.apiKey,
    baseURL: config.baseUrl,
    timeout: config.requestTimeoutMs,
    // Retries are handled here so callers get retry events and jittered backoff.
    maxRetries: 0,
  });

  return {
    model: config.model,
    async complete({ messages, tools, signal, onRetry }) {
      let lastError: unknown;
      for (let attempt = 0; attempt <= config.maxRetries; attempt += 1) {
        try {
          const completion = await client.chat.completions.create(
            {
              model: config.model,
              temperature: config.temperature,
              messages: messages as OpenAI.Chat.Completions.ChatCompletionMessageParam[],
              ...(tools !== undefined && tools.length > 0
                ? { tools: tools as OpenAI.Chat.Completions.ChatCompletionTool[], tool_choice: 'auto' as const }
                : {}),
            },
            { signal },
          );
          const choice = completion.choices[0];
          if (choice === undefined) throw new Error('Model returned no choices.');
          return choice.message;
        } catch (error) {
          lastError = error;
          if (signal.aborted || !isRetryable(error) || attempt === config.maxRetries) break;
          const delayMs = Math.round(2 ** attempt * 500 * (1 + Math.random()));
          onRetry?.(attempt + 1, delayMs, error instanceof Error ? error.message : String(error));
          await sleep(delayMs, signal);
        }
      }
      if (isContextOverflow(lastError)) {
        throw new ContextOverflowError('Provider rejected the request: prompt exceeds the context window.', lastError);
      }
      throw lastError instanceof Error ? lastError : new Error(String(lastError));
    },
  };
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolvePromise, rejectPromise) => {
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort);
      resolvePromise();
    }, ms);
    function onAbort(): void {
      clearTimeout(timer);
      rejectPromise(new Error('Aborted while backing off.'));
    }
    signal.addEventListener('abort', onAbort, { once: true });
  });
}
