import type OpenAI from 'openai';
import type { z } from 'zod';

export type ChatMessage = OpenAI.Chat.Completions.ChatCompletionMessageParam;
export type AssistantMessage = OpenAI.Chat.Completions.ChatCompletionAssistantMessageParam;
export type ToolMessage = OpenAI.Chat.Completions.ChatCompletionToolMessageParam;
export type ToolSpec = OpenAI.Chat.Completions.ChatCompletionTool;

/** Everything a tool is allowed to touch at run time. */
export interface ToolContext {
  /** Absolute path every filesystem/shell operation is confined to. */
  readonly workspace: string;
  readonly commandTimeoutMs: number;
  readonly maxOutputChars: number;
  readonly mcpServerUrl: string | undefined;
  readonly signal: AbortSignal;
}

export interface ToolResult {
  readonly ok: boolean;
  readonly content: string;
  /** Marks bulky output (logs, file dumps) that compaction may drop first. */
  readonly heavy?: boolean;
}

export interface ToolDefinition<Schema extends z.ZodType = z.ZodType> {
  readonly name: string;
  readonly description: string;
  readonly schema: Schema;
  execute(input: z.output<Schema>, context: ToolContext): Promise<ToolResult>;
}

/** A tool whose input type has been erased so tools can live in one registry. */
export type AnyToolDefinition = ToolDefinition<z.ZodType>;

export type AgentStatus = 'completed' | 'max_iterations' | 'context_overflow' | 'aborted' | 'failed';

export interface AgentResult {
  readonly status: AgentStatus;
  readonly output: string;
  readonly iterations: number;
  readonly messages: readonly ChatMessage[];
}

export type AgentEvent =
  | { readonly type: 'iteration'; readonly iteration: number; readonly tokens: number }
  | { readonly type: 'assistant'; readonly content: string }
  | { readonly type: 'tool_call'; readonly name: string; readonly args: string }
  | { readonly type: 'tool_result'; readonly name: string; readonly ok: boolean; readonly content: string }
  | { readonly type: 'compaction'; readonly before: number; readonly after: number; readonly dropped: number }
  | { readonly type: 'retry'; readonly attempt: number; readonly delayMs: number; readonly reason: string }
  | { readonly type: 'error'; readonly message: string }
  | { readonly type: 'done'; readonly status: AgentStatus; readonly iterations: number };

export type AgentEventHandler = (event: AgentEvent) => void;
