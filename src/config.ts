import { resolve } from 'node:path';
import { z } from 'zod';

const numberFromEnv = (fallback: number) =>
  z
    .string()
    .optional()
    .transform((raw) => (raw === undefined || raw.trim() === '' ? fallback : Number(raw)))
    .pipe(z.number().finite());

const envSchema = z.object({
  OPENAI_API_KEY: z.string().min(1).default('sk-local'),
  OPENAI_BASE_URL: z.string().url().default('https://api.openai.com/v1'),
  AGENT_MODEL: z.string().min(1).default('gpt-4o-mini'),
  AGENT_MAX_ITERATIONS: numberFromEnv(20).pipe(z.number().int().positive()),
  AGENT_CONTEXT_TOKENS: numberFromEnv(32_768).pipe(z.number().int().positive()),
  AGENT_COMPACTION_THRESHOLD: numberFromEnv(0.85).pipe(z.number().gt(0).lte(1)),
  AGENT_KEEP_RECENT_TURNS: numberFromEnv(6).pipe(z.number().int().positive()),
  AGENT_WORKSPACE: z.string().default(process.cwd()),
  AGENT_COMMAND_TIMEOUT_MS: numberFromEnv(60_000).pipe(z.number().int().positive()),
  AGENT_MAX_OUTPUT_CHARS: numberFromEnv(20_000).pipe(z.number().int().positive()),
  AGENT_REQUEST_TIMEOUT_MS: numberFromEnv(120_000).pipe(z.number().int().positive()),
  AGENT_MAX_RETRIES: numberFromEnv(3).pipe(z.number().int().nonnegative()),
  AGENT_TEMPERATURE: numberFromEnv(0).pipe(z.number().min(0).max(2)),
  MCP_SERVER_URL: z.string().url().optional().or(z.literal('').transform(() => undefined)),
});

export interface AgentConfig {
  readonly apiKey: string;
  readonly baseUrl: string;
  readonly model: string;
  readonly maxIterations: number;
  readonly contextTokens: number;
  readonly compactionThreshold: number;
  readonly keepRecentTurns: number;
  readonly workspace: string;
  readonly commandTimeoutMs: number;
  readonly maxOutputChars: number;
  readonly requestTimeoutMs: number;
  readonly maxRetries: number;
  readonly temperature: number;
  readonly mcpServerUrl: string | undefined;
}

/**
 * Builds the agent configuration from the environment, applying `overrides`
 * (typically CLI flags) last. Throws a readable error on invalid values.
 */
export function loadConfig(
  env: NodeJS.ProcessEnv = process.env,
  overrides: Partial<AgentConfig> = {},
): AgentConfig {
  const parsed = envSchema.safeParse(env);
  if (!parsed.success) {
    const details = parsed.error.issues
      .map((issue) => `  - ${issue.path.join('.') || '(root)'}: ${issue.message}`)
      .join('\n');
    throw new Error(`Invalid agent configuration:\n${details}`);
  }

  const value = parsed.data;
  const config: AgentConfig = {
    apiKey: value.OPENAI_API_KEY,
    baseUrl: value.OPENAI_BASE_URL,
    model: value.AGENT_MODEL,
    maxIterations: value.AGENT_MAX_ITERATIONS,
    contextTokens: value.AGENT_CONTEXT_TOKENS,
    compactionThreshold: value.AGENT_COMPACTION_THRESHOLD,
    keepRecentTurns: value.AGENT_KEEP_RECENT_TURNS,
    workspace: resolve(value.AGENT_WORKSPACE),
    commandTimeoutMs: value.AGENT_COMMAND_TIMEOUT_MS,
    maxOutputChars: value.AGENT_MAX_OUTPUT_CHARS,
    requestTimeoutMs: value.AGENT_REQUEST_TIMEOUT_MS,
    maxRetries: value.AGENT_MAX_RETRIES,
    temperature: value.AGENT_TEMPERATURE,
    mcpServerUrl: value.MCP_SERVER_URL,
  };

  const merged: AgentConfig = { ...config, ...stripUndefined(overrides) };
  return { ...merged, workspace: resolve(merged.workspace) };
}

function stripUndefined(overrides: Partial<AgentConfig>): Partial<AgentConfig> {
  return Object.fromEntries(
    Object.entries(overrides).filter(([, entryValue]) => entryValue !== undefined),
  );
}
