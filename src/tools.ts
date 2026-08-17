import { spawn } from 'node:child_process';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { isAbsolute, relative, resolve } from 'node:path';
import { z } from 'zod';
import type { AnyToolDefinition, ToolContext, ToolResult, ToolSpec } from './types.ts';

/** Resolves `candidate` inside the workspace, refusing any path escape. */
export function resolveInWorkspace(workspace: string, candidate: string): string {
  const absolute = isAbsolute(candidate) ? resolve(candidate) : resolve(workspace, candidate);
  const rel = relative(workspace, absolute);
  if (rel.startsWith('..') || isAbsolute(rel)) {
    throw new Error(`Path "${candidate}" is outside the agent workspace (${workspace}).`);
  }
  return absolute;
}

export function truncate(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  const omitted = text.length - maxChars;
  return `${text.slice(0, maxChars)}\n… [${omitted} characters truncated]`;
}

const executeCommandSchema = z.object({
  command: z.string().min(1).describe('Shell command to run, e.g. "npm test -- --run".'),
  cwd: z
    .string()
    .optional()
    .describe('Working directory relative to the workspace root. Defaults to the root.'),
  timeout_ms: z
    .number()
    .int()
    .positive()
    .max(600_000)
    .optional()
    .describe('Override the default command timeout. Capped at 10 minutes.'),
});

const executeCommand: AnyToolDefinition = {
  name: 'execute_command',
  description:
    'Run a shell command inside the workspace and return its exit code, stdout and stderr. ' +
    'The command is killed when the timeout elapses. Never use it for interactive programs.',
  schema: executeCommandSchema,
  async execute(input, context) {
    const { command, cwd, timeout_ms: timeoutOverride } = executeCommandSchema.parse(input);
    const workingDirectory = cwd === undefined ? context.workspace : resolveInWorkspace(context.workspace, cwd);
    const timeoutMs = Math.min(timeoutOverride ?? context.commandTimeoutMs, 600_000);
    return runShell(command, workingDirectory, timeoutMs, context);
  },
};

async function runShell(
  command: string,
  cwd: string,
  timeoutMs: number,
  context: ToolContext,
): Promise<ToolResult> {
  return new Promise<ToolResult>((resolvePromise) => {
    const child = spawn(command, {
      cwd,
      shell: true,
      // Detached so the timeout can kill the whole process group, not just the shell.
      detached: process.platform !== 'win32',
      env: process.env,
    });

    const chunks: string[] = [];
    let timedOut = false;
    let settled = false;

    const collect = (chunk: Buffer): void => {
      chunks.push(chunk.toString('utf8'));
    };
    child.stdout.on('data', collect);
    child.stderr.on('data', collect);

    const kill = (): void => {
      if (child.pid === undefined) return;
      try {
        if (process.platform === 'win32') child.kill('SIGKILL');
        else process.kill(-child.pid, 'SIGKILL');
      } catch {
        child.kill('SIGKILL');
      }
    };

    const timer = setTimeout(() => {
      timedOut = true;
      kill();
    }, timeoutMs);
    const onAbort = (): void => kill();
    context.signal.addEventListener('abort', onAbort, { once: true });

    const settle = (result: ToolResult): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      context.signal.removeEventListener('abort', onAbort);
      resolvePromise(result);
    };

    child.on('error', (error: Error) => {
      settle({ ok: false, content: `Failed to start command: ${error.message}` });
    });

    child.on('close', (code, signal) => {
      const output = truncate(chunks.join('').trimEnd(), context.maxOutputChars);
      if (timedOut) {
        settle({
          ok: false,
          heavy: true,
          content: `Command timed out after ${timeoutMs}ms and was killed.\n--- output ---\n${output}`,
        });
        return;
      }
      const status = code ?? `signal ${signal ?? 'unknown'}`;
      settle({
        ok: code === 0,
        heavy: true,
        content: `exit_code: ${status}\ncwd: ${cwd}\n--- output ---\n${output || '(no output)'}`,
      });
    });
  });
}

const readFileSchema = z.object({
  path: z.string().min(1).describe('File path relative to the workspace root.'),
  start_line: z.number().int().positive().optional().describe('1-based first line to return.'),
  max_lines: z.number().int().positive().max(5_000).optional().describe('Maximum number of lines to return.'),
});

const readFileTool: AnyToolDefinition = {
  name: 'read_file',
  description:
    'Read a UTF-8 text file from the workspace. Supports line windows so large files can be paged instead of flooding the context.',
  schema: readFileSchema,
  async execute(input, context) {
    const { path, start_line: startLine, max_lines: maxLines } = readFileSchema.parse(input);
    const absolute = resolveInWorkspace(context.workspace, path);
    try {
      const stats = await stat(absolute);
      if (!stats.isFile()) return { ok: false, content: `"${path}" is not a regular file.` };
      const raw = await readFile(absolute, 'utf8');
      if (startLine === undefined && maxLines === undefined) {
        return { ok: true, heavy: true, content: truncate(raw, context.maxOutputChars) };
      }
      const lines = raw.split('\n');
      const from = (startLine ?? 1) - 1;
      const slice = lines.slice(from, from + (maxLines ?? lines.length));
      const numbered = slice.map((line, index) => `${from + index + 1}\t${line}`).join('\n');
      return { ok: true, heavy: true, content: truncate(numbered, context.maxOutputChars) };
    } catch (error) {
      return { ok: false, content: `Failed to read "${path}": ${errorMessage(error)}` };
    }
  },
};

const writeFileSchema = z.object({
  path: z.string().min(1).describe('File path relative to the workspace root.'),
  content: z.string().describe('Full UTF-8 content to write.'),
  append: z.boolean().optional().describe('Append instead of overwriting. Defaults to false.'),
});

const writeFileTool: AnyToolDefinition = {
  name: 'write_file',
  description:
    'Create or overwrite a UTF-8 text file in the workspace, creating parent directories as needed. Always write the complete intended file content.',
  schema: writeFileSchema,
  async execute(input, context) {
    const { path, content, append } = writeFileSchema.parse(input);
    const absolute = resolveInWorkspace(context.workspace, path);
    try {
      await mkdir(resolve(absolute, '..'), { recursive: true });
      await writeFile(absolute, content, { encoding: 'utf8', flag: append === true ? 'a' : 'w' });
      const verb = append === true ? 'Appended to' : 'Wrote';
      return { ok: true, content: `${verb} "${path}" (${content.length} characters).` };
    } catch (error) {
      return { ok: false, content: `Failed to write "${path}": ${errorMessage(error)}` };
    }
  },
};

const mcpCallSchema = z.object({
  server: z.string().min(1).describe('Logical name of the MCP server to talk to.'),
  tool: z.string().min(1).describe('Tool exposed by that MCP server.'),
  arguments: z.record(z.string(), z.unknown()).default({}).describe('JSON arguments for the MCP tool.'),
});

/**
 * Placeholder MCP bridge. It speaks JSON-RPC 2.0 `tools/call` over HTTP when
 * `MCP_SERVER_URL` is configured; swap in the official MCP client (stdio or
 * streamable HTTP transport, plus server discovery) when wiring real servers.
 */
const mcpCall: AnyToolDefinition = {
  name: 'mcp_call',
  description:
    'Invoke a tool on an external Model Context Protocol (MCP) server. Only use it when the task explicitly needs an external MCP capability.',
  schema: mcpCallSchema,
  async execute(input, context) {
    const { server, tool, arguments: args } = mcpCallSchema.parse(input);
    if (context.mcpServerUrl === undefined) {
      return {
        ok: false,
        content:
          'mcp_call is not configured: set MCP_SERVER_URL to an MCP JSON-RPC endpoint. ' +
          `Requested ${server}.${tool}.`,
      };
    }
    try {
      const response = await fetch(context.mcpServerUrl, {
        method: 'POST',
        headers: { 'content-type': 'application/json', accept: 'application/json' },
        body: JSON.stringify({
          jsonrpc: '2.0',
          id: crypto.randomUUID(),
          method: 'tools/call',
          params: { name: tool, arguments: args, _meta: { server } },
        }),
        signal: AbortSignal.any([context.signal, AbortSignal.timeout(context.commandTimeoutMs)]),
      });
      const body = await response.text();
      return {
        ok: response.ok,
        heavy: true,
        content: truncate(`status: ${response.status}\n${body}`, context.maxOutputChars),
      };
    } catch (error) {
      return { ok: false, content: `MCP call to ${server}.${tool} failed: ${errorMessage(error)}` };
    }
  },
};

export const builtinTools: readonly AnyToolDefinition[] = [
  executeCommand,
  readFileTool,
  writeFileTool,
  mcpCall,
];

export class ToolRegistry {
  readonly #tools = new Map<string, AnyToolDefinition>();

  constructor(tools: readonly AnyToolDefinition[] = builtinTools) {
    for (const tool of tools) this.register(tool);
  }

  register(tool: AnyToolDefinition): void {
    if (this.#tools.has(tool.name)) throw new Error(`Duplicate tool name: ${tool.name}`);
    this.#tools.set(tool.name, tool);
  }

  get(name: string): AnyToolDefinition | undefined {
    return this.#tools.get(name);
  }

  names(): string[] {
    return [...this.#tools.keys()];
  }

  /** OpenAI function-tool specs derived from the Zod schemas — no drift possible. */
  specs(): ToolSpec[] {
    return [...this.#tools.values()].map((tool) => ({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description,
        parameters: z.toJSONSchema(tool.schema, { target: 'draft-7', io: 'input' }),
      },
    }));
  }

  /**
   * Validates raw JSON arguments against the tool schema and runs the tool.
   * Validation and runtime failures are returned as tool results (never thrown)
   * so the model can see the error and correct itself on the next iteration.
   */
  async execute(name: string, rawArguments: string, context: ToolContext): Promise<ToolResult> {
    const tool = this.#tools.get(name);
    if (tool === undefined) {
      return { ok: false, content: `Unknown tool "${name}". Available tools: ${this.names().join(', ')}.` };
    }

    let parsedJson: unknown;
    try {
      parsedJson = rawArguments.trim() === '' ? {} : JSON.parse(rawArguments);
    } catch (error) {
      return { ok: false, content: `Arguments for "${name}" are not valid JSON: ${errorMessage(error)}` };
    }

    const parsed = tool.schema.safeParse(parsedJson);
    if (!parsed.success) {
      const issues = parsed.error.issues
        .map((issue) => `  - ${issue.path.join('.') || '(root)'}: ${issue.message}`)
        .join('\n');
      return { ok: false, content: `Invalid arguments for "${name}":\n${issues}` };
    }

    try {
      return await tool.execute(parsed.data, context);
    } catch (error) {
      return { ok: false, content: `Tool "${name}" threw: ${errorMessage(error)}` };
    }
  }
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
