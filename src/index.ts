#!/usr/bin/env node
import { readFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { Agent } from './agent.ts';
import { loadConfig, type AgentConfig } from './config.ts';
import { createLogger, type LogLevel } from './logger.ts';
import { ToolRegistry, errorMessage } from './tools.ts';
import type { AgentEvent } from './types.ts';

const USAGE = `Usage: ai-agent [options] "<task>"

Options:
  -t, --task-file <path>   Read the task from a file instead of an argument
  -m, --model <name>       Model id (default: $AGENT_MODEL)
  -b, --base-url <url>     OpenAI-compatible base URL (default: $OPENAI_BASE_URL)
  -w, --workspace <path>   Workspace root the agent is confined to (default: cwd)
  -i, --max-iterations <n> ReAct iteration safeguard (default: 20)
  -c, --context-tokens <n> Context window capacity used for compaction (default: 32768)
      --log-level <level>  debug | info | warn | error | silent (default: info)
  -h, --help               Show this help
`;

async function main(): Promise<number> {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      'task-file': { type: 'string', short: 't' },
      model: { type: 'string', short: 'm' },
      'base-url': { type: 'string', short: 'b' },
      workspace: { type: 'string', short: 'w' },
      'max-iterations': { type: 'string', short: 'i' },
      'context-tokens': { type: 'string', short: 'c' },
      'log-level': { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });

  if (values.help === true) {
    process.stdout.write(USAGE);
    return 0;
  }

  const task =
    values['task-file'] !== undefined
      ? await readFile(values['task-file'], 'utf8')
      : positionals.join(' ').trim();

  if (task.trim() === '') {
    process.stderr.write(`No task provided.\n\n${USAGE}`);
    return 2;
  }

  const overrides: Partial<AgentConfig> = {
    ...(values.model !== undefined ? { model: values.model } : {}),
    ...(values['base-url'] !== undefined ? { baseUrl: values['base-url'] } : {}),
    ...(values.workspace !== undefined ? { workspace: values.workspace } : {}),
    ...(values['max-iterations'] !== undefined
      ? { maxIterations: Number(values['max-iterations']) }
      : {}),
    ...(values['context-tokens'] !== undefined
      ? { contextTokens: Number(values['context-tokens']) }
      : {}),
  };

  const config = loadConfig(process.env, overrides);
  const logger = createLogger((values['log-level'] as LogLevel | undefined) ?? 'info');
  const registry = new ToolRegistry();

  const controller = new AbortController();
  const onSignal = (): void => {
    logger.warn('Interrupt received, aborting run…');
    controller.abort();
  };
  process.once('SIGINT', onSignal);
  process.once('SIGTERM', onSignal);

  logger.info(`model=${config.model} base_url=${config.baseUrl} workspace=${config.workspace}`);

  const agent = new Agent({ config, registry, onEvent: (event) => report(event, logger) });
  const result = await agent.run(task, controller.signal);

  process.stdout.write(`${result.output.trim()}\n`);
  return result.status === 'completed' ? 0 : 1;
}

function report(event: AgentEvent, logger: ReturnType<typeof createLogger>): void {
  switch (event.type) {
    case 'iteration':
      logger.info(`— iteration ${event.iteration} (~${event.tokens} tokens)`);
      break;
    case 'assistant':
      logger.info(`assistant: ${firstLines(event.content, 4)}`);
      break;
    case 'tool_call':
      logger.info(`→ ${event.name} ${firstLines(event.args, 1)}`);
      break;
    case 'tool_result':
      logger.info(`← ${event.name} ${event.ok ? 'ok' : 'failed'}`);
      logger.debug(firstLines(event.content, 20));
      break;
    case 'compaction':
      logger.warn(
        `compacted context: ${event.before} → ${event.after} tokens (${event.dropped} messages pruned)`,
      );
      break;
    case 'retry':
      logger.warn(`retry ${event.attempt} in ${event.delayMs}ms: ${event.reason}`);
      break;
    case 'error':
      logger.error(event.message);
      break;
    case 'done':
      logger.info(`done: ${event.status} after ${event.iterations} iterations`);
      break;
    default:
      break;
  }
}

function firstLines(text: string, count: number): string {
  const lines = text.split('\n');
  const head = lines.slice(0, count).join('\n');
  return lines.length > count ? `${head}\n…` : head;
}

try {
  process.exitCode = await main();
} catch (error) {
  process.stderr.write(`${errorMessage(error)}\n`);
  process.exitCode = 1;
}
