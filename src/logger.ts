const LEVELS = { debug: 10, info: 20, warn: 30, error: 40, silent: 100 } as const;

export type LogLevel = keyof typeof LEVELS;

export interface Logger {
  debug(message: string, ...rest: readonly unknown[]): void;
  info(message: string, ...rest: readonly unknown[]): void;
  warn(message: string, ...rest: readonly unknown[]): void;
  error(message: string, ...rest: readonly unknown[]): void;
}

/**
 * Minimal level-filtered logger. Everything goes to stderr so stdout stays
 * reserved for the agent's final answer and can be piped safely.
 */
export function createLogger(level: LogLevel = 'info'): Logger {
  const threshold = LEVELS[level];
  const write = (messageLevel: LogLevel, prefix: string) =>
    (message: string, ...rest: readonly unknown[]): void => {
      if (LEVELS[messageLevel] < threshold) return;
      process.stderr.write(`${prefix} ${message}\n`);
      for (const extra of rest) {
        process.stderr.write(`${typeof extra === 'string' ? extra : JSON.stringify(extra)}\n`);
      }
    };

  return {
    debug: write('debug', '[debug]'),
    info: write('info', '[info] '),
    warn: write('warn', '[warn] '),
    error: write('error', '[error]'),
  };
}
