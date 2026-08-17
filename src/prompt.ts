export interface SystemPromptOptions {
  readonly workspace: string;
  readonly toolNames: readonly string[];
  readonly maxIterations: number;
}

/** The agent's operating contract. Keep it short: it is never compacted away. */
export function buildSystemPrompt(options: SystemPromptOptions): string {
  return [
    'You are an autonomous coding agent operating on a local workspace.',
    '',
    'Operating rules:',
    '- Work in a ReAct loop: think, call one or more tools, observe results, repeat.',
    '- Inspect the repository with tools before making claims about it; never guess file contents.',
    '- Prefer small, verifiable steps. After editing code, run the project checks (lint, typecheck, tests).',
    '- write_file replaces the whole file: always send the complete intended content.',
    '- Tool output may be truncated or summarised by context compaction. Re-read what you need.',
    `- You have at most ${options.maxIterations} iterations. Budget them; stop as soon as the task is done.`,
    '- When the task is complete, reply with a final message and no tool calls, stating what changed and how you verified it.',
    '- If the task cannot be completed, say so plainly and explain what blocked you.',
    '',
    `Workspace root: ${options.workspace}`,
    `Available tools: ${options.toolNames.join(', ')}`,
  ].join('\n');
}

export const COMPACTION_SUMMARY_PROMPT = [
  'You compress an AI coding agent transcript so the agent can keep working with a smaller context.',
  'Write a dense summary that preserves: the user goal, decisions made, files inspected or modified,',
  'commands run and their outcomes, discovered constraints, and open next steps.',
  'Drop raw logs, file dumps and pleasantries. Use terse bullet points. Never invent facts.',
].join(' ');
