/**
 * Drive squad's runAgent() over a list of tasks and dump full message
 * histories to data/trajectories_raw/full/{runId}.json.
 *
 * Why this exists: squad's on-disk session transcripts are summary-only
 * today (see DECISIONS.md). The full Anthropic-format messages live in
 * the in-memory Session object, which we capture here via spec.session.
 *
 * Run:
 *   pnpm tsx data_gen/squad_runner.ts \
 *     --tasks data/tasks/all.jsonl \
 *     --out data/trajectories_raw/full \
 *     --model anthropic/claude-sonnet-4-5 \
 *     --concurrency 8 \
 *     --max-turns 20
 *
 * Requires:
 *   - SQUAD_REPO env var pointing to the squad checkout
 *   - ANTHROPIC_API_KEY (or whichever provider --model picks)
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";

/**
 * Resolve squad's repo path from (in priority order):
 *   1. process.env.SQUAD_REPO (explicit override)
 *   2. $SQUAD_HOME/config (the env-style file written by `squad onboard`,
 *      contains SQUAD_REPO=...)
 *   3. ~/.squad/config
 */
function resolveSquadRepo(): string {
  if (process.env["SQUAD_REPO"]) return process.env["SQUAD_REPO"];
  const home = process.env["SQUAD_HOME"] ?? join(homedir(), ".squad");
  const cfgPath = join(home, "config");
  if (existsSync(cfgPath)) {
    for (const raw of readFileSync(cfgPath, "utf8").split("\n")) {
      const line = raw.trim();
      if (!line || line.startsWith("#")) continue;
      const eq = line.indexOf("=");
      if (eq < 0) continue;
      const k = line.slice(0, eq).trim();
      const v = line.slice(eq + 1).trim().replace(/^["']|["']$/g, "");
      if (k === "SQUAD_REPO") return v;
    }
  }
  console.error(`Could not find SQUAD_REPO in env or ${cfgPath}`);
  process.exit(2);
}

const SQUAD_REPO = resolveSquadRepo();

// We use dynamic import here because top-level static imports would fail
// at parse time if SQUAD_REPO isn't on the resolution path.
const runnerPath = resolve(SQUAD_REPO, "packages/runner/dist/index.js");
const toolsPath = resolve(SQUAD_REPO, "packages/tools/dist/index.js");

interface RawTaskSpec {
  task_id: string;
  prompt: string;
  cwd?: string;
  difficulty?: string;
}

interface CliArgs {
  tasksPath: string;
  outDir: string;
  model: string;
  concurrency: number;
  maxTurns: number;
  toolNames: string[];
  limit: number | null;
}

function parseArgs(): CliArgs {
  const args = process.argv.slice(2);
  const get = (flag: string): string | undefined => {
    const i = args.indexOf(flag);
    return i >= 0 ? args[i + 1] : undefined;
  };
  return {
    tasksPath: get("--tasks") ?? "data/tasks/all.jsonl",
    outDir: get("--out") ?? "data/trajectories_raw/full",
    model: get("--model") ?? "anthropic/claude-sonnet-4-5",
    concurrency: Number(get("--concurrency") ?? "8"),
    maxTurns: Number(get("--max-turns") ?? "20"),
    toolNames: (get("--tools") ?? "read,edit,write,ls,grep,exec,create_task,update_task,list_tasks,get_config,set_config,list_config_paths").split(","),
    limit: get("--limit") ? Number(get("--limit")) : null,
  };
}

function readTasks(path: string, limit: number | null): RawTaskSpec[] {
  const lines = readFileSync(path, "utf8").split("\n").filter(Boolean);
  const tasks = lines.map((l) => JSON.parse(l) as RawTaskSpec);
  return limit ? tasks.slice(0, limit) : tasks;
}

async function runOne(
  task: RawTaskSpec,
  args: CliArgs,
  runner: typeof import("../../squad/packages/runner/dist/index.js"),
): Promise<void> {
  const cwd = task.cwd ?? process.cwd();

  // Capture the message history live by passing in our own Session.
  const session = new runner.Session([{ role: "user", content: task.prompt }]);

  const spec = {
    task: task.prompt,
    agent: "finetune-data",
    model: args.model,
    cwd,
    tools: args.toolNames,
    maxTurns: args.maxTurns,
    session,
    context: { taskId: task.task_id },
  };

  const startedAt = new Date().toISOString();
  let result: Awaited<ReturnType<typeof runner.runAgent>>;
  try {
    result = await runner.runAgent(spec);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    result = {
      succeeded: false,
      output: "",
      error: msg,
      usage: { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 },
      costUsd: 0,
      turns: 0,
      runId: `err_${task.task_id}`,
    } as Awaited<ReturnType<typeof runner.runAgent>>;
  }

  const dump = {
    meta: {
      runId: result.runId,
      taskId: task.task_id,
      agentType: "finetune-data",
      model: args.model,
      succeeded: result.succeeded,
      totalTurns: result.turns,
      stopReason: result.error ? "error" : (result.succeeded ? "end_turn" : "unknown"),
      durationSeconds: 0,
      timestamp: startedAt,
      toolsOffered: args.toolNames,
      systemPrompt: "",  // we let squad use the default; rendered at training time
      usage: result.usage,
      costUsd: result.costUsd,
      error: result.error,
      difficulty: task.difficulty,
      cwd,
    },
    messages: session._ref(),
  };

  const outPath = join(args.outDir, `${result.runId}.json`);
  writeFileSync(outPath, JSON.stringify(dump));
  process.stderr.write(
    `[${task.task_id}] ${result.succeeded ? "ok" : "fail"} turns=${result.turns} cost=$${result.costUsd.toFixed(4)}\n`,
  );
}

async function runPool<T>(
  items: T[],
  concurrency: number,
  worker: (item: T) => Promise<void>,
): Promise<void> {
  let next = 0;
  const workers: Promise<void>[] = [];
  for (let w = 0; w < concurrency; w++) {
    workers.push(
      (async () => {
        while (next < items.length) {
          const i = next++;
          const item = items[i];
          if (item === undefined) continue;
          try {
            await worker(item);
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            process.stderr.write(`[runner-error] ${msg}\n`);
          }
        }
      })(),
    );
  }
  await Promise.all(workers);
}

async function main(): Promise<void> {
  const args = parseArgs();
  mkdirSync(args.outDir, { recursive: true });

  const runner = (await import(runnerPath)) as typeof import("../../squad/packages/runner/dist/index.js");
  // Touch tools so the registry is populated as a side-effect of import.
  await import(toolsPath);

  const tasks = readTasks(args.tasksPath, args.limit);
  process.stderr.write(`running ${tasks.length} tasks @ concurrency=${args.concurrency}, model=${args.model}\n`);

  await runPool(tasks, args.concurrency, (task) => runOne(task, args, runner));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
