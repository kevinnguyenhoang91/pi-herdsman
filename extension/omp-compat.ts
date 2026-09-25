// omp compatibility layer.
//
// omp (oh-my-pi) is a pi fork whose bundled legacy-pi host shims diverge from
// upstream pi in a few places pi-herdsman relies on:
//   1. `contentText` is not re-exported from the legacy pi-ai shim.
//   2. `buildSessionProjection` was removed from the legacy pi-coding-agent
//      shim (omp only ships the whole-branch `buildSessionContext`).
//   3. There is no sync `loadProjectContextFiles` (omp's version is async and
//      capability-based).
//   4. The extension context has no `registerEntryRenderer` (omp only exposes
//      `registerMessageRenderer` for `custom_message` entries).
//   5. herdr launches omp with `--kind omp` and reports sessions as
//      `herdr:omp` / agent `omp` (upstream integration reports `herdr:pi`).
//
// Everything here falls back to native behavior when the host provides it, so
// the plugin keeps working under both runtimes.

import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve, sep } from "node:path";
import {
  getAgentDir,
  sessionEntryToContextMessages,
  type ExtensionAPI,
  type SessionEntry,
} from "@earendil-works/pi-coding-agent";

interface ProjectedEntry {
  sourceEntry: SessionEntry;
  messages: unknown[];
}

interface SessionProjection {
  entries: ProjectedEntry[];
  messages: unknown[];
  thinkingLevel: string;
  model: { provider: string; modelId: string } | null;
}

/** Equivalent of upstream `contentText` from pi-ai `utils/text.ts`. */
export function contentText(
  content: string | readonly { type: string; text: string }[],
  separator = "\n",
): string {
  if (typeof content === "string") return content;
  return content
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join(separator);
}

function buildEntryIndex(
  entries: readonly SessionEntry[],
  byId?: Map<string, SessionEntry>,
): Map<string, SessionEntry> {
  if (byId) return byId;
  const index = new Map<string, SessionEntry>();
  for (const entry of entries) index.set(entry.id, entry);
  return index;
}

function buildSessionPath(
  entries: readonly SessionEntry[],
  leafId?: string | null,
  byId?: Map<string, SessionEntry>,
): SessionEntry[] {
  const index = buildEntryIndex(entries, byId);
  let leaf: SessionEntry | undefined;
  if (leafId === null) return [];
  if (leafId) leaf = index.get(leafId);
  leaf ??= entries[entries.length - 1];
  if (!leaf) return [];
  const path: SessionEntry[] = [];
  let current: SessionEntry | undefined = leaf;
  while (current) {
    path.push(current);
    current = current.parentId ? index.get(current.parentId) : undefined;
  }
  path.reverse();
  return path;
}

function buildContextEntries(
  entries: readonly SessionEntry[],
  leafId?: string | null,
  byId?: Map<string, SessionEntry>,
): SessionEntry[] {
  const path = buildSessionPath(entries, leafId, byId);
  let compaction: SessionEntry | null = null;
  for (const entry of path) {
    if (entry.type === "compaction") compaction = entry;
  }
  if (!compaction) return path;
  const compactionIdx = path.findIndex((entry) => entry.id === compaction!.id);
  if (compactionIdx < 0) return path;
  const contextEntries: SessionEntry[] = [compaction];
  let foundFirstKept = false;
  for (let i = 0; i < compactionIdx; i++) {
    const entry = path[i]!;
    if (entry.id === compaction!.firstKeptEntryId) foundFirstKept = true;
    if (
      foundFirstKept &&
      !(entry.type === "message" && entry.message.role === "system")
    ) {
      contextEntries.push(entry);
    }
  }
  contextEntries.push(...path.slice(compactionIdx + 1));
  return contextEntries;
}

interface ContextEditEntry {
  type: "context_edit";
  targetId: string;
  replacement: { content: string | unknown[] } | null;
}

function isContextEditEntry(entry: unknown): entry is ContextEditEntry {
  return (
    typeof entry === "object" &&
    entry !== null &&
    "type" in entry &&
    entry.type === "context_edit" &&
    "targetId" in entry &&
    "replacement" in entry
  );
}

function projectContextEntry(entry: SessionEntry, edit: unknown): unknown[] {
  const messages = sessionEntryToContextMessages(entry) as unknown[];
  if (!isContextEditEntry(edit)) return messages;
  const replacement = edit.replacement;
  if (replacement === null) return [];
  return messages.map((message) => {
    if (typeof message !== "object" || message === null) return message;
    const role = (message as { role?: unknown }).role;
    if (
      role !== "user" &&
      role !== "assistant" &&
      role !== "toolResult" &&
      role !== "custom"
    ) {
      return message;
    }
    const content =
      (role === "assistant" || role === "toolResult") &&
      typeof replacement.content === "string"
        ? [{ type: "text", text: replacement.content }]
        : replacement.content;
    return { ...message, content };
  });
}

/** Port of upstream pi's provenance-preserving, compaction-aware projection. */
export function buildSessionProjection(
  entries: readonly SessionEntry[],
  leafId?: string | null,
  byId?: Map<string, SessionEntry>,
): SessionProjection {
  const path = buildSessionPath(entries, leafId, byId);
  let thinkingLevel = "off";
  let model: { provider: string; modelId: string } | null = null;
  for (const entry of path) {
    if (entry.type === "thinking_level_change") {
      thinkingLevel = entry.thinkingLevel ?? "off";
    } else if (entry.type === "model_change") {
      // omp records `model_change` as `{ model: "provider/id" }`; upstream pi
      // as `{ provider, modelId }`. Handle both.
      if (typeof entry.model === "string") {
        const slash = entry.model.indexOf("/");
        model =
          slash === -1
            ? { provider: entry.model, modelId: entry.model }
            : {
                provider: entry.model.slice(0, slash),
                modelId: entry.model.slice(slash + 1),
              };
      } else if (entry.provider || entry.modelId) {
        model = { provider: entry.provider, modelId: entry.modelId };
      }
    } else if (entry.type === "message" && entry.message.role === "assistant") {
      model = {
        provider: entry.message.provider,
        modelId: entry.message.model,
      };
    }
  }
  const contextEntries = buildContextEntries(entries, leafId, byId);
  const edits = new Map<string, unknown>();
  for (const entry of contextEntries) {
    if (entry.type === "context_edit") edits.set(entry.targetId, entry);
  }
  const projectedEntries = contextEntries.map((sourceEntry, index) => ({
    sourceEntry,
    // Only the newest compaction at index zero contributes a checkpoint and
    // summary; older retained compactions contribute no messages.
    messages:
      sourceEntry.type === "compaction" && index > 0
        ? []
        : projectContextEntry(sourceEntry, edits.get(sourceEntry.id)),
  }));
  return {
    entries: projectedEntries,
    messages: projectedEntries.flatMap((entry) => entry.messages),
    thinkingLevel,
    model,
  };
}

// ponytail: sync context-file discovery (AGENTS.md/CLAUDE.md in cwd ancestors
// plus agentDir); @-import expansion and GEMINI.md-style providers are skipped —
// port omp's capability-based loadProjectContextFiles if spawned-agent prompts
// look thin.
/** Test seam: cwds passed to loadProjectContextFiles. */
export const projectContextCwds: string[] = [];

export function loadProjectContextFiles(options: {
  cwd?: string;
  agentDir?: string;
} = {}): Array<{ path: string; content: string; depth: number }> {
  const files: Array<{ path: string; content: string; depth: number }> = [];
  const seen = new Set<string>();
  const add = (path: string, depth: number): void => {
    try {
      if (seen.has(path) || !existsSync(path)) return;
      seen.add(path);
      files.push({ path, content: readFileSync(path, "utf8"), depth });
    } catch {
      // unreadable context file: skip
    }
  };
  let dir = resolve(options.cwd ?? process.cwd());
  projectContextCwds.push(options.cwd ?? process.cwd());
  let depth = 0;
  const root = (dir.match(/^[A-Za-z]:[/\\]|^\//) ?? ["/"])[0]!;
  for (;;) {
    add(join(dir, "AGENTS.md"), depth);
    add(join(dir, "CLAUDE.md"), depth);
    depth += 1;
    const parent = dirname(dir);
    if (parent === dir || parent.length <= root.length) break;
    dir = parent;
  }
  const agentDir = options.agentDir;
  if (agentDir) {
    add(join(agentDir, "AGENTS.md"), 0);
    add(join(agentDir, "CLAUDE.md"), 0);
  }
  return files.sort((a, b) => (b.depth ?? -1) - (a.depth ?? -1));
}

/** Register a custom-entry renderer, falling back on hosts without the API. */
export function registerEntryRenderer(
  pi: ExtensionAPI,
  customType: string,
  renderer: (entry: unknown, options: unknown, theme: unknown) => unknown,
): void {
  const host = pi as unknown as {
    registerEntryRenderer?: (
      customType: string,
      renderer: (entry: unknown, options: unknown, theme: unknown) => unknown,
    ) => void;
    registerMessageRenderer: (
      customType: string,
      renderer: (message: unknown, options: unknown, theme: unknown) => unknown,
    ) => void;
  };
  if (typeof host.registerEntryRenderer === "function") {
    host.registerEntryRenderer(customType, renderer);
    return;
  }
  // omp renders custom_message entries; adapt `details` to the `data` field
  // upstream entry renderers expect.
  host.registerMessageRenderer(customType, (message, options, theme) => {
    const adapted =
      typeof message === "object" &&
      message !== null &&
      !("data" in message) &&
      "details" in message
        ? { ...message, data: message.details }
        : message;
    return renderer(adapted, options, theme);
  });
}

/**
 * True when the host agent directory belongs to omp rather than pi.
 * omp defaults to `~/.omp/agent`, pi to `~/.pi/agent`; both honor
 * PI_CODING_AGENT_DIR via their getAgentDir().
 */
export function isOmpHost(): boolean {
  return getAgentDir().split(sep).includes(".omp");
}

export function herdrAgentKind(): "pi" | "omp" {
  return isOmpHost() ? "omp" : "pi";
}

/** herdr installs a state extension per agent kind; select by host kind. */
export function herdrAgentStateExtensionPath(agentDir: string): string {
  const omp = herdrAgentKind() === "omp";
  return join(
    agentDir,
    "extensions",
    omp ? "herdr-omp-agent-state.ts" : "herdr-agent-state.ts",
  );
}

/** Accept a herdr agent-session identity from either the pi or omp kind. */
export function isHerdrAgentSession(
  source: unknown,
  agent: unknown,
): boolean {
  return (
    (source === "herdr:pi" && agent === "pi") ||
    (source === "herdr:omp" && agent === "omp")
  );
}