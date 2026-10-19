import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";

// Surface-probe extension. It plays two roles, and both are observable in
// a *plain* `omp --print` run — see jobs/omp-config-surfaces.yaml for the map.
//
// 1. A surface in its own right: it registers a slash command and an
//    eagerly-loaded tool, so `harbor_hello` appears in the tool roster of the
//    request the CLI sends.
// 2. The reporter for the surfaces that are otherwise invisible in a trial's
//    artifacts. `AGENTS.md`, `rules/`, `skills/` and `agents/` all take effect
//    by being folded into the *request* — the system prompt and the `task`
//    tool's agent roster — and omp writes neither to `agent/omp.txt` nor to
//    `agent/trajectory.json`. So on every `before_provider_request` the
//    extension greps the request it is about to send and prints one line per
//    surface to stderr, which `run()` merges into `agent/omp.txt` with `2>&1`.
//
// Every report line names the exact marker it looked for and prints ABSENT when
// the surface is missing, so a config that failed to bootstrap cannot pass
// silently.
//
// ## Why `skills` and `skills-body` are separate surfaces
//
// `omp://skills.md` (probed against the installed omp) exposes skills to the
// model as:
//   "lightweight metadata in the system prompt (name + description)"
//   "on-demand content via the `read` tool against `skill://...`"
// and states plainly that "`globs` and `alwaysApply` are metadata here, not
// automatic skill-invocation controls" — unlike a rule, a skill has no
// frontmatter flag that injects its body. There is no setting either: every
// `skills.*` key `omp config list` prints is `registryUrl`, `enabled`, the
// `enable*` source toggles, `enableSkillCommands`, `customDirectories`,
// `ignoredSkills` and `includeSkills` — none of them inject a body.
//
// So in a plain `omp --print` run the roster line is present unconditionally
// (the `read` tool advertises skill URIs) but the *body* only reaches the model
// if the model chooses to `read skill://harbor-hello` (or the user invokes
// `/skill:harbor-hello`). Reporting the roster line as "skills present" would
// therefore over-claim the card's bar ("resolvable by name AND its body reaches
// the model"), so the two facts get their own verdicts: `skills` is the roster
// entry, `skills-body` is the body text itself.
//
// ## Reporting once
//
// A surface is reported on first sight and not repeated, so a long run is not
// flooded. The one deliberate exception is an upgrade: a surface reported
// ABSENT on an early request is reported again the first time it is seen
// present, because a skill body arrives only after the model reads it — in a
// request later than the first. A surface already seen present is never
// re-reported.

const SURFACES: ReadonlyArray<readonly [string, string]> = [
  ["AGENTS.md", "Marker: HARBOR_OMP_FIXTURE_CONFIG"],
  ["rules", "Marker: HARBOR_OMP_FIXTURE_RULE"],
  ["skills", "- harbor-hello: Greeting skill"],
  ["skills-body", "HARBOR_OMP_FIXTURE_SKILL"],
  ["agents", "Hello subagent"],
  ["extensions", "harbor_hello"],
];

const TOOL_MARKER = "harbor_hello";

export type SurfaceEmit = (line: string) => void;

// The reporter core, kept separate from the extension so a test can drive it
// over crafted payloads without loading omp. State is per-instance, so one
// reporter spans every request of a single session.
export function createSurfaceReporter(emit: SurfaceEmit): (payload: unknown) => void {
  const reportedAbsent = new Set<string>();
  const reportedPresent = new Set<string>();

  return (payload: unknown) => {
    let request: string;
    try {
      request = payload === undefined ? "" : JSON.stringify(payload);
    } catch (error) {
      emit(`HARBOR_OMP_FIXTURE_SURFACE report failed: ${String(error)}`);
      return;
    }

    for (const [surface, marker] of SURFACES) {
      if (reportedPresent.has(surface)) continue;
      if (!request.includes(marker)) {
        // Report ABSENT once, then stay quiet until (if ever) it appears.
        if (reportedAbsent.has(surface)) continue;
        reportedAbsent.add(surface);
        emit(
          `HARBOR_OMP_FIXTURE_SURFACE ${surface} ABSENT marker=${JSON.stringify(marker)}`,
        );
        continue;
      }
      reportedPresent.add(surface);
      emit(
        `HARBOR_OMP_FIXTURE_SURFACE ${surface} present marker=${JSON.stringify(marker)}`,
      );
    }
  };
}

export default function harborHello(pi: ExtensionAPI) {
  console.error("HARBOR_OMP_FIXTURE_EXTENSION loaded extensions/harbor-hello.ts");

  pi.registerCommand("harbor-hello", {
    description: "Report the hello marker",
    handler: async (_args, ctx) => {
      ctx.ui.notify("HARBOR_OMP_FIXTURE_EXTENSION", "info");
    },
  });

  pi.registerTool({
    name: TOOL_MARKER,
    label: "Hello",
    description: "Echoes the hello marker.",
    loadMode: "essential",
    parameters: pi.zod.object({}),
    async execute() {
      return {
        content: [{ type: "text", text: "HARBOR_OMP_FIXTURE_TOOL" }],
        details: {},
      };
    },
  });

  const reportSurfaces = createSurfaceReporter((line) => console.error(line));

  pi.on("before_provider_request", async (event) => {
    // Fail open: a reporter fault must never block or alter the request.
    try {
      reportSurfaces(event.payload);
    } catch (error) {
      console.error(`HARBOR_OMP_FIXTURE_SURFACE report failed: ${String(error)}`);
    }
  });
}
