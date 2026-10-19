import type { HookAPI } from "@oh-my-pi/pi-coding-agent/extensibility/hooks";

// Surface-probe hook: registered under hooks/pre/ so ambient native
// discovery picks it up. It reports on stderr at module evaluation and again
// from its `session_start` handler, because stderr is the one channel a plain
// `omp --print` run carries into the trial's `agent/omp.txt` (`run()` merges it
// with `2>&1`). The `pi.logger.debug` line alone would prove nothing: omp
// writes logs under the state root's `logs/` directory, which a trial does not
// collect, so a hook that only logs leaves no evidence that it ever ran.
export default function harborHelloHook(pi: HookAPI) {
  console.error("HARBOR_OMP_FIXTURE_HOOK loaded hooks/pre/harbor-hello.ts");
  pi.on("session_start", () => {
    console.error("HARBOR_OMP_FIXTURE_HOOK session_start handler ran");
    pi.logger.debug("HARBOR_OMP_FIXTURE_HOOK loaded");
  });
}
