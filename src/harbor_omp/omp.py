"""Install OMP (oh-my-pi) into a task container and solve the task with it.

Places the configured OMP config root where the CLI resolves it, runs the prompt
headlessly, and reads the run back as an ATIF trajectory with token and cost
accounting.
"""

import base64
import binascii
import json
import re
import shlex
from collections.abc import Iterator
from itertools import count
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, override

from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import (
    BaseInstalledAgent,
    with_prompt_template,
)
from harbor.agents.model_connection import ModelConnectionSpec
from harbor.agents.options import Cli, InstalledAgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent,
    ContentPart,
    FinalMetrics,
    ImageSource,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.utils.trajectory_utils import format_trajectory_json
from pydantic import Field

from harbor_omp.bun_install import bun_install_snippet

_OMP_PACKAGE = "@oh-my-pi/pi-coding-agent"

# Both are needed, and both are honoured for the default profile only:
# ``PI_CODING_AGENT_DIR`` alone silently drops subagents, commands and tools.
_OMP_AGENT_DIR_ENV = "PI_CODING_AGENT_DIR"
_OMP_CONFIG_DIR_ENV = "PI_CONFIG_DIR"

# Removed once the bootstrap finishes; named so that, if it ever survives, it
# does not tell the agent under test that it is being evaluated.
_REMOTE_ROOT = PurePosixPath("/tmp/omp-config-staging")
_REMOTE_AGENT_DIR = _REMOTE_ROOT / "agent"
_REMOTE_ARCHIVE = _REMOTE_ROOT / "config.tar.gz"
_REMOTE_EXTRACT_DIR = _REMOTE_ROOT / "extract"

_MCP_FILENAME = "mcp.json"
_MCP_SCHEMA_URL = (
    "https://raw.githubusercontent.com/can1357/oh-my-pi/main/"
    "packages/coding-agent/src/config/mcp-schema.json"
)

# A single, safe path component: the name is both a ``$HOME`` segment and the
# ``PI_CONFIG_DIR`` value (a leading dot is fine; ``.``, ``..`` are not).
_CONFIG_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_CONFIG_NAME_RESERVED = frozenset({".", ".."})

# Wraps the agent: a sampled memory trace for the whole agent phase, and a
# post-mortem dump on a non-zero exit only. ``oom_kill`` in ``memory.events`` is
# the field that separates an OOM kill from any other SIGKILL. Both files land in
# ``/logs/agent``, collected as ``agent/``. Why: docs/operations.md, defect D6.
_DIAGNOSTICS_SNIPPET = """\
C=/sys/fs/cgroup
probe() {
  while :; do
    {
      printf '%s ' "$(date -u +%FT%TZ)"
      printf 'cur=%s ' "$(cat $C/memory.current 2>/dev/null)"
      printf 'max=%s ' "$(cat $C/memory.max 2>/dev/null)"
      printf 'events=%s ' "$(cat $C/memory.events 2>/dev/null | tr '\\n' ',')"
      printf 'free=%s\\n' "$(free -m 2>/dev/null | awk 'NR==2{print $3"/"$2}')"
    } >> "$PROBE_LOG" 2>&1
    sleep 2
  done
}
probe & PROBE_PID=$!
postmortem() {
  {
    echo "=== agent exited $1 at $(date -u +%FT%TZ)"
    echo "-- memory.events"; cat $C/memory.events 2>/dev/null
    echo "-- memory.max"; cat $C/memory.max 2>/dev/null
    echo "-- free -m"; free -m 2>/dev/null
    echo "-- procs by rss"
    ps -eo pid,rss,comm --sort=-rss 2>/dev/null | head -12
    echo "-- dmesg tail"; dmesg 2>/dev/null | tail -30 || echo none
    echo "-- journalctl tail"
    journalctl -n 40 --no-pager 2>/dev/null | tail -40 || echo none
  } >> "$POSTMORTEM_LOG" 2>&1
}
"""

_ImageMediaType = Literal["image/png", "image/jpeg", "image/gif", "image/webp"]
_IMAGE_EXTENSIONS: dict[_ImageMediaType, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
}


class OmpOptions(InstalledAgentOptions):
    """Options for the OMP agent, each mapped to the CLI's own surface."""

    thinking: Annotated[
        Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"] | None,
        Cli("--thinking"),
    ] = Field(default=None, description="Thinking level.")
    advisor: Annotated[bool | None, Cli("--advisor")] = Field(
        default=None,
        description="Enable the advisor runtime (passively reviews each turn).",
    )
    max_time: Annotated[str | None, Cli("--max-time")] = Field(
        default=None,
        description="Stop the session after this duration (e.g. '10m', '1h').",
    )
    config_dir: str | None = Field(
        default=None,
        description=(
            "Host directory uploaded as the OMP agent directory "
            "(the shape of ~/.omp/agent)."
        ),
    )
    config_url: str | None = Field(
        default=None,
        description=(
            "Archive URL of the OMP config root (the shape of ~/.omp, i.e. the "
            "omp-home repository). Fetched and extracted in the task environment."
        ),
    )
    config_sha256: str | None = Field(
        default=None,
        description="Expected sha256 of config_url; verified before extraction.",
    )
    config_ref: str | None = Field(
        default=None,
        description=(
            "Human-readable provenance label for config_url/config_dir "
            "(e.g. a git tag or commit). Recorded in trial metadata."
        ),
    )
    config_name: str = Field(
        # OMP's own default: a bootstrapped root sits where an ordinary install,
        # and the bundled-defaults arm, keep theirs, and its name does not
        # announce the harness.
        default=".omp",
        description=(
            "Directory name (under $HOME) the config_url archive extracts to; "
            "also the PI_CONFIG_DIR value. Must be a single path component."
        ),
    )
    config_subdir: str = Field(
        default="agent",
        description=(
            "Path to the agent directory inside the config_url archive. "
            "Use '.' when the archive's top level is the agent directory."
        ),
    )


class Omp(BaseInstalledAgent):
    """Install oh-my-pi into the task container and run it with ``omp --print``."""

    capabilities = AgentCapabilities(
        atif=True,
        resume=True,
        skills=True,
        mcp_servers=True,
    )
    # Passthrough exposes exactly the env OMP reads its provider key from, so no
    # key rewriting is needed.
    MODEL_CONNECTION = ModelConnectionSpec(passthrough=True)

    _OUTPUT_FILENAME = "omp.txt"
    _TRAJECTORY_FILENAME = "trajectory.json"
    _SESSIONS_DIRECTORY = Path("omp/sessions")
    # Kill attribution (docs/operations.md, defect D6); collected as ``agent/``.
    _PROBE_FILENAME = "resource-probe.log"
    _POSTMORTEM_FILENAME = "postmortem.txt"

    options_model = OmpOptions
    options: OmpOptions

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the agent and its per-session resume bookkeeping."""
        super().__init__(*args, **kwargs)
        # Per-session-file line counts taken before a resume, so metrics and
        # copied context cover only the turns this run added.
        self._session_usage_start: dict[str, int] = {}

    @staticmethod
    @override
    def name() -> str:
        return "omp"

    @override
    def get_version_command(self) -> str | None:
        return 'export PATH="$HOME/.bun/bin:$PATH"; omp --version'

    @override
    def parse_version(self, stdout: str) -> str:
        # ``omp --version`` prints e.g. ``omp/18.10.1``.
        return stdout.strip().splitlines()[-1].strip().removeprefix("omp/").strip()

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        # ``unzip`` is not optional: ``bun.sh/install`` extracts the release zip
        # with it and aborts without it.
        await self.ensure_system_dependencies(
            environment, ("curl", "bash", "coreutils", "git", "tar", "unzip")
        )
        version_spec = f"@{self._version}" if self._version else "@latest"
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"{bun_install_snippet()} && "
                f"bun install -g --ignore-scripts "
                f"{shlex.quote(_OMP_PACKAGE)}{version_spec} && "
                'export PATH="$HOME/.bun/bin:$PATH" && '
                "omp --version"
            ),
        )
        await self._bootstrap_config(environment)

    # ------------------------------------------------------------------
    # Configuration bootstrap + provenance
    # ------------------------------------------------------------------

    def _validated_config_name(self) -> str:
        name = self.options.config_name
        if name in _CONFIG_NAME_RESERVED or not _CONFIG_NAME_RE.match(name):
            raise ValueError(
                "config_name must be a single path component (letters, digits, "
                f"'.', '_', '-'; not '.' or '..'); got {name!r}"
            )
        return name

    def _agent_subdir(self) -> str:
        return self.options.config_subdir.strip().strip("/")

    def _config_root_expr(self) -> str:
        """Shell expression for the bootstrapped config root.

        May contain ``$HOME``; callers must not shell-quote the result.
        """
        return f'"$HOME/{self._validated_config_name()}"'

    def _agent_dir_expr(self) -> str:
        """Shell expression for the agent directory inside the config root.

        May contain ``$HOME``; callers must not shell-quote the result.
        """
        name = self._validated_config_name()
        subdir = self._agent_subdir()
        if subdir in ("", "."):
            return f'"$HOME/{name}"'
        return f'"$HOME/{name}/{subdir}"'

    def _config_lineage(self) -> dict[str, Any]:
        """Provenance record for the configuration this run used."""
        if self.options.config_url:
            source = "archive"
        elif self.options.config_dir:
            source = "directory"
        else:
            source = "bundled-defaults"
        lineage: dict[str, Any] = {
            "source": source,
            "package": _OMP_PACKAGE,
            "url": self.options.config_url,
            "ref": self.options.config_ref,
            "sha256": self.options.config_sha256,
            "dir": self.options.config_dir,
            "config_name": None,
            "agent_dir": None,
        }
        if source != "bundled-defaults":
            lineage["config_name"] = self.options.config_name
            lineage["agent_dir"] = self._agent_dir_expr()
        return lineage

    async def _bootstrap_config(self, environment: BaseEnvironment) -> None:
        staging = shlex.quote(_REMOTE_ROOT.as_posix())
        await self.exec_as_agent(
            environment,
            command=f"mkdir -p {staging} && chmod 700 {staging}",
        )
        # Provenance goes to ``AgentContext.metadata`` only, never into the
        # container: anything the agent can read must not name the harness.
        if self.options.config_url:
            await self._fetch_config_archive(environment)
        elif self.options.config_dir:
            await self._upload_config_dir(environment)
        await self.exec_as_agent(environment, command=f"rm -rf {staging}")

    async def _fetch_config_archive(self, environment: BaseEnvironment) -> None:
        url = self.options.config_url
        if url is None:
            raise ValueError("config_url is not set")
        name = self._validated_config_name()
        archive = shlex.quote(_REMOTE_ARCHIVE.as_posix())
        extract = shlex.quote(_REMOTE_EXTRACT_DIR.as_posix())
        dest = f'"$HOME/{name}"'

        verify = ""
        if self.options.config_sha256:
            verify = (
                f'printf "%s  %s\\n" {shlex.quote(self.options.config_sha256)} '
                f"{archive} | sha256sum -c - && "
            )

        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"curl -fsSL {shlex.quote(url)} -o {archive} && "
                f"{verify}"
                f"rm -rf {extract} {dest} && "
                f"mkdir -p {extract} && "
                f"tar -xzf {archive} -C {extract} --strip-components=1 && "
                f"mv {extract} {dest} && "
                f"rm -f {archive}"
            ),
        )

    async def _upload_config_dir(self, environment: BaseEnvironment) -> None:
        # Absolute, because Harbor's ``upload_dir`` resolves a relative path
        # against the *task's* directory, not the invoking one.
        source = Path(self.options.config_dir or "").expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"OMP config directory not found: {source}")
        staging = shlex.quote(_REMOTE_AGENT_DIR.as_posix())
        await self.exec_as_agent(environment, command=f"rm -rf {staging}")
        await environment.upload_dir(source, _REMOTE_AGENT_DIR.as_posix())
        # Move the upload under the config root so both accessors agree, then
        # prove the tree arrived: empty means OMP falls back to its defaults.
        agent_dir = self._agent_dir_expr()
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"mkdir -p {self._config_root_expr()} && "
                f"rm -rf {agent_dir} && "
                f"mv {staging} {agent_dir} && "
                f'if [ -z "$(ls -A {agent_dir})" ]; then '
                f'echo "OMP config bootstrap: {agent_dir} is empty after upload" >&2; '
                "exit 1; "
                "fi"
            ),
        )
        if environment.default_user is not None:
            user = shlex.quote(str(environment.default_user))
            await self.exec_as_root(
                environment,
                command=f"chown -R {user} {self._config_root_expr()}",
            )

    async def _register_skills(self, environment: BaseEnvironment) -> None:
        """Copy task-provided skills into the bootstrapped ``skills/`` dir.

        OMP discovers skills from ``<agent_dir>/skills/*/SKILL.md``, so the
        task's skills sit alongside the operator's own skills.
        """
        if not self.skills_dir:
            return
        skills_dir = f"{self._agent_dir_expr()}/skills"
        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {skills_dir} && "
                f"cp -r {shlex.quote(self.skills_dir)}/. {skills_dir}/ "
                "2>/dev/null || true"
            ),
        )

    async def _write_mcp_config(self, environment: BaseEnvironment) -> None:
        """Merge task-provided MCP servers into OMP's native ``mcp.json``.

        The user ``mcp.json`` is a single file, so a bootstrapped config's
        servers must be merged rather than overwritten.
        """
        if not self.mcp_servers:
            return
        path = f"{self._agent_dir_expr()}/{_MCP_FILENAME}"
        existing = await self.exec_as_agent(
            environment, command=f"cat {path} 2>/dev/null || true"
        )
        document: dict[str, Any] = {"$schema": _MCP_SCHEMA_URL, "mcpServers": {}}
        try:
            parsed = json.loads(existing.stdout or "")
            if isinstance(parsed, dict):
                document = parsed
        except json.JSONDecodeError:
            self.logger.debug("Ignoring unparseable existing OMP mcp.json")
        servers = document.setdefault("mcpServers", {})
        if not isinstance(servers, dict):
            servers = {}
            document["mcpServers"] = servers
        for server in self.mcp_servers:
            if server.transport == "stdio":
                servers[server.name] = {
                    "type": "stdio",
                    "command": server.command,
                    "args": list(server.args),
                }
            else:
                servers[server.name] = {"type": "http", "url": server.url}
        document.setdefault("$schema", _MCP_SCHEMA_URL)
        content = json.dumps(document, indent=2) + "\n"
        await self.exec_as_agent(
            environment,
            command=(
                f"printf '%s' {shlex.quote(content)} > {path} && chmod 600 {path}"
            ),
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    @override
    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        escaped_instruction = shlex.quote(instruction)

        if not self.model_name:
            raise ValueError("OMP requires a model name (provider/model).")

        access = self.model_connection
        env = dict(access.env)

        await self._register_skills(environment)
        await self._write_mcp_config(environment)

        config_env = ""
        if self.options.config_url or self.options.config_dir:
            # Both are set: each reaches a different consumer.
            config_env = (
                f"export {_OMP_CONFIG_DIR_ENV}="
                f"{shlex.quote(self._validated_config_name())}; "
                f"export {_OMP_AGENT_DIR_ENV}={self._agent_dir_expr()}; "
            )

        cli_flags = self.build_cli_flags()
        flags = f"{cli_flags} " if cli_flags else ""
        resume_flag = "--continue " if self._resume else ""
        model_arg = f"--model {shlex.quote(self.model_name)} "

        session_dir = self.environment_logs_dir / self._SESSIONS_DIRECTORY
        output_path = self.environment_logs_dir / self._OUTPUT_FILENAME
        probe_log = self.environment_logs_dir / self._PROBE_FILENAME
        postmortem_log = self.environment_logs_dir / self._POSTMORTEM_FILENAME

        self._session_usage_start = (
            await self._session_positions(environment) if self._resume else {}
        )

        await self.exec_as_agent(
            environment,
            command=(
                'export PATH="$HOME/.bun/bin:$PATH"; '
                f"{config_env}"
                f"PROBE_LOG={shlex.quote(str(probe_log))}; "
                f"POSTMORTEM_LOG={shlex.quote(str(postmortem_log))}; "
                f"mkdir -p {shlex.quote(str(session_dir))} || exit $?; "
                f"{_DIAGNOSTICS_SNIPPET}"
                "omp --print --mode json "
                f"--session-dir {shlex.quote(str(session_dir))} "
                "--auto-approve "
                f"{resume_flag}{model_arg}{flags}"
                f"{escaped_instruction} "
                f'2>&1 </dev/null | grep -v \'"type":"message_update"\' '
                f"| stdbuf -oL tee {shlex.quote(str(output_path))}; "
                # The pipeline is no longer the last statement, so its status
                # must be carried out explicitly -- otherwise ``kill`` succeeds
                # and a killed agent is reported to Harbor as a clean exit.
                "rc=$?; "
                'kill "$PROBE_PID" 2>/dev/null; '
                '[ "$rc" -eq 0 ] || postmortem "$rc"; '
                'exit "$rc"'
            ),
            env=env,
        )

    async def _session_positions(self, environment: BaseEnvironment) -> dict[str, int]:
        """Line count per session file, used to skip copied context on resume.

        OMP nests sessions as ``<dir>/<cwd-slug>/<timestamp>_<uuid>.jsonl``
        with subagent files under a sibling ``<timestamp>_<uuid>/`` directory,
        so the count is taken per file across the whole tree.
        """
        session_root = self.environment_logs_dir / self._SESSIONS_DIRECTORY
        result = await self.exec_as_agent(
            environment,
            command=(
                f"find {shlex.quote(str(session_root))} -type f "
                "-name '*.jsonl' -exec wc -l {} +"
            ),
        )
        positions: dict[str, int] = {}
        for line in (result.stdout or "").splitlines():
            count, separator, filename = line.strip().partition(" ")
            if not separator:
                continue
            try:
                positions[PurePosixPath(filename.strip()).name] = int(count)
            except ValueError:
                continue
        return positions

    # ------------------------------------------------------------------
    # Post-run: trajectory + metrics
    # ------------------------------------------------------------------

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        trajectory_path = self.logs_dir / self._TRAJECTORY_FILENAME
        try:
            trajectory_path.unlink(missing_ok=True)
        except OSError as exc:
            self.logger.debug(
                f"Failed to remove stale OMP trajectory file {trajectory_path}: {exc}"
            )

        context.metadata = {
            **(context.metadata or {}),
            "omp_config": self._config_lineage(),
        }

        loaded = self._load_trajectory()
        if loaded is not None:
            trajectory, accounting_metrics = loaded
            try:
                trajectory_path.write_text(
                    format_trajectory_json(trajectory.to_json_dict())
                )
                self.logger.debug(f"Wrote OMP trajectory to {trajectory_path}")
            except OSError as exc:
                self.logger.debug(
                    f"Failed to write trajectory file {trajectory_path}: {exc}"
                )

            context.n_input_tokens = sum(
                metrics.prompt_tokens or 0 for metrics in accounting_metrics
            )
            context.n_output_tokens = sum(
                metrics.completion_tokens or 0 for metrics in accounting_metrics
            )
            context.n_cache_tokens = sum(
                metrics.cached_tokens or 0 for metrics in accounting_metrics
            )
            costs = [
                metrics.cost_usd
                for metrics in accounting_metrics
                if metrics.cost_usd is not None
            ]
            context.cost_usd = sum(costs) if costs else None
            return

        self._populate_context_from_output(context)

    @staticmethod
    def _session_candidates(session_dir: Path) -> list[Path]:
        """Session files excluding subagent transcripts.

        A subagent transcript lives under ``<session>/`` where ``<session>.jsonl``
        exists as a sibling, so any file with an ancestor that is itself a
        session stem is dropped, as is the advisor transcript.
        """
        files = list(session_dir.rglob("*.jsonl"))
        session_roots = {path.with_suffix("") for path in files}
        return [
            path
            for path in files
            if path.name != "__advisor.jsonl"
            and not any(ancestor in session_roots for ancestor in path.parents)
        ]

    def _load_trajectory(self) -> tuple[Trajectory, list[Metrics]] | None:
        session_dir = self.logs_dir / self._SESSIONS_DIRECTORY
        try:
            session_files = self._session_candidates(session_dir)
            if not session_files:
                return None
            session_path = max(
                session_files,
                key=lambda path: (path.stat().st_mtime_ns, path.name),
            )
            lines = session_path.read_text().splitlines()
        except (OSError, UnicodeError) as exc:
            self.logger.debug(f"Failed to read OMP session logs: {exc}")
            return None

        events: list[tuple[int, dict[str, Any]]] = []
        for position, line in enumerate(lines, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append((position, event))

        try:
            copied_until = self._session_usage_start.get(session_path.name, 0)
            trajectory = self._convert_session_events_to_trajectory(
                self._active_session_events(events), copied_until=copied_until
            )
            if trajectory is None:
                return None
            return trajectory, self._accounting_metrics(events, copied_until)
        except Exception:
            self.logger.exception("Failed to convert OMP session to trajectory")
            return None

    def _accounting_metrics(
        self, events: list[tuple[int, dict[str, Any]]], copied_until: int
    ) -> list[Metrics]:
        """Collect usage from every model call made during the current run.

        ``model_usage`` events cover auxiliary calls (titles, judges, memory)
        that never appear as messages, so they are counted too; a resumed run
        skips events at or below the pre-run line count.
        """
        metrics: list[Metrics] = []
        for position, event in events:
            if position <= copied_until:
                continue
            usage: Any = None
            event_type = event.get("type")
            if event_type in {"compaction", "branch_summary", "model_usage"}:
                usage = event.get("usage")
            elif event_type == "message":
                message = event.get("message")
                if isinstance(message, dict) and message.get("role") in {
                    "assistant",
                    "toolResult",
                }:
                    usage = message.get("usage")
            if (event_metrics := self._metrics_from_usage(usage)) is not None:
                metrics.append(event_metrics)
        return metrics

    @staticmethod
    def _active_session_events(
        events: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[int, dict[str, Any]]]:
        """Return the current root-to-leaf path from OMP's append-only tree."""
        entries_by_id: dict[str, tuple[int, dict[str, Any]]] = {}
        leaf_id: str | None = None
        headers: list[tuple[int, dict[str, Any]]] = []

        for positioned_event in events:
            _, event = positioned_event
            if event.get("type") == "session":
                headers.append(positioned_event)
                continue
            event_id = event.get("id")
            if isinstance(event_id, str) and event_id:
                entries_by_id[event_id] = positioned_event
                leaf_id = event_id

        if leaf_id is None:
            return events

        branch: list[tuple[int, dict[str, Any]]] = []
        visited: set[str] = set()
        while leaf_id is not None and leaf_id not in visited:
            visited.add(leaf_id)
            candidate = entries_by_id.get(leaf_id)
            if candidate is None:
                break
            branch.append(candidate)
            parent_id = candidate[1].get("parentId")
            leaf_id = parent_id if isinstance(parent_id, str) and parent_id else None

        branch.reverse()
        return [*headers, *branch]

    def _save_image(
        self, image_data: str, mime_type: str, image_index: int
    ) -> tuple[str, _ImageMediaType] | tuple[None, None]:
        match = next(
            (item for item in _IMAGE_EXTENSIONS.items() if item[0] == mime_type),
            None,
        )
        if match is None:
            self.logger.debug("Unsupported OMP image MIME type: %s", mime_type)
            return None, None
        valid_mime_type, extension = match

        image_path = self.logs_dir / "images" / f"omp_image_{image_index}.{extension}"
        try:
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(base64.b64decode(image_data, validate=True))
        except (binascii.Error, OSError, ValueError):
            self.logger.debug("Failed to save OMP image", exc_info=True)
            return None, None
        return f"images/{image_path.name}", valid_mime_type

    def _content(
        self, content: Any, *, next_image_index: Iterator[int]
    ) -> str | list[ContentPart]:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        parts: list[ContentPart] = []
        has_image = False
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text" and isinstance(block.get("text"), str):
                parts.append(ContentPart(type="text", text=block["text"]))
                continue
            if block_type != "image":
                continue
            image_data = block.get("data")
            mime_type = block.get("mimeType")
            if not isinstance(image_data, str) or not isinstance(mime_type, str):
                continue
            image_path, valid_mime_type = self._save_image(
                image_data, mime_type, next(next_image_index)
            )
            if image_path is None or valid_mime_type is None:
                continue
            has_image = True
            parts.append(
                ContentPart(
                    type="image",
                    source=ImageSource(media_type=valid_mime_type, path=image_path),
                )
            )
        if has_image:
            return parts
        return "\n".join(part.text or "" for part in parts).strip()

    @staticmethod
    def _integer(value: Any) -> int:
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    @classmethod
    def _metrics_from_usage(cls, usage: Any) -> Metrics | None:
        if not isinstance(usage, dict):
            return None

        known_fields = {
            "input",
            "output",
            "cacheRead",
            "cacheWrite",
            "reasoning",
            "totalTokens",
            "cost",
        }
        if not known_fields.intersection(usage):
            return None

        input_tokens = cls._integer(usage.get("input"))
        output_tokens = cls._integer(usage.get("output"))
        cache_read_tokens = cls._integer(usage.get("cacheRead"))
        cache_write_tokens = cls._integer(usage.get("cacheWrite"))
        reasoning_tokens = cls._integer(usage.get("reasoning"))
        total_tokens = cls._integer(usage.get("totalTokens"))
        cost = usage.get("cost")
        cost_total = cost.get("total") if isinstance(cost, dict) else None
        cost_usd = (
            float(cost_total)
            if isinstance(cost_total, int | float) and cost_total > 0
            else None
        )

        return Metrics(
            prompt_tokens=input_tokens + cache_read_tokens + cache_write_tokens,
            completion_tokens=output_tokens,
            cached_tokens=cache_read_tokens,
            cost_usd=cost_usd,
            extra={
                "cache_write_tokens": cache_write_tokens,
                "reasoning_tokens": reasoning_tokens,
                "omp_total_tokens": total_tokens,
            },
        )

    @classmethod
    def _sum_metrics(cls, first: Metrics, second: Metrics) -> Metrics:
        first_extra = first.extra or {}
        second_extra = second.extra or {}
        costs = [cost for cost in (first.cost_usd, second.cost_usd) if cost is not None]
        return Metrics(
            prompt_tokens=(first.prompt_tokens or 0) + (second.prompt_tokens or 0),
            completion_tokens=(first.completion_tokens or 0)
            + (second.completion_tokens or 0),
            cached_tokens=(first.cached_tokens or 0) + (second.cached_tokens or 0),
            cost_usd=sum(costs) if costs else None,
            extra={
                key: cls._integer(first_extra.get(key))
                + cls._integer(second_extra.get(key))
                for key in (
                    "cache_write_tokens",
                    "reasoning_tokens",
                    "omp_total_tokens",
                )
            },
        )

    def _convert_session_events_to_trajectory(
        self,
        events: list[tuple[int, dict[str, Any]]],
        copied_until: int = 0,
    ) -> Trajectory | None:
        session_id = "unknown"
        default_model_name = self.model_name
        steps: list[Step] = []
        call_steps: dict[str, Step] = {}

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cached_tokens = 0
        total_cache_write_tokens = 0
        total_reasoning_tokens = 0
        total_omp_tokens = 0
        total_cost = 0.0
        image_indexes = count(1)

        def add_usage(usage: Any, *, copied: bool) -> Metrics | None:
            nonlocal total_prompt_tokens
            nonlocal total_completion_tokens
            nonlocal total_cached_tokens
            nonlocal total_cache_write_tokens
            nonlocal total_reasoning_tokens
            nonlocal total_omp_tokens
            nonlocal total_cost

            metrics = self._metrics_from_usage(usage)
            if metrics is None:
                return None
            if not copied:
                total_prompt_tokens += metrics.prompt_tokens or 0
                total_completion_tokens += metrics.completion_tokens or 0
                total_cached_tokens += metrics.cached_tokens or 0
                extra = metrics.extra or {}
                total_cache_write_tokens += self._integer(
                    extra.get("cache_write_tokens")
                )
                total_reasoning_tokens += self._integer(extra.get("reasoning_tokens"))
                total_omp_tokens += self._integer(extra.get("omp_total_tokens"))
                total_cost += metrics.cost_usd or 0.0
            return metrics

        for position, event in events:
            event_type = event.get("type")
            if event_type == "session":
                event_session_id = event.get("id")
                if isinstance(event_session_id, str) and event_session_id:
                    session_id = event_session_id
                continue
            if event_type == "model_change":
                # OMP records a single ``model`` field, not provider/modelId.
                model_id = event.get("model")
                if isinstance(model_id, str) and model_id:
                    default_model_name = model_id
                continue
            if event_type == "custom_message":
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        timestamp=self._event_timestamp(event),
                        source="system",
                        message=self._content(
                            event.get("content"), next_image_index=image_indexes
                        ),
                        is_copied_context=True if position <= copied_until else None,
                        extra={
                            **self._event_ids(event),
                            "omp_event_type": "custom_message",
                            "custom_type": event.get("customType"),
                            "display": event.get("display"),
                            "details": event.get("details"),
                        },
                    )
                )
                continue
            if event_type in {"compaction", "branch_summary"}:
                summary = event.get("summary")
                if not isinstance(summary, str) or not summary.strip():
                    continue
                is_copied = position <= copied_until
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        timestamp=self._event_timestamp(event),
                        source="agent",
                        model_name=default_model_name,
                        message=summary.strip(),
                        metrics=add_usage(event.get("usage"), copied=is_copied),
                        is_copied_context=True if is_copied else None,
                        llm_call_count=1,
                        extra={
                            "omp_event_type": event_type,
                            "omp_event_id": event.get("id"),
                            "tokens_before": self._integer(event.get("tokensBefore")),
                            "from_id": event.get("fromId"),
                            "from_hook": bool(event.get("fromHook", False)),
                        },
                    )
                )
                continue
            if event_type != "message":
                continue

            message = event.get("message")
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role in {"user", "developer"}:
                # ``developer`` carries OMP's system notices (continue/resume
                # prompts); both are non-agent-authored context.
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        timestamp=self._event_timestamp(event),
                        source="system" if role == "developer" else "user",
                        message=self._content(
                            message.get("content"), next_image_index=image_indexes
                        ),
                        is_copied_context=True if position <= copied_until else None,
                        extra=self._event_ids(event),
                    )
                )
                continue
            if role == "assistant":
                text_blocks: list[str] = []
                reasoning_blocks: list[str] = []
                tool_calls: list[ToolCall] = []
                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        block_type = block.get("type")
                        if block_type == "text" and isinstance(block.get("text"), str):
                            text_blocks.append(block["text"])
                        elif block_type == "thinking" and isinstance(
                            block.get("thinking"), str
                        ):
                            reasoning_blocks.append(block["thinking"])
                        elif block_type == "toolCall":
                            tool_call_id = block.get("id")
                            function_name = block.get("name")
                            if not isinstance(tool_call_id, str) or not isinstance(
                                function_name, str
                            ):
                                continue
                            arguments = block.get("arguments")
                            if not isinstance(arguments, dict):
                                arguments = {}
                            tool_calls.append(
                                ToolCall(
                                    tool_call_id=tool_call_id,
                                    function_name=function_name,
                                    arguments=arguments,
                                )
                            )

                if (
                    message.get("stopReason") == "aborted"
                    and not text_blocks
                    and not reasoning_blocks
                    and not tool_calls
                ):
                    continue

                is_copied = position <= copied_until
                step = Step(
                    step_id=len(steps) + 1,
                    timestamp=self._event_timestamp(event),
                    source="agent",
                    model_name=default_model_name,
                    message="\n".join(text_blocks).strip(),
                    reasoning_content="\n".join(reasoning_blocks).strip() or None,
                    tool_calls=tool_calls or None,
                    observation=Observation(results=[]) if tool_calls else None,
                    metrics=add_usage(message.get("usage"), copied=is_copied),
                    is_copied_context=True if is_copied else None,
                    llm_call_count=1,
                    extra={
                        **self._event_ids(event),
                        "provider": message.get("provider"),
                        "api": message.get("api"),
                        "response_id": message.get("responseId"),
                        "response_model": message.get("responseModel"),
                        "stop_reason": message.get("stopReason"),
                        "raw_stop_reason": message.get("rawStopReason"),
                    },
                )
                steps.append(step)
                for tool_call in tool_calls:
                    call_steps[tool_call.tool_call_id] = step
                continue
            if role != "toolResult":
                continue

            tool_call_id = message.get("toolCallId")
            if not isinstance(tool_call_id, str):
                continue
            call_step = call_steps.get(tool_call_id)
            if call_step is None:
                continue
            if call_step.observation is None:
                call_step.observation = Observation(results=[])
            call_step.observation.results.append(
                ObservationResult(
                    source_call_id=tool_call_id,
                    content=self._content(
                        message.get("content"), next_image_index=image_indexes
                    ),
                    extra={
                        "tool_name": message.get("toolName"),
                        "is_error": bool(message.get("isError", False)),
                    },
                )
            )
            tool_metrics = add_usage(
                message.get("usage"), copied=position <= copied_until
            )
            if tool_metrics is not None:
                if step.is_copied_context is True and position > copied_until:
                    steps.append(
                        Step(
                            step_id=len(steps) + 1,
                            timestamp=self._event_timestamp(event),
                            source="agent",
                            model_name=default_model_name,
                            message="",
                            metrics=tool_metrics,
                            llm_call_count=1,
                            extra={
                                **self._event_ids(event),
                                "omp_event_type": "tool_result_usage",
                                "source_call_id": tool_call_id,
                                "tool_name": message.get("toolName"),
                            },
                        )
                    )
                else:
                    step.metrics = (
                        tool_metrics
                        if step.metrics is None
                        else self._sum_metrics(step.metrics, tool_metrics)
                    )
                    step.llm_call_count = (step.llm_call_count or 0) + 1

        if not steps:
            return None

        has_copied_steps = any(step.is_copied_context is True for step in steps)
        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=session_id,
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=default_model_name,
            ),
            steps=steps,
            notes=(
                "final_metrics excludes steps marked is_copied_context; copied "
                "steps remain serialized to preserve resumed-session context."
                if has_copied_steps
                else None
            ),
            final_metrics=FinalMetrics(
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_cached_tokens=total_cached_tokens,
                total_cost_usd=total_cost if total_cost > 0 else None,
                total_steps=sum(step.is_copied_context is not True for step in steps),
                extra={
                    "cache_write_tokens": total_cache_write_tokens,
                    "reasoning_tokens": total_reasoning_tokens,
                    "omp_total_tokens": total_omp_tokens,
                },
            ),
        )

    @staticmethod
    def _event_timestamp(event: dict[str, Any]) -> str | None:
        timestamp = event.get("timestamp")
        return timestamp if isinstance(timestamp, str) else None

    @staticmethod
    def _event_ids(event: dict[str, Any]) -> dict[str, Any]:
        return {
            "omp_event_id": event.get("id"),
            "omp_parent_id": event.get("parentId"),
        }

    def _populate_context_from_output(self, context: AgentContext) -> None:
        output_file = self.logs_dir / self._OUTPUT_FILENAME
        if not output_file.exists():
            return

        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_read_tokens = 0
        total_cache_write_tokens = 0
        total_cost = 0.0

        for line in output_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("type") == "message_end":
                    message = event.get("message") or {}
                    if message.get("role") in {"assistant", "toolResult"}:
                        usage = message.get("usage") or {}
                        total_input_tokens += usage.get("input", 0)
                        total_output_tokens += usage.get("output", 0)
                        total_cache_read_tokens += usage.get("cacheRead", 0)
                        total_cache_write_tokens += usage.get("cacheWrite", 0)
                        cost = usage.get("cost") or {}
                        total_cost += cost.get("total", 0.0)
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue

        context.n_input_tokens = (
            total_input_tokens + total_cache_read_tokens + total_cache_write_tokens
        )
        context.n_output_tokens = total_output_tokens
        context.n_cache_tokens = total_cache_read_tokens
        context.cost_usd = total_cost if total_cost > 0 else None
