"""In-container verification of the OMP config bootstrap.

Asserts the two things a host-side test cannot see:

* the bootstrapped agent directory contains the fixture tree (R6), and
* task-declared MCP servers were **merged** into the bootstrapped ``mcp.json``
  rather than overwriting it (R5).

Both bootstrap paths place the config root at ``$HOME/<config_name>``
(``PI_CONFIG_DIR``); a ``config_dir`` upload passes through a staging
directory that the bootstrap removes once the tree is in place.
"""

import json
from pathlib import Path

MARKER = "HARBOR_OMP_FIXTURE_CONFIG"

_FIXTURE_FILES = (
    "AGENTS.md",
    "config.yml",
    "skills/harbor-hello/SKILL.md",
    "agents/harbor-hello.md",
    "extensions/harbor-hello.ts",
    "hooks/pre/harbor-hello.ts",
    "rules/harbor-hello.md",
    "commands/harbor-hello.md",
)


def _candidate_agent_dirs() -> list[Path]:
    """Locations the OMP agent directory can occupy after a bootstrap.

    Both bootstrap paths place the config root at ``$HOME/<config_name>``
    (``.omp`` by default), so the agent directory is its ``agent/``
    child. ``/root`` and ``/home/*`` are covered because the verifier may run
    as a different user than the agent.
    """
    candidates = [
        Path.home() / ".omp" / "agent",
        Path("/root/.omp/agent"),
    ]
    home_root = Path("/home")
    if home_root.is_dir():
        candidates.extend(sorted(home_root.glob("*/.omp/agent")))
    return candidates


def _agent_dir() -> Path:
    for candidate in _candidate_agent_dirs():
        if (candidate / "AGENTS.md").is_file():
            return candidate
    raise AssertionError(
        f"no bootstrapped OMP agent dir found among {_candidate_agent_dirs()}"
    )


def test_agent_directory_is_bootstrapped():
    agent_dir = _agent_dir()
    assert MARKER in (agent_dir / "AGENTS.md").read_text()
    missing = [name for name in _FIXTURE_FILES if not (agent_dir / name).is_file()]
    assert not missing, f"bootstrapped agent dir is missing {missing}"


def test_task_mcp_servers_merged_into_bootstrapped_mcp_json():
    agent_dir = _agent_dir()
    servers = json.loads((agent_dir / "mcp.json").read_text())["mcpServers"]
    assert "harbor-fixture-existing" in servers, "bootstrapped server was lost"
    assert "harbor-task-server" in servers, "task MCP server was not merged in"
