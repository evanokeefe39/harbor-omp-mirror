"""Shared shell snippet for installing Bun in agent environments."""

BUN_VERSION = "1.3.14"


def bun_install_snippet(version: str = BUN_VERSION) -> str:
    """Return a shell snippet that installs Bun and leaves it on ``PATH``.

    Leaves ``BUN_INSTALL`` exported so callers can chain
    ``bun install -g ...`` with ``&&``.

    Requires ``curl``, ``unzip`` and a glibc-based distro: the official Bun
    binaries do not run on musl (e.g. Alpine).
    """
    return (
        'export BUN_INSTALL="$HOME/.bun" && '
        'export PATH="$BUN_INSTALL/bin:$PATH" && '
        f'curl -fsSL https://bun.sh/install | bash -s "bun-v{version}" && '
        "command -v bun &>/dev/null || "
        "{ echo 'Error: bun failed to load' >&2; exit 1; } && "
        "bun --version"
    )
