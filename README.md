# harbor-omp-mirror

The eval agent for OMP — a Harbor installed agent that runs the `omp` coding CLI
inside a task container — published at each release of its private source
repository, so a trial can install and run it from a tag instead of a private
checkout.

**This is a build artifact: do not open pull requests here.** Every file below is
written by the publisher at release time; an edit here is overwritten by the next
release and lost. Change the source repository and cut a release.

## Install

    uv add "harbor-omp @ git+https://github.com/evanokeefe39/harbor-omp-mirror@mirror-preview.1"

Harbor resolves an installed agent by import path, so a job runs straight from a
checkout of this tag:

    uv run harbor run -c jobs/<job>.yaml --yes -o jobs

## Contents

- `src/harbor_omp/` — the adapter (`Omp(BaseInstalledAgent)`), the Bun install and
  the config bootstrap
- `jobs/*.yaml` — the job configs a trial is launched from
- `fixtures/` — the config fixtures and the probe task those jobs bootstrap
- `pyproject.toml`, `LICENSE`, `.gitattributes` — the package, as an install reads it

## Not included

The source repository's tests, docs, backlog, ledgers and `scripts/`, and every
`jobs/<job_name>/` run directory (a trial writes its trajectory, logs and result
there). None of them is part of the agent.

Published at `mirror-preview.1`.
