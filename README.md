# harbor-omp-mirror

The eval agent for OMP — a Harbor installed agent that runs the `omp` coding CLI
inside a task container — published at each release of its private source
repository, so a trial can install and run it from a tag instead of a private
checkout.

**This is a build artifact: do not open pull requests here.** Every file below is
written by the publisher at release time; an edit here is overwritten by the next
release and lost. Change the source repository and cut a release.

## Install

    uv add "harbor-omp @ git+https://github.com/evanokeefe39/harbor-omp-mirror@v0.4.0"

Harbor resolves an installed agent by import path. A job that runs this agent
names it as `harbor_omp:Omp`; the job config itself belongs to the runner, not to
this mirror:

    uv run --with "harbor-omp @ git+https://github.com/evanokeefe39/harbor-omp-mirror@v0.4.0" harbor run -c <job>.yaml --yes -o <jobs-dir>

## Contents

- `src/harbor_omp/` — the adapter (`Omp(BaseInstalledAgent)`), the Bun install and
  the config bootstrap
- `pyproject.toml`, `LICENSE`, `.gitattributes` — the package, as an install reads it

This is the Installed Agent only. The job configs and the fixtures a job
bootstraps belong to the runner: they name the task set, the environment, the
model and the harness config to measure, all of which change per run and per
question, so they cannot be pinned by an agent release.

## Not included

The source repository's tests, docs, backlog, ledgers and `scripts/`, its job
configs and fixtures, and every `jobs/<job_name>/` run directory (a trial writes
its trajectory, logs and result there). None of them is part of the agent.

Published at `v0.4.0`.
