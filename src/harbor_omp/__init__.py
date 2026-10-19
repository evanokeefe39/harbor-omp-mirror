"""Make OMP (oh-my-pi) available to Harbor as an out-of-tree installed agent.

Harbor resolves an agent that is not built into the framework from an import
path::

    uv run harbor run -t <task> -a harbor_omp:Omp -m <provider>/<model>

``harbor_omp.omp:Omp`` is the same thing spelled out; the re-export is the
shorter form.
"""

from harbor_omp.omp import Omp, OmpOptions

__all__ = ["Omp", "OmpOptions"]
