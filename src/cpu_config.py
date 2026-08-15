"""CPU allocation and thread-pool configuration.

Stdlib only, and deliberately free of numpy/pymdp imports: BLAS and OpenMP read
their thread limits from the environment the first time numpy is imported, so
entry points import this module and call `limit_blas_threads()` before anything
that pulls in numpy. That is also why this lives at the package root rather than
under src/utils/ -- importing `src.utils.anything` drags numpy in via
src/utils/__init__.py.

Core-count detection handles the three ways this code gets run:

  * under a scheduler (SLURM), where the job was granted a specific number of
    CPUs that may be far smaller than the node it landed on;
  * inside a container or cpuset, where a cgroup quota or an affinity mask caps
    us below the machine size;
  * on a plain local machine, where we should leave the user some headroom.

`os.cpu_count()` reports the machine, not the grant, so using it directly
oversubscribes a SLURM allocation and gets the job throttled by the cgroup.
"""
import os
import re

# Every thread-pool knob numpy might consult, depending on which BLAS it was
# built against. One compute thread per worker process is what we want: the
# arrays here are tiny, and 100+ workers each spinning up their own thread pool
# is pure contention.
_THREAD_ENV_VARS = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
)

WORKER_ENV_VAR = 'MODEL_COMP_MAX_WORKERS'


def limit_blas_threads(n=1):
    """Pin BLAS/OpenMP thread pools to `n` threads per process.

    Must be called before numpy is first imported. Uses setdefault so an
    explicitly exported OMP_NUM_THREADS (etc.) still wins.
    """
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, str(n))


def _positive_int(value):
    """Parse a positive int, returning None for anything else."""
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _slurm_cpus():
    """CPUs granted by SLURM as (count, source), or (None, None) if not under SLURM."""
    for var in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE'):
        n = _positive_int(os.environ.get(var))
        if n:
            return n, var

    # SLURM_JOB_CPUS_PER_NODE is formatted like '36' or '36(x2)' for a job
    # spanning several nodes; the leading integer is this node's share.
    raw = os.environ.get('SLURM_JOB_CPUS_PER_NODE')
    if raw:
        match = re.match(r'\s*(\d+)', raw)
        if match:
            n = _positive_int(match.group(1))
            if n:
                return n, 'SLURM_JOB_CPUS_PER_NODE'
    return None, None


def _cgroup_cpus():
    """CPU quota imposed by a cgroup (containers, some schedulers), or None.

    A quota is a fraction of wall time, not a set of cores, so it is converted
    to a whole-core equivalent.
    """
    try:  # cgroup v2
        with open('/sys/fs/cgroup/cpu.max') as fh:
            quota_s, period_s = fh.read().split()[:2]
        if quota_s != 'max':
            quota, period = int(quota_s), int(period_s)
            if quota > 0 and period > 0:
                return max(1, quota // period)
    except (OSError, ValueError, IndexError):
        pass

    try:  # cgroup v1
        with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us') as fh:
            quota = int(fh.read().strip())
        with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us') as fh:
            period = int(fh.read().strip())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return None


def detect_available_cpus():
    """Return (n_cpus, source, is_allocation).

    `is_allocation` is True when the count came from an explicit grant -- a
    SLURM variable, a cgroup quota, or an affinity mask narrower than the
    machine -- rather than from the raw machine size. Callers use it to decide
    whether reserving headroom makes sense: cores handed out by a scheduler are
    already exclusively ours, so holding some back just wastes the request.
    """
    machine = os.cpu_count() or 1

    n, var = _slurm_cpus()
    if n:
        return min(n, machine), var, True

    affinity = None
    if hasattr(os, 'sched_getaffinity'):  # Linux only
        try:
            affinity = len(os.sched_getaffinity(0))
        except OSError:
            affinity = None

    # An affinity mask or cgroup quota below the machine size is a hard cap
    # someone placed on us; honour the tightest one.
    limits = [
        (count, source)
        for count, source in ((affinity, 'cpu affinity'), (_cgroup_cpus(), 'cgroup quota'))
        if count and count < machine
    ]
    if limits:
        n, source = min(limits)
        return n, source, True

    if affinity:
        return affinity, 'cpu affinity', False
    return machine, 'os.cpu_count', False


def default_reserved_cores(n_cpus, is_allocation):
    """Cores to leave free when the caller does not say.

    Under an explicit allocation, none: the scheduler already fenced these off
    for us. On a shared or personal machine, keep a little headroom so the box
    stays usable -- one core on a laptop, a handful on a big login node.
    """
    if is_allocation:
        return 0
    return max(1, n_cpus // 16)


def resolve_worker_count(reserve=None, respect_env=True):
    """Return (workers, description) for the model-fitting process pool.

    `reserve=None` applies `default_reserved_cores`; pass an int to override.
    An explicit MODEL_COMP_MAX_WORKERS always wins, which is both how a user
    forces a worker count and how the entry point hands its decision down to
    the fitting code.
    """
    if respect_env:
        forced = _positive_int(os.environ.get(WORKER_ENV_VAR))
        if forced:
            return forced, f"{forced} workers (set by {WORKER_ENV_VAR})"

    n_cpus, source, is_allocation = detect_available_cpus()
    if reserve is None:
        reserve = default_reserved_cores(n_cpus, is_allocation)
    reserve = max(0, min(int(reserve), n_cpus - 1))
    workers = max(1, n_cpus - reserve)

    kind = 'allocated' if is_allocation else 'detected'
    return workers, (
        f"{workers} workers ({n_cpus} CPUs {kind} via {source}"
        f"{f', {reserve} reserved' if reserve else ''})"
    )
