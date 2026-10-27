"""Native math policy shared by scene episodes and pick-evaluation trials."""

from contextlib import contextmanager


@contextmanager
def single_threaded_blas():
    """Limit loaded BLAS pools for an episode, restoring the caller's settings on exit.

    Enter after Isaac startup and environment construction: SimulationApp overwrites
    OPENBLAS_NUM_THREADS with its CPU thread limit (32 by default), so libraries loaded
    during startup ignore the launcher's value. Discover pools at each entry, including
    NumPy and SciPy loaded since the preceding episode. PhysX/OpenMP are independent.
    """
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1, user_api='blas'):
        yield
