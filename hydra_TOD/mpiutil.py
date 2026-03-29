from __future__ import annotations

"""
MPI parallelisation utilities for distributed TOD processing.

This module provides helper functions for partitioning work across MPI ranks,
parallel map-reduce operations, and thread-limited local parallelism via
joblib.  It is adapted from
https://github.com/radiocosmology/caput/blob/master/caput/mpiutil.py

References
----------
Zhang et al. (2026), RASTI, rzag024.
"""

from typing import Any, Callable, TypeVar

import numpy as np
from numpy.typing import NDArray
import logging


rank: int = 0
size: int = 1
_comm: Any = None
world: Any = None
rank0: bool = True

logger = logging.getLogger(__name__)

# import mpi4py.rc
# mpi4py.rc.initialize = False  # Disables auto-initialization of MPI

# Add MPI initialization control
_mpi_initialized: bool = False

T = TypeVar("T")


def init_mpi() -> None:
    """
    Initialise MPI once at module import.

    This function is called automatically when the module is first imported.
    Subsequent calls are no-ops.
    """
    global _mpi_initialized
    if not _mpi_initialized:
        from mpi4py import MPI

        if not MPI.Is_initialized():
            MPI.Init()
        _mpi_initialized = True


# Initialize MPI when module is imported
try:
    init_mpi()

    from mpi4py import MPI

    _comm = MPI.COMM_WORLD
    world = _comm
    rank = int(_comm.Get_rank())
    size = int(_comm.Get_size())

    if size > 1:
        logger.debug("Starting MPI rank=%i [size=%i]", rank, size)

    rank0 = rank == 0
except (ImportError, TypeError, AttributeError):
    # Fallback for environments where mpi4py is mocked or unavailable
    _comm = None
    world = None
    rank = 0
    size = 1
    rank0 = True
    MPI = None  # type: ignore[assignment]

from joblib import Parallel, delayed  # noqa: E402


def get_parallel_pool() -> Parallel:
    """
    Create a thread-limited joblib parallel pool that respects MPI resources.

    On Linux, the pool size is determined from CPU affinity; on macOS, it is
    computed as ``floor(total_cores / mpi_size) - 1``.  At least one thread
    is always reserved for MPI communication.

    Returns
    -------
    pool : joblib.Parallel
        A joblib ``Parallel`` instance configured with ``prefer="threads"``.
    """
    from math import floor
    from os import cpu_count
    import psutil

    # Get the current process
    process = psutil.Process()

    # Get total physical cores (not logical CPUs)
    total_cores = psutil.cpu_count(logical=False) or cpu_count()

    if hasattr(process, "cpu_affinity"):
        # Linux: Use actual assigned cores
        available_cores = len(process.cpu_affinity())
        safe_cores = max(1, available_cores - 1)  # Explicitly reserve 1 core
    else:
        # MacOS: Calculate fair share
        mpi_size = size if _comm else 1
        cores_per_rank = max(1, floor(total_cores / mpi_size))
        safe_cores = max(1, cores_per_rank - 1)  # Reserve 1 core per MPI process

    return Parallel(n_jobs=safe_cores, prefer="threads")


local_parallel_pool = get_parallel_pool()


def local_parallel_func(
    func: Callable[..., T],
    input_list: list[Any],
    single_arg: bool = True,
) -> list[T]:
    """
    Execute *func* over *input_list* using the local thread pool.

    Parameters
    ----------
    func : callable
        Function to apply to each element (or unpacked tuple) of
        *input_list*.
    input_list : list
        Items to iterate over.
    single_arg : bool, optional
        If ``True`` (default), each element is passed as a single argument.
        If ``False``, each element is unpacked as ``func(*item)``.

    Returns
    -------
    results : list
        List of return values, one per input item.
    """
    with local_parallel_pool as parallel:
        if single_arg:
            return parallel(delayed(func)(item) for item in input_list)
        else:  # multiple inputs for func
            return parallel(delayed(func)(*item) for item in input_list)


def partition_list(
    full_list: list[T] | NDArray,
    i: int,
    n: int,
    method: str = "con",
) -> list[T] | NDArray:
    """
    Partition a list into *n* pieces and return the *i*-th partition.

    Parameters
    ----------
    full_list : list or NDArray
        The list to partition.
    i : int
        Zero-based index of the partition to return.
    n : int
        Total number of partitions.
    method : {'con', 'alt', 'rand'}, optional
        Partitioning strategy:

        * ``'con'`` (default) -- contiguous chunks.
        * ``'alt'`` -- alternating / interleaved selection.
        * ``'rand'`` -- random permutation.

    Returns
    -------
    partition : list or NDArray
        The *i*-th partition of *full_list*.

    Raises
    ------
    ValueError
        If *method* is not one of the recognised strategies.
    """

    def _partition(N, n, i):
        # If partiion `N` numbers into `n` pieces,
        # return the start and stop of the `i` th piece
        base = N // n
        rem = N % n
        num_lst = rem * [base + 1] + (n - rem) * [base]
        cum_num_lst = np.cumsum([0] + num_lst)

        return cum_num_lst[i], cum_num_lst[i + 1]

    N = len(full_list)
    start, stop = _partition(N, n, i)

    if method == "con":
        return full_list[start:stop]
    elif method == "alt":
        return full_list[i::n]
    elif method == "rand":
        choices = np.random.permutation(N)[start:stop]
        return [full_list[i] for i in choices]
    else:
        raise ValueError("Unknown partition method %s" % method)


def partition_list_mpi(
    full_list: list[T] | NDArray,
    method: str = "con",
    comm: Any = _comm,
) -> list[T] | NDArray:
    """
    Return the partition of a list assigned to the current MPI rank.

    A convenience wrapper around :func:`partition_list` that uses the
    rank and size from the given communicator.

    Parameters
    ----------
    full_list : list or NDArray
        The list to partition.
    method : {'con', 'alt', 'rand'}, optional
        Partitioning strategy (see :func:`partition_list`).
    comm : MPI.Comm, optional
        MPI communicator. Default is ``MPI.COMM_WORLD``.

    Returns
    -------
    partition : list or NDArray
        The partition of *full_list* for this rank.
    """
    if comm is not None:
        rank = comm.rank
        size = comm.size

    return partition_list(full_list, rank, size, method=method)


def parallel_map_gather(
    func: Callable[..., T],
    glist: list[Any],
    multi_inputs: bool = False,
    root: int | None = None,
    method: str = "con",
    comm: Any = _comm,
) -> list[T] | None:
    """
    Apply a function to a list in parallel across MPI ranks and gather results.

    Each rank operates on its partition of *glist*.  Results are gathered
    (optionally to a single root) and returned in the original list order.

    Parameters
    ----------
    func : callable
        Function to apply to each element of *glist*.
    glist : list
        Global list to map over.  Must be identical on all ranks.
    multi_inputs : bool, optional
        If ``True``, each element of *glist* is unpacked as
        ``func(*item)``. Default is ``False``.
    root : int or None, optional
        If ``None`` (default), all ranks receive the full result list.  If
        an integer, only that rank receives the results (others return
        ``None``).
    method : {'con', 'alt', 'rand'}, optional
        Partitioning strategy. Default is ``'con'``.
    comm : MPI.Comm, optional
        MPI communicator. Default is ``MPI.COMM_WORLD``.

    Returns
    -------
    results : list or None
        Ordered list of ``func`` return values.  ``None`` on non-root ranks
        when *root* is specified.
    """

    # Synchronize
    barrier(comm=comm)

    # If we're only on a single node, then just perform without MPI
    if comm is None or comm.size == 1:
        if multi_inputs:
            return [func(*item) for item in glist]
        else:
            return [func(item) for item in glist]

    # Pair up each list item with its position.
    zlist = list(enumerate(glist))

    # Partition list based on MPI rank
    llist = partition_list_mpi(zlist, method=method, comm=comm)

    # Operate on sublist
    if multi_inputs:
        flist = [(ind, func(*item)) for ind, item in llist]
    else:
        flist = [(ind, func(item)) for ind, item in llist]

    barrier(comm=comm)

    rlist = None
    if root is None:
        # Gather all results onto all ranks
        rlist = comm.allgather(flist)
    else:
        # Gather all results onto the specified rank
        rlist = comm.gather(flist, root=root)

    if rlist is not None:
        # Flatten the list of results
        flatlist = [item for sublist in rlist for item in sublist]

        # Sort into original order
        sortlist = sorted(flatlist, key=(lambda item: item[0]))

        # Synchronize
        barrier(comm=comm)

        # Extract the return values into a list
        return [item for ind, item in sortlist]
    else:
        return None


def parallel_map_sum(
    func: Callable[..., Any],
    glist: list[Any],
    multi_inputs: bool = False,
    root: int | None = None,
    method: str = "con",
    comm: Any = _comm,
) -> Any:
    """
    Apply a function to a list in parallel and reduce results by summation.

    Each rank evaluates *func* on its partition and the partial sums are
    reduced across ranks.

    Parameters
    ----------
    func : callable
        Function to apply.  Must return a value that supports ``+``.
    glist : list
        Global list to map over.
    multi_inputs : bool, optional
        If ``True``, unpack each element as ``func(*item)``.
    root : int or None, optional
        If ``None``, all ranks receive the sum.  Otherwise only the
        specified rank.
    method : {'con', 'alt', 'rand'}, optional
        Partitioning strategy.
    comm : MPI.Comm, optional
        MPI communicator.

    Returns
    -------
    result : Any
        Global sum of all ``func`` return values.
    """

    # Synchronize
    barrier(comm=comm)

    # If we're only on a single node, then just perform without MPI
    if comm is None or comm.size == 1:
        if multi_inputs:
            result = [func(*item) for item in glist]
        else:
            result = [func(item) for item in glist]
        # get the sum of the results
        result = sum(result)
        return result

    # Partition list based on MPI rank
    llist = partition_list_mpi(glist, method=method, comm=comm)

    # Operate on sublist
    if multi_inputs:
        fsum = sum([func(*item) for item in llist])
    else:
        fsum = sum([func(item) for item in llist])

    barrier(comm=comm)

    if root is None:
        # Reduce all results onto all ranks
        rsum = comm.allreduce(fsum, op=MPI.SUM)
    else:
        # Reduce all results onto the specified rank
        rsum = comm.reduce(fsum, op=MPI.SUM, root=root)

    return rsum


def parallel_jobs_no_gather_no_return(
    func: Callable[..., Any],
    glist: list[Any],
    method: str = "con",
    comm: Any = _comm,
) -> list[Any] | None:
    """
    Apply a function across MPI ranks without gathering results.

    Useful for side-effect-only operations (e.g. writing files) where return
    values are not needed.

    Parameters
    ----------
    func : callable
        Function to apply.
    glist : list
        Global zipped list to map over.
    method : {'con', 'alt', 'rand'}, optional
        Partitioning strategy.
    comm : MPI.Comm, optional
        MPI communicator.

    Returns
    -------
    results : list or None
        Return values when running on a single node; ``None`` otherwise.
    """

    # Synchronize
    barrier(comm=comm)

    # If we're only on a single node, then just perform without MPI
    if comm is None or comm.size == 1:
        return [func(item) for item in glist]

    # Partition list based on MPI rank
    llist = partition_list_mpi(glist, method=method, comm=comm)

    # Operate on sublist
    for zipped_item in llist:
        func(zip(*zipped_item))

    # Synchronize
    barrier(comm=comm)
    return None


def barrier(comm: Any = _comm) -> None:
    """
    Synchronise all MPI processes.

    A no-op when running with a single rank or without MPI.

    Parameters
    ----------
    comm : MPI.Comm, optional
        MPI communicator. Default is ``MPI.COMM_WORLD``.
    """
    if comm is not None and comm.size > 1:
        comm.Barrier()
