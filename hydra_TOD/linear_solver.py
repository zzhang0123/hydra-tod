from __future__ import annotations

# This is exactly the linear_solver in Hydra.
try:
    from mpi4py.MPI import SUM as MPI_SUM
    from mpi4py.MPI import LAND as MPI_LAND
except:
    pass

from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mpi4py import MPI


def matvec_mpi(
    comm_row: MPI.Intracomm,
    mat_block: NDArray[np.floating],
    vec_block: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Perform a distributed matrix-vector product for one row of a block matrix.

    Each block in the matrix row is multiplied by the corresponding row block
    of the vector. The result on each worker is then summed together to give
    the result for the corresponding row of the result vector.

    All workers in the row will possess the result for the same row of the
    result vector.

    For example, for the first row of this (block) linear system::

        ( A B C )     ( x )     ( r0 )
        ( D E F )  .  ( y )  =  ( r1 )
        ( G H I )     ( z )     ( r2 )

    workers 0, 1, and 2 will compute Ax, By, and Cz respectively. They will
    then collectively sum over their results to obtain ``r0 = Ax + By + Cz``.
    The three workers will all possess copies of ``r0``.

    Parameters
    ----------
    comm_row : MPI.Intracomm
        MPI group communicator for a row of the block matrix.
    mat_block : NDArray[np.floating]
        Block of the matrix belonging to this worker, shape ``(m, n)``.
    vec_block : NDArray[np.floating]
        Block of the vector belonging to this worker, shape ``(n,)``.

    Returns
    -------
    res_block : NDArray[np.floating]
        Block of the result vector corresponding to this row, shape ``(m,)``.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    # Do matrix-vector product for the available blocks
    y = mat_block @ vec_block

    # Do reduce to all members of this column group
    ytot = np.zeros_like(y)
    comm_row.Allreduce(y, ytot, op=MPI_SUM)
    return ytot


# Backward-compatible alias
matvec_mpi_v2 = matvec_mpi


def setup_mpi_blocks(
    comm: MPI.Comm,
    matrix_shape: tuple[int, int],
    split: int = 1,
) -> tuple[
    tuple[MPI.Intracomm, ...] | None,
    dict[int, tuple[int, int]],
    tuple[int, int] | dict,
]:
    """
    Set up a scheme for dividing a linear system into MPI blocks.

    This function determines the number and size of the blocks, creates a map
    between MPI workers and blocks, and sets up MPI communicator groups needed
    by the CG solver to communicate intermediate results.

    The linear system matrix operator is assumed to be square, and the blocks
    must also be square. The blocks will be zero-padded at the edges if the
    operator cannot be evenly divided into the blocks.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator object for all active workers.
    matrix_shape : tuple of int
        The shape ``(N, N)`` of the linear operator matrix that is to be
        divided into blocks.
    split : int, optional
        How many rows and columns to split the matrix into. For instance,
        ``split=2`` will split the matrix into 2 rows and 2 columns, for a
        total of 4 blocks. Default is 1.

    Returns
    -------
    comm_groups : tuple of MPI.Intracomm or None
        Group communicators ``(active, row, col, diag)``.  These correspond to
        the MPI workers that are active, and the ones for each row, each
        column, and along the diagonal of the block structure, respectively.
        Each worker returns its own set of communicators; where it is not a
        member of a relevant group, ``None`` is returned instead.  Returns
        ``None`` entirely for inactive workers.
    block_map : dict
        Dictionary mapping worker ID to ``(row_id, col_id)`` of the block it
        manages.
    block_shape : tuple of int
        Shape of the square blocks ``(block_rows, block_cols)`` that the full
        matrix and RHS vector should be split into.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    assert len(matrix_shape) == 2, "'matrix_shape' must be a tuple with 2 entries"
    assert (
        matrix_shape[0] == matrix_shape[1]
    ), "Only square matrices are currently supported"
    myid = comm.Get_rank()
    nworkers = comm.Get_size()

    # Check that enough workers are available
    nblocks = split * split
    assert nworkers >= nblocks, "Specified more blocks than workers"
    workers = np.arange(nblocks).reshape((split, split))

    # Handle workers that do not have an assigned block
    if myid >= nblocks:
        return None, {}, {}

    # Construct map of block row/column IDs vs worker IDs
    block_map: dict[int, tuple[int, int]] = {}
    for w in workers.flatten():
        _row, _col = np.where(workers == w)
        block_map[w] = (_row[0], _col[0])
    myrow, mycol = block_map[myid]

    # Setup communicator groups for each row, columns, and the diagonals
    grp_active = workers.flatten()
    grp_row = workers[myrow, :]
    grp_col = workers[:, mycol]
    grp_diag = np.diag(workers)
    comm_active = comm.Create(comm.group.Incl(grp_active))
    comm_row = comm.Create(comm.group.Incl(grp_row))
    comm_col = comm.Create(comm.group.Incl(grp_col))
    comm_diag = comm.Create(comm.group.Incl(grp_diag))

    # Calculate block size (all blocks must have the same shape, so some zero-
    # padding will be done if the matrix operator shape is not exactly
    # divisible by 'split')
    block_rows = int(np.ceil(matrix_shape[0] / split))  # rows per block
    block_cols = int(np.ceil(matrix_shape[1] / split))  # cols per block
    block_shape = (block_rows, block_cols)
    assert (
        block_rows == block_cols
    ), "Current implementation assumes that blocks are square"

    comms = (comm_active, comm_row, comm_col, comm_diag)
    return comms, block_map, block_shape


def collect_linear_sys_blocks(
    comm: MPI.Comm,
    block_map: dict[int, tuple[int, int]],
    block_shape: tuple[int, int],
    Amat: NDArray[np.floating] | None = None,
    bvec: NDArray[np.floating] | None = None,
    verbose: bool = False,
) -> tuple[NDArray[np.floating] | None, NDArray[np.floating] | None]:
    """
    Distribute LHS operator matrix and RHS vector blocks to assigned workers.

    The root worker (rank 0) splits the full matrix and vector into blocks and
    sends each block to its assigned worker via point-to-point MPI
    communication.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator object for all active workers.
    block_map : dict
        Dictionary mapping worker ID to ``(row_id, col_id)`` of the block it
        manages, as returned by :func:`setup_mpi_blocks`.
    block_shape : tuple of int
        Shape ``(block_rows, block_cols)`` of the square blocks that the full
        matrix operator and RHS vector should be split into.
    Amat : NDArray[np.floating] or None, optional
        The full LHS matrix operator, which will be split into blocks.  Only
        required on the root worker.
    bvec : NDArray[np.floating] or None, optional
        The full right-hand side vector, which will be split into blocks.  Only
        required on the root worker.
    verbose : bool, optional
        If ``True``, print status messages when MPI communication is complete.

    Returns
    -------
    my_Amat : NDArray[np.floating] or None
        The single block of the matrix operator belonging to this worker with
        shape ``block_shape``.  Zero-padded at edges if the matrix cannot be
        exactly divided.  Returns ``None`` if worker is not active.
    my_bvec : NDArray[np.floating] or None
        The single block of the RHS vector belonging to this worker.  Workers
        in the same column receive the same block.  Returns ``None`` if worker
        is not active.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    myid = comm.Get_rank()
    dtype = bvec.dtype

    # Determine whether this worker is participating
    workers_used = np.array(list(block_map.keys()))
    workers_used.sort()
    if myid not in workers_used:
        return None, None

    # Initialise blocks of A matrix and b vector
    block_rows, block_cols = block_shape
    my_Amat = np.zeros((block_rows, block_cols), dtype=dtype)
    my_bvec = np.zeros((block_rows,), dtype=dtype)

    # Send blocks from root worker
    if myid == 0:
        reqs = []
        for w in workers_used:

            # Get row and column indices for this worker
            wrow, wcol = block_map[w]

            # Start and end indices of block (handles edges)
            ii = wrow * block_rows
            jj = wcol * block_cols
            iip = ii + block_rows
            jjp = jj + block_cols
            if iip > Amat.shape[0]:
                iip = Amat.shape[0]
            if jjp > Amat.shape[1]:
                jjp = Amat.shape[1]

            if w == 0:
                # Block belongs to root worker
                my_Amat[: iip - ii, : jjp - jj] = Amat[ii:iip, jj:jjp]
                my_bvec[: jjp - jj] = bvec[jj:jjp]

            else:
                # Send blocks to other worker
                # Handles zero-padding of blocks at edge of matrix
                # FIXME: Do we have to copy here, to get contiguous memory?
                Amat_buf = np.zeros_like(my_Amat)
                bvec_buf = np.zeros_like(my_bvec)

                Amat_buf[: iip - ii, : jjp - jj] = Amat[ii:iip, jj:jjp]
                bvec_buf[: jjp - jj] = bvec[jj:jjp]

                # The flattened Amat_buf array is reshaped into 2D when received
                comm.Send(Amat_buf.flatten().copy(), dest=w)
                comm.Send(bvec_buf, dest=w)

        if verbose:
            print("All send operations completed.")
    else:
        # Receive this worker's assigned blocks of Amat and bvec
        comm.Recv(my_Amat, source=0)
        comm.Recv(my_bvec, source=0)

        if verbose:
            print("Worker %d finished receive" % myid)

    return my_Amat, my_bvec


def cg_mpi(
    comm_groups: tuple[MPI.Intracomm, ...] | None,
    Amat_block: NDArray[np.floating],
    bvec_block: NDArray[np.floating],
    vec_size: int,
    block_map: dict[int, tuple[int, int]],
    maxiters: int = 1000,
    abs_tol: float = 1e-8,
) -> NDArray[np.floating] | None:
    """
    Distributed Conjugate Gradient solver for block-partitioned linear systems.

    The linear operator matrix is split into square blocks, each handled by a
    single MPI worker.  The algorithm follows the standard CG iteration but
    with distributed matrix-vector products and scalar reductions across ranks.

    Parameters
    ----------
    comm_groups : tuple of MPI.Intracomm or None
        Group communicators ``(active, row, col, diag)`` as set up by
        :func:`setup_mpi_blocks`.  If ``None``, the worker is inactive and
        the function returns immediately.
    Amat_block : NDArray[np.floating]
        The block of the matrix operator belonging to this worker.
    bvec_block : NDArray[np.floating]
        The block of the right-hand side vector corresponding to this
        worker's matrix operator block.
    vec_size : int
        The size of the total result vector across all blocks.
    block_map : dict
        Dictionary mapping worker ID to ``(row_id, col_id)`` of the block it
        manages.
    maxiters : int, optional
        Maximum number of solver iterations. Default is 1000.
    abs_tol : float, optional
        Absolute tolerance on each element of the residual. Once this
        tolerance has been reached for all entries of the residual vector,
        the solution is considered to have converged. Default is 1e-8.

    Returns
    -------
    x : NDArray[np.floating] or None
        Solution vector for the full system.  Only workers on the diagonal
        have the correct solution vector; other workers return ``None``.

    Notes
    -----
    The initial guess is always the zero vector.  For non-zero initial
    guesses, the residual computation would need to be adjusted.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    if comm_groups is None:
        # FIXME: Need to fix this so non-active workers are ignored without hanging
        return None

    comm_active, comm_row, comm_col, comm_diag = comm_groups
    # grp_active, grp_row, grp_col, grp_diag = groups
    myid = comm_active.Get_rank()
    myrow, mycol = block_map[myid]

    # Initialise solution vector
    x_block = np.zeros_like(bvec_block)

    # Calculate initial residual (if x0 is not 0, need to rewrite this, i.e.
    # be careful with x_block ordering)
    r_block = bvec_block[:]  # - matvec_mpi(comm_row, Amat_block, x_block)
    pvec_block = r_block[:]

    # Iterate
    niter = 0
    finished = False
    while niter < maxiters and not finished:

        # Check convergence criterion from all workers
        converged = np.all(np.abs(r_block) < abs_tol)

        # Check if convergence is reached
        # (reduce with logical-AND operation)
        converged = comm_active.allreduce(converged, op=MPI_LAND)
        if converged:
            finished = True
            break

        # Distribute pvec_block to all workers (identical in each column)
        # NOTE: Assumes that the rank IDs in comm_col are in the same order as
        # for the original comm_world communicator, in which case the rank with
        # ID = mycol will be the one on the diagonal that we want to broadcast
        # the up-to-date value of pvec_block from
        if myrow != mycol:
            pvec_block *= 0.0
        comm_col.Bcast(pvec_block, root=mycol)

        # Calculate matrix operator product with p-vector (returns result for
        # this row)
        A_dot_p = matvec_mpi(comm_row, Amat_block, pvec_block)

        # Only workers with mycol == myrow will give correct updates, so only
        # calculate using those
        if mycol == myrow:
            # Calculate residual norm, summed across all (diagonal) workers
            r_dot_r = comm_diag.allreduce(np.dot(r_block.T, r_block), op=MPI_SUM)

            # Calculate quadratic, summed across all (diagonal) workers
            pAp = comm_diag.allreduce(np.dot(pvec_block.T, A_dot_p), op=MPI_SUM)

            # Calculate alpha (valid on all diagonal workers)
            alpha = r_dot_r / pAp

            # Update solution vector and residual for this worker
            x_block = x_block + alpha * pvec_block
            r_block = r_block - alpha * A_dot_p

            # Calculate updated residual norm
            rnew_dot_rnew = comm_diag.allreduce(np.dot(r_block.T, r_block), op=MPI_SUM)

            # Calculate beta (valid on all diagonal workers)
            beta = rnew_dot_rnew / r_dot_r

            # Update pvec_block (valid on all diagonal workers)
            pvec_block = r_block + beta * pvec_block

        comm_active.barrier()

        # Increment iteration
        niter += 1

    # Gather all the blocks into a single array (on diagonal workers only)
    if myrow == mycol:
        x_all = np.zeros((vec_size), dtype=x_block.dtype)
        x_all_blocks = np.zeros(
            (x_block.size * comm_diag.Get_size()), dtype=x_block.dtype
        )  # needed for zero-padding
        comm_diag.Allgather(x_block, x_all_blocks)

        # Remove zero padding if necessary
        x_all[:] = x_all_blocks[:vec_size]
    else:
        x_all = None

    comm_active.barrier()
    return x_all


def cg(
    Amat: NDArray[np.floating] | None,
    bvec: NDArray[np.floating],
    maxiters: int = 1000,
    abs_tol: float = 1e-18,
    use_norm_tol: bool = False,
    x0: NDArray[np.floating] | None = None,
    linear_op: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None,
    comm: MPI.Comm | None = None,
) -> NDArray[np.floating]:
    """
    Serial Conjugate Gradient solver with optional MPI broadcasting.

    Implements the standard CG algorithm on a single process, with optional
    broadcasting of the solution to all MPI workers.  Uses the same algorithm
    as :func:`cg_mpi` and can be used for testing and comparison.

    Parameters
    ----------
    Amat : NDArray[np.floating] or None
        Linear operator matrix.  Ignored if ``linear_op`` is provided.
    bvec : NDArray[np.floating]
        Right-hand side vector.
    maxiters : int, optional
        Maximum number of solver iterations. Default is 1000.
    abs_tol : float, optional
        Absolute tolerance on each element of the residual (or on the norm
        if ``use_norm_tol=True``). Default is 1e-18.
    use_norm_tol : bool, optional
        If ``True``, check convergence on the norm of the residual rather
        than per-element. Default is ``False``.
    x0 : NDArray[np.floating] or None, optional
        Initial guess for the solution vector.  Defaults to zeros.
    linear_op : callable or None, optional
        If specified, this function ``f(x) -> Ax`` is used instead of
        explicit matrix multiplication.
    comm : MPI.Comm or None, optional
        If specified, the CG solver runs only on rank 0 and the solution is
        broadcast to all workers.

    Returns
    -------
    x : NDArray[np.floating]
        Solution vector for the full system.

    Notes
    -----
    Threading within NumPy is still permitted even though the algorithm
    itself is serial.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    # MPI worker ID
    myid = 0
    if comm is not None:
        myid = comm.Get_rank()

    # Use Amat as the linear operator if function not specified
    if linear_op is None:
        linear_op = lambda v: Amat @ v

    # Initialise solution vector
    if x0 is None:
        x = np.zeros_like(bvec)
    else:
        assert x0.shape == bvec.shape, "Initial guess x0 has a different shape to bvec"
        assert x0.dtype == bvec.dtype, "Initial guess x0 has a different type to bvec"
        x = x0.copy()

    # Calculate initial residual
    # NOTE: linear_op may have internal MPI calls; we assume that it
    # handles synchronising its input itself, but that only the root
    # worker receives a correct return value.

    r = bvec - linear_op(x)
    pvec = r[:]

    # Blocks indexed by i,j: y = A . x = Sum_j A_ij b_j
    niter = 0
    finished = False
    while niter < maxiters and not finished:

        try:
            if myid == 0:
                # Root worker checks for convergence
                if use_norm_tol:
                    # Check tolerance on norm of r
                    if np.linalg.norm(r) < abs_tol:
                        finished = True
                else:
                    # Check tolerance per array element
                    if np.all(np.abs(r) < abs_tol):
                        finished = True

            # Broadcast finished flag to all workers (need to use a non-immutable type)
            finished_arr = np.array(
                [
                    finished,
                ]
            )
            if comm is not None:
                comm.Bcast(finished_arr, root=0)
                finished = bool(finished_arr[0])
            if finished:
                break

            # Do CG iteration
            r_dot_r = np.dot(r.T, r)
            A_dot_p = linear_op(pvec)  # root worker will broadcast correct pvec

            # Only root worker needs to do these updates; other workers idle
            if myid == 0:
                pAp = pvec.T @ A_dot_p
                alpha = r_dot_r / pAp

                x = x + alpha * pvec
                r = r - alpha * A_dot_p

                # Update pvec
                beta = np.dot(r.T, r) / r_dot_r
                pvec = r + beta * pvec

            # Update pvec on all workers
            if comm is not None:
                comm.Bcast(pvec, root=0)

            # Increment iteration
            niter += 1
        except:
            raise

    if comm is not None:
        comm.barrier()

    # Synchronise solution across all workers
    if comm is not None:
        comm.Bcast(x, root=0)
    return x


try:
    import torch

    # Specify device (MPS, CPU, or CUDA)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
except ImportError:
    torch = None
    device = None


def pytorch_lin_solver(
    A: NDArray[np.floating],
    b: NDArray[np.floating],
    device: torch.device | None = device,
) -> NDArray[np.floating]:
    """
    Solve a linear system ``Ax = b`` using PyTorch.

    Uses ``torch.linalg.solve`` for a direct solution. Handles the MPS
    backend by falling back to CPU (MPS does not support ``linalg.solve``
    in double precision).

    Parameters
    ----------
    A : NDArray[np.floating]
        Coefficient matrix of shape ``(N, N)``.
    b : NDArray[np.floating]
        Right-hand side vector of shape ``(N,)`` or matrix ``(N, K)``.
    device : torch.device or None, optional
        Torch device to use (``cpu``, ``cuda``, or ``mps``).  Defaults to
        the best available device detected at import time.

    Returns
    -------
    x : NDArray[np.floating]
        Solution vector (or matrix) as a NumPy array.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """
    if torch is None:
        raise ImportError(
            "PyTorch is required for pytorch_lin_solver. Install with: pip install torch"
        )
    # Convert numpy arrays to PyTorch tensors
    if device.type == "mps":
        # Use CPU for lstsq operation
        with torch.device("cpu"):
            A_torch = torch.tensor(A, dtype=torch.float32)
            b_torch = torch.tensor(b, dtype=torch.float32)
            x_torch = torch.linalg.solve(A_torch, b_torch)
    else:
        # Use specified device
        A_torch = torch.tensor(A, dtype=torch.float64).to(device)
        b_torch = torch.tensor(b, dtype=torch.float64).to(device)
        x_torch = torch.linalg.solve(A_torch, b_torch)

    return x_torch.cpu().numpy()


def pytorch_nnls(
    A: NDArray[np.floating],
    b: NDArray[np.floating],
    device: torch.device | None = device,
    max_iter: int = 1000,
) -> NDArray[np.floating]:
    """
    Solve non-negative least squares using PyTorch with L-BFGS optimisation.

    Minimises ``||Ax - b||^2`` subject to ``x >= 0`` by first running
    unconstrained L-BFGS and then clamping the solution to non-negative
    values.

    Parameters
    ----------
    A : NDArray[np.floating]
        Coefficient matrix of shape ``(M, N)``.
    b : NDArray[np.floating]
        Right-hand side vector of shape ``(M,)``.
    device : torch.device or None, optional
        Torch device to use. Defaults to the best available device.
    max_iter : int, optional
        Maximum number of L-BFGS iterations. Default is 1000.

    Returns
    -------
    x : NDArray[np.floating]
        Non-negative solution vector of shape ``(N,)``.

    Notes
    -----
    The non-negativity constraint is enforced by clamping after
    optimisation, so the solution is only approximately NNLS-optimal.
    """
    A_torch = torch.tensor(A, dtype=torch.float32).to(device)
    b_torch = torch.tensor(b, dtype=torch.float32).to(device)

    # Initialize with zeros
    x = torch.zeros(A.shape[1], dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS([x], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = torch.sum((A_torch @ x - b_torch) ** 2)
        loss.backward()
        return loss

    optimizer.step(closure)

    # Apply non-negativity constraint
    with torch.no_grad():
        x = torch.clamp(x, min=0)

    return x.cpu().numpy()
