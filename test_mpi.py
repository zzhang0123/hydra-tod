

from mpi4py import MPI
import numpy as np
from linear_solver import setup_mpi_blocks, collect_linear_sys_blocks, cg_mpi

# Initialize MPI
comm = MPI.COMM_WORLD


# Define matrix and vector
N = 1000  # Size of the matrix
Amat = np.random.rand(N, N)  # Random matrix
bvec = np.random.rand(N)     # Random RHS vector

# Set up MPI blocks
split = 2  # Number of blocks per row/column
comm_groups, block_map, block_shape = setup_mpi_blocks(comm, (N, N), split)

# Distribute matrix and vector blocks
my_Amat, my_bvec = collect_linear_sys_blocks(comm, block_map, block_shape, Amat, bvec)

# Solve the linear system
x = cg_mpi(comm_groups, my_Amat, my_bvec, N, block_map, maxiters=1000, abs_tol=1e-8)

# Gather the solution
if comm.Get_rank() == 0:
    print(block_shape)
    print("Solution vector:", x)

