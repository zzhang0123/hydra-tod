import numpy as np
import mpiutil
from mpi4py import MPI
from full_Gibbs_sampler import get_Tsys_operator

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def test_get_Tsys_operator():
    # Create test data based on rank
    if rank == 0:
        # Rank 0 has 2 TODs
        U_list = [np.zeros((3, 2)), np.zeros((3, 2))]  # Tsky operators
        R_list = [np.ones((3, 1)), np.ones((3, 2))]  # Trec operators
    else:
        # Rank 1 has 1 TOD
        U_list = [np.zeros((4, 2))]
        R_list = [np.ones((4, 3))]
    
    # Get combined Tsys operators
    Tsys_ops = get_Tsys_operator(U_list, R_list)
    
    # Verify results
    if rank == 1:
        print("Tsys_ops: \n", Tsys_ops[0])
       
if __name__ == "__main__":
    test_get_Tsys_operator()

