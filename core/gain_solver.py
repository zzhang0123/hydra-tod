import numpy as np


def sample_from_multiple_distributions(b_values, std_dev_values):
    # Convert b_values and std_dev_values to numpy arrays if they are not already
    b_values = np.array(b_values)
    std_dev_values = np.array(std_dev_values)
    
    # Generate one sample for each pair of b and std_dev using vectorized numpy.random.normal
    samples = np.random.normal(loc=b_values, scale=std_dev_values)
    
    return samples


def mean_gains_linear_solver(per_chunk_N_mat, full_gain_vec):
    """
    Sample per-chunk mean gains using a linear solver.
    """
    chunk_size = len(per_chunk_N_mat)
    n_chunks = int(len(full_gain_vec)/chunk_size)
    N_inv = np.linalg.inv(per_chunk_N_mat)
    diag_mat = np.sum(N_inv) * np.ones(n_chunks)
    Amatrix = np.diag(diag_mat)

    gain_array = full_gain_vec.reshape((n_chunks, chunk_size))
    qVector = np.einsum('ij,aj->a', N_inv, gain_array)

    Lmatr = np.linalg.cholesky(Amatrix)
    bVector = qVector + Lmatr@np.random.normal(0, 1, n_chunks)
    result = np.linalg.solve(Amatrix, bVector)
    return result


def mean_gains_ML_solution(per_chunk_N_mat, full_gain_vec):
    """
    Solve for the maximum likelihood mean gains using the linear solver.
    """
    chunk_size = len(per_chunk_N_mat)
    n_chunks = int(len(full_gain_vec)/chunk_size)
    N_inv = np.linalg.inv(per_chunk_N_mat)
    diag_mat = np.sum(N_inv) * np.ones(n_chunks)
    Amatrix = np.diag(diag_mat)

    gain_array = full_gain_vec.reshape((n_chunks, chunk_size))
    qVector = np.einsum('ij,aj->a', N_inv, gain_array)

    result = np.linalg.solve(Amatrix, qVector)
    return result