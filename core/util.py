import numpy as np


def calculate_function_values(var1_values, var2_values, your_function):
    """
    Calculate function values for all combinations of two input variable arrays using NumPy.

    Parameters:
    var1_values: NumPy array of the first input variable values
    var2_values: NumPy array of the second input variable values
    your_function: The function to calculate, which takes two NumPy arrays as input.

    Returns:
    A NumPy array containing the function values for all combinations.
    """
    # Create a grid of all combinations of var1 and var2
    var1_grid, var2_grid = np.meshgrid(var1_values, var2_values, indexing='ij')

    # Use your_function to calculate the function values for all combinations
    result = your_function(var1_grid, var2_grid)

    return result

def chunk_data(data, width):
    """
    Chunk a list of sequenced data into sublists by a given width.
    In each sublist, the difference between the maximum and minimum values is less than or equal to the width.

    Note that the width should be integer times of the timestep of the data.

    """

    # Initialize an empty list to hold the chunks
    chunks = []
    indices = []
    
    counter = 0
    
    # Loop until all data is processed
    while data:
        start_value = data[0]

        # Calculate the start and end range for the current chunk
        chunk_start = start_value
        chunk_end = start_value + width
        
        # Create a chunk by including all elements in the range [chunk_start, chunk_end)
        chunk = [x for x in data if chunk_start <= x < chunk_end]
        # Find the corresponding indices for the chunk elements
        chunk_length = len(chunk)
        chunk_indices = [x + counter for x in range(chunk_length)] 
        counter += chunk_length
        
        # Append the chunk to the result
        chunks.append(chunk)

        # Append the indices to the result
        indices.append(chunk_indices)
        
        # Remove the used elements from data
        data = [x for x in data if x >= chunk_end]
        
    return chunks, indices

def get_chunk_indices(list_size, num_chunks):
    """
    Calculate the indices of the chunks for a list of a given size and number of chunks.

    Parameters:
    list_size (int): The size of the list to be chunked.
    num_chunks (int): The number of chunks to create.

    Returns:
    numpy array: The indices of the chunks.

    Example:
    chunked_list_indices(10, 3) -> array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    """
    chunk_sizes = np.full(num_chunks, list_size // num_chunks)
    chunk_sizes[:list_size % num_chunks] += 1
    chunk_indices = np.repeat(np.arange(num_chunks), chunk_sizes)
    return chunk_indices

def chunked_list(list, num_chunks):
    """"
    Get a list of chunks from a given list.
    """
    chunk_indices = get_chunk_indices(len(list), num_chunks)
    chunked_list = [list[chunk_indices == i] for i in np.unique(chunk_indices)]
    return chunk_indices, chunked_list

def project_onto_tensor_basis(data, eigvecs_t, eigvecs_f):
    """
    Project a data matrix onto the tensor product basis.

    Parameters:
    data (numpy array): 2D array of shape (Nt, Nf).
    eigvecs_t (numpy array): Nt x Nt orthogonal matrix of eigenvectors.
    eigvecs_f (numpy array): Nf x Nf orthogonal matrix of eigenvectors.

    Returns:
    numpy array: The projection of the data vector onto the tensor basis, shape (Nt, Nf).
    """
    Nt, _ = eigvecs_t.shape
    Nf, _ = eigvecs_f.shape
    Nt_data, Nf_data = data.shape

    assert Nt_data == Nt and Nf_data == Nf, "Data chunk has the wrong shape."
    
    # Project onto the tensor product basis using the orthogonal properties
    proj_t = eigvecs_t.T @ data
    proj_tf = proj_t @ eigvecs_f
    
    return proj_tf
