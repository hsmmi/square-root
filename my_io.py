import os
import numpy as np


def create_adjacency_matrix(matrix_size: int, decimal: int) -> np.ndarray:
    matrix = np.random.rand(matrix_size, matrix_size)
    row_sum = np.sum(matrix, axis=1)
    adjacency_matrix = (matrix.T / row_sum).T
    adjacency_matrix = np.round(adjacency_matrix, decimal)
    cell = np.random.randint(0, matrix_size, matrix_size)
    for i in range(matrix_size):
        adjacency_matrix[i][cell[i]] += 1 - np.round(
            np.sum(adjacency_matrix[i]), decimal)
    adjacency_matrix = np.round(adjacency_matrix, decimal)
    return adjacency_matrix


def create_random_walk_matrix(matrix_size: int) -> np.ndarray:
    adjacency_matrix = np.random.choice([0, 1], (matrix_size, matrix_size))
    row_sum = np.sum(adjacency_matrix, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        random_walk = (adjacency_matrix.T / row_sum).T
    random_walk[np.isnan(random_walk)] = 0
    return random_walk


def generate_dataset(
     matrix_size: int, file: str = None, decimal: int = 3) -> np.ndarray:
    adjacency_matrix = create_adjacency_matrix(matrix_size, decimal)
    while(np.sum(adjacency_matrix) != matrix_size):
        adjacency_matrix = create_adjacency_matrix(matrix_size, decimal)
    if(file is not None):
        path = os.path.join(os.path.dirname(__file__), file)
        data = np.vstack((np.arange(matrix_size), adjacency_matrix))
        np.savetxt(path, data, delimiter=",")
    return adjacency_matrix
