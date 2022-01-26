import os
import numpy as np


def create_adjacency_matrix(matrix_size: int, decimal: int) -> np.ndarray:
    matrix = np.random.rand(matrix_size, matrix_size)
    A = np.sum(matrix, axis=1)
    adjacency_matrix = (matrix.T / A).T
    adjacency_matrix = np.round(adjacency_matrix, decimal)
    cell = np.random.randint(0, matrix_size, matrix_size)
    for i in range(matrix_size):
        adjacency_matrix[i][cell[i]] += 1 - np.round(
            np.sum(adjacency_matrix[i]), decimal)
    return adjacency_matrix


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
