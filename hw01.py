import numpy as np
from my_io import generate_dataset
import warnings

warnings.filterwarnings('ignore')

matrix_size = 5

error = 0

for _ in range(100):
    B = generate_dataset(matrix_size)
    # print(f'B:\n{B}\n')

    # Eigendecomposition of a matrix
    eigen_values, eigen_vectors = np.linalg.eig(B)
    eigen_values = np.diag(eigen_values)

    eigen_values_square = np.sqrt(eigen_values, dtype=np.complex_)
    A = (eigen_vectors @ eigen_values_square @ np.linalg.inv(eigen_vectors))
    A = A.astype(float)
    reconstruct_B_with_rooted_square = np.round((A @ A), 3)
    print(np.sum(A, axis=1))
    # print(f'A @ A:\n{reconstruct_B_with_rooted_square}\n')

    error += np.linalg.norm(
        B - reconstruct_B_with_rooted_square, 'fro')

print(f'The average error in 100 random matrices is {error/100}')
print('pause')
