from copy import deepcopy
import numpy as np
from my_io import generate_dataset
import warnings

warnings.filterwarnings('ignore')

matrix_size = 5

decimal_point = 4

random_test = 100


def err_improve(B, A, i, j, k, x):
    Ap = deepcopy(A)
    Ap[k, i] += x
    Ap[k, j] -= x
    return np.linalg.norm(B - A@A, 'fro') - np.linalg.norm(B - Ap@Ap, 'fro')


def find_root_square(B, method, decimal_point):
    A = deepcopy(B)

    matrix_size = B.shape[0]
    old_err = 1000
    new_err = 100
    no_improve = 0

    best_err = 1000
    best_A = None

    learning_rate = 1

    iteration = 0

    if(method == 'adaptive_lerning_rate'):
        while(learning_rate >= 10**-decimal_point):
            iteration += 1

            A2 = A@A
            E = B - A2

            i, j, k = np.random.randint(0, matrix_size, 3)
            while(i == j):
                i, j, k = np.random.randint(0, matrix_size, 3)

            if(err_improve(B, A, i, j, k, learning_rate) > 0
                    and A[k, j] >= learning_rate):
                A[k, i] += learning_rate
                A[k, j] -= learning_rate

            if(err_improve(B, A, i, j, k, -learning_rate) > 0
                    and A[k, i] >= learning_rate):
                A[k, i] += -learning_rate
                A[k, j] -= -learning_rate

            new_err = np.linalg.norm(B - A@A, 'fro')
            # print(new_err)

            if(no_improve > 100):
                learning_rate /= 2
                learning_rate = np.round(learning_rate, decimal_point)
                no_improve = 0

            if(best_err - new_err >= 10**-decimal_point):
                best_err = new_err
                no_improve = 0
                best_A = deepcopy(A)
            else:
                no_improve += 1

    if(method == 'closed_form'):
        while(no_improve < 1000):
            iteration += 1

            A2 = A@A
            E = B - A2

            i, j, k = np.random.randint(0, matrix_size, 3)
            while(i == j):
                i, j, k = np.random.randint(0, matrix_size, 3)
            # i, j, k = 1, 3, 1
            C = np.zeros((matrix_size, matrix_size))
            C[k][i], C[k][j] = 1, -1

            a4, a3, a2, a1, a0 = [0]*5

            sum_new_val = 0

            for row in range(matrix_size):
                for col in range(matrix_size):
                    new_row = A[row, :] + C[row, :]
                    new_col = A[:, col] + C[:, col]
                    new_val = sum(new_row * new_col)
                    sum_new_val += new_val
                    delta_val = new_val - A2[row, col]
                    a2 += delta_val**2
                    a1 += 2 * delta_val * E[row, col]
                    a0 += E[row, col]**2

            roots = np.roots((a2, a1, a0))

            if(roots[0] != 0):
                root = roots[0]
            else:
                root = roots[1]
            if(type(root) == np.complex_):
                root = root.astype(float)
            A[k][i] -= root
            A[k][j] += root

            old_err = new_err
            new_err = np.linalg.norm(B - A@A, 'fro')
            # if(new_err > old_err):
            #     print(new_err)
            if(best_err - new_err >= 10**-decimal_point):
                best_err = new_err
                no_improve = 0
                best_A = deepcopy(A)
            else:
                no_improve += 1

    return best_A, best_err, iteration


avg_err_adaptive_lerning_rate = 0
avg_err_closed_form = 0
avg_iter_adaptive_lerning_rate = 0
avg_iter_closed_form = 0

for _ in range(random_test):
    B = generate_dataset(5, decimal=decimal_point)
    best_A, adaptive_lerning_rate_err, iteration_adaptive_lerning_rate =\
        find_root_square(B, 'adaptive_lerning_rate', decimal_point)
    best_A, closed_form_err, iteration_closed_form =\
        find_root_square(B, 'closed_form', decimal_point)
    avg_err_adaptive_lerning_rate += adaptive_lerning_rate_err
    avg_iter_adaptive_lerning_rate += iteration_adaptive_lerning_rate
    avg_err_closed_form += closed_form_err
    avg_iter_closed_form += iteration_closed_form

avg_err_adaptive_lerning_rate /= random_test
avg_iter_adaptive_lerning_rate /= random_test
avg_err_closed_form /= random_test
avg_iter_closed_form /= random_test

print(
    f'The average error in {random_test}',
    f'random matrices is by adaptive lerning rate is {avg_err_adaptive_lerning_rate}')
print(f'The average iteration in {random_test} random matrices by adaptive lerning rate is {avg_iter_adaptive_lerning_rate}')
print(
    f'The average error in {random_test}',
    f'random matrices is by closed form rate is {avg_err_closed_form}')
print(f'The average iteration in {random_test} random matrices by closed form rate is {avg_iter_closed_form}')
print('pause')
