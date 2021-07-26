from sympy import Matrix, latex
import numpy as np
new_matrix = [[1, 0, 1, 0, 0, 0, 0, 1],
              [0, 1, 0, 1, 0, 0, 0, 1],
              [1, 0, 1, 0, 1, 0, 0, 0],
              [0, 1, 0, 1, 0, 1, 0, 0],
              [0, 0, 1, 0, 1, 0, 1, 0],
              [0, 0, 0, 1, 0, 1, 0, 1],
              [1, 0, 0, 0, 1, 0, 1, 0],
              [1, 0, 0, 0, 0, 1, 0, 1]
              ]

new_matrix = Matrix(new_matrix)

# print(int(np.linalg.det(new_matrix)) %2)
new_matrix_inverse = new_matrix.inv_mod(2)
print(latex(new_matrix_inverse))






