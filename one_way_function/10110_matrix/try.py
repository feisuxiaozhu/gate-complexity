# We want to check that matrix_110 + lower triangular = high rank matrix
from numpy.linalg import matrix_rank
import numpy as np
import random


def flipCoin():
    return random.choice([True, False])


matrix_110 = [[1, 0, 1, 1, 0, 1, 1, 0],
              [0, 1, 0, 1, 1, 0, 1, 1],
              [1, 0, 1, 0, 1, 1, 0, 1],
              [1, 1, 0, 1, 0, 1, 1, 0],
              [0, 1, 1, 0, 1, 0, 1, 1],
              [1, 0, 1, 1, 0, 1, 0, 1],
              [1, 1, 0, 1, 1, 0, 1, 0],
              [0, 1, 1, 0, 1, 1, 0, 1]
              ]
matrix_110 = np.array(matrix_110)

lower_triangular = [[1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1]]
lower_triangular = np.array(lower_triangular)

dic = {}
for i in range(len(matrix_110)+1):
    dic[i]=0

counter = 0
while counter < 1000000:
    temp = lower_triangular.copy()
    for i in range(len(temp[0])):
        for j in range(i+1):
            if flipCoin():
                temp[i, j] = 0

    result = (temp + matrix_110) % 2
    rank = matrix_rank(result)
    dic[rank] += 1
    counter += 1
    if counter % 10000 == 0:
        print(counter) 
    
print(dic)




