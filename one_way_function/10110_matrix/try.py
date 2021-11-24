# We want to check that matrix_110 + lower triangular = high rank matrix
from numpy.linalg import matrix_rank
import numpy as np
import random
from random import randrange
import multiprocessing


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

#######################################################################
# Permute the rows and columns of a matrix that has non-zero entries only in its button half.


def permute_for_random_L(procnum, return_dict):
    counter = 0
    counter_set = set()
    iter0 = 500/8
    iter = 10000
    while counter < iter0:
        temp = lower_triangular.copy()
        for i in range(len(temp[0])):
            for j in range(i+1):
                if flipCoin():
                    temp[i, j] = 0
        counter += 1
        counter_2 = 0
        while counter_2 < iter:
            i = randrange(len(matrix_110[0]))
            j = randrange(len(matrix_110[0]))
            k = randrange(len(matrix_110[0]))
            l = randrange(len(matrix_110[0]))
            temp[[i, j], :] = temp[[j, i], :]
            temp[:, [k, l]] = temp[:, [l, k]]
            result = (temp + matrix_110) % 2
            rank = matrix_rank(result)
            counter_set.add(rank)
            counter_2 += 1
            if counter_2 % (iter/10) == 0:
                print('Checking random L number '+str(counter) +
                      ' on permutation number ' + str(counter_2))
    return_dict[procnum] =  counter_set


jobs = []
manager = multiprocessing.Manager()
return_dict = manager.dict()
for i in range(8):
    p = multiprocessing.Process(
        target=permute_for_random_L, args=(i, return_dict))
    jobs.append(p)
    p.start()

for proc in jobs:
    proc.join()
print(return_dict.values())

#######################################################################
# # Permute the rows and columns of a perfect lower triangular matrix.
# temp = lower_triangular.copy()
# counter = 0
# counter_set = set()
# while counter < 10000000:
#     i = randrange(len(matrix_110[0]))
#     j = randrange(len(matrix_110[0]))
#     k = randrange(len(matrix_110[0]))
#     l = randrange(len(matrix_110[0]))
#     temp[[i, j], :] = temp[[j, i], :]
#     temp[:, [k, l]] = temp[:, [l, k]]
#     result = (temp + matrix_110) % 2
#     rank = matrix_rank(result)
#     counter_set.add(rank)
#     if counter % 10000 == 0:
#         print(counter)
#     counter += 1
# print(counter_set)


#######################################################################
# # Sample the rank of L + M, where L are matrix with non-zero entries only in the bottom half.

# dic = {}
# for i in range(len(matrix_110)+1):
#     dic[i]=0

# counter = 0
# while counter < 1000000:
#     temp = lower_triangular.copy()
#     for i in range(len(temp[0])):
#         for j in range(i+1):
#             if flipCoin():
#                 temp[i, j] = 0

#     result = (temp + matrix_110) % 2
#     rank = matrix_rank(result)
#     dic[rank] += 1
#     counter += 1
#     if counter % 10000 == 0:
#         print(counter)

# print(dic)
