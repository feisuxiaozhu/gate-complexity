import numpy as np
import itertools
import pickle
import random
import galois

# creat a random matrix that can be computed at most by m gates on n inputs
n = 20
m = 30
def matrix_generator(n,m):
    GF2 = galois.GF(2)
    counter = 0
    while True:
        counter += 1
        print(counter)
        candidates = []

        for i in range(n): # creat first n unit vectors of length n + m
            temp = np.zeros((n,1), dtype=int)
            temp[i][0] = 1
            candidates.append(temp)

        for i in range(m): # add m gates
            number_of_candidate = len(candidates)
            random_indices = random.sample(range(0,number_of_candidate), 2) # randomly pick two
            one = candidates[random_indices[0]]
            two = candidates[random_indices[1]]
            sum = (one + two) % 2
            candidates.append(sum)

        matrix_indices = random.sample(range(0, m+n), n) # build matrix
        result_matrix = candidates[matrix_indices[0]].T
        for i in range(1,n):
            result_matrix = np.concatenate((result_matrix, candidates[matrix_indices[i]].T), axis=0)
        if np.linalg.matrix_rank(result_matrix) == n: # check matrix is invertible  
            result_matrix_inv = GF2(result_matrix)
            return result_matrix, result_matrix_inv
    
A, B = matrix_generator(n,m)
print(A,B)