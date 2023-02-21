import numpy as np
import random
import galois
from typing import List, Any
import os

def weighted_sample(choices: List[Any], probs: List[float]):
    """
    Sample from `choices` with probability according to `probs`
    """
    probs = np.concatenate(([0], np.cumsum(probs)))
    r = random.random()
    for j in range(len(choices) + 1):
        if probs[j] < r <= probs[j + 1]:
            return choices[j]

# creat a random matrix that can be computed at most by m gates on n inputs
# randomness cuttoff c is between 0 to m
n = 20
m = 30
c=1
def matrix_generator(n,m,c):
    GF2 = galois.GF(2)
    counter = 0
    while True:
        counter += 1
        if counter % 1000 == 0:
            os.system('cls')
            print(counter)
        candidates = []

        for i in range(n): # creat first n unit vectors of length n + m
            temp = np.zeros((n,1), dtype=int)
            temp[i][0] = 1
            candidates.append(temp)

        for i in range(c): # add c gates uniformally
            number_of_candidate = len(candidates)
            random_indices = random.sample(range(0,number_of_candidate), 2) # randomly pick two
            one = candidates[random_indices[0]]
            two = candidates[random_indices[1]]
            sum = (one + two) % 2
            candidates.append(sum)

        for i in range(c, m): # add m-c gates using Hamming weight distribution of previous candidates
            Hamming = []
            for candidate in candidates:
                Hamming.append(np.sum(candidate))
            normed = [float(i)/np.sum(Hamming) for i in Hamming]
            all_indices = [i for i in range(len(candidates))]
            index1 = weighted_sample(all_indices, normed)
            index2 = weighted_sample(all_indices, normed)
            while index2 == index1:
                index2 = weighted_sample(all_indices, normed)
            random_indices = [index1, index2]
            # print(random_indices)
            one = candidates[random_indices[0]]
            two = candidates[random_indices[1]]
            sum = (one + two) % 2
            candidates.append(sum)
            # print(normed)


        matrix_indices = random.sample(range(0, m+n), n) # build matrix
        result_matrix = candidates[matrix_indices[0]].T
        for i in range(1,n):
            result_matrix = np.concatenate((result_matrix, candidates[matrix_indices[i]].T), axis=0)
        if np.linalg.matrix_rank(result_matrix) == n: # check matrix is invertible  
            result_matrix_inv = GF2(result_matrix)
            return result_matrix, result_matrix_inv
        
        # break
    
A, B = matrix_generator(n,m,c)
print(A,B)





