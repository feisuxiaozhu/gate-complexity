import numpy as np
import random
import galois
from typing import List, Any
import os
import pickle

with open("./one_way_function/sparse_matrix_creater/A.pickle","wb") as f:
    pickle.dump([], f)

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
# random output
n = 20
m = 30
c=1
def matrix_generator(n,m,c):
    GF2 = galois.GF(2)
    counter = 0
    while True:
        counter += 1
        if counter % 10000 == 0:
            # os.system('cls')
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


        # matrix_indices = random.sample(range(0, m+n), n) # build matrix
        # matrix_indices = [i+m for i in range(n)] # use n last gate as output
        matrix_indices = []
        for i in range(n): # pick output based on weight distribution
            index = weighted_sample(all_indices, normed)
            while index in matrix_indices:
                index = weighted_sample(all_indices, normed)
            matrix_indices.append(index)
        # print(len(matrix_indices))
        # print(matrix_indices)
        result_matrix = candidates[matrix_indices[0]].T
        for i in range(1,n):
            result_matrix = np.concatenate((result_matrix, candidates[matrix_indices[i]].T), axis=0)
        if np.linalg.matrix_rank(result_matrix) == n: # check matrix is invertible  
            result_matrix_inv = GF2(result_matrix)
            return result_matrix, result_matrix_inv
        
        # break
    
for j in range(10): 
    print('finding matrix: ' + str(j+1))   
    A, B = matrix_generator(n,m,c)
    print(A,B)
    with open('./one_way_function/sparse_matrix_creater/A.pickle', 'rb') as handle:
        b = pickle.load(handle)
    b.append(A)
    with open('./one_way_function/sparse_matrix_creater/A.pickle', 'wb') as fp:
        pickle.dump(b, fp)






