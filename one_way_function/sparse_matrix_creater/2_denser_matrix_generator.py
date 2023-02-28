import numpy as np
import random
import galois
from typing import List, Any
import os
import pickle
import multiprocessing



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

def matrix_generator(n,m,c):
    GF2 = galois.GF(2)
    counter = 0
    while True:
        counter += 1
        # if counter % 10000 == 0:
            # os.system('cls')
            # print(counter)
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
                Hamming.append((np.sum(candidate))**1) # non-linear distribution
            normed = [float(i)/np.sum(Hamming) for i in Hamming]
            all_indices = [j for j in range(len(candidates))]
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
        all_indices = [j for j in range(len(candidates))]

        # matrix_indices = random.sample(range(0, m+n), n) # build matrix
        # matrix_indices = [i+m for i in range(n)] # use n last gate as output
        
        Hamming = []
        for candidate in candidates:
            Hamming.append(np.sum(candidate))
        normed = [float(i)/np.sum(Hamming) for i in Hamming] # linear norm distribution for output
        matrix_indices = []
        for i in range(n): # pick output based on weight distribution
            index = weighted_sample(all_indices, normed)
            while index in matrix_indices:
                index = weighted_sample(all_indices, normed)
            matrix_indices.append(index)
        result_matrix = candidates[matrix_indices[0]].T
        for i in range(1,n):
            result_matrix = np.concatenate((result_matrix, candidates[matrix_indices[i]].T), axis=0)
        if np.linalg.matrix_rank(result_matrix) == n: # check matrix is invertible  
            result_matrix_inv = GF2(result_matrix)
            return result_matrix, result_matrix_inv
        
def worker(n,m,c,i):
    with open('./one_way_function/sparse_matrix_creater/B'+str(i)+'.pickle',"wb") as f:
        pickle.dump([], f)
    for j in range(5): 
        print('Thread '+str(i)+' finding matrix: ' + str(j+1))   
        A, B = matrix_generator(n,m,c)
        print(A,B)
        with open('./one_way_function/sparse_matrix_creater/B'+str(i)+'.pickle', 'rb') as handle:
            b = pickle.load(handle)
        b.append(B) # will check if it is dense afterwards
        with open('./one_way_function/sparse_matrix_creater/B'+str(i)+'.pickle', 'wb') as fp:
            pickle.dump(b, fp)


n = 20
m = 30
c=5

if __name__ == '__main__':
    jobs = [] # list of jobs
    jobs_num = 6 # number of workers
    for i in range(jobs_num):
        # Declare a new process and pass arguments to it
        p1 = multiprocessing.Process(target=worker, args=(n,m,c,i))
        jobs.append(p1)
        # Declare a new process and pass arguments to it
        # p2 = multiprocessing.Process(target=worker, args=(4,6,0,))
        # jobs.append(p2)
        p1.start() # starting workers
        # p2.start() # starting workers

