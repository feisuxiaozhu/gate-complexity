import numpy as np
import random
import galois
from typing import List, Any
import os
import pickle
import multiprocessing
from sympy import Matrix, latex
from itertools import combinations, product, permutations

def weighted_sample(choices: List[Any], probs: List[float]):
    """
    Sample from `choices` with probability according to `probs`
    """
    probs = np.concatenate(([0], np.cumsum(probs)))
    r = random.random()
    for j in range(len(choices) + 1):
        if probs[j] < r <= probs[j + 1]:
            return choices[j]

def sidant_test(n, m):
  j = 0
  while True:
    
    B_inv, B = matrix_generator(n,m)
    # print(B)
    A = np.matrix(B)

    # j += 1
    # print(j)
    flag = 0
    row_sum = np.array(np.sum(np.matrix(A), axis = 0))[0]
    if np.any([i in row_sum for i in range(6)]): # first sidhant test
      continue
    print('passed column denstiy check!')
    for ind in pair_cols: # second sidhant test
      
      if np.sum(np.array(B[:, ind[0]] + B[:, ind[1]])) <= 5:
        flag = 1
        break 
    if flag: 
      continue
    
    return (B_inv, B)
       

def matrix_generator(n,m,k):
    GF2 = galois.GF(2)
    counter = 0
    c=int(n/2)
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

        for i in range(c): # add n/2 gates to include all inputs
            one = candidates[2*i]
            two = candidates[2*i+1]
            sum = (one + two) % 2
            candidates.append(sum)
        
        for i in range(c, m): # add m-c gates using Hamming weight distribution of previous candidates
            Hamming = []
            for candidate in candidates:
                Hamming.append((np.sum(candidate))**(k)) # non-linear distribution
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
        
      
        i = m+n-1
        result_matrix = candidates[i].T
        counter = 1
        while counter < n:
            counter += 1
            temp_matrix = []
            while np.linalg.matrix_rank(temp_matrix) < counter and i>=n:
                i -= 1
                # if i<n:
                #     matrix_generator
                    # return np.zeros((n,n)), np.zeros((n,n))
                    # continue
                temp_matrix = np.concatenate((result_matrix, candidates[i].T), axis=0)
                temp_matrix = GF2(temp_matrix)
            result_matrix = temp_matrix
            if i == n-1:
               break

        if i == n-1:
           continue
        result_matrix = GF2(result_matrix)
        result_matrix_inv = np.linalg.inv(result_matrix)
        row_sum = np.array(np.sum(np.matrix(result_matrix_inv), axis = 0))[0]
        if np.any([i in row_sum for i in range(6)]): # must pass the fist Sidhant test!
            continue
        return result_matrix, result_matrix_inv


def worker(n,m,k,i):
    print('Thread '+str(i)+' working') 
    A,B = matrix_generator(n,m,k)
    print(A)
    print(B)



if __name__ == '__main__':
    n=20
    m=30
    pair_cols = [list(i) for i in list(combinations(range(n), 2))]

    jobs = [] # list of jobs
    jobs_num = 10 # number of workers
    for i in range(jobs_num):
        k = 0.1*i + 0.1
        p1 = multiprocessing.Process(target=worker, args=(n,m,k,i))
        jobs.append(p1)
        p1.start() # starting workers



    # A,B = matrix_generator(n,m)
    # print(A)
    # print(B)
    # print('A density: ' + str(np.count_nonzero(A)/n))
    # print('B density: ' + str(np.count_nonzero(B)/n))

















