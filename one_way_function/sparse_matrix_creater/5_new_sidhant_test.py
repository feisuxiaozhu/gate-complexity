import numpy as np
import galois
import random
from typing import List, Any
from itertools import combinations
# For any paired matrix M of size 20x5 of B_inv, check rank(M + 1-row-sparse) > 1.


def weighted_sample(choices: List[Any], probs: List[float]):
    """
    Sample from `choices` with probability according to `probs`
    """
    probs = np.concatenate(([0], np.cumsum(probs)))
    r = random.random()
    for j in range(len(choices) + 1):
        if probs[j] < r <= probs[j + 1]:
            return choices[j]

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
        # if np.any([i in row_sum for i in range(6)]): # must pass the fist Sidhant test!
        #     continue
        return result_matrix, result_matrix_inv
    

def new_sidhant_test(B,n, pair_dimension):
    GF2 = galois.GF(2)
    print(B)
    full_col = [i for i in range(n)]
    pair_cols = [list(i) for i in list(combinations(range(n), pair_dimension))]
    # To save time, just check all nxd submatrices has at least 1 row HW > 1
    for pair_col in pair_cols:
        dummy = np.zeros((n, pair_dimension))
        dummy = B[:, pair_col]
        first_check = 0
        for k in range(n):
            row_sum = 0
            for ell in range(pair_dimension):
                if dummy[k][ell] == 1:
                    row_sum += 1
            if row_sum > 1:
                first_check =1
                break
        if first_check == 0:
            print('fail first check at columns ' + str(pair_col) )
            return False
    
    print('pass first check!')
           
        

    # If all the submatrices are fine, we then do the actual pairing
    for pair_col in pair_cols:
        print('checking columns: ' + str(pair_col))
        remain_col = list(set(full_col)-set(pair_col))
        for i in range(len(pair_col)+1):
            added_columns = [list(i) for i in list(combinations(remain_col, i))] # choose columns that will be added to the paired matrix
            for added_column in added_columns:
                for j in range(len(added_column),len(added_column)+1): # permutation when adding columns to paired matrix
                    inserting_columns = [list(i) for i in list(combinations(range(len(pair_col)), j))]
                    for inserting_col in inserting_columns:
                        dummy_add = np.zeros((n, pair_dimension), dtype=int)
                        dummy_add[:,inserting_col] = B[:,added_column]
                        dummy_pair = np.zeros((n, pair_dimension), dtype=int)
                        dummy_pair = B[:,pair_col]
                        dummy_add = GF2(dummy_add)
                        dummy_pair = GF2(dummy_pair)
                        dummy_pair = dummy_pair + dummy_add # finished pairing

                        # check 1-sparsity, i.e., there exists a row of HW>1
                        check_pass = 0
                        for k in range(n):
                            row_sum = 0
                            for ell in range(pair_dimension):
                                if dummy_pair[k][ell] == 1:
                                    row_sum += 1

                            if row_sum > 1:
                                check_pass =1
                                break
                        
                        if check_pass == 0:
                            print('fail second check at columns ' + str(pair_col) )
                            return False
    print('pass second check!')                 
    return True
        
    


# n=20
# m=30
n=20
m=30
pair_dimension = 5
power=1
flag = False
counter = 0
while (not flag):
    A,B = matrix_generator(n,m,power)
    flag = new_sidhant_test(B,n,pair_dimension)
    counter += 1
    print('tried number of matrix: '+ str(counter))

print(B)

# full_list = [i for i in range(n)]
# print(full_list)
# partial = [1,3,5]
# print(list(set(full_list)-set(partial)))
# pair_cols = [list(i) for i in list(combinations([5,6,8,10], 2))]
# print(pair_cols)
# restriction = [0,3]
# print(B[:,restriction])

# d=3
# GF2 = galois.GF(2)
# A = np.array([[1,0,0],[1,1,0],[1,1,1]])
# A = GF2(A)
# print(A)
# B = np.zeros((3,3), dtype=int)
# print(B)
# B[:,[1,2]] = A[:,[0,1]]
# print(B)
# s = 0
# for i in range(d):
#     if A[0][i] == 1:
#         s+=1
# print(s)