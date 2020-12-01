import numpy as np
import itertools
import pickle

# This file generate some invertible n-by-n zero-one matrices that are sufficient for checking A = L + S

def generate_non_zero_vectors(n):
    non_zero_vectors = []
    index_pool = []
    for i in range(n):
        index_pool.append(i)
    for numer_of_ones in range(1,n+1):
        one_indices_pool = itertools.combinations(index_pool,numer_of_ones)
        for one_indices in one_indices_pool:
            temp_vec = []
            for i in range(n):
                temp_vec.append(0)
            for one_index in one_indices:
                temp_vec[one_index] = 1
            non_zero_vectors.append(temp_vec)
    return non_zero_vectors

def generate_matrix_A(n):
    result = []
    non_zero_vectors = generate_non_zero_vectors(n)
    pools = itertools.combinations(non_zero_vectors, n)
    for vector_set in pools:
        matrix = []
        for vector in vector_set:
            matrix.append(vector)
        result.append(matrix)
    with open('matrix_A'+'_'+str(n), 'wb') as fp:
        pickle.dump(result, fp)
    return result

# Do NOT SET n >= 6 !
generate_matrix_A(6)

# the following commented code is for reading the file. 
# with open ('matrix_A', 'rb') as fp:
#     itemlist = pickle.load(fp)
# print(len(itemlist))