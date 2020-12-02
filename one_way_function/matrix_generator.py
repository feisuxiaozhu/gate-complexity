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
        matrix = np.array(matrix)
        if (np.abs(np.linalg.det(matrix))==1): # Only include invertible matrices that has determinant 1
            result.append(matrix)
    with open('./one_way_function/matrix_A'+'_'+str(n), 'wb') as fp:
        pickle.dump(result, fp)
    return result

# Generate all n-by-n rank <= r matrices
def generate_matrix_L(n,r):
    result = []
    indices_pool = [] # First generate all r-by-n indices
    for i in range(n):
        for j in range(r):
            index = [j,i]
            indices_pool.append(index)
    if r==n: # If rank is the same as dimension, return all possible zero-one matrices
        for i in range(1,r*r+1):
            chosen_non_one_indices_pool = itertools.combinations(indices_pool, i)
            for chosen_non_one_indices in chosen_non_one_indices_pool:
                temp_matrix = np.zeros((n,n),dtype=int)
                for chosen_non_one_index in chosen_non_one_indices:
                    temp_matrix[chosen_non_one_index[0]][chosen_non_one_index[1]] = 1
                result.append(temp_matrix)
        result.append(np.zeros((n,n),dtype=int))
        return result
    
    # If rank < n, then return all possible non-repetitive zero-one matrices of rank <= r
    temp_result = []
    candidate_r_by_n_matrices = []
    for i in range(1, r*n+1):
        chosen_non_one_indices_pool = itertools.combinations(indices_pool, i)
        for chosen_non_one_indices in chosen_non_one_indices_pool:
            temp_matrix = np.zeros((r,n),dtype=int)
            for chosen_non_one_index in chosen_non_one_indices:
                    temp_matrix[chosen_non_one_index[0]][chosen_non_one_index[1]] = 1
            candidate_r_by_n_matrices.append(temp_matrix)
    for r_by_n in candidate_r_by_n_matrices:
        for temp in candidate_r_by_n_matrices:
            n_by_r = np.transpose(temp)
            matrix = np.matmul(n_by_r,r_by_n)
            matrix = tuple(map(tuple, matrix))
            temp_result.append(matrix) 
    temp_result = set(temp_result) # remove duplicated matrices represented as tuple (only tuple is hashable so we have to conver array/list into tuple)
    for temp in temp_result:
        result.append(np.asarray(temp))
    
    with open('./one_way_function/matrix_L'+'_n'+str(n)+'_r'+str(r), 'wb') as fp:
        pickle.dump(result, fp)
    return result






# Do NOT SET n >= 6 !
generate_matrix_A(5)
generate_matrix_L(3,1)

# the following commented code is for reading the file. 
# with open ('./one_way_function/matrix_L_n3_r1', 'rb') as fp:
#     itemlist = pickle.load(fp)
#     print(len(itemlist))
#     for item in itemlist:
#         print(item.tolist())


