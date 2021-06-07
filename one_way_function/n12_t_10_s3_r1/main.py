import numpy as np
import itertools
import pickle
import random
from sympy import Matrix, latex

# a helper function checks whether rank = 2
def rank_checker(matrix):
    row_1 = matrix[0]
    row_2 = matrix[1]
    first_sum = ((row_1+row_2)%2).tolist()[0]
    sum_1=sum(first_sum)
    rank2 = True
    if sum_1 == 0:
        rank2 = False

    return rank2

def create_candidate_vectors(n,s):
    indices = [x for x in range(n)]
    candidate_vectors = []
    for number_of_non_zero in range(s,n+1):
        chosen_non_one_indices_pool = itertools.combinations(indices, number_of_non_zero)
        for non_zero_indices in chosen_non_one_indices_pool:
            vector = np.zeros((n))
            for non_zero_index in non_zero_indices:
                vector[non_zero_index] = 1
            candidate_vectors.append(vector)
    return candidate_vectors

def check_hamming_distance(x,y):
    z = x-y
    weight = 0
    for i in range(len(z)):
        weight += abs(z[i])
    return weight


# creat a set of vectors that have pairwise Hamming distance >2*(s-1)
def pairwise_vector_set(n,s):
    candidate_vectors = create_candidate_vectors(n,s)
    random_number = random.randint(0,len(candidate_vectors)-1)
    result = [] 
    initilized_vector = candidate_vectors[random_number]
    result.append(initilized_vector)

    for candidate in candidate_vectors:
        outside_ball = True
        for found_vector in result:
            if check_hamming_distance(candidate, found_vector) <= 2*(s-1):
                outside_ball = False
                break
        if outside_ball:
            result.append(candidate)
    return result

def pairwise_vector_set_for_initial_vector(n,s,initial_vector,candidate_vectors):
    result = [] 
    result.append(initial_vector)
    for candidate in candidate_vectors:
        outside_ball = True
        for found_vector in result:
            if check_hamming_distance(candidate, found_vector) <= 2*(s-1):
                outside_ball = False
                break
        if outside_ball:
            result.append(candidate)
    return result    

def check_rigidity(matrix):
    survivor_count = 0
    for i,j in row_number_set:
        survivor_count += 1
        survivor_set.add(survivor_count)
        for s in S_matrices:
            s_matrix = np.asmatrix(s)
            row_1 = matrix[i].tolist()
            row_2 = matrix[j].tolist()
            new_matrix = [row_1,row_2]
            new_matrix = np.asmatrix(new_matrix)
            substracted_matrix = (new_matrix + s_matrix) % 2
            if not rank_checker(substracted_matrix):
                # print(new_matrix)
                # print(s_matrix)
                # print(substracted_matrix)
                return False
    return True

# load all S matrices
with open('./matrix_S', 'rb') as fp:
    S_matrices = pickle.load(fp)

# pick all possible 2 rows from 12 rows
row_number = [0,1,2,3,4,5,6,7,8,9,10,11]
row_number_set = list(itertools.combinations(row_number,2))
survivor_set = set()

'''
--------------------------------------------------------------- 
'''

# s = 3
# n = 12
# vector_set = pairwise_vector_set(n,s)
# # create a matrix from the first 10 vectors
# matrix = []
# for i in range(n):
#     matrix.append(vector_set[i].tolist())
# matrix = np.array(matrix)

# new_matrix=[]
# for i in range(n):
#     row = []
#     for j in range(n):
#         if matrix[i][j]==0.:
#             row.append(0)
#         else:
#             row.append(1)
#     new_matrix.append(row)




# if int(np.linalg.det(new_matrix)) %2 == 1:
#     new_matrix = Matrix(new_matrix)
#     print(latex(new_matrix))
#     new_matrix_inverse = new_matrix.inv_mod(2)
#     print(latex(new_matrix_inverse))
#     new_matrix_inverse= np.array(new_matrix_inverse)
#     print(check_rigidity(new_matrix_inverse))
# else:
#     print('the matrix chosen in this run is not invertible')