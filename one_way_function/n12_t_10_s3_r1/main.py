import numpy as np
import itertools
import pickle
import random
from sympy import Matrix

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

# load all S matrices
with open('./matrix_S', 'rb') as fp:
    S_matrices = pickle.load(fp)

s = 3
n = 12
vector_set = pairwise_vector_set(n,s)
# create a matrix from the first 10 vectors
matrix = []
for i in range(n):
    matrix.append(vector_set[i].tolist())
matrix = np.array(matrix)

print(matrix)



