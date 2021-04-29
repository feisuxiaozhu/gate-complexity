import numpy as np
from main import create_candidate_vectors,  pairwise_vector_set_for_initial_vector, check_rigidity
import itertools
import pickle
from sympy import Matrix
# We want to exhaust all possible n=12 (r,s,t)-rigidt matrices, for r=1, s=3, t=10
# But notice our method is semi-exhaustive, since pairwise_vector_set_for_initial_vector uses greedy algorithm once
# and cannot cover all possible pair-wise independent set.
counter = 0
# We will only save the rigid matrices with inverse who has row sparsity two
def createAllPossibleRigidMatrices(pairwise_set, n, number):
    result = []
    indices = [i for i in range(1,len(pairwise_set))]
    indices_pool =itertools.combinations(indices, n-1)
    counter = 1
    for indices in indices_pool:
        indices=list(indices)
        indices=[0] + indices
        matrix = []
        for index in indices:
            matrix.append(pairwise_set[index].tolist())
        matrix = np.array(matrix)
        new_matrix=[]
        for i in range(n):
            row = []
            for j in range(n):
                if matrix[i][j]==0.:
                    row.append(0)
                else:
                    row.append(1)
            new_matrix.append(row)

# The following part is time consuming, maybe we can optimize the code below to make it run
# faster! But I think it is intrinsically time consuming to find inverses.
        if int(np.linalg.det(new_matrix)) %2 == 1:
            try:
                new_matrix = Matrix(new_matrix)
                new_matrix_inverse = new_matrix.inv_mod(2)
                new_matrix_inverse = np.array(new_matrix_inverse)
                new_matrix = np.array(new_matrix)
                for i in range(n):
                    if sum(new_matrix_inverse[i])<=2:
                        print(new_matrix_inverse)
                        result.append(new_matrix_inverse)
                        break
            except Exception as ex:
                print(ex)
        counter += 1
        print('Working on candidate vector number: ' + str(number+1))
        print('Number of rigid matrices checked: ' + str(counter))
        
    return result


'''
--------------------------------------------------------------- 
'''
result = []
n = 12
s = 3
candidate_vectors = create_candidate_vectors(n, s)
for i in range(len(candidate_vectors)):
# for i in range(1,2):
    initilized_vector = candidate_vectors[i]
    pairwise_set = pairwise_vector_set_for_initial_vector(
        n, s, initilized_vector, candidate_vectors[i:])
    result = result + createAllPossibleRigidMatrices(pairwise_set, n, i)

with open('./sparse_inverses', 'wb') as fp:
        pickle.dump(result, fp)
