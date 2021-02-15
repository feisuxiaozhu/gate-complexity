import numpy as np
import itertools
import pickle
# Set n=10, r=2, s=4, t=7
# a helper function checks whether rank = 3
def rank_checker(matrix):
    row_1 = matrix[0]
    row_2 = matrix[1]
    first_sum = ((row_1+row_2)%2).tolist()[0]
    sum_1=sum(first_sum)
    rank2 = True
    if sum_1 == 0:
        rank2 = False

    return rank2

# first generate a random 10-by-10 matrix
# matrix = np.random.randint(2, size=(10, 10))

# pick all possible 2 rows from 10 rows
row_number = [0,1,2,3,4,5,6,7,8,9]
row_number_set = list(itertools.combinations(row_number,2))

# load all S matrices
with open('./matrix_S', 'rb') as fp:
    S_matrices = pickle.load(fp)


survivor_set = set()

def check_rigidity(matrix):
    survivor_count = 0
    for i,j in row_number_set:
        print(i,j)
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

counter = 0
result = []
found_rigid = False

while not found_rigid:
    counter += 1
    print(counter, ' number of random matrices generated')
    print(survivor_set)
    matrix = np.random.randint(2, size=(10, 10))
    if check_rigidity(matrix):
        print(matrix)
        result.append(np.array(matrix.tolist()))
        found_rigid =  True    

with open('./rigid_matrix', 'wb') as fp:
    pickle.dump(result, fp)      
    










