import numpy as np
import itertools
import pickle
# Set n=10, r=2, s=4, t=7
# a helper function checks whether rank = 3
def rank_checker(matrix):
    row_1 = matrix[0]
    row_2 = matrix[1]
    row_3 = matrix[2]
    first_sum = ((row_1+row_2)%2).tolist()[0]
    second_sum = ((row_1+row_3)%2).tolist()[0]
    third_sum = ((row_2+row_3)%2).tolist()[0]
    sum_1 = sum(first_sum)
    sum_2 = sum(second_sum)
    sum_3 = sum(third_sum)
    rank3 = True
    if sum_1 == 0:
        rank3 = False
    elif sum_2 == 0:
        rank3 = False
    elif sum_3 == 0:
        rank3 = False
    return rank3

# first generate a random 10-by-10 matrix
# matrix = np.random.randint(2, size=(10, 10))

# pick all possible 3 rows from 10 rows
row_number = [0,1,2,3,4,5,6,7,8,9]
row_number_set = list(itertools.combinations(row_number,3))

# load all S matrices
with open('./one_way_function/n_10_random/matrix_S', 'rb') as fp:
    S_matrices = pickle.load(fp)

def check_rigidity(matrix):
    for i,j,k in row_number_set:
        print(i,j,k)
        for s in S_matrices:
            s_matrix = np.asmatrix(s)
            row_1 = matrix[i].tolist()
            row_2 = matrix[j].tolist()
            row_3 = matrix[k].tolist()
            new_matrix = [row_1,row_2,row_3]
            new_matrix = np.asmatrix(new_matrix)
            substracted_matrix = (new_matrix + s_matrix) % 2
            if not rank_checker(substracted_matrix):
                return False
    return True


counter = 0
result = []
found_rigid = False
while not found_rigid:
    counter += 1
    print(counter, ' number of random matrices generated')
    matrix = np.random.randint(2, size=(10, 10))
    if check_rigidity(matrix):
        print(matrix)
        result.append(np.array(matrix.tolist()))
        found_rigid =  True
        

with open('./one_way_function/n_10_random/rigid_matrix', 'wb') as fp:
    pickle.dump(result, fp)      
    










