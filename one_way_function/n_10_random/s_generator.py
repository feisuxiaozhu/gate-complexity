import numpy as np
import itertools
import pickle

# Generate all 3-by-10 S matrices that each row has at most 3 non-zero entries
indices = [0,1,2,3,4,5,6,7,8,9]
row_pools = []

zero_row = [0,0,0,0,0,0,0,0,0,0]
# zero row
row_pools.append(zero_row)
# one non-zero entry
for i in range(10):
    zero_row = [0,0,0,0,0,0,0,0,0,0]
    zero_row[i]=1
    row_pools.append(zero_row)

# two non-zero entries
# choose 2 non-zero entries:
indices_pool =  itertools.combinations(indices,2)
for i,j in indices_pool:
    zero_row = [0,0,0,0,0,0,0,0,0,0]
    zero_row[i]=1
    zero_row[j]=1
    row_pools.append(zero_row)

# three non-zero entries
# choose 3 non-zero entries:
indices_pool = itertools.combinations(indices,3)
for i,j,k in indices_pool:
    zero_row = [0,0,0,0,0,0,0,0,0,0]
    zero_row[i] = 1
    zero_row[j] = 1
    zero_row[k] = 1
    row_pools.append(zero_row)

# generate indices for all possible rows
possible_row_indices = []
for i in range(len(row_pools)):
    possible_row_indices.append(i)

result = []
# generate all possible S matrices
for i,j,k in itertools.product(possible_row_indices, repeat=3):
    matrix = [row_pools[i],row_pools[j],row_pools[k]]
    matrix = np.array(matrix)
    result.append(matrix)

with open('./one_way_function/n_10_random/matrix_S', 'wb') as fp:
        pickle.dump(result, fp)






