from group_classifier import check_equal, find_index_in_list, tr, mult, inv
import numpy as np


Y_120 = np.load('./Y120_element.npy',allow_pickle='TRUE')
M = np.zeros((120,120),dtype=int)

repeat_counter = 0
for i in range(120):
    A = Y_120[i]
    B = inv(A)
    index = find_index_in_list(B, Y_120)
    M[index][i]=1
    print(i)
    print(index)



