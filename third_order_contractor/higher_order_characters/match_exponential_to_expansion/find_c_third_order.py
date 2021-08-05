import sympy as sym
import numpy as np
import _pickle as pickle


read_dictionary = np.load(
    './c_for_i_j_result.npy', allow_pickle='TRUE').item()

result = {}
for i in range(17):
    for j in range(17):
        for k in range(17):
            first_two_index = str(i+1) + '_' + str(j+1)
            first_two_index_values = read_dictionary[first_two_index]
            three_index = str(i+1) + '_' + str(j+1)+'_'+str(k+1)
            temp_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            for counter in range(17):
                temp_array = temp_array + first_two_index_values[counter] * np.array(read_dictionary[str(counter+1)+'_'+str(k+1)])
            result[three_index] = temp_array.tolist()


print(result)
with open('./c_for_i_j_k_result.npy', 'wb') as f:
    np.save(f, result)