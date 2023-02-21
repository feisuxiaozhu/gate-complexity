import pickle

with open('./one_way_function/sparse_matrix_creater/A.pickle', 'rb') as handle:
        b = pickle.load(handle)

print(b)