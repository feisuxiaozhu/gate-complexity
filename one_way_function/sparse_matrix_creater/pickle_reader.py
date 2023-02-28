import pickle

with open('./one_way_function/sparse_matrix_creater/B5.pickle', 'rb') as handle:
        b = pickle.load(handle)
print(b)
print(len(b))