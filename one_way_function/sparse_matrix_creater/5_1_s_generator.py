import numpy as np
import pickle
import galois
import random
from typing import List, Any
from itertools import combinations, product

def s_generator(n, d):
    indices = [i for i in range(n)]
    for i in range(n+1): # choose how many rows are non-zero
        non_zero_rows_list = [list(i) for i in list(combinations(range(n), i))] #choose non-zero rows
        for non_zero_rows in non_zero_rows_list:
            non_zero_cols_list = [list(p) for p in product(range(d), repeat=len(non_zero_rows))]
            for non_zero_cols in non_zero_cols_list:
                print(non_zero_rows, non_zero_cols)

n=5
d=2
s_generator(n,d)