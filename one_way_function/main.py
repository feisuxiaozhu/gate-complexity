import pickle
import numpy as np

def main(n,r,s,t):
    L_candidates = []
    result=[]
    for i in range(r):
        rank = i+1
        L_file_name = 'matrix_L_n'+str(n)+'_r'+str(rank)
        with open ('./one_way_function/'+L_file_name, 'rb') as fp:
            L_candidates_temp =  pickle.load(fp)
        L_candidates = L_candidates + L_candidates_temp
    
    A_file_name = 'matrix_A_'+str(n)
    with open ('./one_way_function/'+ A_file_name, 'rb') as fp:
        A_candidates =  pickle.load(fp)
    
    print('There are '+str(len(A_candidates)) + ' number of matrices A, and '+str(len(L_candidates)) + ' number of matrices L')
    number_of_operation = len(A_candidates)*len(L_candidates)
    print('There are '+str(number_of_operation)+ ' number of operations to check' )
    result = []
    counter = 0
    for A in A_candidates:
        counter += 1
        print('matrix number '+ str(counter))
        trigger = False
        for L in L_candidates:
            S = A^L
            s_t_result = check_S(S,s,t)
            if s_t_result:
                trigger = True
                break
        if not trigger:
            print(A)
            result.append(A)

    return result

def check_S(matrix,s,t):
    counter = 0
    for row in matrix:
        density = sum(row)
        if density >= s:
            counter += 1
    if counter <= t:
        return True # matrix is (s,t)-sparse
    else:
        return False # matrix is not (s,t)-sparse
n=4
t = 3
s = 0
r = 3
result = main(n,r,s,t)   
print(result)
print(len(result))


