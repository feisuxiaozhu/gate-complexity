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
    temp_result = []
    counter = 0
    for A in A_candidates:
        for L in L_candidates:
            counter += 1
            if counter % 10000 == 0:
                print('operation number '+ str(counter))
                print(str('{:02.2f}'.format(counter/number_of_operation)) +' completed')
            S = A^L
            s_t_result = check_S(S,s,t)
            if not s_t_result:
                temp_result.append(tuple(map(tuple, A)))
                print('--------------------------------')
                print(A,L,S)
                
    temp_result = set(temp_result)  
    for temp in temp_result:
        result.append(np.asarray(temp)) 
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
s = 3
r = 1
result = main(n,r,s,t)   
print(len(result))


