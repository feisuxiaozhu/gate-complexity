import math

#n number of terms, S the total number they sum up to
#this function calculates all non-negative solutions to x_1+..+x_n=S
def f(array,n,S,res): 
    res = res
    s = sum(array)
    if n == 1:
        res.append(array + [S-s])
    else:
        for i in xrange(0,S+1-s):
            f(array + [i],n-1,S,res)

def find_nest_commutator(combinatorics_coefficients):
    res=''
    length = len(combinatorics_coefficients)
    for i in range(length-1):
        res += combinatorics_coefficients[i][0].capitalize()
    if res[-1]== 'A':
        res += 'B'
    else:
        res += 'A'
    return res

def grab_value_for_coefficient(x):
    index = int(x[-1])-1
    if (x[0]=='a'):
        return a_coefficients[index]
    elif  (x[0]=='b'):
        return b_coefficients[index]
    elif (x[0]=='c'):
        temp = 0
        for i in range(index+1):
            temp+= a_coefficients[i]
        return temp
    else:
        temp = 0
        for i in range(index+1):
            temp+= b_coefficients[i]
        return temp
     
def find_numerical_coefficient(combinatorics_coefficients):
    result = 1./5
    counter_set = {}
    length = len(combinatorics_coefficients)
    for i in range(length-1):
        if combinatorics_coefficients[i] in counter_set.keys():
            counter_set[combinatorics_coefficients[i]] += 1
        else:
            counter_set[combinatorics_coefficients[i]] = 1
    for i,j in counter_set.items():
        result *= 1./math.factorial(int(j))
    for i in combinatorics_coefficients:
        result *= grab_value_for_coefficient(i)
    return abs(result)

def helper(conjugators,commutators):
    n = len(conjugators)
    S = 3
    combinatorics = []
    f([],n,S,combinatorics)
    combinatorics_coefficients = []
    for combo in combinatorics:
        temp = []
        for index in range(len(combo)):
            repeat = combo[index]
            for j in range(repeat):
                temp.append(conjugators[index])
        temp.append(commutators[0])
        temp.append(commutators[1])
        combinatorics_coefficients.append(temp)
    for combo in combinatorics_coefficients:
        nest_commutator = find_nest_commutator(combo)
        numerical_contribution = find_numerical_coefficient(combo)
        if nest_commutator in final_result.keys():
            final_result[nest_commutator] += numerical_contribution
        else:
            final_result[nest_commutator] = numerical_contribution


k = 2
p=4
p_k = 1./(4-4**(1./(2*k-1)))
global a_coefficients 
a_coefficients = [1/2.*p_k,p_k,1/2.*(1-3.*p_k),1/2.*(1.-3.*p_k),p_k,1/2.*p_k]
global b_coefficients 
b_coefficients = [p_k,p_k,1-4.*p_k,p_k,p_k,0]
global final_result 
final_result = {}

conjugators_list_1 = [['a4','b4','a5','b5','a6'],['a4','b4','a5','b5'],['a4','b4','a5'],['a4','b4'],['a4']]
commutators_list_1 = [['a6','d5'],['b5','c5'],['a5','d4'],['b4','c4'],['a4','d3']]
conjugators_list_2 = [['b3'],['b3','a3'],['b3','a3','b2'],['b3','a3','b2','a2'],['b3','a3','b2','a2','b1']]
commutators_list_2 = [['b3','c3'],['a3','d2'],['b2','c2'],['a2','d1'],['b1','c1']]

for i in range(len(conjugators_list_1)):
    conjugators = conjugators_list_1[i]
    commutators = commutators_list_1[i]
    helper(conjugators,commutators)

for i in range(len(conjugators_list_2)):
    conjugators = conjugators_list_2[i]
    commutators = commutators_list_2[i]
    helper(conjugators,commutators)

combined_result = {}
for i,j in final_result.items():
    prefix = i[0:3]
    if prefix in combined_result.keys():
        combined_result[prefix] += j
    else:
        combined_result[prefix] = j
print(combined_result)



