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
    print(counter_set)
    for i,j in counter_set.items():
        result *= 1./math.factorial(int(j))
    for i in combinatorics_coefficients:
        print(i)
        print(grab_value_for_coefficient(i))
        result *= grab_value_for_coefficient(i)
    print(result)

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
    print(combinatorics_coefficients)


k = 2
p=4
p_k = 1/float(4-4**(1/(2*k-1)))
global a_coefficients 
a_coefficients = [1/2.*p_k,p_k,1/2.*(1-3.*p_k),1/2.*(1.-3.*p_k),p_k,1/2.*p_k]
global b_coefficients 
b_coefficients = [p_k,p_k,1-4.*p_k,p_k,p_k,0]


# conjugators = ['a4','b4']
# commutators = ['b4','c4']

# helper(conjugators,commutators)
print(a_coefficients)
print(b_coefficients)
find_numerical_coefficient(['b4', 'b4', 'b4', 'b4', 'c4'])


