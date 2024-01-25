import numpy as np

def f(x): # a function of degree at most len(x)-1
    length = len(x)
    result = 0
    for i in range(length):
        result += x[i]**i
    return result

n = 10 # number of features
k = 10000 # number of samples from normal distribution
mean = []
for i in range(n):
    mean.append(0)
A = np.random.rand(n,n)
AT = np.transpose(A)
covA = np.matmul(A,AT)

samples = list(np.random.multivariate_normal(mean, covA,k))
values = []
for sample in samples:
    value = f(sample)
    values.append(value)

print(values)






