from sympy import *

V1 = Symbol('V1')
V2 = Symbol('V2')
V11=Symbol('V11')
V1_1=Symbol('V1_1')
V3=Symbol('V3')
V21=Symbol('V21')
V111=Symbol('V111')
V2_1=Symbol('V2_1')
V11_1 = Symbol('V11_1')
N = Symbol('N')
beta=Symbol('beta')


beta2r = 1/2*V1**6*(1/4*V2+1/4*V11+1/2*V1_1-V1**2)*beta**2
beta2i = 1/2*V1**6*(-1/4*V2-1/4*V11+1/2*V1_1)*beta**2
beta2t = 1/(8*N)*V1**6*(V2-V11)*beta**2
beta2u = 1/(4*N**2)*V1**6*(1-V1_1)*beta**2
beta0 = 1/(4*N**2)*(1-V1**8)*beta**2
beta1 = V1**4*beta+1/(8*N**2)*V1**4*(4*V1**8-V11**4-2*V1_1**4-V2**4)*beta**3
beta2 = (N+1)/(8*N)*(V2**4-V1**8)*beta**2
beta3 = (N+1)*(N+2)/(6*N**2)*(1/24*V3**4+1/12*V1**(12)-1/8*V1**4*V2**4)*beta**3


dictSU3 = {V1: 0.8342, V2:0.6299, V11:0.8342, V1_1:0.6599, V3:0.4222, V21:0.6599, V111:1.0, V2_1:0.4679, V11_1:0.629, N:3}
print('SU(3):')
print('beta2r: '+ str(beta2r.subs(dictSU3)))
print('beta2i: '+ str(beta2i.subs(dictSU3)))
print('beta2t: '+ str(beta2t.subs(dictSU3)))
print('beta2u: '+ str(beta2u.subs(dictSU3)))

dictSU2 = {V1:0.9648, V2:0.9078, V11:1, V1_1:0.9078, V3:0.8325, V21:0.9648, N:2}
print('SU(2):')
print('beta2r: '+ str(beta2r.subs(dictSU2)))
print('beta2i: '+ str(beta2i.subs(dictSU2)))
print('beta2t: '+ str(beta2t.subs(dictSU2)))
print('beta2u: '+ str(beta2u.subs(dictSU2)))



