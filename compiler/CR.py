from mpmath import pslq, im, re, sqrt, matrix, norm
import numpy as np

omega = -0.5 + 0.866025j

def isInR3(a):
# Chek if the input is in the ring R_{3,chi}
  imgPart = im(a)
  realPart = re(a)

  if abs(imgPart-0)<0.001:
    a1=0
  elif pslq([imgPart,-0.866025],tol=0.01):
    a1 =pslq([imgPart,-0.866025],tol=0.01)[1]
    flagImg = pslq([imgPart,-0.866025], tol=0.01)[0]

    if flagImg != 1:
      return 0
  else:
    return 0

  if abs(realPart+a1/2-0)<0.001:
    a0 = 0
  elif pslq([realPart+a1/2,-1],tol=0.01):
    a0 = pslq([realPart+a1/2,-1],tol=0.01)[1]
    flagReal = pslq([realPart+a1/2,-1],tol=0.01)[0]

    if flagReal != 1:
      return 0
  else:
    return 0
  # print('linear combination is:' + str(a0)+str(a1)+'omega')
  return 1

def sde(z):
  if abs(z-0)<0.001:
    return 0
  # notice that z has to be in the ring R_{3,chi}, or it will become an infinite loop
  f = 0
  chi = sqrt(-3)
  a = z * chi**f
  while not isInR3(a):
    f = f + 1
    a = z * chi**f
  return f

def DMatrix(a,b,c):
  temp = matrix([[omega**a,0,0],[0,omega**b,0],[0,0,omega**c]])
  return temp

def sdeReduceList(z):
  H = matrix([[1,1,1],[1,omega,omega**2],[1,omega**2,omega]])/sqrt(-3)
  S = matrix([[1,0,0],[0,omega,0],[0,0,1]])
  R = matrix([[1,0,0],[0,1,0],[0,0,-1]])
  X = matrix([[0,0,1],[1,0,0],[0,1,0]])
  original_sde = [sde(z[0]),sde(z[1]),sde(z[2])]
  print('original sde is:' + str(original_sde))
  a_list = [0,1,2]
  epsilon_list = [0,1]
  delta_list = [0,1,2]
  for a_0 in a_list:
    for a_1 in a_list:
      for a_2 in a_list:
        for epsilon in epsilon_list:
          for delta in delta_list:
            D = DMatrix(a_0,a_1,a_2)
            new_z = H*D*R**epsilon*X**delta*S*z
            new_z1 = new_z[0]
            new_z2 = new_z[1]
            new_z3 = new_z[2]
            new_sde = [sde(new_z1),sde(new_z2),sde(new_z3)]
            print(new_sde)

def sdeReduceOneRound(z):
  H = matrix([[1,1,1],[1,omega,omega**2],[1,omega**2,omega]])/sqrt(-3)
  S = matrix([[1,0,0],[0,omega,0],[0,0,1]])
  R = matrix([[1,0,0],[0,1,0],[0,0,-1]])
  X = matrix([[0,0,1],[1,0,0],[0,1,0]])
  original_sde = [sde(z[0]),sde(z[1]),sde(z[2])]
  print('input sde is:' + str(original_sde))
  a_list = [0,1,2]
  epsilon_list = [0,1]
  delta_list = [0,1,2]
  for a_0 in a_list:
    for a_1 in a_list:
      for a_2 in a_list:
        for epsilon in epsilon_list:
          for delta in delta_list:
            D = DMatrix(a_0,a_1,a_2)
            new_z = H*D*R**epsilon*X**delta*S*z
            new_z1 = new_z[0]
            new_z2 = new_z[1]
            new_z3 = new_z[2]
            new_sde = [sde(new_z1),sde(new_z2),sde(new_z3)]
            if new_sde[0] == original_sde[0]-1:
              result = {'reduced_z': new_z, 'sde':new_sde, 'a_0':a_0, 'a_1':a_1, 'a_2':a_2, 'epsilon':epsilon, 'delta': delta}
              return result

def sdeReduceIteration(z):
  tempsde = [sde(z[0]),sde(z[1]),sde(z[2])]
  sdeFlag =  tempsde[0] == tempsde[1] and tempsde[0] == tempsde[2]
  normFlag = abs(norm(z)-1) < 0.001
  if sdeFlag and normFlag: #check input z to be a unit vector and has legit sde
    while tempsde[0]>0 and sdeFlag:
      result = sdeReduceOneRound(z)
      z = result['reduced_z']
      tempsde = [sde(z[0]),sde(z[1]),sde(z[2])]
      sdeFlag =  tempsde[0] == tempsde[1] and tempsde[0] == tempsde[2]
    return z
  else:
    return 0

testy = matrix([[1],[1],[1]])/sqrt(-3)
# testy = matrix([[191-82*omega],[4+1*omega],[15+8*omega]])/sqrt(-3)**10
# print(sdeReduce(testy))
# testy2 = sdeReduce(testy)['reduced_z']
# testy3 = sdeReduce(testy2)['reduced_z']

z = sdeReduceIteration(testy)



