import sys
from mpmath import pslq, im, re, sqrt, matrix, norm
import numpy as np
import csv
import io

# Follow the theoretical work from https://arxiv.org/pdf/2311.08696
omega = -0.5 + 0.8660254037844387j

def read_approx(file_name):
    with open(file_name, 'r') as file:
        for line in file:
        # Remove leading/trailing whitespace
            line = line.strip()

        # Skip empty lines
            if not line:
                continue

        # Split the line into parts based on a delimiter (e.g., comma, space)
            parts = line.split(',')
            # print(parts)
            if len(parts)==9:
                return parts

def array_to_csv_string(array):
    with io.StringIO() as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(array)
        return csvfile.getvalue().strip()

def isInR3(a):
# Check if the input is in the ring R_{3}
  imgPart = im(a)
  realPart = re(a)
  # print(imgPart, realPart)
  if abs(imgPart-0)<1e-9:
    a1=0
    
  elif check_integer_division(imgPart,0.8660254037844387):
    a1 =check_integer_division(imgPart,0.8660254037844387)
    
  else:
    return 0
  

  if abs(realPart+a1/2)<1e-9:
    a0 = 0
  elif check_integer_division(realPart+a1/2,1):
    a0 = check_integer_division(realPart+a1/2,1)
  else:
    return 0
  # print('linear combination is:' + str(a0)+str(a1)+'omega')
  return 1

def sde(z):
  if abs(z-0)<1e-9:
    return 0
#  print("z: ",z)
  # notice that z has to be in the ring R_{3,chi}, or it will become an infinite loop
  f = 0
  chi = sqrt(-3)
  a = z * chi**f
  while not isInR3(a):
    # print(f)
    f = f + 1
    a = z * chi**f
  return f

def DMatrix(a,b,c):
  temp = matrix([[omega**a,0,0],[0,omega**b,0],[0,0,omega**c]])
  return temp

# def sdeReduceList(z):
#   H = matrix([[1,1,1],[1,omega,omega**2],[1,omega**2,omega]])/sqrt(-3)
#   S = matrix([[1,0,0],[0,omega,0],[0,0,1]])
#   R = matrix([[1,0,0],[0,1,0],[0,0,-1]])
#   X = matrix([[0,0,1],[1,0,0],[0,1,0]])
#   original_sde = [sde(z[0]),sde(z[1]),sde(z[2])]
#   print('original sde is:' + str(original_sde))
#   a_list = [0,1,2]
#   epsilon_list = [0,1]
#   delta_list = [0,1,2]
#   for a_0 in a_list:
#     for a_1 in a_list:
#       for a_2 in a_list:
#         for epsilon in epsilon_list:
#           for delta in delta_list:
#             D = DMatrix(a_0,a_1,a_2)
#             new_z = H*D*R**epsilon*X**delta*S*z
#             new_z1 = new_z[0]
#             new_z2 = new_z[1]
#             new_z3 = new_z[2]
#             print(new_z)
#             new_sde = [sde(new_z1),sde(new_z2),sde(new_z3)]
#             # print(new_sde)

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
            # print(new_z)
            new_sde = [sde(new_z1),sde(new_z2),sde(new_z3)]
            # print(new_sde)
            if new_sde[0] == original_sde[0]-1:
              result = {'reduced_z': new_z, 'sde':new_sde, 'a_0':a_0, 'a_1':a_1, 'a_2':a_2, 'epsilon':epsilon, 'delta': delta}
              return result

def sdeReduceIteration(z):
  count = 0
  tempsde = [sde(z[0]),sde(z[1]),sde(z[2])]
  print(str(tempsde[0])+',',end='')
  sdeFlag =  tempsde[0] == tempsde[1] and tempsde[0] == tempsde[2]
  normFlag = abs(norm(z)-1) < 0.0001
  if sdeFlag and normFlag: #check input z to be a unit vector and has legit sde
    while tempsde[0]>0 and sdeFlag:
      result = sdeReduceOneRound(z)
      count=count+1
      z = result['reduced_z']
      tempsde = [sde(z[0]),sde(z[1]),sde(z[2])]
      sdeFlag =  tempsde[0] == tempsde[1] and tempsde[0] == tempsde[2]
    return z,count
  else:
    return 0

def check_integer_division(a, divisor,tol=1e-3):
    # divisor = 0.8660254038
    result = a / divisor
    
    # Check if result is within tol of an integer
    rounded_result = round(result)
    if abs(result - rounded_result) < tol:
        return rounded_result
    else:
        return None

#testy = matrix([[1],[1],[1]])/sqrt(-3)
testy = matrix([[191-82*omega],[4+1*omega],[15+8*omega]])/sqrt(-3)**10
# print(sdeReduce(testy))
# testy2 = sdeReduce(testy)['reduced_z']
# testy3 = sdeReduce(testy2)['reduced_z']

if len(sys.argv) > 1:
    file_name = sys.argv[1]
#    print("file_name:", file_name)
else:
    print("Error, input file)")

parts=read_approx(file_name)
testy = matrix([[float(parts[4])],[float(parts[5])+float(parts[6])*1j],[float(parts[7])+float(parts[8])*1j]])
# print(sde(testy[0]),sde(testy[1]),sde(testy[2]))

# problem = (0.345602804445033 - 0.0641549186501988j)
# print(sde(problem))
# print(isInR3(problem))
# result = sdeReduceOneRound(testy)
# testy = result['reduced_z']
# result = sdeReduceOneRound(testy)
# print(result)

# for part in parts:
#     print(part+",",end='')
z,count = sdeReduceIteration(testy)
print(z)
# print(count)



