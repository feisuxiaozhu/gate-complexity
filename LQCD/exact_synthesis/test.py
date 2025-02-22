from CR_class import *
import sys
import csv
import io

H = HMatrix()
S = SMatrix()
R = RMatrix()
X = XMatrix()


def sdeReduceOneRound(z):
    original_sde = ExtractSDE(z)
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
                        new_sde = ExtractSDE(new_z)
                        if new_sde[0] < original_sde[0]:
                            result = {'reduced_z': new_z, 'sde':new_sde, 'a_0':a_0, 'a_1':a_1, 'a_2':a_2, 'epsilon':epsilon, 'delta': delta}
                            return result

def sdeReduceIteration(z):
  count = 0
  tempsde = ExtractSDE(z)
  print(str(tempsde[0])+',',end='')
  sdeFlag =  tempsde[0] == tempsde[1] and tempsde[0] == tempsde[2]
  if sdeFlag: #check input z to be a unit vector and has legit sde
    while tempsde[0]>0 and sdeFlag:
      result = sdeReduceOneRound(z)
      count=count+1
      z = result['reduced_z']
      tempsde = ExtractSDE(z)
      sdeFlag =  tempsde[0] == tempsde[1] and tempsde[0] == tempsde[2]
    return z,count
  else:
    return 0


testy = EisensteinVector3(EisensteinFraction(EisensteinInteger(191,-82), 10), EisensteinFraction(EisensteinInteger(4,1), 10), EisensteinFraction(EisensteinInteger(15,8), 10))
z,count = sdeReduceIteration(testy)



