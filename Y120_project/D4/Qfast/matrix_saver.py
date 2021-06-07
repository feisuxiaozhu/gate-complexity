import numpy as np

## D4 inversion
# U = np.matrix([[1, 0, 0, 0, 0, 0, 0, 0],
#               [0, 1, 0, 0, 0, 0, 0, 0],
#               [0, 0, 1, 0, 0, 0, 0, 0],
#               [0, 0, 0, 1, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 1, 0],
#               [0, 0, 0, 0, 0, 1, 0, 0],
#               [0, 0, 0, 0, 1, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 1]],dtype=np.complex128)

# np.savetxt('U.unitary',U)

# Y120 inversion
U = np.zeros((128,128), dtype=np.complex128)

for i in range(118):
    if i% 2 != 1:
        U[i][i+1] = 1
    else:
        U[i][i-1] = 1

for i in range(118,128):
    U[i][i]=1

np.savetxt('U.unitary',U)






