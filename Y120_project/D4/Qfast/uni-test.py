import numpy as np
from qfast import synthesize

U = np.loadtxt( "U.unitary", dtype = np.complex128 )

# U = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
#               [0, 1, 0, 0, 0, 0, 0, 0],
#               [0, 0, 1, 0, 0, 0, 0, 0],
#               [0, 0, 0, 1, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 1, 0],
#               [0, 0, 0, 0, 0, 1, 0, 0],
#               [0, 0, 0, 0, 1, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.complex128)


# print(U)
print(synthesize( U ) )






