#import linalg package of the SciPy module for the LU decomp
import scipy.linalg as linalg 
#import NumPy
import numpy as np
#define A same as before

# A = np.array([[x, 1] for x in range(1,4)], dtype=float)
A = np.array([[2, 1], [5, 1]], dtype=float)
B = np.array([5, 8], dtype=float)
# B = A[:, 0]+1

# A = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])  
# B = np.array([1,0,0,1,1])  
# xPredicted = np.array(([0,1,0], [1,0,0]), dtype=float)

#call the lu_factor function
LU = linalg.lu_factor(A) 

#solve given LU and B
x = linalg.lu_solve(LU, B) 
print("Solutions:\n{}\n".format(x))

#now we want to see how A has been factorized, P is the so called Permutation matrix
P, L, U = linalg.lu(A) 

print (P)
print (L)
print (U)


