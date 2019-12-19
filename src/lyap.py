#!/usr/bin/env python3
import numpy as np
from numpy.linalg import inv

def solve(L,J):
    '''
    solves the 4D matrix equation L=-JX-XJ. Accepted J can be
    1: a len 4 array of diagonal elements of J,
    2: a 4x4 matrix for J
    3: a 6x6 matrix obtained by lyap.inverse(J)
    '''
    # A would be much easier if J is diagonal
    if J.shape==(4,):
        return -L/(J[:,None]+J)
    
    if J.shape==(4,4):
        J=inverse(J)
    
    l=np.array([L[0,1],L[0,2],L[0,3],L[1,2],L[1,3],L[2,3]])
    w=J.dot(l) # this can be stored so we don't need to recompute it
    X=np.array([[0,w[0],w[1],w[2]],
                [-w[0],0,w[3],w[4]],
                [-w[1],-w[3],0,w[5]],
                [-w[2],-w[4],-w[5],0]])
    return X

def inverse(J):
    '''
    returns the 6x6 inverse used to solve the 4D matrix equation L=-JX-XJ.
    '''
    A=np.array([[-J[0,0]-J[1,1],-J[1,2],-J[1,3],J[0,2],J[0,3],0],
                [-J[1,2],-J[0,0]-J[2,2],-J[2,3],-J[0,1],0,J[0,3]],
                [-J[1,3],-J[2,3],-J[0,0]-J[3,3],0,-J[0,1],-J[0,2]],
                [J[0,2],-J[0,1],0,-J[1,1]-J[2,2],-J[2,3],J[1,3]],
                [J[0,3],0,-J[0,1],-J[2,3],-J[1,1]-J[3,3],-J[1,2]],
                [0,J[0,3],-J[0,2],J[1,3],-J[1,2],-J[2,2]-J[3,3]]])
    return inv(A)

if __name__=='__main__':
    # Tests
    L=np.array([[0,1,2,3],
                [-1,0,5,8],
                [-2,-5,0,9],
                [-3,-8,-9,0]])
    Jdiag=np.array([1,2,3,4])
    print(solve(L,Jdiag))
    print(solve(L,np.diag(Jdiag)))
    J=np.array([[1,2,3,4],
                [2,4,6,7],
                [3,6,10,15],
                [4,7,15,19]])
    print(solve(L,J))
    J_inv=inverse(J)
    print(solve(L,J_inv))
