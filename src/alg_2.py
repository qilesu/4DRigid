import numpy as np
import quat
import rot
import lyap
from numpy.linalg import inv

'''
This is the same algorithm as alg_1, except all calculation is done in the
body frame
'''

def getKE(ql,qr,L,J0):
    Omega=solveOmega(ql,qr,L,J0)
    R = rot.as_matrix(ql,qr)
    L0=(R.T).dot(L).dot(R)
    return 1/4*np.trace(Omega.dot(L0))

def solveOmega(ql,qr,L,J0):
    # This function solves the lyapunov equation L=-J*Omega-Omega*J
    # by expanding out the equation into the form l=A*w
    R = rot.as_matrix(ql,qr)
    #J = R.dot(J0).dot(R.T)
    L0=(R.T).dot(L).dot(R)
    return lyap.solve(L0,J0)

def step(ql,qr,L,J0,dt):
    Omega=solveOmega(ql,qr,L,J0)
    ql,qr=rot.rotate_quat(ql,qr,dt*Omega,'body')
    
    ql = quat.normalize(ql)
    qr = quat.normalize(qr)
    
    return ql,qr
