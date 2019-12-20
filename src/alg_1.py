import numpy as np
import quat
import rot
import lyap
from numpy.linalg import inv

def getKE(ql,qr,L,J0):
    Omega=solveOmega(ql,qr,L,J0)
    return 1/4*np.trace(Omega.dot(L))

def solveOmega(ql,qr,L,J0):
    # This function solves the lyapunov equation L=-J*Omega-Omega*J
    # by expanding out the equation into the form l=A*w
    R = rot.as_matrix(ql,qr)
    J = R.dot(J0).dot(R.T)
    return lyap.solve(L,J)

def step(ql,qr,L,J0,dt):
    Omega=solveOmega(ql,qr,L,J0)
    #ql,qr=rot.rotate_quat(ql,qr,dt*Omega,'after')
    ql,qr=rot.rotate_quat(ql,qr,dt*Omega,'world')
    
    ql = quat.normalize(ql)
    qr = quat.normalize(qr)
    
    return ql,qr
