import numpy as np
import quat
import rot
import lyap
from numpy.linalg import inv

'''
This is the leapfrog in body frame
'''

def getKE(Omega,J0):
    return -1/2*np.trace(Omega.dot(Omega).dot(J0))

def solveAccel(Omega,J0):
    # This function solves the lyapunov equation LHS=-J*A-A*J
    LHS=Omega.dot(Omega).dot(J0)-J0.dot(Omega).dot(Omega)
    return lyap.solve(LHS,J0)

def step(ql,qr,Omega,J0,dt):
    A=solveAccel(Omega,J0)
    Omega=Omega+A*dt/2
    
    ql,qr=rot.rotate_quat(ql,qr,dt*Omega,'body')
    ql = quat.normalize(ql)
    qr = quat.normalize(qr)

    A=solveAccel(Omega,J0)
    Omega=Omega+A*dt/2
    
    return ql,qr,Omega
