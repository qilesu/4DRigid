import numpy as np
import quat
import lyap
from numpy.linalg import inv

def getKE(ql,qr,L,J0):
    Omega=solveOmega(ql,qr,L,J0)
    return 1/4*np.trace(Omega.dot(L))

def solveOmega(ql,qr,L,J0):
    # This function solves the lyapunov equation L=-J*Omega-Omega*J
    # by expanding out the equation into the form l=A*w
    R = quat.get4Dmatrix(ql,qr)
    J = R.dot(J0).dot(R.T)
    return lyap.solve(L,J)

def step(ql,qr,L,J0,dt):
    Omega=solveOmega(ql,qr,L,J0)

    # find the infinitesimal quaternions dql and dqr
    dql=[0,
         0.5*dt*(Omega[1,0]+Omega[3,2]),
         0.5*dt*(Omega[2,0]+Omega[1,3]),
         0.5*dt*(Omega[3,0]+Omega[2,1])]
    dqr=[0,
         0.5*dt*(Omega[1,0]-Omega[3,2]),
         0.5*dt*(Omega[2,0]-Omega[1,3]),
         0.5*dt*(Omega[3,0]-Omega[2,1])]

    ql = quat.normalize(quat.mult(quat.exp(dql),ql))
    qr = quat.normalize(quat.mult(qr,quat.exp(dqr)))
    
    return ql,qr
