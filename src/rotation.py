import numpy as np
import quaternion as quat
from numpy.linalg import inv

def getKE(ql,qr,L,J0):
    Omega=solveOmega(ql,qr,L,J0)
    return 1/4*np.trace(Omega.dot(L))

def solveOmega(ql,qr,L,J0):
    # This function solves the lyapunov equation L=-J*Omega-Omega*J
    # by expanding out the equation into the form l=A*w
    R = quat.get4Dmatrix(ql,qr)
    J = R.dot(J0).dot(R.T)
    # A would be much easier if J is diagonal
    A=np.array([[-J[0,0]-J[1,1],-J[1,2],-J[1,3],J[0,2],J[0,3],0],
                [-J[1,2],-J[0,0]-J[2,2],-J[2,3],-J[0,1],0,J[0,3]],
                [-J[1,3],-J[2,3],-J[0,0]-J[3,3],0,-J[0,1],-J[0,2]],
                [J[0,2],-J[0,1],0,-J[1,1]-J[2,2],-J[2,3],J[1,3]],
                [J[0,3],0,-J[0,1],-J[2,3],-J[1,1]-J[3,3],-J[1,2]],
                [0,J[0,3],-J[0,2],J[1,3],-J[1,2],-J[2,2]-J[3,3]]])
    l=np.array([L[0,1],L[0,2],L[0,3],L[1,2],L[1,3],L[2,3]])
    w=inv(A).dot(l) # this can be stored so we don't need to recompute it
    return np.array([[0,w[0],w[1],w[2]],
                     [-w[0],0,w[3],w[4]],
                     [-w[1],-w[3],0,w[5]],
                     [-w[2],-w[4],-w[5],0]])

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
