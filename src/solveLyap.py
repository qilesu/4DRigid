import numpy as np
from scipy.linalg import solve_lyapunov
from numpy.linalg import inv
from rotation import quatNormalize,quatMult
from rotation import stepQuat as step

J=np.array([[1,2,3,4],[2,3,5,7],[3,5,10,5],[4,7,5,12]])
L=np.array([[0,1,0,0],[-1,0,2,3],[0,-2,0,9],[0,-3,-9,0]])
J=(J+J.T)/2
L=(L-L.T)/2
ql=np.array([1,0,0,0])
qr=np.array([1,0,0,0])

def solveOmega(L,J):
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
def stepQuat(ql,qr,L,J,dt):
    Omega=solveOmega(L,J)
    dql=0.5*dt*np.array([0,
                    Omega[1,0]+Omega[3,2],
                    Omega[2,0]+Omega[1,3],
                    Omega[3,0]+Omega[2,1]])
    dql[0]=np.sqrt(1-dql[1]**2-dql[2]**2-dql[3]**2)
    dqr=0.5*dt*np.array([0,
                    Omega[1,0]-Omega[3,2],
                    Omega[2,0]-Omega[1,3],
                    Omega[3,0]-Omega[2,1]])
    dqr[0]=np.sqrt(1-dqr[1]**2-dqr[2]**2-dqr[3]**2)
    ql = quatMult(dql,ql)
    qr = quatMult(qr,dqr)
    return quatNormalize(ql),quatNormalize(qr)
