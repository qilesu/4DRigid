import numpy as np
from numpy.linalg import inv

def simulate(dt,steps):
    # Rotation quaternions
    ql = np.array([1,0,0,0])
    qr = np.array([1,0,0,0])

    # principal 2nd moments
    J0 = np.diag([1,2,3,4])
    
    # Anglular momentum, specify the upper trangle
    L = np.array([[0,1,20,3],[0,0,10,10],[0,0,0,1],[0,0,0,0]])
    L = L-L.T # completes the lower half
    
    for i in range(steps):
        ql,qr=stepQuat(ql,qr,L,J0,dt)
        print(getKE(ql,qr,L,J0),L[0,2])

def getKE(ql,qr,L,J0):
    Omega=solveOmega(ql,qr,L,J0)
    return 1/4*np.trace(Omega.dot(L))

def solveOmega(ql,qr,L,J0):
    # This function solves the lyapunov equation L=-J*Omega-Omega*J
    # by expanding out the equation into the form l=A*w
    R = getR(ql,qr)
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

    # find the incremental rotations dql and dqr
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

    R=getR(dql,dqr)
    
    ql = quatNormalize(quatMult(dql,ql))
    qr = quatNormalize(quatMult(qr,dqr))
    
    return ql,qr

def getR(ql,qr):
    # converts left and right quaternions into a single rotation matrix
    matL = np.array([[ql[0],-ql[1],-ql[2],-ql[3]],
                     [ql[1],ql[0],-ql[3],ql[2]],
                     [ql[2],ql[3],ql[0],-ql[1]],
                     [ql[3],-ql[2],ql[1],ql[0]]])
    matR = np.array([[qr[0],-qr[1],-qr[2],-qr[3]],
                     [qr[1],qr[0],qr[3],-qr[2]],
                     [qr[2],-qr[3],qr[0],qr[1]],
                     [qr[3],qr[2],-qr[1],qr[0]]])
    return matL.dot(matR)

def quatMult(q1, q0):
    w0,x0,y0,z0 = q0
    w1,x1,y1,z1 = q1
    return np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                     x1*w0 + y1*z0 - z1*y0 + w1*x0,
                     -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                     x1*y0 - y1*x0 + z1*w0 + w1*z0])

def quatNormalize(q):
    return q/np.sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3])
