import numpy as np

def get4Dmatrix(ql,qr):
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

def mult(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return [-x1*x2 - y1*y2 - z1*z2 + w1*w2,
            x1*w2 + y1*z2 - z1*y2 + w1*x2,
            -x1*z2 + y1*w2 + z1*x2 + w1*y2,
            x1*y2 - y1*x2 + z1*w2 + w1*z2]
def exp(q):
    norm=np.exp(q[0])
    v=np.sqrt(q[1]*q[1]+q[2]*q[2]+q[3]*q[3])
    factor=norm*np.sinc(v/np.pi)
    return [norm*np.cos(v),
            factor*q[1],
            factor*q[2],
            factor*q[3]]

def norm(q):
    return np.sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3])

def normalize(q):
    normq=norm(q)
    return [q[0]/normq,q[1]/normq,q[2]/normq,q[3]/normq]
