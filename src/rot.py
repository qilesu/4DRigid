#!/usr/bin/env python3
import numpy as np
import quat

def as_matrix(ql,qr):
    '''converts left and right quaternions into a single rotation matrix''' 
    matL = np.array([[ql[0],-ql[1],-ql[2],-ql[3]],
                     [ql[1],ql[0],-ql[3],ql[2]],
                     [ql[2],ql[3],ql[0],-ql[1]],
                     [ql[3],-ql[2],ql[1],ql[0]]])
    matR = np.array([[qr[0],-qr[1],-qr[2],-qr[3]],
                     [qr[1],qr[0],qr[3],-qr[2]],
                     [qr[2],-qr[3],qr[0],qr[1]],
                     [qr[3],qr[2],-qr[1],qr[0]]])
    return matL.dot(matR)

def rotate(ql,qr,theta,order='before'):
    '''
    rotate ql and qr according to the infinitesimal rotation represented by
    the skew symmetric matrix theta. Equivalent to multiplication by the matrix
    exp(theta) on the left/right depending on order
    if order=='before', new rotation is applied before the old one
    if order=='after', new rotation is applied after the old one    
    returns the rotated quaternions
    '''
    ql_incr,qr_incr=as_quat_exp(theta)
    if order=='before':
        ql=quat.mult(ql,ql_inc)
        qr=quat.mult(qr_inc,qr)
        return ql,qr
    elif order=='after':
        ql=quat.mult(ql_inc,ql)
        qr=quat.mult(qr,qr_inc)
        return ql,qr

def as_quat_exp(mat):
    '''
    returns the left and right quaternion representation of the exponential
    of the 4D skew symmetric matrix mat
    '''
    dql=[0,
         (mat[1,0]+mat[3,2])/2,
         (mat[2,0]+mat[1,3])/2,
         (mat[3,0]+mat[2,1])/2]
    dqr=[0,
         (mat[1,0]-mat[3,2])/2,
         (mat[2,0]-mat[1,3])/2,
         (mat[3,0]-mat[2,1])/2]
    
    return quat.exp(dql),quat.exp(dqr)

if __name__=='__main__':
    mat=np.array([[0,2,3,4],
                  [-2,0,4,8],
                  [-3,-4,0,9],
                  [-4,-8,-9,0]])
    from scipy.linalg import expm
    print(expm(mat))
    ql,qr=as_quat_exp(mat)
    print(as_matrix(ql,qr))
