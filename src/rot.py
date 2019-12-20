#!/usr/bin/env python3
import numpy as np
import quat

'''
This class should not depend on whether quaternions are implemented with lists
tuples or numpy arrays.
'''

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

def rotate_quat(ql,qr,theta,frame):
    '''
    rotate ql and qr according to the infinitesimal rotation represented by
    the skew symmetric matrix theta. Equivalent to multiplication by the matrix
    exp(theta) on the left/right depending on the frame used
    if frame=='world', new rotation is applied in the world frame
    if frame=='body', new rotation is applied in the body frame    
    returns the rotated quaternions
    '''
    ql_incr,qr_incr=as_quat_exp(theta)
    if frame=='body':
        ql=quat.mult(ql,ql_incr)
        qr=quat.mult(qr_incr,qr)
        return ql,qr
    elif frame=='world':
        ql=quat.mult(ql_incr,ql)
        qr=quat.mult(qr,qr_incr)
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
    print('correct exponentiation')
    mat=np.array([[0,2,3,4],
                  [-2,0,4,8],
                  [-3,-4,0,9],
                  [-4,-8,-9,0]])
    from scipy.linalg import expm
    print('scipy version')
    print(expm(mat))
    print('my version')
    ql,qr=as_quat_exp(mat)
    print(as_matrix(ql,qr))

    print('noncommutativity of rotation')
    matxy=np.array([[0,-1,0,0],
                    [1,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0]])
    matxz=np.array([[0,0,-1,0],
                    [0,0,0,0],
                    [1,0,0,0],
                    [0,0,0,0]])

    print('ROTATIONS SPECIFIED IN WORLD FRAME')
    ql=[1,0,0,0]
    qr=[1,0,0,0]
    print('rotate xy, 90 deg')
    ql,qr=rotate_quat(ql,qr,np.pi/2*matxy,'world')
    print(as_matrix(ql,qr))
    print('then rotate xz, 90 deg')
    ql,qr=rotate_quat(ql,qr,np.pi/2*matxz,'world')
    print(as_matrix(ql,qr))

    print('reset')
    ql=[1,0,0,0]
    qr=[1,0,0,0]
    print('rotate xz, 90 deg')
    ql,qr=rotate_quat(ql,qr,np.pi/2*matxz,'world')
    print(as_matrix(ql,qr))
    print('then rotate xy, 90 deg')
    ql,qr=rotate_quat(ql,qr,np.pi/2*matxy,'world')
    print(as_matrix(ql,qr))

    print('ROTATIONS SPECIFIED IN BODY FRAME')
    ql=[1,0,0,0]
    qr=[1,0,0,0]
    print('rotate xy, 90 deg')
    ql,qr=rotate_quat(ql,qr,np.pi/2*matxy,'body')
    print(as_matrix(ql,qr))
    print('then rotate xz, 90 deg')
    ql,qr=rotate_quat(ql,qr,np.pi/2*matxz,'body')
    print(as_matrix(ql,qr))

    print('reset')
    ql=[1,0,0,0]
    qr=[1,0,0,0]
    print('rotate xz, 90 deg')
    ql,qr=rotate_quat(ql,qr,np.pi/2*matxz,'body')
    print(as_matrix(ql,qr))
    print('then rotate xy, 90 deg')
    ql,qr=rotate_quat(ql,qr,np.pi/2*matxy,'body')
    print(as_matrix(ql,qr))
