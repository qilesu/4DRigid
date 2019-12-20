#!/usr/bin/env python3
import numpy as np
'''
All of the below methods returns a list. While the arguments passed can be
tuples or numpy arrays, it's better to stick with lists
'''

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

if __name__=='__main__':
    q=[5,0,12,0]
    print(norm(q))
    q=normalize(q)
    print(norm(q))

    q=[0,1,2,3]
    q=normalize(q)
    print(q)
    q=mult(q,q)
    print(q)

    q1=[1,2,3,4]
    q2=[5,6,7,8]
    q3=[9,10,11,12]
    print(mult(q1,mult(q2,q3))==mult(mult(q1,q2),q3))

    print(np.exp(9-10j))
    q=exp([9,6,8,0])
    print(q)
    print(np.sqrt(q[1]*q[1]+q[2]*q[2]+q[3]*q[3]))
