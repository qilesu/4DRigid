#!/usr/bin/env python3
import numpy as np

class Rotor:
    def __init__(self,a,xy,xz,xw,yz,yw,zw):
        self.a=a
        self.xy=xy
        self.xz=xz
        self.xw=xw
        self.yz=yz
        self.yw=yw
        self.zw=zw
    def __str__(self):
        return '({}, ({}, {}, {}; {}, {}; {}))'.format(self.a,
                                                       self.xy,self.xz,self.xw,
                                                       self.yz,self.yw,
                                                       self.zw)
    def norm(self):
        return pow(self.a**2+
                   self.xy**2+self.xz**2+self.xw**2+
                   self.yz**2+self.yw**2+
                   self.zw**2,
                   0.5)
    def normalize(self):
        '''a mutator method which normalizes the rotor'''
        norm=self.norm()
        self.a/=norm
        self.xy/=norm
        self.xz/=norm
        self.xw/=norm
        self.yz/=norm
        self.yw/=norm
        self.zw/=norm
    def normal(self):
        norm=self.norm()
        return self/norm
    def reverse(self):
        return Rotor(self.a,
                     -self.xy,-self.xz,-self.xw,
                     -self.yz,-self.yw,
                     -self.zw)
    def __mul__(self,q):
        if type(q) == Rotor:
            p = self
            a  = p.a*q.a  - p.xy*q.xy - p.xz*q.xz - p.xw*q.xw - p.yz*q.yz - p.yw*q.yw - p.zw*q.zw
            xy = p.a*q.xy + p.xy*q.a  + p.yz*q.xz - p.xz*q.yz + p.yw*q.xw - p.xw*q.yw
            xz = p.a*q.xz + p.xz*q.a  + p.xy*q.yz - p.yz*q.xy + p.zw*q.xw - p.xw*q.zw
            xw = p.a*q.xw + p.xw*q.a  + p.xy*q.yw - p.yw*q.xy + p.xz*q.zw - p.zw*q.xz
            yz = p.a*q.yz + p.yz*q.a  + p.xz*q.xy - p.xy*q.xz + p.zw*q.yw - p.yw*q.zw
            yw = p.a*q.yw + p.yw*q.a  + p.xw*q.xy - p.xy*q.xw + p.yz*q.zw - p.zw*q.yz
            zw = p.a*q.zw + p.zw*q.a  + p.xw*q.xz - p.xz*q.xw + p.yw*q.yz - p.yz*q.yw
            return Rotor(a,xy,xz,xw,yz,yw,zw)
        else:
            return Rotor(self.a*q,
                     self.xy*q,self.xz*q,self.xw*q,
                     self.yz*q,self.yw*q,
                     self.zw*q)
    def __truediv__(self,q):
        return Rotor(self.a/q,
                     self.xy/q,self.xz/q,self.xw/q,
                     self.yz/q,self.yw/q,
                     self.zw/q)
    def rotate(self,v):
        p=self 
        #left mult
        x   =  p.a*v[0] + p.xy*v[1] + p.xz*v[2] + p.xw*v[3]
        y   =  p.a*v[1] - p.xy*v[0] + p.yz*v[2] + p.yw*v[3]
        z   =  p.a*v[2] - p.xz*v[0] - p.yz*v[1] + p.zw*v[3]
        w   =  p.a*v[3] - p.xw*v[0] - p.yw*v[1] - p.zw*v[2]
        xyz = p.xy*v[2] - p.xz*v[1] + p.yz*v[0]
        yzw = p.yz*v[3] - p.yw*v[2] + p.zw*v[1]
        zwx = p.xz*v[3] - p.xw*v[2] + p.zw*v[0]
        wxy = p.xy*v[3] - p.xw*v[1] + p.yw*v[0]
        #right mult by reverse
        p=p.reverse()
        u=[0,0,0,0]
        u[0] = x*p.a - y*p.xy - z*p.xz - w*p.xw - xyz*p.yz - wxy*p.yw - zwx*p.zw
        u[1] = y*p.a + x*p.xy - z*p.yz - w*p.yw + xyz*p.xz + wxy*p.xw - yzw*p.zw
        u[2] = z*p.a + x*p.xz + y*p.yz - w*p.zw - xyz*p.xy + zwx*p.xw + yzw*p.yw
        u[3] = w*p.a + x*p.xw + y*p.yw + z*p.zw - wxy*p.xy - zwx*p.xz - yzw*p.yz
        return u
        
    def as_matrix(self):
        A=[self.rotate([1,0,0,0]),
           self.rotate([0,1,0,0]),
           self.rotate([0,0,1,0]),
           self.rotate([0,0,0,1])]
        return np.array(A).T

# static methods
def exp(r):
    expa=np.exp(r.a)
    bvnorm=pow(r.xy**2+r.xz**2+r.xw**2+
               r.yz**2+r.yw**2+
               r.zw**2,
               0.5)
    bvscale=expa*np.sinc(bvnorm/np.pi)
    return Rotor(expa*np.cos(bvnorm),
                 bvscale*r.xy, bvscale*r.xz, bvscale*r.xw,
                 bvscale*r.yz, bvscale*r.yw,
                 bvscale*r.zw)
def get_rotor(vFrom,vTo):
    a=1+vFrom[0]*vTo[0]+vFrom[1]*vTo[1]+vFrom[2]*vTo[2]+vFrom[3]*vTo[3]
    xy=vTo[0]*vFrom[1]-vTo[1]*vFrom[0]
    xz=vTo[0]*vFrom[2]-vTo[2]*vFrom[0]
    xw=vTo[0]*vFrom[3]-vTo[3]*vFrom[0]
    yz=vTo[1]*vFrom[2]-vTo[2]*vFrom[1]
    yw=vTo[1]*vFrom[3]-vTo[3]*vFrom[1]
    zw=vTo[2]*vFrom[3]-vTo[3]*vFrom[2]
    r=Rotor(a,xy,xz,xw,yz,yw,zw)
    r.normalize()
    return r



if __name__=='__main__':
    print('test __init__()')
    r=Rotor(10,1,1,2,3,5,7)
    print(r)
    
    print('test norm()')
    r=Rotor(0,3,4,0,0,0,0)
    print(r.norm())
    r=Rotor(5,12,0,0,0,0,0)
    print(r.norm())

    print('test normalize(), mutator')
    r=Rotor(0,3,4,0,0,0,0)
    r.normalize()
    print(r)
    r=Rotor(4,3,0,0,0,0,0)
    r.normalize()
    print(r)

    print('test normal()')
    r=Rotor(1,1,0,0,0,0,0)
    print(r.normal())
    r=Rotor(0,3,4,0,0,0,0)
    print(r.normal())

    print('test reverse()')
    r=Rotor(1,2,3,4,5,6,7)
    print(r.reverse())
    
    print('test __mul__()')
    a=Rotor(0,3,4,0,0,0,0)
    b=Rotor(5,12,0,0,0,0,0)
    print(a*b)
    print(a*2)
    a=Rotor(0,9,8,7,6,5,4)
    a.normalize()
    print(a*a*(-1))

    print('test __truediv__()')
    a=Rotor(0,3,4,0,0,0,0)
    print(a/2)
    
    print('test get_rotor()')
    vFrom=[1,0,0,0]
    vTo=[0,1,0,0]
    r=get_rotor(vFrom,vTo)
    print(r)
    print(r*r)
    
    print('test rotate()')
    r=get_rotor([1,0,0,0],[0,1,0,0])
    print(r.rotate([1,0,0,0]))
    print(((r*r).normal()).rotate([1,0,0,0]))
    print(((r*r*r).normal()).rotate([1,0,0,0]))
    print(((r*r*r*r).normal()).rotate([1,0,0,0]))
    r=get_rotor([0,0,1,0],[0,1,0,0])
    print(r.rotate([1,2,3,4]))
    print(((r*r).normal()).rotate([1,2,3,4]))
    print(((r*r*r).normal()).rotate([1,2,3,4]))
    print(((r*r*r*r).normal()).rotate([1,2,3,4]))
    
    print('test as_matrix()')
    r=get_rotor([1,0,0,0],[0,1,0,0])
    print(r.as_matrix())
    r=get_rotor([0,0,0,1],[0,1,0,0])
    print(r.as_matrix())

    print('test exp()')
    r=Rotor(5,12,0,0,0,0,0)
    print(exp(r))
    print(exp(Rotor(0,12,0,0,0,0,0)))
    print(exp(r).normal())
    a=Rotor(0,1,0,0,0,0,0)
    r=exp(a*np.pi/4)#angle of rotation is 2x norm of bvector. expect 90 deg rot
    print(r)
    print(r.as_matrix())
    a=Rotor(0,9,8,7,6,5,4)
    a.normalize()
    r=exp(a*np.pi/4)
    v=[10,-20,30,-40]
    u=(r*r*r*r).normal().rotate(v)
    print(u)
