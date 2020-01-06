#!/usr/bin/env python3

class Bivector:
    def __init__(self,xy,xz,xw,yz,yw,zw):
        self.xy=xy
        self.xz=xz
        self.xw=xw
        self.yz=yz
        self.yw=yw
        self.zw=zw
    def __str__(self):
        return '({}, {}, {}; {}, {}; {})'.format(self.xy,self.xz,self.xw,
                                                 self.yz,self.yw,
                                                 self.zw,)
    def __iter__(self):
        return iter([self.xy,
                     self.xz,
                     self.xw,
                     self.yz,
                     self.yw,
                     self.zw])
    def norm(self):
        return pow(self.xy**2+self.xz**2+self.xw**2+
                   self.yz**2+self.yw**2+
                   self.zw**2,
                   0.5)
def wedge(u,v):
    return Bivector(u[0]*v[1]-u[1]*v[0],
                    u[0]*v[2]-u[2]*v[0],
                    u[0]*v[3]-u[3]*v[0],
                    u[1]*v[2]-u[2]*v[1],
                    u[1]*v[3]-u[3]*v[1],
                    u[2]*v[3]-u[3]*v[2])

if __name__=='__main__':
    print('test __init__()')
    b=Bivector(1,2,3,4,5,6)
    print(b)

    print('test __iter__()')
    b=Bivector(2,4,6,8,10,12)
    print(*b) # unpacking it

    print('test norm()')
    b=Bivector(3,4,0,0,0,0)
    print(b.norm())

    print('test wedge()')
    u=[1,0,0,0]
    v=[0,1,0,0]
    print(wedge(u,v))
    print(wedge(v,u))
    u=[1,-4,9,-16]
    print(wedge(u,u))
