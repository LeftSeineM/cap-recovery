from pathlib import Path
import math
import numpy as np
from .finish_primitives import rotation,transform,pose
from .rightmost_target import arr
ROOT=Path(__file__).resolve().parent
def qdist(a,b):return float(2*np.degrees(np.arccos(np.clip(abs(np.dot(a/np.linalg.norm(a),b/np.linalg.norm(b))),0,1))))
def angle(q,axis):return float(np.degrees(np.arccos(np.clip((rotation(q)@axis)[2],-1,1))))
def count(a,b,spacing,deg=3):return max(1,math.ceil(np.linalg.norm(a[:3]-b[:3])/spacing),math.ceil(qdist(a[3:],b[3:])/deg))
def interp(a,b,n):
    qa=a[3:]/np.linalg.norm(a[3:]);qb=b[3:]/np.linalg.norm(b[3:]);dot=float(np.dot(qa,qb))
    if dot<0:qb=-qb;dot=-dot
    theta=math.acos(float(np.clip(dot,-1,1)))
    for i in range(1,n+1):
        t=i/n;q=(1-t)*qa+t*qb if theta<1e-8 else (np.sin((1-t)*theta)*qa+np.sin(t*theta)*qb)/np.sin(theta)
        yield np.r_[(1-t)*a[:3]+t*b[:3],q/np.linalg.norm(q)]

def template(target_pose):
    with np.load(ROOT/'validated_push_style.npz',allow_pickle=False) as z:
        source=arr(z['source_object_pose']);style=arr(z['link6_world'])
    source_inv=np.linalg.inv(transform(source));target_T=transform(target_pose)
    result=np.array([pose(target_T@source_inv@transform(v)) for v in style])
    forward=rotation(target_pose[3:])@rotation(source[3:]).T@np.array([0.,1.,0.])
    forward[2]=0;forward/=np.linalg.norm(forward)
    pre=result[0].copy();pre[:3]+=-forward*.012+np.array([0,0,.008])
    return result,pre,forward
