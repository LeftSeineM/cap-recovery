"""Legal nearest 2pi representatives; pure CPU math plus read-only runtime limits."""
import math
import xml.etree.ElementTree as ET
import numpy as np


def nearest_legal(raw, current, types, urdf_limits, hard_limits, soft_limits):
    raw=np.asarray(raw,dtype=float);current=np.asarray(current,dtype=float)
    u=np.asarray(urdf_limits,dtype=float);h=np.asarray(hard_limits,dtype=float);s=np.asarray(soft_limits,dtype=float)
    if raw.shape!=current.shape or u.shape!=raw.shape+(2,) or h.shape!=u.shape or s.shape!=u.shape:
        raise ValueError("Joint/limit shape mismatch")
    if not np.isfinite(raw).all() or not np.isfinite(current).all() or any(np.isnan(x).any() for x in [u,h,s]):
        raise ValueError("Invalid joints or limits")
    lo=np.maximum(np.maximum(u[:,0],h[:,0]),s[:,0]);hi=np.minimum(np.minimum(u[:,1],h[:,1]),s[:,1])
    result=[];shifts=[]
    for i,(r,c,typ) in enumerate(zip(raw,current,types)):
        if lo[i]>hi[i]:raise ValueError(f"Empty legal interval at joint {i}")
        if typ in ("revolute","continuous"):
            center=int(round((c-r)/(2*math.pi)))
            kmin=math.ceil((lo[i]-r)/(2*math.pi)) if np.isfinite(lo[i]) else center-1
            kmax=math.floor((hi[i]-r)/(2*math.pi)) if np.isfinite(hi[i]) else center+1
            if kmin>kmax:raise ValueError(f"No legal equivalent at joint {i}")
            k=max(kmin,min(kmax,center));v=r+2*math.pi*k
        else:
            k=0;v=r
        if not lo[i]<=v<=hi[i]:raise ValueError(f"No legal value at joint {i}")
        result.append(v);shifts.append(k)
    return np.array(result),{"shift_turns":shifts,"legal_lower":lo.tolist(),"legal_upper":hi.tolist(),"raw_delta":(raw-current).tolist(),"normalized_delta":(np.array(result)-current).tolist()}


def runtime_limits(rm, robot, env_idx=0):
    def cpu(x):
        if hasattr(x,"detach"):x=x.detach().cpu().numpy()
        return np.asarray(x,dtype=float)
    key=rm.robot_key[rm.robot_list.index(robot)]
    indices=list(robot.arm_joint_indices)
    # Documented live IsaacLab fields; fail rather than synthesize unavailable limits.
    hard=cpu(key.data.joint_pos_limits)[env_idx,indices,:]
    soft=cpu(key.data.soft_joint_pos_limits)[env_idx,indices,:]
    names=[key.joint_names[i] for i in indices]
    tree=ET.parse(robot.urdf_path);joints={j.attrib["name"]:j for j in tree.getroot().findall("joint")}
    types=[];limits=[]
    for name in names:
        j=joints[name];typ=j.attrib["type"];types.append(typ)
        if typ=="continuous":limits.append([-math.inf,math.inf])
        else:
            lim=j.find("limit")
            if lim is None:raise ValueError("Missing URDF limit: "+name)
            limits.append([float(lim.attrib["lower"]),float(lim.attrib["upper"])])
    return names,types,np.array(limits),hard,soft
