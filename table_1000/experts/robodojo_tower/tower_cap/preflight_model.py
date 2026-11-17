"""Case-independent CPU kinematics/convex preflight. No simulator state writes."""
import json,hashlib,copy,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R,Slerp
from scipy.optimize import least_squares,linprog
from scipy.spatial import ConvexHull
from .entry_state import transform

def pose(m):
 q=R.from_matrix(m[:3,:3]).as_quat();return np.r_[m[:3,3],q[3],q[:3]]
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def world(v,m):return np.asarray(v)@m[:3,:3].T+m[:3,3]
def overlap(v,w):
 if np.any(v.max(0)<w.min(0)) or np.any(w.max(0)<v.min(0)):return 0.
 eq=np.vstack([ConvexHull(v).equations,ConvexHull(w).equations]);r=linprog([0,0,0,-1],A_ub=np.c_[eq[:,:3],np.ones(len(eq))],b_ub=-eq[:,3],bounds=[(None,None)]*4,method='highs')
 if not r.success:raise RuntimeError('convex geometry LP failed')
 return max(0.,float(r.x[3]))
class PreflightFailure(RuntimeError):pass
class ModelContext:
 """Prepared read-only static geometry; actual state and its complete digest remain fresh."""
 def __init__(self,calibration,scene,table_height,freeze_data=False):
  self.source_calibration=calibration;self.source_scene=scene;self.table=table_height
  self.digest_prefix='{"calibration":'+json.dumps(calibration,sort_keys=True,separators=(',',':'))+',"scene":'+json.dumps(scene,sort_keys=True,separators=(',',':'))+',"state":'
  self.digest_suffix=',"table_height":'+json.dumps(table_height,separators=(',',':'))+'}'
  if freeze_data:
   from types import MappingProxyType
   def frozen(value):
    if isinstance(value,dict):return MappingProxyType({k:frozen(v) for k,v in value.items()})
    if isinstance(value,list):return tuple(frozen(v) for v in value)
    return value
   self.calibration=frozen(calibration);self.scene=frozen(scene)
  else:self.calibration=calibration;self.scene=scene
  self.geom=self.calibration['geometry'];self.joints=[ET.fromstring(calibration['urdf']).find("joint[@name='joint%d']"%i) for i in range(1,7)]
  self.arm_vertices={k:np.asarray(v)[ConvexHull(v).vertices] for k,v in self.geom['arm_vertices'].items()}
  for v in self.arm_vertices.values():v.setflags(write=False)
 def input_digest(self,state):
  return hashlib.sha256((self.digest_prefix+json.dumps(state,sort_keys=True,separators=(',',':'))+self.digest_suffix).encode()).hexdigest()

class Model:
 def __init__(self,state,calibration,scene,table_height,context=None):
  if context is None:context=ModelContext(calibration,scene,table_height)
  elif context.source_calibration is not calibration or context.source_scene is not scene or context.table!=table_height:raise RuntimeError('prepared static geometry identity mismatch')
  self.context=context;self.input=dict(state=copy.deepcopy(state),calibration=context.calibration,scene=context.scene,table_height=table_height);self.input_sha256=context.input_digest(state);self.scene=context.scene;self.table=table_height;self.geom=context.geom;self.joints=context.joints;self.arm_vertices=context.arm_vertices
  self.q={a:np.asarray(q,dtype=float) for a,q in state['qs'].items()};self.tools={a:transform(p) for a,p in state['tools'].items()};self.base={a:self.tools[a]@np.linalg.inv(self.fk(self.q[a])) for a in self.q};self.objects={k:transform(v['pose']) for k,v in state['objects'].items()};self.grips=state['grips'].copy();self.observed_grip_commands=state['grips'].copy();self.measured_gripper_joints=copy.deepcopy(state.get('actual_gripper_joints',{}));self.steps=[];self.checked_samples=0;self.held=None;self.refs={}
 def fork(self):
  # Frozen asset/scene/calibration data are read-only in every model operation.
  # Duplicate mutable state and path containers, retaining identical digest and geometry.
  other=copy.copy(self);other.input=dict(self.input,state=copy.deepcopy(self.input['state']))
  other.q={k:v.copy() for k,v in self.q.items()};other.tools={k:v.copy() for k,v in self.tools.items()};other.base={k:v.copy() for k,v in self.base.items()};other.objects={k:v.copy() for k,v in self.objects.items()};other.grips=self.grips.copy();other.observed_grip_commands=self.observed_grip_commands.copy();other.measured_gripper_joints=copy.deepcopy(self.measured_gripper_joints);other.refs={k:v.copy() for k,v in self.refs.items()};other.steps=self.steps.copy()
  return other
 def fk(self,q):
  m=np.eye(4)
  for j,v in zip(self.joints,q):
   o=j.find('origin');t=np.eye(4);t[:3,3]=np.fromstring(o.attrib['xyz'],sep=' ');t[:3,:3]=R.from_euler('xyz',np.fromstring(o.attrib['rpy'],sep=' ')).as_matrix();r=np.eye(4);r[:3,:3]=R.from_rotvec(np.fromstring(j.find('axis').attrib['xyz'],sep=' ')*v).as_matrix();m=m@t@r
  return m
 def parts(self,a,q,g):
  m=self.base[a].copy();p={}
  for i,(j,v) in enumerate(zip(self.joints,q),1):
   o=j.find('origin');t=np.eye(4);t[:3,3]=np.fromstring(o.attrib['xyz'],sep=' ');t[:3,:3]=R.from_euler('xyz',np.fromstring(o.attrib['rpy'],sep=' ')).as_matrix();r=np.eye(4);r[:3,:3]=R.from_rotvec(np.fromstring(j.find('axis').attrib['xyz'],sep=' ')*v).as_matrix();m=m@t@r;p['arm'+str(i)]=world(self.arm_vertices['link'+str(i)],m)
  for k,part in self.geom['gripper']['parts'].items():
   v=np.array(part['hull_vertices'])
   if k in ('link7','link8'):
    index=0 if k=='link7' else 1;sign=1 if index==0 else -1;nominal=-.01+.054*g
    measured=self.measured_gripper_joints.get(a)
    if measured is not None:
     if len(measured)!=2 or not np.isfinite(measured).all():raise RuntimeError('invalid measured two-finger joint geometry')
     actual=float(measured[index]);actual_v=v.copy();actual_v[:,1]+=sign*actual
     if abs(g-self.observed_grip_commands[a])<=1e-9:v=actual_v
     else:
      v[:,1]+=sign*nominal;v=np.vstack([actual_v,v]) # Conservative sweep envelope, not claimed predicted contact.
    else:v[:,1]+=sign*nominal # CPU hypothetical state without observation, explicitly not physical evidence.
   p[k]=world(v,m)
  return p
 def check(self,qq,grips,target=None,contact=False):
  parts={a:self.parts(a,qq[a],grips[a]) for a in qq};objs={k:v.copy() for k,v in self.objects.items()}
  if self.held:
   a=next(iter(self.refs));objs[self.held]=self.base[a]@self.fk(qq[a])@self.refs[a]
  verts={k:world(self.scene[k]['vertices'],p) for k,p in objs.items()};bad=[]
  margin=float(getattr(self,'clearance_margin',0.)) if self.stage=='empty_home' else 0.
  if margin:
   corners=np.array([[x,y,z] for x in (-margin,margin) for y in (-margin,margin) for z in (-margin,margin)])
   verts={k:(v[:,None,:]+corners[None,:,:]).reshape(-1,3) for k,v in verts.items()}
  for a,p in parts.items():
   for k,v in p.items():
    if v[:,2].min()<self.table-1e-5:bad.append(('table',a,k,float(self.table-v[:,2].min())))
    for name,w in verts.items():
     if name==target and k in ('link7','link8') and (contact or self.held==name):continue
     d=overlap(v,w)
     if d>1e-6:bad.append(('robot_object',a,k,name,d))
   for i in range(1,7):
    for j in range(i+2,7):
     d=overlap(p['arm'+str(i)],p['arm'+str(j)])
     if d>1e-6:bad.append(('self',a,i,j,d))
   for k in ('link6','link7','link8'):
    for i in range(1,5):
     d=overlap(p[k],p['arm'+str(i)])
     if d>1e-6:bad.append(('self_hand',a,k,i,d))
  aa=sorted(parts)
  for i,a in enumerate(aa):
   for b in aa[i+1:]:
    for k,v in parts[a].items():
     for k2,w in parts[b].items():
      d=overlap(v,w)
      if d>1e-6:bad.append(('interarm',a,k,b,k2,d))
  if self.held:
   v=verts[self.held]
   if v[:,2].min()<self.table-.0005:bad.append(('held_table',self.held,float(self.table-v[:,2].min())))
   for name,w in verts.items():
    if name==self.held:continue
    # During lateral transport keep real-contact clearance, not allowed penetration.
    if self.stage=='held_transport' or (self.stage=='support_lower' and name not in getattr(self,'support_ids',())):
     pad=.002;corners=np.array([[x,y,z] for x in (-pad,pad) for y in (-pad,pad) for z in (-pad,pad)]);w=(w[:,None,:]+corners[None,:,:]).reshape(-1,3)
    d=overlap(v,w)
    limit=1e-6 if self.stage in ('held_transport','physical_hold_lift') or (self.stage=='support_lower' and name not in getattr(self,'support_ids',())) else .0005
    if self.stage=='held_clearance_escape':
     # Escape must not deepen the actual initial contact; never use the old500um allowance.
     initial_v=world(self.scene[self.held]['vertices'],self.objects[self.held]);initial_w=world(self.scene[name]['vertices'],self.objects[name]);limit=overlap(initial_v,initial_w)+1e-6
    if d>limit:bad.append(('held_peer',self.held,name,d))
  self.checked_samples+=1
  if bad:raise PreflightFailure(json.dumps(dict(stage=self.stage,conflicts=bad)))
 def advance(self,desired,stage,target=None,contact=False,depth=0):
  self.stage=stage;picked={}
  for a,end in desired.items():
   tar=np.linalg.inv(self.base[a])@end
   def err(q):
    f=self.fk(q);return np.r_[5*(f[:3,3]-tar[:3,3]),R.from_matrix(f[:3,:3]@tar[:3,:3].T).as_rotvec()]
   rr=least_squares(err,self.q[a],bounds=([-10]*5+[-3.14],[10]*5+[3.14]),max_nfev=120,ftol=1e-9,xtol=1e-9,gtol=1e-9)
   if np.linalg.norm(err(rr.x))>.001 or max(abs(rr.x-self.q[a]))>.2:
    if depth>=7:raise PreflightFailure(stage+' reach/discontinuity')
    mid={}
    for arm,p in desired.items():
     m=self.tools[arm].copy();m[:3,3]=(m[:3,3]+p[:3,3])/2;m[:3,:3]=Slerp([0,1],R.from_matrix([self.tools[arm][:3,:3],p[:3,:3]]))(.5).as_matrix();mid[arm]=m
    self.advance(mid,stage,target,contact,depth+1);self.advance(desired,stage,target,contact,depth+1);return
   picked[a]=rr.x
  for u in (.5,1.):self.check({a:self.q[a]*(1-u)+picked.get(a,self.q[a])*u for a in self.q},self.grips,target,contact)
  for a,q in picked.items():self.q[a]=q;self.tools[a]=desired[a]
  self.steps.append(dict(kind='move',stage=stage,tools={a:pose(p).tolist() for a,p in desired.items()},q={a:q.tolist() for a,q in picked.items()}))
 def move(self,ends,stage,spacing=.012,degrees=8,target=None,contact=False):
  start={a:self.tools[a].copy() for a in ends};n=max(max(1,int(np.ceil(np.linalg.norm(end[:3,3]-start[a][:3,3])/spacing)),int(np.ceil(R.from_matrix(start[a][:3,:3].T@end[:3,:3]).magnitude()/np.deg2rad(degrees)))) for a,end in ends.items())
  for t in np.linspace(0,1,n+1)[1:]:
   desired={}
   for a,end in ends.items():
    p=np.eye(4);p[:3,3]=(1-t)*start[a][:3,3]+t*end[:3,3];p[:3,:3]=Slerp([0,1],R.from_matrix([start[a][:3,:3],end[:3,:3]]))(t).as_matrix();desired[a]=p
   self.advance(desired,stage,target,contact)
 def close(self,values,target,steps=10):
  start=self.grips.copy()
  for i in range(1,steps+1):
   for a,g in values.items():self.grips[a]=start[a]+(g-start[a])*i/steps
   self.stage='grip';self.check(self.q,self.grips,target,True);self.steps.append(dict(kind='grip',values=self.grips.copy(),stage='grip'))
 def certificate(self,request,source_sha256):return dict(input_sha256=self.input_sha256,request_sha256=digest(request),planner_source_sha256=source_sha256,path_sha256=digest(self.steps),path=self.steps,checked_samples=self.checked_samples,all_samples_clear=True,actions=len(self.steps),scope='CPU exact convex half/full joint samples and ideal tracking, no dynamics or force proof')
