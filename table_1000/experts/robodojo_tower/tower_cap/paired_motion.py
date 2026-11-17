"""Candidate synchronized S0; dynamics require native validation."""
from copy import deepcopy
import numpy as np
from .motion import Motion,OfficialSuccess
from .style import count,interp,qdist
from .rightmost_target import arr
from .multisolution import continuous_world_ik
from .legal_joint_representative import nearest_legal
from .finish_primitives import transform

class PairedMotion:
 def __init__(self,env,emit,last):
  self.env,self.emit=env,emit
  self.arms={a:Motion(env,emit,last,a) for a in ('left_arm','right_arm')}
  self.cmd=deepcopy(next(iter(self.arms.values())).cmd)
  self.guard=lambda:None
  self.rigid_target=None
  self.diagnose=getattr(emit,'diagnose',None)
 def tools(self):return {a:m.tool() for a,m in self.arms.items()}
 def send(self,stage):
  before=int(self.env.take_action_cnt[0])
  if self.env.end_flag[0] or before>=self.env.step_lim:raise RuntimeError('native terminal/budget')
  keep=self.emit(deepcopy(self.cmd),{'stage':stage,'paired_arms':list(self.arms)})
  if int(self.env.take_action_cnt[0])!=before+1:raise RuntimeError('paired emitter must advance exactly one action')
  self.guard()
  if self.env.end_flag[0]:
   if self.env.success[0]:raise OfficialSuccess()
   raise RuntimeError('native terminal failure')
  if not keep:raise RuntimeError('native emitter stopped')
 def _advance(self,goals,stage,depth=0,rigid=None):
  picked={};actual=self.tools();failure=None
  for a,m in self.arms.items():
   transform(goals[a])
   current=arr(m.rm.get_joint(m.arm,[0])[0]).reshape(-1)
   try:
    def log(row):
     if self.diagnose:self.diagnose(dict(stage=stage,native_step=int(self.env.take_action_cnt[0]),arm=a,event='paired_ik',candidate=row))
    raw=continuous_world_ik(m.rm,m.arm,goals[a],current,m.limits,log)
    q,info=nearest_legal(arr(raw).reshape(-1),current,*m.limits)
    if not np.isfinite(q).all() or np.max(np.abs(q-current))>.20:raise RuntimeError('joint discontinuity/nonfinite')
    picked[a]=q
   except Exception as e:failure=str(e);break
  if failure:
   if depth>=7:raise RuntimeError('paired IK bounded subdivision: '+failure)
   tiny=all(np.linalg.norm(goals[a][:3]-actual[a][:3])<=.00025 and qdist(actual[a][3:],goals[a][3:])<=.15 for a in self.arms)
   if tiny:raise RuntimeError('paired IK unreachable: '+failure)
   if rigid is None:
    mid={a:next(interp(actual[a],goals[a],2)) for a in self.arms};left_rigid=right_rigid=None
   else:
    target,read_object,refs=rigid
    anchor=arr(read_object()).reshape(7) if self.rigid_target is None else self.rigid_target.copy()
    middle=next(interp(anchor,target,2))
    mid={a:refs[a].link6_goal(middle) for a in self.arms}
    left_rigid=(middle,read_object,refs);right_rigid=rigid
   self._advance(mid,stage,depth+1,left_rigid)
   self._advance(goals,stage,depth+1,right_rigid);return
  for a,m in self.arms.items():self.cmd[m.arm_key]=picked[a].tolist()
  self.send(stage)
  if rigid is not None:self.rigid_target=arr(rigid[0]).reshape(7).copy()
  for a,p in self.tools().items():
   if np.linalg.norm(p[:3]-goals[a][:3])>.015 or qdist(p[3:],goals[a][3:])>15:raise RuntimeError('paired tracking error '+a)
 def move(self,ends,stage,spacing=.004,degrees=3):
  if set(ends)!=set(self.arms):raise ValueError('both tool targets required')
  starts=self.tools();n=max(count(starts[a],ends[a],spacing,degrees) for a in self.arms)
  sequences={a:list(interp(starts[a],ends[a],n)) for a in self.arms}
  for i in range(n):self._advance({a:sequences[a][i] for a in self.arms},stage)
 def move_rigid(self,end,read_object,refs,stage,spacing=.004,degrees=3):
  if set(refs)!=set(self.arms):raise ValueError('both immutable references required')
  start=arr(read_object()).reshape(7) if self.rigid_target is None else self.rigid_target.copy();end=arr(end).reshape(7)
  transform(start);transform(end)
  n=count(start,end,spacing,degrees)
  # Endpoint tool translation can include a large rotation lever arm.
  for a in self.arms:n=max(n,count(refs[a].link6_goal(start),refs[a].link6_goal(end),spacing,degrees))
  for target in interp(start,end,n):
   self._advance({a:refs[a].link6_goal(target) for a in self.arms},stage,rigid=(target,read_object,refs))
