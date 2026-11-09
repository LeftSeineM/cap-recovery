"""Empty receiver prepares during left pickup; no emitter and no hand-tile movement."""
from pathlib import Path
import json,math
import numpy as np
from .finish_primitives import rotation
from .legal_joint_representative import nearest_legal,runtime_limits
from .json_safety import plain
def arr(x):
 if hasattr(x,'detach'):x=x.detach().cpu().numpy()
 return np.asarray(x,dtype=float)
def qdist(a,b):
 a=arr(a)/np.linalg.norm(a);b=arr(b)/np.linalg.norm(b);return float(2*np.degrees(np.arccos(np.clip(abs(np.dot(a,b)),0,1))))
def count(a,b):return max(1,math.ceil(np.linalg.norm(a[:3]-b[:3])/.010),math.ceil(qdist(a[3:],b[3:])/5))
def poses(a,b,n):
 qa=a[3:]/np.linalg.norm(a[3:]);qb=b[3:]/np.linalg.norm(b[3:])
 if np.dot(qa,qb)<0:qb=-qb
 for i in range(1,n+1):
  t=i/n;q=(1-t)*qa+t*qb;q/=np.linalg.norm(q);yield np.r_[a[:3]*(1-t)+b[:3]*t,q]
class RightPrepare:
 def __init__(self,env,out,prior,available_left_actions):
  self.env=env;self.rm=env.robot_manager;self.r=self.rm.get_robot_by_arm_name('right_arm');self.out=Path(out)/'right_prepare';self.out.mkdir(exist_ok=False)
  with np.load(prior,allow_pickle=False) as z:self.goal=z['entry_right'].copy()
  self.status='running';self.n=0;self.pending=None;self.final_q=None
  self.limits=runtime_limits(self.rm,self.r,0);now=self.tool();high=self.goal.copy();high[2]=max(now[2]+.03,self.goal[2]+.08,1.08)
  a=list(poses(now,high,count(now,high)));b=list(poses(high,self.goal,count(high,self.goal)));self.queue=a+b
  if len(self.queue)>available_left_actions or len(self.queue)>96:raise ValueError('parallel receiver prepare cannot fit left pickup')
  self.dump('plan.json',{'goal_source':'same geometry validated handover entry414','entry':now,'high':high,'goal':self.goal,'actions':len(self.queue),'available_left_actions':available_left_actions,'no_tile_moves':True,'unverified_full_robot_clearance':True})
 def tool(self):return arr(self.rm.get_real_endpose(self.r,[0],is_relative=False)[0]).reshape(7)
 def dump(self,n,x):(self.out/n).write_text(json.dumps(plain(x),indent=2),encoding='utf-8')
 def before_action(self,left):
  if self.pending is not None:raise ValueError('previous right prepare action not observed')
  merged={k:list(v) for k,v in left.items()}
  if self.status=='complete':q=self.final_q;desired=self.goal
  else:
   desired=self.queue.pop(0);ik=self.rm.solve_ik(target_pose=desired.tolist(),env_idx=0,robot=self.r,trans='world')
   if ik.get('status')!='Success':raise ValueError('right_prepare_ik_failed')
   _,types,u,h,s=self.limits;current=arr(self.rm.get_joint(self.r,[0])[0]).reshape(-1);q,meta=nearest_legal(arr(ik['joint_value']).reshape(-1),current,types,u,h,s)
   if np.max(np.abs(q-current))>.20:raise ValueError('right_prepare_delta_over.20')
  merged[self.rm.process_name(self.r.arm_name)]=q.tolist();merged[self.rm.process_name(self.r.gripper_name)]=[1.]
  self.pending=(int(self.env.take_action_cnt[0]),desired,q);return merged
 def after_action(self):
  if self.pending is None:raise ValueError('no pending right prepare action')
  before,desired,q=self.pending;self.pending=None
  if int(self.env.take_action_cnt[0])!=before+1:raise ValueError('single emitter did not advance once')
  got=self.tool()
  if np.linalg.norm(got[:3]-desired[:3])>.015 or qdist(got[3:],desired[3:])>12:raise ValueError('right prepare tracking failed')
  if self.status!='complete':
   self.n+=1;self.dump(f'state_{self.n:03d}.json',{'native_step':int(self.env.take_action_cnt[0]),'desired':desired,'actual':got})
   if not self.queue:self.final_q=q.copy();self.status='complete'
 def require_ready(self):
  if self.status!='complete':raise ValueError('right not prepared before handover')
  got=self.tool()
  if np.linalg.norm(got[:3]-self.goal[:3])>.005 or qdist(got[3:],self.goal[3:])>3:raise ValueError('receiver original414 entry mismatch')
