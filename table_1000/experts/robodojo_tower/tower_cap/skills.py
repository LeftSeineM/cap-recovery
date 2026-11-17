"""Runtime boundary: S1 plans, existing native S0 emitter, measured postconditions.
This module is a candidate until same-SHA physical regression completes.
"""
import json,copy
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from .preflight_model import Model,ModelContext,transform,pose,digest
from .planner import plan_carry,plan_acquire,plan_clear,plan_symmetric_transport,shape_profile
from .skill_contract import SkillCall,SkillInapplicable,source_hash
from .measured_geometry import measured_geometry_references,observed_geometry_references,reference_digest
from .support_release import release_on_support as release_impl,stability_report

class Runtime:
 def __init__(self,env,emit,last,out,scene,calibration,table_height,read_object,home,template):
  from tower_cap.paired_motion import PairedMotion
  self.env=env;self.native_emit=emit;self.motion=PairedMotion(env,self.checked_emit,last);self.out=Path(out);self.scene=scene;self.calibration=calibration;self.table=table_height;self.read_object=read_object;self.home=home;self.template=template;self.references={};self.held=None;self.active_arms=[];self.sha,self.source_files=source_hash(Path(__file__).parent)
  self.context=ModelContext(calibration,scene,table_height,freeze_data=True)
  self.motion.guard=self.guard;self.motion.diagnose=getattr(emit,'diagnose',None)
  for m in self.motion.arms.values():m.diagnose=getattr(emit,'diagnose',None)
 def checked_emit(self,action,metadata=None):
  # S0 solves from actual joints, including its original bounded subdivision. Every resulting native command gets a fresh geometry check.
  s=self.snapshot();live=Model(s,self.calibration,self.scene,self.table,context=self.context);live.clearance_margin=float(getattr(self,'action_clearance_margin',0.));qq={};gg={}
  baseline_hash=reference_digest(self.references)
  if self.held is not None:
   live.held=self.held;live.support_ids=tuple(getattr(self,'protected_during_carry',{}))
   if not getattr(self,'release_phase',False):
    self.guard() # Capture baseline and its original limits stay authoritative during holding.
    live.refs=measured_geometry_references(s,self.held,self.references)
    self.record(stage='measured_held_collision_geometry',guard_reference_sha256=baseline_hash,geometry_reference_sha256=reference_digest(live.refs),capture_reference_updated=False)
   else:
    # Opening ends hold-integrity enforcement in the existing release procedure.
    # Its supported/stable postconditions remain unchanged; predict actual geometry.
    live.refs=observed_geometry_references(s,self.held,self.references)
  for a,m in self.motion.arms.items():
   qq[a]=np.asarray(action[m.arm_key],dtype=float);gg[a]=float(action[m.grip_key][0])
   if not np.isfinite(qq[a]).all() or max(abs(qq[a]-live.q[a]))>.20:raise RuntimeError('actual joint delta exceeds existing S0 bound')
  live.stage=metadata.get('stage','native_action') if metadata else 'native_action';target=getattr(self,'contact_target',None);contact=getattr(self,'contact_allowed',False)
  for t in (.5,1.):live.check({a:live.q[a]*(1-t)+qq[a]*t for a in qq},{a:live.grips[a]*(1-t)+gg[a]*t for a in gg},self.held or target,bool(self.held is not None or contact))
  result=self.native_emit(action,metadata)
  if reference_digest(self.references)!=baseline_hash:raise RuntimeError('capture reference changed during a native command')
  return result
 def snapshot(self):
  from tower_cap.rightmost_target import arr
  return dict(qs={a:arr(m.rm.get_joint(m.arm,[0])[0]).reshape(-1).tolist() for a,m in self.motion.arms.items()},tools={a:p.tolist() for a,p in self.motion.tools().items()},objects={k:dict(pose=np.asarray(self.read_object(k)).tolist()) for k in self.scene},grips={a:float(self.motion.cmd[m.grip_key][0]) for a,m in self.motion.arms.items()},actual_gripper_joints={a:arr(m.rm.get_end_effector_real_val(m.arm,[0])[0]).reshape(-1).tolist() for a,m in self.motion.arms.items()})
 def record(self,**data):
  with (self.out/'s1_runtime.jsonl').open('a') as f:f.write(json.dumps(dict(native_step=int(self.env.take_action_cnt[0]),**data),default=lambda x:x.tolist())+'\n')
 def guard(self):
  for k,p in getattr(self,'protected_during_carry',{}).items():
   current=np.asarray(self.read_object(k));dp=float(np.linalg.norm(current[:3]-p[:3]));dq=float(np.rad2deg(R.from_matrix(transform(current)[:3,:3]@transform(p)[:3,:3].T).magnitude()))
   self.record(stage='protected_support_during_carry',object=k,position_change_m=dp,angle_change_deg=dq)
   if dp>.002 or dq>2.:raise RuntimeError('carry disturbed supporting structure; no release')
  if self.held is None:return
  obj=transform(self.read_object(self.held));tools=self.motion.tools();reports={}
  for a,h in self.references.items():
   actual=np.linalg.inv(transform(tools[a]))@obj;dp=float(np.linalg.norm(actual[:3,3]-h[:3,3]));dq=float(np.rad2deg(R.from_matrix(actual[:3,:3]@h[:3,:3].T).magnitude()));reports[a]=dict(position_drift_m=dp,angle_drift_deg=dq)
  self.record(stage='immutable_hold',object=self.held,metrics=reports)
  if any(v['position_drift_m']>.012 or v['angle_drift_deg']>12 for v in reports.values()):raise RuntimeError('immutable held reference drift; reference not recaptured')
 def adopt_entry(self,history):
  from .entry_state import classify_entry
  if len(history)<2:raise SkillInapplicable('relative-motion entry needs at least two real prefix states')
  current=self.snapshot()
  for k in self.scene:
   if np.linalg.norm(np.asarray(history[-1]['objects'][k]['pose'])-np.asarray(current['objects'][k]['pose']))>1e-6:raise SkillInapplicable('prefix history does not end at actual current state')
  result=classify_entry(history,current['grips'],self.scene,self.calibration['geometry']['gripper'])
  self.record(stage='actual_entry_classified',report=result)
  if result['mode']=='empty':return result
  if result['mode'] not in ('single_held_candidate','paired_held_candidate'):raise SkillInapplicable('ambiguous or unsupported holding entry')
  for object_id,arms in result['arms_by_object'].items():self.capture(object_id,arms)
  return result
 def capture(self,object_id,arms):
  if self.held is not None:raise RuntimeError('existing held reference cannot be silently recaptured')
  obj=transform(self.read_object(object_id));tools=self.motion.tools();self.held=object_id;self.active_arms=list(arms);self.references={a:np.linalg.inv(transform(tools[a]))@obj for a in arms};self.record(stage='capture_once_before_lift',object=object_id,references=self.references)
 def execute(self,cert,call):
  from tower_cap.legal_joint_representative import nearest_legal
  from tower_cap.style import qdist
  if cert['planner_source_sha256']!=self.sha or cert['path_sha256']!=digest(cert['path']):raise RuntimeError('preflight source/path identity mismatch')
  self.action_clearance_margin=float(cert.get('collision_clearance_m',0.))
  actual=self.snapshot();model=Model(actual,self.calibration,self.scene,self.table,context=self.context)
  if model.input_sha256!=cert['input_sha256']:raise RuntimeError('actual state changed since preflight; regenerate before actions')
  if len(cert['path'])>int(self.env.step_lim)-int(self.env.take_action_cnt[0]):raise SkillInapplicable('checked path exceeds remaining native budget')
  evidence=self.out/('preflight_'+call.row['call_id']+'_'+str(int(self.env.take_action_cnt[0]))+'.json');evidence.write_text(json.dumps(cert));call.row['evidence'].append(str(evidence))
  for step in cert['path']:
   call.row['stage']=step['stage'];self.guard()
   # checked_emit retains a fresh actual-state Model before every native action.
   self.contact_target=cert.get('object');self.contact_allowed=bool(self.held is not None or step['stage'] in ('contact','grip'))
   if step['kind']=='move':
    # Preflight certifies the intended tool path; real S0 must re-solve each endpoint from actual joints and subdivide when necessary.
    goals=self.motion.tools()
    for a,p in step['tools'].items():goals[a]=np.asarray(p,dtype=float)
    if self.held is not None and len(self.references)==2:
     from types import SimpleNamespace
     control_refs=self.references
     if cert.get('control_geometry_references') is not None:
      control_refs={a:np.asarray(h) for a,h in cert['control_geometry_references'].items()}
      if set(control_refs)!=set(self.references) or cert.get('guard_reference_sha256')!=reference_digest(self.references):raise RuntimeError('paired control/guard reference identity mismatch')
      measured=measured_geometry_references(actual,self.held,self.references)
      from .pair_control import control_references
      expected=control_references(measured,self.references)
      if cert.get('control_mode')!='original_hand_pair_anchor_feedback' or reference_digest(measured)!=cert.get('measured_geometry_reference_sha256') or reference_digest(control_refs)!=cert.get('control_geometry_reference_sha256') or any(not np.allclose(control_refs[a],expected[a],rtol=0.,atol=1e-9) for a in expected):raise RuntimeError('paired control references violate original hand-pair constraint')
     refs={a:SimpleNamespace(link6_goal=lambda p,h=h:pose(transform(p)@np.linalg.inv(h))) for a,h in control_refs.items()}
     anchor=next(iter(control_refs));object_goal=pose(transform(goals[anchor])@control_refs[anchor]);self.motion.rigid_target=np.asarray(self.read_object(self.held)).copy();self.motion._advance(goals,step['stage'],rigid=(object_goal,lambda:np.asarray(self.read_object(self.held)),refs))
    else:self.motion._advance(goals,step['stage'])
   else:
    for a,g in step['values'].items():self.motion.cmd[self.motion.arms[a].grip_key]=[float(g)]
    self.motion.send(step['stage'])
   self.record(stage='executed_preflight_step',planned_stage=step['stage'],actual=self.snapshot())
 def call(self,name,params,operation):
  call=SkillCall(self.out,name,params,self.env);call.observe=self.snapshot
  try:
   from threadpoolctl import threadpool_limits
   with threadpool_limits(limits=1,user_api='blas'):return call.invoke(lambda:operation(call))
  except Exception:
   self.record(stage='skill_exception',skill=name,actual=self.snapshot());raise

def carry_held_object(rt,object_id,arms,goal,carry_options=None):
 params=dict(object=object_id,arms=list(arms),goal=list(goal),**(carry_options or {}))
 def run(call):
  if rt.held is None:raise SkillInapplicable('inherited entry must first pass actual-history classification; no blind reference capture')
  if rt.held!=object_id or set(rt.references)!=set(arms):raise SkillInapplicable('requested object/hands differ from immutable held entry')
  rt.protected_during_carry={k:np.asarray(rt.read_object(k)).copy() for k in params.get('protect',[])}
  request=dict(params,immutable_references={a:h.tolist() for a,h in rt.references.items()});state=rt.snapshot();initial=float(state['objects'][object_id]['pose'][2])
  if params.get('feedback_transport_single') or params.get('observed_object_feedback'):
   from .feedback_transport import plan_observed_transport
   cert=plan_observed_transport(state,rt.calibration,rt.scene,rt.table,request,rt.sha,rt.references,True,rt.context)['certificate']
  else:cert=plan_carry(state,rt.calibration,rt.scene,rt.table,request,rt.sha)
  # Split at real lift boundary to verify actual object rise before any transport.
  split=max(i for i,s in enumerate(cert['path']) if s['stage']=='physical_hold_lift')+1
  first=copy.deepcopy(cert);first['path']=first['path'][:split];first['path_sha256']=digest(first['path']);rt.execute(first,call)
  rise=float(rt.read_object(object_id)[2]-initial)
  if rise<.018:raise RuntimeError('physical held-object rise below existing18mm condition')
  # Actual state after lift is new evidence. Fresh planning, retaining original immutable references.
  state=rt.snapshot()
  if params.get('feedback_transport_single') or params.get('observed_object_feedback'):
   from .feedback_transport import execute_observed_transport
   feedback=execute_observed_transport(rt,request,call);evaluated_goal=feedback['selected_goal'];call.row['feedback']=feedback
  elif params.get('goal_yaw_symmetries'):
   rest=plan_symmetric_transport(state,rt.calibration,rt.scene,rt.table,request,rt.sha,False);rt.execute(rest,call);evaluated_goal=rest['selected_goal']
  else:
   state=rt.snapshot();m=Model(state,rt.calibration,rt.scene,rt.table,context=rt.context);m.held=object_id;m.support_ids=tuple(params.get('protect',[]));m.refs={a:h.copy() for a,h in rt.references.items()}
   from .planner import rigid_move
   high=m.objects[object_id].copy();high[2,3]=max(high[2,3],float(goal[2])+.06);rigid_move(m,high,'held_vertical_clearance');above=transform(goal);above[2,3]=high[2,3];rigid_move(m,above,'held_transport');rigid_move(m,transform(goal),'support_lower',.0015 if len(arms)==2 else .004,3 if len(arms)==2 else 4)
   rest=m.certificate(request,rt.sha);rest.update(object=object_id,arms=list(arms));rt.execute(rest,call)
   evaluated_goal=goal
  actual=np.asarray(rt.read_object(object_id));position_error=float(np.linalg.norm(actual[:3]-np.asarray(evaluated_goal)[:3]));angle_error=float(np.rad2deg(R.from_matrix(transform(actual)[:3,:3]@transform(evaluated_goal)[:3,:3].T).magnitude()))
  if position_error>.015 or angle_error>15:raise RuntimeError('held destination tracking outside S0 limits')
  rt.guard();rt.protected_during_carry={}
  return dict(actual_state=rt.snapshot(),physical_lift_m=rise,position_error_m=position_error,angle_error_deg=angle_error)
 return rt.call('carry_held_object',params,run)

def release_on_support(rt,object_id,support_ids,required_count=1,allow_table=False):
 params=dict(object=object_id,supports=list(support_ids),required_count=required_count,allow_table=allow_table)
 def run(call):
  if rt.held!=object_id:raise SkillInapplicable('release requires currently verified held object')
  arms={a:rt.motion.arms[a] for a in rt.active_arms}
  class ReleaseMotion:
   def __init__(self):self.arms=arms;self.cmd=rt.motion.cmd
   @property
   def guard(self):return rt.motion.guard
   @guard.setter
   def guard(self,value):rt.motion.guard=value
   def send(self,stage):
    call.row['stage']=stage;rt.contact_target=object_id;rt.contact_allowed=True;rt.motion.send(stage)
  closed={a:float(rt.motion.cmd[m.grip_key][0]) for a,m in arms.items()}
  rt.release_phase=True
  try:
   report=release_impl(ReleaseMotion(),lambda:np.asarray(rt.read_object(object_id)),rt.scene[object_id]['vertices'],lambda:[dict(id=k,pose=rt.read_object(k),vertices=rt.scene[k]['vertices']) for k in support_ids],closed,rt.table if allow_table else None,required_count,record=rt.record)
  finally:rt.release_phase=False
  rt.held=None;rt.references={};rt.motion.guard=rt.guard
  if not report['postconditions_met']:raise RuntimeError('shared support/release stability postconditions failed')
  return dict(actual_state=rt.snapshot(),release=report)
 return rt.call('release_on_support',params,run)

def withdraw_after_release(rt,arms,protect=(),return_home=True):
 params=dict(arms=list(arms),home=rt.home,protect=list(protect),return_home=bool(return_home))
 def run(call):
  if rt.held is not None:raise SkillInapplicable('cannot park with active held reference')
  before={k:np.asarray(rt.read_object(k)).copy() for k in protect}
  # First separate fingers vertically using the existing checked staging route.
  # Home rotation is replanned from the resulting measured physical state.
  stages=[dict(params,return_home=False),params] if return_home else [params]
  for stage_request in stages:
   cert=plan_clear(rt.snapshot(),rt.calibration,rt.scene,rt.table,stage_request,rt.sha);rt.execute(cert,call)
   for k,p in before.items():
    actual=np.asarray(rt.read_object(k));dp=float(np.linalg.norm(actual[:3]-p[:3]));dq=float(np.rad2deg(R.from_matrix(transform(actual)[:3,:3]@transform(p)[:3,:3].T).magnitude()))
    if dp>.002 or dq>2.:raise RuntimeError('withdrawal disturbed protected structure')
  stability={}
  for k,p in before.items():
   actual=np.asarray(rt.read_object(k));dp=float(np.linalg.norm(actual[:3]-p[:3]));dq=float(np.rad2deg(R.from_matrix(transform(actual)[:3,:3]@transform(p)[:3,:3].T).magnitude()));stability[k]=dict(stable=dp<=.002 and dq<=2.,position_change_m=dp,angle_change_deg=dq,scope='before/after withdrawal drift, not three sampled observations')
  if any(not x['stable'] for x in stability.values()):raise RuntimeError('withdrawal disturbed protected structure')
  return dict(actual_state=rt.snapshot(),protected_structure=stability)
 return rt.call('clear_and_park' if return_home else 'withdraw_to_staging',params,run)

def clear_and_park(rt,arms,protect=()):return withdraw_after_release(rt,arms,protect,True)

def _pick_place(rt,object_id,arms,goal,support_ids,required_count,allow_table,name,goal_yaw_symmetries=None,return_home=True,task_upright_yaw_free=False):
 params=dict(object=object_id,arms=list(arms),goal=list(goal),supports=list(support_ids),required_count=required_count,allow_table=allow_table,return_home=bool(return_home),home=rt.home)
 if len(arms)==1:
  prism=shape_profile(rt.scene[object_id]['vertices'])=='triangular_prism'
  params.update(feedback_transport_single=True,feedback_translation_step_m=.006 if prism else .012,feedback_rotation_step_deg=2. if prism else 4.)
 else:params.update(observed_object_feedback=True,feedback_translation_step_m=.006,feedback_rotation_step_deg=2.)
 if task_upright_yaw_free:params['task_upright_yaw_free']=True
 if goal_yaw_symmetries:
  params['goal_yaw_symmetries']=list(goal_yaw_symmetries)
  if shape_profile(rt.scene[object_id]['vertices'])=='triangular_prism':
   params['grasp_family']='parallel_prism_end_faces';params['feedback_transport_single']=True
 def require(row):
  if row['status']=='inapplicable':raise SkillInapplicable('child skill '+row['skill']+': '+str(row.get('reason')))
  if row['status']!='completed':raise RuntimeError('child skill '+row['skill']+': '+str(row.get('reason')))
 def run(call):
  if rt.held is None:
   cert=plan_acquire(rt.snapshot(),rt.calibration,rt.scene,rt.table,params,rt.sha,rt.template);rt.execute(cert,call);rt.capture(object_id,arms)
  require(carry_held_object(rt,object_id,arms,goal,dict(protect=list(support_ids),**{k:params[k] for k in ('goal_yaw_symmetries','carry_clearances_m','task_upright_yaw_free','home','feedback_transport_single','observed_object_feedback','feedback_translation_step_m','feedback_rotation_step_deg') if k in params})));require(release_on_support(rt,object_id,support_ids,required_count,allow_table));require(withdraw_after_release(rt,list(rt.motion.arms) if return_home else arms,protect=[object_id]+list(support_ids),return_home=return_home))
  return rt.snapshot()
 return rt.call(name,params,run)

def pick_place_block(rt,object_id,arm,goal,support_ids=(),allow_table=False,goal_yaw_symmetries=None,return_home=True,task_upright_yaw_free=False):return _pick_place(rt,object_id,[arm],goal,support_ids,1,allow_table,'pick_place_block',goal_yaw_symmetries,return_home,task_upright_yaw_free)
def place_board_bimanual(rt,object_id,arms,goal,support_ids,return_home=True):
 if len(arms)!=2 or len(set(arms))!=2 or len(support_ids)!=2:raise SkillInapplicable('two distinct grippers and support objects required')
 return _pick_place(rt,object_id,arms,goal,support_ids,2,False,'place_board_bimanual',goal_yaw_symmetries=[0.,180.],return_home=return_home)
