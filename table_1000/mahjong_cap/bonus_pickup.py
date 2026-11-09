"""Finite post-three-tiles left bonus pickup. Explicit runner invocation only.
No take_action, simulator startup, reset, state writes, reward calls or resources.
emit_action exclusively owns physics, native reward, recorder and action budget.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np
from .json_safety import plain
from .right_prepare import RightPrepare
from .grip_continuity import read_same_job_hold_grips
from .legal_joint_representative import nearest_legal,runtime_limits

ROOT=Path(__file__).parent
from .rightmost_target import HAND_LABELS
ALL=list(HAND_LABELS)
BONUS='mahjong9_0'
TRANSFER_RESERVE=80  # planning/reporting allowance; not an extra native entry barrier
REFERENCE=np.array([-.39969244599342346,-.050051696598529816,.8145018815994263,
 -5.77009768676362e-6,.7078030705451965,.7064098119735718,2.9887480195611715e-5])

def arr(v):
 if hasattr(v,'detach'):v=v.detach().cpu().numpy()
 return np.asarray(v,dtype=float)
def rotation(q):
 q=arr(q).reshape(4);q=q/np.linalg.norm(q);w,x,y,z=q
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
 [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
 [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def axis_angle(q,axis):
 return float(np.degrees(np.arccos(np.clip((rotation(q)@np.asarray(axis))[2],-1,1))))
def qdistance(a,b):
 a=arr(a)/np.linalg.norm(a);b=arr(b)/np.linalg.norm(b)
 return float(2*np.degrees(np.arccos(np.clip(abs(np.dot(a,b)),0,1))))
def matrix(p):
 p=arr(p).reshape(7);T=np.eye(4);T[:3,:3]=rotation(p[3:]);T[:3,3]=p[:3];return T
def samples(a,b,translation_step=.004):
 return max(1,math.ceil(np.linalg.norm(arr(a)[:3]-arr(b)[:3])/translation_step),math.ceil(qdistance(arr(a)[3:],arr(b)[3:])/3))
def interpolate(a,b,n):
 a=arr(a);b=arr(b);qa=a[3:]/np.linalg.norm(a[3:]);qb=b[3:]/np.linalg.norm(b[3:])
 if np.dot(qa,qb)<0:qb=-qb
 for i in range(1,n+1):
  t=i/n;q=(1-t)*qa+t*qb;q/=np.linalg.norm(q)
  yield np.r_[a[:3]*(1-t)+b[:3]*t,q]

def run_bonus(env,emit_action,out):
 TARGETS=[str(row[0]) for row in env.kong]
 root_out=Path(out)
 out=root_out/'mahjong_bonus_pickup';out.mkdir(exist_ok=False)
 e=0;rm=env.robot_manager;lm=env.scene_manager.layout_manager
 arms=[r for r in rm.robot_list if r.type=='target']
 left=rm.get_robot_by_arm_name('left_arm');right=rm.get_robot_by_arm_name('right_arm')
 others=[x for x in ALL if x not in TARGETS];count=0;sequence=0;reference=None;certified=False
 begin=int(env.take_action_cnt[e]);stop=None
 def object_state(obj):
  s=obj.get_current_dynamic_state();p=arr(s.position).reshape(3);q=arr(s.orientation).reshape(4)
  return {'position':p.tolist(),'quat_wxyz':q.tolist(),'linear_velocity':arr(s.linear_velocity if s.linear_velocity is not None else [0,0,0]).tolist(),'angular_velocity':arr(s.angular_velocity if s.angular_velocity is not None else [0,0,0]).tolist(),'Yup_deg':axis_angle(q,[0,1,0]),'Zup_deg':axis_angle(q,[0,0,1])}
 def obj(label):return lm.get_scene_object(env_idx=e,inst_name=lm.get_instance_name(env_idx=e,label=label))
 def tool():return arr(rm.get_real_endpose(left,[e],is_relative=False)[e]).reshape(7)
 def snapshot(stage,extra=None):
  nonlocal sequence
  objects={label:object_state(obj(label)) for label in ALL+[BONUS]}
  # Includes every lower stack tile even when no mahjong label is assigned.
  rigid={str(inst):object_state(o) for inst,o in env.scene_manager._rigid_and_dynamic_objects[e].items() if hasattr(o,'get_current_dynamic_state')}
  rec={'stage':stage,'native_steps':int(env.take_action_cnt[e]),'sent':count,'objects':objects,'all_rigid_objects':rigid,
   'robots':{r.arm_name:{'link6_world':arr(rm.get_real_endpose(r,[e],is_relative=False)[e]).tolist(),'joints':arr(rm.get_joint(r,[e])[e]).tolist(),'gripper_actual':arr(rm.get_end_effector_real_val(r,[e])[e]).tolist()} for r in arms},
   'reward_remaining':repr(env.reward_manager.check_list[e]),'extra':extra}
  B=objects[BONUS];T=np.linalg.inv(matrix(tool()))@matrix(B['position']+B['quat_wxyz'])
  rec['bonus_in_left_link6']=T.tolist()
  if reference is not None:
   D=np.linalg.inv(reference)@T;rec['fixed_grasp_error']={'position_m':float(np.linalg.norm(D[:3,3])),'rotation_deg':float(np.degrees(np.arccos(np.clip((np.trace(D[:3,:3])-1)/2,-1,1))))}
  (out/f'state_{sequence:04d}.json').write_text(json.dumps(plain(rec),indent=2),encoding='utf8');sequence+=1;return rec
 def conditions(rec):
  return all(rec['objects'][x]['Zup_deg']<30 for x in TARGETS) and all(rec['objects'][x]['Yup_deg']<7 for x in others)
 def result(status,extra=None):
  r={'status':status,'counts_as_success':False,'native_begin':begin,'native_end':int(env.take_action_cnt[e]),'bonus_actions':count,'remaining_native_budget':600-int(env.take_action_cnt[e]),'advisory_transfer_final_steps':TRANSFER_RESERVE,'advisory80_remaining':600-int(env.take_action_cnt[e])>=TRANSFER_RESERVE,'fixed_reference_created':reference is not None,'pickup_reference_certified_by_lift':certified,'next':'no_handoff_or_final_placement_in_this_block','extra':extra}
  (out/'result.json').write_text(json.dumps(plain(r),indent=2),encoding='utf8');return r
 try:
  identity=json.loads((root_out/'identity.json').read_text(encoding='utf-8'))
  if not identity.get('case','').startswith('make_kong__'):return result('same_job_case_mismatch')
  continuity=read_same_job_hold_grips(env,root_out,identity['job'])
  baseline=snapshot('entry')
  instance=str(lm.get_instance_name(env_idx=e,label=BONUS))
  if not instance.startswith('mahjong_15_'):return result('actual_bonus_category_is_not_bound15')
  shape=json.loads((ROOT/'enabled_collision_bounds.json').read_text(encoding='utf-8'))['15']
  if shape['sha256']!='677c2b6123e5197f59ed7d2e8b4f41b814626d1560a63d72e78debf74f93bd57':return result('actual15_bound_asset_hash_mismatch')
  (out/'bonus_actual_geometry.json').write_text(json.dumps(plain({'instance':instance,'bounds':shape})),encoding='utf-8')
  if env.num_envs!=1 or set(str(row[e]) for row in env.kong)!=set(TARGETS):
   return result('case_identity_mismatch')
  if int(env.step_lim)!=600:return result('unexpected_native_step_limit')
  if not conditions(baseline):return result('three_target_or_nine_neighbor_condition_failed')
  pending=env.reward_manager.check_list[e]
  if len(pending)!=1 or 'is_qpos_close' in repr(pending) or 'is_A_xy_distance_close_to_pos' not in repr(pending) or 'is_all_gripper_open' not in repr(pending):
   return result('native_first_reward_stage_not_verified_consumed')
  b=baseline['objects'][BONUS];bp=arr(b['position']+b['quat_wxyz'])
  mismatch={'position_m':float(np.linalg.norm(bp[:3]-REFERENCE[:3])),'quaternion_deg':qdistance(bp[3:],REFERENCE[3:])}
  if mismatch['position_m']>.005 or mismatch['quaternion_deg']>3:return result('bonus_initial_pose_differs_from_prior',mismatch)
  spec={'prior_sha256': '3025c50b31c0b5e92612d5abd1eabccbf9ae53fc0b5507dbc3f614a19d2fbc60'};prior=ROOT/'grasp_prior_380_415.npz'
  if hashlib.sha256(prior.read_bytes()).hexdigest()!=spec['prior_sha256']:return result('prior_hash_mismatch')
  with np.load(prior,allow_pickle=False) as z:
   entry=z['entry_left_link6'].copy();poses=z['left_link6'].copy();grips=z['left_grip'][:,0].copy();rows=z['original_row'].copy()
  names,types,ul,hl,sl=runtime_limits(rm,left,e)
  # Freeze idle right target ONCE; do not replay its original trajectory or drift with it.
  right_q=arr(rm.get_joint(right,[e])[e]).copy()
  def fraction(r):
   v=float(np.mean(arr(rm.get_end_effector_real_val(r,[e])[e])));lo,hi=map(float,r.gripper_scale)
   return float(np.clip((v-lo)/(hi-lo) if r.gripper_move['sign']==1 else (hi-v)/(hi-lo),0,1))
  right_grip=continuity['right_hold_command'];left_grip=continuity['left_hold_command']
  current=tool();high=max(float(current[2])+.04,1.055)
  up=current.copy();up[2]=high
  back=up.copy();back[1]=min(float(current[1])-.06,-.32)
  # Elevated diagonal keeps all lateral motion at the chosen high Z.
  # Rotation occurs continuously during the diagonal, never at low row height.
  above=entry.copy();above[2]=high
  route=[('clear_up',up,.010),('high_diagonal_to_above_original_entry',above,.010),('descend_to_original_row379_entry',entry,.004)]
  estimate=math.ceil(abs(1-left_grip)/.1);prev=current;costs=[]
  for name,p,spacing in route:
   n=samples(prev,p,spacing);estimate+=n;costs.append({'phase':name,'translation_step_m':spacing,'actions':n});prev=p
  # One IK command per recorded pose preserves candidate original pose timing.
  # Illegal/discontinuous direct solutions subdivide; this is a lower bound.
  estimate+=len(poses);remaining=600-begin
  (out/'plan.json').write_text(json.dumps(plain({'route':[(n,p.tolist(),spacing) for n,p,spacing in route],'phase_costs':costs,'estimated_native_actions_before_possible_prior_subdivision':estimate,'estimate_is_lower_bound_if_prior_subdivision_needed':True,'remaining_original_budget':remaining,'advisory_actions_for_transfer_and_final':TRANSFER_RESERVE,'predicted_remaining_after_pickup':remaining-estimate,'advisory80_fits':remaining-estimate>=TRANSFER_RESERVE,'strategy':'high_diagonal_8mm_then_contact4mm; original_pose_timing_if_legal_IK_else_subdivide','right_hold_q':right_q.tolist(),'right_hold_grip':right_grip,'matching_tolerance_m':.005,'matching_tolerance_deg':3}),indent=2),encoding='utf8')
  if remaining<estimate+2:return result('insufficient_remaining_native_budget',{'needed_with_margin':estimate+2,'remaining':remaining,'advisory_transfer_reserve':TRANSFER_RESERVE,'estimate_may_increase_if_prior_IK_subdivision_required':True})
  prep=RightPrepare(env,out,ROOT/'handover_prior_415_490.npz',available_left_actions=estimate)
  def action(q,g):
   return {rm.process_name(left.arm_name):q.tolist(),rm.process_name(left.gripper_name):[float(g)],rm.process_name(right.arm_name):right_q.tolist(),rm.process_name(right.gripper_name):[right_grip]}
  def send(q,g,stage,desired=None,extra=None):
   nonlocal count,stop,certified
   if count>=180 or int(env.take_action_cnt[e])>=600 or env.end_flag[e]:stop='finite_pickup180_or_native_end';return False
   before=int(env.take_action_cnt[e]);merged=prep.before_action(action(q,g));keep=emit_action(merged,{'stage':'bonus_'+stage,'desired_link6_world':desired.tolist() if desired is not None else None,'extra':extra});after=int(env.take_action_cnt[e]);prep.after_action()
   if after!=before+1:
    snapshot('emitter_did_not_advance_once',{'before':before,'after':after});stop='emitter_no_single_action';return False
   count+=1;rec=snapshot(stage,{'desired_link6_world':desired.tolist() if desired is not None else None,'grip_command':float(g),'extra':extra})
   if not conditions(rec):stop='three_target_or_nine_neighbor_condition_broken';return False
   if desired is not None:
    got=tool()
    if np.linalg.norm(got[:3]-desired[:3])>.015 or qdistance(got[3:],desired[3:])>12:stop='link6_tracking_failed';return False
   if prep.status=='complete' and np.max(np.abs(arr(rm.get_joint(right,[e])[e])-prep.final_q))>.08:stop='prepared_right_tracking_failed';return False
   if reference is not None:
    err=rec['fixed_grasp_error']
    if err['position_m']>.010 or err['rotation_deg']>10:stop='fixed_grasp_reference_tracking_failed';return False
    if rec['objects'][BONUS]['position'][2]-baseline['objects'][BONUS]['position'][2]>.010:certified=True
   if not keep:stop='emitter_native_end_or_budget';return False
   return True
  def move(goal,g,stage,translation_step=.004,original_pose_timing=False):
   now=tool()
   if original_pose_timing:
    direct=rm.solve_ik(target_pose=goal.tolist(),env_idx=e,robot=left,trans='world')
    if direct.get('status')=='Success':
     current_q=arr(rm.get_joint(left,[e])[e]);q,normalization=nearest_legal(arr(direct['joint_value']).reshape(-1),current_q,types,ul,hl,sl)
     if np.max(np.abs(q-current_q))<=.20:
      if not send(q,g,stage,goal,{'original_pose_timing':True,'legal_representative':normalization}):return False,stop
      return True,None
    # No action emitted for rejected direct target. Finite contact subdivision.
    snapshot('prior_direct_target_needs_subdivision',{'phase':stage,'desired':goal.tolist(),'direct_status':str(direct.get('status'))})
   n=samples(now,goal,translation_step)
   if original_pose_timing:n=max(2,n)
   for i,desired in enumerate(interpolate(now,goal,n),1):
    ik=rm.solve_ik(target_pose=desired.tolist(),env_idx=e,robot=left,trans='world')
    if ik.get('status')!='Success':
     snapshot('ik_failed',{'phase':stage,'desired':desired.tolist(),'result':str(ik)});return False,'ik_failed'
    current_q=arr(rm.get_joint(left,[e])[e]);q,normalization=nearest_legal(arr(ik['joint_value']).reshape(-1),current_q,types,ul,hl,sl)
    if np.max(np.abs(q-current_q))>.20:
     snapshot('ik_discontinuity',{'phase':stage,'normalized':normalization});return False,'ik_discontinuity'
    if not send(q,g,stage,desired,{'sample':i,'total':n,'legal_representative':normalization}):return False,stop
   return True,None
  for phase,goal,spacing in route:
   # Open only at elevated stack waypoint; keep the current hand during retreat.
   if phase=='descend_to_original_row379_entry':
    for g in np.linspace(left_grip,1,math.ceil(abs(1-left_grip)/.1)+1)[1:]:
     if not send(arr(rm.get_joint(left,[e])[e]),float(g),'open_above_stack',tool()):return result(stop)
    left_grip=1.
   ok,why=move(goal,left_grip,phase,translation_step=spacing)
   if not ok:return result(why)
  check=snapshot('prior_entry')
  new=check['objects'][BONUS]
  if np.linalg.norm(arr(new['position'])-bp[:3])>.005 or qdistance(new['quat_wxyz'],bp[3:])>3:return result('bonus_disturbed_before_grasp')
  for row,goal,g in zip(rows,poses,grips):
   ok,why=move(goal,float(np.clip(g,0,1)),f'original_pose_row_{int(row)}',original_pose_timing=True)
   if not ok:return result(why)
   if int(row)==397:
    # First candidate grasp reference, fixed BEFORE the lift; never recaptured.
    rec=snapshot('first_closed_grasp_candidate');reference=np.asarray(rec['bonus_in_left_link6'])
    (out/'first_grasp_reference.json').write_text(json.dumps({'original_row':397,'native_step':int(env.take_action_cnt[e]),'object_in_left_link6':reference.tolist(),'status':'candidate_until_actual_lift','reset_allowed':False},indent=2),encoding='utf8')
  prep.require_ready()
  final=snapshot('pickup_endpoint_after_original_row414')
  return result('bonus_pickup_lift_observed' if certified else 'no_verified_bonus_lift',{'endpoint':final,'physical_success_not_proven_by_video_alone':True})
 except Exception as error:
  try:snapshot('exception',{'error':repr(error)})
  except Exception:pass
  return result('stopped_exception',{'error':repr(error)})
