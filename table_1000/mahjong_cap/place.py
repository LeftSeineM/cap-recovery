"""Finite same-table placement candidate. No simulator startup or direct actuation."""
from pathlib import Path
from .grip_continuity import read_same_job_hold_grips
import json,math,hashlib
import numpy as np
import re
from .json_safety import plain
from .rightmost_target import runtime_target,rotated_offsets,HAND_LABELS
from .finish_primitives import rotation,quaternion,transform,pose,grip_fraction
from .legal_joint_representative import nearest_legal,runtime_limits

def arr(x):
 if hasattr(x,'detach'):x=x.detach().cpu().numpy()
 return np.asarray(x,dtype=float)
def angle(q,axis):return float(np.degrees(np.arccos(np.clip((rotation(q)@axis)[2],-1,1))))
def qdist(a,b):return float(2*np.degrees(np.arccos(np.clip(abs(np.dot(a/np.linalg.norm(a),b/np.linalg.norm(b))),0,1))))
def count(a,b,spacing,rotation_step=6):return max(1,math.ceil(np.linalg.norm(a[:3]-b[:3])/spacing),math.ceil(qdist(a[3:],b[3:])/rotation_step))
def interp(a,b,n):
 qa=a[3:]/np.linalg.norm(a[3:]);qb=b[3:]/np.linalg.norm(b[3:])
 dot=float(np.dot(qa,qb))
 if dot<0:qb=-qb;dot=-dot
 theta=math.acos(float(np.clip(dot,-1,1)))
 for i in range(1,n+1):
  t=i/n
  q=(1-t)*qa+t*qb if theta<1e-8 else (np.sin((1-t)*theta)*qa+np.sin(t*theta)*qb)/np.sin(theta)
  q/=np.linalg.norm(q)
  yield np.r_[(1-t)*a[:3]+t*b[:3],q]

def swept_box_clear(a,b,obstacles,bounds,margin=.001):
 """Conservative AABB enclosure of exact translation+shortest quaternion SLERP.
 Each segment center lies between endpoints. All rotated corner deviations from
 midpoint rotation are bounded by 2*r*sin(theta/4); this encloses the whole arc.
 It checks held tile against measured mahjong boxes only, not robot/hand/table.
 """
 midpoint=next(interp(a,b,2));theta=math.radians(qdist(a[3:],b[3:]));offsets=rotated_offsets(midpoint[3:],bounds)
 radius=float(max(np.linalg.norm(x) for x in rotated_offsets([1,0,0,0],bounds)))
 delta=2*radius*math.sin(theta/4)
 lo=np.minimum(a[:3],b[:3])+offsets.min(axis=0)-delta-margin;hi=np.maximum(a[:3],b[:3])+offsets.max(axis=0)+delta+margin
 for name,c,o_half in obstacles:
  if np.all(lo<c+o_half) and np.all(hi>c-o_half):return False,{'obstacle':name,'held_swept_min':lo.tolist(),'held_swept_max':hi.tolist(),'obstacle_center':c.tolist(),'obstacle_half':o_half.tolist()}
 return True,None

def route_clear(start,route,obstacles,bounds):
 now=start;fail=None
 for name,end,spacing in route:
  for p in interp(now,end,count(now,end,spacing)):
   ok,fail=swept_box_clear(now,p,obstacles,bounds)
   if not ok:return False,{'phase':name,'collision':fail}
   now=p
 return True,None

def object_route(start,goal,obstacles,bounds):
 """Prefer a combined diagonal iff its full held-box sweep has actual clearance.
 If not, use a known-box-top plus corner-radius clearance height for XY/rotation, then descend at final clear XY only.
 One finite alternative, no optimizing search or scene/planner framework.
 """
 above=goal.copy();above[2]+=.020
 direct=[('combined_transit_upright_to_above20',above,.012),('supported_table_descent',goal,.003)]
 ok,why=route_clear(start,direct,obstacles,bounds)
 if ok:return direct,{'choice':'direct_checked_sweep','rejected_direct':None}
 # Conservative clearance height bounds every corner for any orientation.
 high=goal.copy();high[2]=max(goal[2]+.080,max((float(c[2]+h[2]) for _,c,h in obstacles),default=goal[2])+float(np.linalg.norm((arr(bounds["max"])-arr(bounds["min"]))/2))+.002)
 fallback=[('high_transit_upright',high,.012),('at_clear_slot_down_to_above20',above,.012),('supported_table_descent',goal,.003)]
 ok,fail=route_clear(start,fallback,obstacles,bounds)
 return (fallback if ok else None),{'choice':'high_then_slot_descent' if ok else 'no_clear_candidate','rejected_direct':why,'fallback_rejection':fail}

def run_place(env,emit_action,out,receiver_reference,expected_job_id):
 """Reference dict/path: schema,arm_tag,object_label,object_in_link6,evidence_id.

 Schema='immutable_receiver_reference_v1', arm_tag='right_arm',
 object_label='mahjong9_0', object_in_link6=4x4 or flat16, evidence_id nonempty.
 Supplied reference must originate at verified handover. It is never recaptured.
 """
 targets=[str(row[0]) for row in env.kong]
 others=[name for name in HAND_LABELS if name not in targets]
 source_output=Path(out)
 out=source_output/'mahjong_rightmost_place';out.mkdir(exist_ok=False)
 e=0;rm=env.robot_manager;lm=env.scene_manager.layout_manager
 right=rm.get_robot_by_arm_name('right_arm');left=rm.get_robot_by_arm_name('left_arm')
 begin=int(env.take_action_cnt[e]);sent=0;seq=0;ref=None;released=False;stop=None
 def dump(name,value):
  (out/name).write_text(json.dumps(plain(value),ensure_ascii=False,indent=2),encoding='utf8')
 def obj(label):return lm.get_scene_object(env_idx=e,inst_name=lm.get_instance_name(env_idx=e,label=label))
 def read(o):
  s=o.get_current_dynamic_state();p=arr(s.position).reshape(3);q=arr(s.orientation).reshape(4)
  v=arr(s.linear_velocity if s.linear_velocity is not None else [np.nan]*3).reshape(3)
  w=arr(s.angular_velocity if s.angular_velocity is not None else [np.nan]*3).reshape(3)
  return {'pose':np.r_[p,q].tolist(),'linear_velocity':v.tolist(),'angular_velocity':w.tolist(),'Yup_deg':angle(q,[0,1,0]),'Zup_deg':angle(q,[0,0,1])}
 def tool():return arr(rm.get_real_endpose(right,[e],is_relative=False)[e]).reshape(7)
 def snap(stage,extra=None):
  nonlocal seq
  labeled={x:read(obj(x)) for x in list(HAND_LABELS)+['mahjong9_0']}
  rigid={str(k):read(o) for k,o in env.scene_manager._rigid_and_dynamic_objects[e].items() if hasattr(o,'get_current_dynamic_state')}
  r={'stage':stage,'native_step':int(env.take_action_cnt[e]),'sent':sent,'objects':labeled,'all_rigid_objects':rigid,'right_link6_world':tool().tolist(),'right_joint':arr(rm.get_joint(right,[e])[e]).tolist(),'right_gripper_actual':arr(rm.get_end_effector_real_val(right,[e])[e]).tolist(),'left_gripper_actual':arr(rm.get_end_effector_real_val(left,[e])[e]).tolist(),'native_end':bool(env.end_flag[e]),'native_success':bool(env.success[e]),'remaining_reward':repr(env.reward_manager.check_list[e]),'contact_force_evidence':None,'contact_force_limitation':'No verified force/contact API; bottom/velocity observations are retained, not claimed as force proof.','extra':extra}
  if ref is not None:
   actual=np.linalg.inv(transform(tool()))@transform(labeled['mahjong9_0']['pose']);D=np.linalg.inv(ref)@actual
   r['fixed_reference_error']={'position_m':float(np.linalg.norm(D[:3,3])),'rotation_deg':float(np.degrees(np.arccos(np.clip((np.trace(D[:3,:3])-1)/2,-1,1))))}
  dump(f'state_{seq:04d}.json',r);seq+=1;return r
 def common(r):return all(r['objects'][x]['Zup_deg']<30 for x in targets) and all(r['objects'][x]['Yup_deg']<7 for x in others)
 def finish(status,extra=None):
  v={'status':status,'counts_as_success':bool(env.end_flag[e] and env.success[e]),'native_begin':begin,'native_end':int(env.take_action_cnt[e]),'place_actions':sent,'remaining_budget':600-int(env.take_action_cnt[e]),'released':released,'native_terminal_success':bool(env.end_flag[e] and env.success[e]),'extra':extra};dump('result.json',v);return v
 try:
  spec=json.loads(Path(receiver_reference).read_text(encoding='utf8')) if isinstance(receiver_reference,(str,Path)) else json.loads(json.dumps(receiver_reference))
  if spec.get('schema')!='immutable_receiver_reference_v1' or spec.get('arm_tag')!='right_arm' or spec.get('object_label')!='mahjong9_0' or not spec.get('evidence_id'):return finish('invalid_receiver_reference_identity')
  ref=arr(spec['object_in_link6']).reshape(4,4).copy();ref.setflags(write=False)
  if not np.allclose(ref[3],[0,0,0,1]) or not np.allclose(ref[:3,:3].T@ref[:3,:3],np.eye(3),atol=1e-5) or np.linalg.det(ref[:3,:3])<.999:return finish('invalid_receiver_rigid_transform')
  dump('supplied_immutable_receiver_reference.json',spec)
  baseline=snap('entry')
  if env.num_envs!=1 or int(env.step_lim)!=600:return finish('native_case_or_budget_mismatch')
  if env.end_flag[e]:return finish('already_native_terminal_at_entry')
  if not common(baseline):return finish('common_conditions_failed')
  pending=env.reward_manager.check_list[e]
  if len(pending)!=1 or 'is_qpos_close' in repr(pending) or 'is_A_xy_distance_close_to_pos' not in repr(pending) or 'is_all_gripper_open' not in repr(pending):return finish('final_native_stage_not_verified')
  err=baseline['fixed_reference_error']
  if err['position_m']>.010 or err['rotation_deg']>10:return finish('receiver_reference_not_current',err)
  target_spec=runtime_target(env)
  dump('actual_rightmost_target_before_planning.json',target_spec)
  category=str(target_spec['runtime_label_binding']['mahjong9_0']['category_idx'])
  bounds_by_category=json.loads((Path(__file__).parent/'enabled_collision_bounds.json').read_text(encoding='utf-8'))
  bounds=bounds_by_category[category]
  goal=arr(target_spec['goal_pose']);table_z=target_spec['table_z_from_measured_peer']
  peer=baseline['objects'][target_spec['upright_peer_label']]
  if not target_spec['upright_peer_available']:return finish('no_upright_geometry_peer')
  def actual_box(label,state):
   instance=str(lm.get_instance_name(env_idx=e,label=label));match=re.fullmatch(r'mahjong_(\d+)_(\d+)',instance)
   if match is None:raise ValueError('unknown runtime instance bounds')
   bounds=bounds_by_category[str(int(match.group(1)))];p=arr(state['pose']);o=rotated_offsets(p[3:],bounds)
   return p[:3]+(o.min(axis=0)+o.max(axis=0))/2,(o.max(axis=0)-o.min(axis=0))/2
  bonus_offsets=rotated_offsets(goal[3:],bounds);gcenter=goal[:3]+(bonus_offsets.min(axis=0)+bonus_offsets.max(axis=0))/2;ghalf=(bonus_offsets.max(axis=0)-bonus_offsets.min(axis=0))/2
  def clear(r):
   for label in HAND_LABELS:
    c,h=actual_box(label,r['objects'][label])
    if all(np.abs(c[:2]-gcenter[:2])<h[:2]+ghalf[:2]+.0015):return False
   return True
  if not clear(baseline):return finish('actual_rightmost_slot_not_clear')
  limits=runtime_limits(rm,right,e);_,types,ul,hl,sl=limits
  leftq=arr(rm.get_joint(left,[e])[e]).copy()
  continuity=read_same_job_hold_grips(env,source_output,expected_job_id)
  rg=continuity['right_hold_command'];lg=continuity['left_hold_command']
  continuity['measured_grip_fraction_for_observation_only']={'right':grip_fraction(right,arr(rm.get_end_effector_real_val(right,[e])[e])),'left':grip_fraction(left,arr(rm.get_end_effector_real_val(left,[e])[e]))}
  dump('same_job_grip_command_continuity.json',continuity)
  start=arr(baseline['objects']['mahjong9_0']['pose'])
  bonus_instance=str(lm.get_instance_name(env_idx=e,label='mahjong9_0'))
  def measured_obstacles(r):
   return [(label,*actual_box(label,r['objects'][label])) for label in HAND_LABELS]
  route,route_info=object_route(start,goal,measured_obstacles(baseline),bounds)
  dump('swept_clearance_plan.json',{'route_info':route_info,'obstacle_boxes':[{'instance':n,'center':c.tolist(),'half':h.tolist()} for n,c,h in measured_obstacles(baseline)],'half_source':'measured enabled mahjong collision mesh shared extent, existing asset review','margin_m':.001,'coverage':'held mahjong vs recorded mahjong boxes only; not gripper/robot/table certification'})
  if route is None:return finish('held_tile_swept_box_route_not_clear',route_info)
  costs=[];prev=start
  for name,p,spacing in route:
   n=count(prev,p,spacing);costs.append({'phase':name,'actions':n,'spacing_m':spacing,'object_goal':p.tolist()});prev=p
  release_steps=max(1,math.ceil(max(1-rg,1-lg)/.15));estimate=sum(x['actions'] for x in costs)+release_steps+1
  dump('plan.json',{'route':costs,'estimated_actions':estimate,'remaining_budget':600-begin,'release_steps':release_steps,'support_observation_hold_steps':0,'postrelease_observation_steps':1,'goal':goal.tolist(),'table_z_from_actual_peer':float(table_z),'slot_AABB_margin_m':.002,'entry_reference_evidence_id':spec['evidence_id'],'geometry_is_not_full_robot_collision_validation':True,'route_selection':route_info,'high_rotation_step_deg':6})
  if 600-begin<estimate+1:return finish('insufficient_native_budget_for_place',{'estimated':estimate,'remaining':600-begin})
  def supported(r):
   s=r['objects']['mahjong9_0'];p=arr(s['pose']);bottom=p[2]+rotated_offsets(p[3:],bounds)[:,2].min()
   return abs(bottom-table_z)<.003 and s['Yup_deg']<7 and np.linalg.norm(p[:2]-goal[:2])<.015
  def send(q,rightg,leftg,stage,desired=None,check_grasp=True):
   nonlocal sent,stop
   if sent>=100 or int(env.take_action_cnt[e])>=600 or env.end_flag[e]:stop='finite_place100_or_native_end';return None
   action={rm.process_name(right.arm_name):q.tolist(),rm.process_name(right.gripper_name):[float(rightg)],rm.process_name(left.arm_name):leftq.tolist(),rm.process_name(left.gripper_name):[float(leftg)]}
   before=int(env.take_action_cnt[e]);keep=emit_action(action,{'stage':'place_'+stage,'desired_right_link6_world':None if desired is None else desired.tolist(),'immutable_receiver_evidence_id':spec['evidence_id']});after=int(env.take_action_cnt[e]);sent+=1
   r=snap(stage,{'desired_link6_world':None if desired is None else desired.tolist(),'right_grip':rightg,'left_grip':leftg})
   if after!=before+1:stop='emitter_not_one_native_step';return None
   if bool(env.end_flag[e]) and bool(env.success[e]):
    r['official_terminal_success']=True;dump('official_terminal_success.json',r);return r
   if not common(r) or not clear(r):stop='common_or_slot_condition_broken';return None
   latest=runtime_target(env)
   if np.linalg.norm(arr(latest['goal_pose'])[:3]-goal[:3])>.002:stop='rightmost_hand_target_moved_after_plan';return None
   if np.max(np.abs(arr(rm.get_joint(left,[e])[e])-leftq))>.08:stop='left_hold_drift';return None
   if check_grasp:
    error=r['fixed_reference_error']
    if error['position_m']>.010 or error['rotation_deg']>10:stop='immutable_receiver_grasp_drift';return None
   if desired is not None:
    actual=tool()
    if np.linalg.norm(actual[:3]-desired[:3])>.015 or qdist(actual[3:],desired[3:])>12:stop='right_tracking_error';return None
   if not keep and not env.end_flag[e]:stop='emitter_stopped';return None
   return r
  for name,object_goal,spacing in route:
   now=arr(read(obj('mahjong9_0'))['pose'])
   for target in interp(now,object_goal,count(now,object_goal,spacing)):
    before_snapshot=snap('pre_action_clearance',{'phase':name})
    physical_now=arr(before_snapshot['objects']['mahjong9_0']['pose'])
    ok,collision=swept_box_clear(physical_now,target,measured_obstacles(before_snapshot))
    if not ok:return finish('held_tile_preaction_swept_box_blocked',{'phase':name,'collision':collision})
    desired=pose(transform(target)@np.linalg.inv(ref));ik=rm.solve_ik(target_pose=desired.tolist(),env_idx=e,robot=right,trans='world')
    if ik.get('status')!='Success':return finish('right_place_ik_failed',{'phase':name,'target':target.tolist()})
    current=arr(rm.get_joint(right,[e])[e]);q,normalization=nearest_legal(arr(ik['joint_value']).reshape(-1),current,types,ul,hl,sl)
    if np.max(np.abs(q-current))>.20:return finish('right_place_ik_discontinuity',{'phase':name,'normalization':normalization})
    observation=send(q,rg,lg,name,desired)
    if observation is None:return finish(stop)
    if observation.get('official_terminal_success'):return finish('official_native_terminal_success',{'support_observation':None,'native_authority':True})
  # Original command-preserving gradual release after route endpoint; no added support/velocity opening gate.
  for i in range(1,release_steps+1):
   t=i/release_steps;r=send(arr(rm.get_joint(right,[e])[e]),rg+(1-rg)*t,lg+(1-lg)*t,'supported_release_both_open',check_grasp=False)
   if r is None:return finish(stop)
   if r.get('official_terminal_success'):
    released=True;return finish('official_native_terminal_success',{'support_metrics_diagnostic_only':supported(r),'native_authority':True})
   dump(f'release_observation_{i:02d}.json',{'support_metrics_diagnostic_only':supported(r),'does_not_gate_opening':True})
   if env.end_flag[e]:released=True;return finish('native_terminal_during_supported_release',{'support_metrics_diagnostic_only':supported(r),'force_contact_not_available':True})
  released=True
  observations=[]
  for _ in range(1):
   r=send(arr(rm.get_joint(right,[e])[e]),1.,1.,'postrelease_support_observation',check_grasp=False)
   if r is None:return finish(stop)
   if r.get('official_terminal_success'):return finish('official_native_terminal_success',{'support_metrics_diagnostic_only':supported(r),'native_authority':True})
   observations.append(supported(r))
   if env.end_flag[e]:break
  return finish('supported_release_observed' if all(observations) else 'postrelease_support_unstable',{'support_observations':observations,'force_contact_not_available':True})
 except Exception as exc:
  try:snap('exception',{'error':repr(exc)})
  except Exception:pass
  return finish('place_exception',{'error':repr(exc)})
