"""Finite dual-arm handover; emit_action owns native stepping and checks."""
from pathlib import Path
import json,hashlib,math
import numpy as np
from .json_safety import plain
ROOT=Path(__file__).parent
from .rightmost_target import HAND_LABELS
ALL=list(HAND_LABELS)
BONUS='mahjong9_0'
def arr(v):
 if hasattr(v,'detach'):v=v.detach().cpu().numpy()
 return np.asarray(v,dtype=float)
def rotation(q):
 q=arr(q).reshape(4);q=q/np.linalg.norm(q);w,x,y,z=q
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def axis_angle(q,axis):return float(np.degrees(np.arccos(np.clip((rotation(q)@np.asarray(axis))[2],-1,1))))
def qdistance(a,b):
 a=arr(a)/np.linalg.norm(a);b=arr(b)/np.linalg.norm(b);return float(2*np.degrees(np.arccos(np.clip(abs(np.dot(a,b)),0,1))))
def matrix(p):
 p=arr(p).reshape(7);T=np.eye(4);T[:3,:3]=rotation(p[3:]);T[:3,3]=p[:3];return T
def samples(a,b,translation_step=.004):return max(1,math.ceil(np.linalg.norm(arr(a)[:3]-arr(b)[:3])/translation_step),math.ceil(qdistance(arr(a)[3:],arr(b)[3:])/3))
def interpolate(a,b,n):
 a=arr(a);b=arr(b);qa=a[3:]/np.linalg.norm(a[3:]);qb=b[3:]/np.linalg.norm(b[3:])
 if np.dot(qa,qb)<0:qb=-qb
 for i in range(1,n+1):
  t=i/n;q=(1-t)*qa+t*qb;q/=np.linalg.norm(q);yield np.r_[a[:3]*(1-t)+b[:3]*t,q]

"""Legal nearest 2pi representatives; pure CPU math plus read-only runtime limits."""
import math
import xml.etree.ElementTree as ET
import numpy as np
from .json_safety import plain


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


def run_handover(env,emit_action,out,pickup_subdir="mahjong_bonus_pickup"):
 TARGETS=[str(row[0]) for row in env.kong]
 """Explicit finite block. Does not reset, attach, call reward, or finalize placement."""
 root_out=Path(out);folder=root_out/'mahjong_bonus_handover';folder.mkdir(exist_ok=False)
 rm=env.robot_manager;lm=env.scene_manager.layout_manager;e=0
 left=rm.get_robot_by_arm_name('left_arm');right=rm.get_robot_by_arm_name('right_arm')
 arms=[left,right];left_ref=None;right_ref=None;count=0;seq=0;stop=None
 begin=int(env.take_action_cnt[e]);current_row=414;observations=[];right_ref_row=None;right_ref_native=None
 right_candidate_stable=False;release_anchor=None;right_carried=False
 def dump(name,x): (folder/name).write_text(json.dumps(plain(x),indent=2),encoding='utf-8')
 def tool(r):return arr(rm.get_real_endpose(r,[e],is_relative=False)[e]).reshape(7)
 def grip(r):
  v=float(np.mean(arr(rm.get_end_effector_real_val(r,[e])[e])));lo,hi=map(float,r.gripper_scale)
  return float(np.clip((v-lo)/(hi-lo) if r.gripper_move['sign']==1 else (hi-v)/(hi-lo),0,1))
 def obj(label):
  o=lm.get_scene_object(env_idx=e,inst_name=lm.get_instance_name(env_idx=e,label=label))
  s=o.get_current_dynamic_state();p=arr(s.position).reshape(3);q=arr(s.orientation).reshape(4)
  return {'position':p.tolist(),'quat_wxyz':q.tolist(),'Yup_deg':axis_angle(q,[0,1,0]),'Zup_deg':axis_angle(q,[0,0,1]),
   'linear_velocity':arr(s.linear_velocity if s.linear_velocity is not None else [0,0,0]).tolist(),
   'angular_velocity':arr(s.angular_velocity if s.angular_velocity is not None else [0,0,0]).tolist()}
 def error(ref,T):
  D=np.linalg.inv(ref)@T
  return {'position_m':float(np.linalg.norm(D[:3,3])),'rotation_deg':float(np.degrees(np.arccos(np.clip((np.trace(D[:3,:3])-1)/2,-1,1))))}
 def capture(stage,extra=None):
  nonlocal seq
  objects={x:obj(x) for x in ALL+[BONUS]};B=objects[BONUS];world=matrix(B['position']+B['quat_wxyz'])
  poses={r.arm_name:tool(r) for r in arms}
  trans={r.arm_name:np.linalg.inv(matrix(poses[r.arm_name]))@world for r in arms}
  rec={'stage':stage,'original_row':current_row,'native_step':int(env.take_action_cnt[e]),'sent':count,'objects':objects,
   'robots':{r.arm_name:{'link6_world':poses[r.arm_name].tolist(),'joint':arr(rm.get_joint(r,[e])[e]).tolist(),'gripper_actual':arr(rm.get_end_effector_real_val(r,[e])[e]).tolist(),'gripper_normalized_actual':grip(r)} for r in arms},
   'object_in_link6':{k:v.tolist() for k,v in trans.items()},
   'left_fixed_error':error(left_ref,trans['left_arm']) if left_ref is not None else None,
   'right_fixed_error':error(right_ref,trans['right_arm']) if right_ref is not None else None,
   'native_remaining':{'check_list':repr(env.reward_manager.check_list[e]),'final_check_list':repr(env.reward_manager.final_check_list[e]),'query_list':repr(env.reward_manager.query_list[e])},
   'native_success_flag':bool(env.success[e]),'native_end_flag':bool(env.end_flag[e]),'extra':extra}
  dump(f'state_{seq:04d}.json',rec);seq+=1;observations.append(rec);return rec
 def common(rec):
  return all(rec['objects'][x]['Zup_deg']<30 for x in TARGETS) and all(rec['objects'][x]['Yup_deg']<7 for x in ALL if x not in TARGETS)
 def within(x,pos=.010,rot=10):return x is not None and x['position_m']<=pos and x['rotation_deg']<=rot
 def result(status,extra=None):
  r={'status':status,'counts_as_success':False,'native_begin':begin,'native_end':int(env.take_action_cnt[e]),'actions':count,'remaining_native_budget':600-int(env.take_action_cnt[e]),'last_original_row':current_row,'left_reference_reset':False,'right_reference_row':right_ref_row,'right_dual_close_stable':right_candidate_stable,'right_only_physical_carry_verified':right_carried,'next':'no_final_placement_in_this_block','extra':extra}
  dump('result.json',r);return r
 try:
  if env.num_envs!=1 or int(env.step_lim)!=600 or set(str(row[e]) for row in env.kong)!=set(TARGETS):return result('case_identity_or_budget_mismatch')
  if pickup_subdir!='mahjong_bonus_pickup':return result('unsupported_explicit_pickup_provenance')
  ref_path=root_out/pickup_subdir/'first_grasp_reference.json'
  pickup_result_path=root_out/pickup_subdir/'result.json'
  if not ref_path.is_file() or not pickup_result_path.is_file():return result('missing_pickup_fixed_reference_or_result')
  prior_pickup=json.loads(pickup_result_path.read_text(encoding='utf-8'))
  if prior_pickup.get('status')!='bonus_pickup_lift_observed' or not prior_pickup.get('pickup_reference_certified_by_lift'):return result('pickup_not_physically_lift_certified')
  rf=json.loads(ref_path.read_text(encoding='utf-8'))
  if rf.get('original_row')!=397 or rf.get('reset_allowed') is not False:return result('invalid_left_fixed_reference_provenance')
  left_ref=np.asarray(rf['object_in_left_link6'],dtype=float)
  if left_ref.shape!=(4,4) or not np.isfinite(left_ref).all():return result('invalid_left_fixed_reference_matrix')
  dump('left_fixed_reference_unchanged.json',{'reference':rf,'sha256':hashlib.sha256(ref_path.read_bytes()).hexdigest(),'source':str(ref_path)})
  spec={'prior_sha256': 'e4d162867d650d8a73a331b08ed89beb8405dc75444ef07ef38d1d7a632dc4ae'};prior=ROOT/'handover_prior_415_490.npz'
  if hashlib.sha256(prior.read_bytes()).hexdigest()!=spec['prior_sha256']:return result('handover_prior_hash_mismatch')
  with np.load(prior,allow_pickle=False) as z: data={k:z[k].copy() for k in z.files}
  base=capture('entry');L=tool(left);R=tool(right)
  entry_errors={r.arm_name:{'position_m':float(np.linalg.norm(tool(r)[:3]-data[key][:3])),'rotation_deg':qdistance(tool(r)[3:],data[key][3:])} for r,key in [(left,'entry_left'),(right,'entry_right')]}
  predicted=matrix(data['entry_left'])@left_ref
  actual=matrix(base['objects'][BONUS]['position']+base['objects'][BONUS]['quat_wxyz'])
  comparison={'tool_entry_errors':entry_errors,'actual_object_in_left':base['object_in_link6']['left_arm'],'fixed_left_reference':left_ref.tolist(),'actual_relative_error':base['left_fixed_error'],'source414_held_object_prediction_from_actual_fixed_reference':predicted.tolist(),'source414_prediction_error':error(predicted,actual),'original_object_prior_at414_available':False,'source_object_prior_limit':'input objstate only captures native120; never invent an original row414 object pose','actual_pickup_endpoint':prior_pickup['extra']['endpoint']['objects'][BONUS]}
  dump('entry_actual_source_comparison.json',comparison)
  if not common(base):return result('common_pose_checks_failed_on_entry')
  if not within(base['left_fixed_error']):return result('existing_left_fixed_reference_not_tracking')
  if entry_errors['left_arm']['position_m']>.015 or entry_errors['left_arm']['rotation_deg']>5:return result('left_endpoint_differs_from_original_row414',comparison)
  if np.linalg.norm(arr(base['objects'][BONUS]['position'])-arr(prior_pickup['extra']['endpoint']['objects'][BONUS]['position']))>.005:return result('bonus_changed_since_pickup_endpoint')
  limits={r.arm_name:runtime_limits(rm,r,e) for r in arms}
  held_left=tool(left);last_grips=[grip(left),grip(right)]
  if entry_errors['right_arm']['position_m']>.005 or entry_errors['right_arm']['rotation_deg']>3:return result('parallel_right_prepare_did_not_match_original414')
  if 600-begin<75:return result('insufficient_budget_for75_handover')
  def solve(goal,r):
   ik=rm.solve_ik(target_pose=goal.tolist(),env_idx=e,robot=r,trans='world')
   if ik.get('status')!='Success':return None,{'status':str(ik.get('status'))}
   try:
    _,types,u,h,s=limits[r.arm_name];cur=arr(rm.get_joint(r,[e])[e]).reshape(-1)
    q,norm=nearest_legal(arr(ik['joint_value']).reshape(-1),cur,types,u,h,s)
    return (q if np.max(np.abs(q-cur))<=.20 else None),norm
   except Exception as err:return None,{'error':repr(err)}
  def send(goals,grips,stage,meta=None):
   nonlocal count,stop
   if count>=100 or int(env.take_action_cnt[e])>=600 or env.end_flag[e]:stop='finite_budget_or_native_end';return False
   qs=[];details=[]
   for r,goal in zip(arms,goals):
    q,detail=solve(goal,r);qs.append(q);details.append(detail)
   if any(q is None for q in qs):return None
   action={}
   for r,q,g in zip(arms,qs,grips):action[rm.process_name(r.arm_name)]=q.tolist();action[rm.process_name(r.gripper_name)]=[float(np.clip(g,0,1))]
   before=int(env.take_action_cnt[e]);keep=emit_action(action,{'stage':stage,'original_row':current_row,'desired_left_link6':goals[0].tolist(),'desired_right_link6':goals[1].tolist(),'legal_ik':details,'metadata':meta})
   after=int(env.take_action_cnt[e]);count+=1
   rec=capture(stage,{'grip_commands':list(map(float,grips)),'legal_ik':details,'metadata':meta})
   if after!=before+1:stop='emitter_not_single_action';return False
   if not common(rec):stop='common_pose_checks_broken';return False
   for r,goal in zip(arms,goals):
    got=tool(r)
    if np.linalg.norm(got[:3]-goal[:3])>.015 or qdistance(got[3:],goal[3:])>12:stop='dual_link6_tracking_failed';return False
   # Left reference is frozen forever; guard ends only at intentional opening.
   if current_row<469 and not within(rec['left_fixed_error']):stop='left_fixed_reference_slipped';return False
   if right_ref is not None and not within(rec['right_fixed_error']):stop='right_fixed_reference_slipped';return False
   if not keep:stop='emitter_native_end_or_budget';return False
   return True
  def advance(goals,grips,stage,depth=0):
   # Try precisely one original-pose action first. Rejected IK emits nothing.
   outcome=send(goals,grips,stage,{'subdivision_depth':depth})
   if outcome is not None:return outcome
   if depth>=6:return False
   start=[tool(r) for r in arms];starts=[grip(r) for r in arms]
   mid=[next(interpolate(a,b,2)) for a,b in zip(start,goals)]
   if not advance(mid,[(a+b)/2 for a,b in zip(starts,grips)],stage+'_subdivide',depth+1):return False
   return advance(goals,grips,stage,depth+1)
  for row,lp,rp,lg,rg in zip(data['rows'],data['left'],data['right'],data['left_grip'],data['right_grip']):
   current_row=int(row)
   if current_row==469:
    if right_ref is None or not right_candidate_stable:return result('right_grasp_not_verified_before_left_release')
    release_anchor=capture('before_left_release')
   if not advance([lp,rp],[float(lg),float(rg)],f'original_dual_pose_row_{current_row}'):return result(stop or 'dual_ik_subdivision_exhausted')
   rec=observations[-1]
   if current_row==466:
    # One immutable candidate, captured before left opens; dual closing is not success.
    T=np.asarray(rec['object_in_link6']['right_arm']);dist=float(np.linalg.norm(T[:3,3]))
    if grip(right)>.65 or not .10<dist<.21:return result('right_close_geometry_not_plausible',{'tool_to_object_m':dist,'right_gripper_actual_normalized':grip(right)})
    right_ref=T.copy();right_ref_row=466;right_ref_native=int(env.take_action_cnt[e])
    dump('right_receiver_reference.json',{'schema':'immutable_receiver_reference_v1','arm_tag':'right_arm','object_label':BONUS,'object_in_link6':T.tolist(),'evidence_id':f'bonus-handover-row466-native{int(env.take_action_cnt[e])}','original_row':466,'native_step':int(env.take_action_cnt[e]),'object_in_right_link6':T.tolist(),'status':'candidate_until_left_open_right_only_carry','reset_allowed':False,'both_fixed_transforms':rec['object_in_link6']})
   if current_row==468:
    rec=capture('dual_hold_before_left_open')
    right_candidate_stable=within(rec['right_fixed_error'],.005,5) and within(rec['left_fixed_error']) and grip(right)<.65
   if current_row==489:
    end=capture('right_only_endpoint_row489')
    anchor=release_anchor;rb=arr(anchor['robots']['right_arm']['link6_world']);re=tool(right)
    object_displacement=float(np.linalg.norm(arr(end['objects'][BONUS]['position'])-arr(anchor['objects'][BONUS]['position'])))
    tool_displacement=float(np.linalg.norm(re[:3]-rb[:3]))
    left_relative_change=error(np.asarray(anchor['object_in_link6']['left_arm']),np.asarray(end['object_in_link6']['left_arm']))
    right_carried=bool(grip(left)>.85 and grip(right)<.65 and within(end['right_fixed_error'],.010,10) and tool_displacement>.010 and object_displacement>.010 and left_relative_change['position_m']>.010 and end['objects'][BONUS]['position'][2]>.835)
    dump('physical_transfer_verification.json',{'right_only_carry_verified':right_carried,'right_tool_displacement_m':tool_displacement,'object_displacement_m':object_displacement,'left_object_relative_change':left_relative_change,'right_fixed_error':end['right_fixed_error'],'gripper_actual_normalized':{'left':grip(left),'right':grip(right)},'bonus_height':end['objects'][BONUS]['position'][2],'right_reference_never_reset':True,'left_reference_preserved_not_guarded_after_release':True})
  if right_carried:
   dump('receiver_reference_certified.json',{'schema':'immutable_receiver_reference_v1','arm_tag':'right_arm','object_label':BONUS,'object_in_link6':right_ref.tolist(),'evidence_id':f'bonus-handover-row466-native{right_ref_native}-verified-row489','reference_original_row':466,'reference_native_step':right_ref_native,'verified_original_row':489,'verified_native_step':int(env.take_action_cnt[e]),'status':'right_only_physical_carry_observed','physical_certificate':'physical_transfer_verification.json','reset_allowed':False})
  return result('right_only_physical_carry_observed' if right_carried else 'handover_unverified_after_row489')
 except Exception as err:
  try:capture('exception',{'error':repr(err)})
  except Exception:pass
  return result('stopped_exception',{'error':repr(err)})
