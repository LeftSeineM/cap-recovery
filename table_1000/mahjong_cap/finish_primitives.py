"""Audited native interface adapters and rigid-transform math; NOT an auto runner.

No Isaac imports or simulator startup. These functions must be called by the unique
runner inside an already authorized live environment. No hidden stage transitions,
grasp geometry, collision clearance, force or contact assumptions are supplied.
"""
from dataclasses import dataclass
import numpy as np

MIDDLE_Q_WXYZ=np.array([0.,1.,1.,0.])/np.sqrt(2.)
FINAL_XY=np.array([.319,-.15])

def rotation(q):
 q=np.asarray(q,dtype=float).reshape(4);n=np.linalg.norm(q)
 if not np.isfinite(q).all() or n<1e-12:raise ValueError('Invalid wxyz quaternion')
 w,x,y,z=q/n
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                  [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                  [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def quaternion(R):
 R=np.asarray(R,dtype=float).reshape(3,3)
 if not np.allclose(R.T@R,np.eye(3),atol=1e-6) or np.linalg.det(R)<0:raise ValueError('Not a rotation')
 # Eigenvector construction avoids a trace singularity at 180 degrees.
 qxyzw=np.linalg.eigh(np.array([
  [R[0,0]-R[1,1]-R[2,2],R[0,1]+R[1,0],R[0,2]+R[2,0],R[2,1]-R[1,2]],
  [R[0,1]+R[1,0],R[1,1]-R[0,0]-R[2,2],R[1,2]+R[2,1],R[0,2]-R[2,0]],
  [R[0,2]+R[2,0],R[1,2]+R[2,1],R[2,2]-R[0,0]-R[1,1],R[1,0]-R[0,1]],
  [R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1],np.trace(R)]])/3)[1][:,-1]
 q=qxyzw[[3,0,1,2]]
 return q if q[0]>=0 else -q

def transform(pose):
 pose=np.asarray(pose,dtype=float).reshape(7)
 if not np.isfinite(pose).all():raise ValueError('Nonfinite pose')
 T=np.eye(4);T[:3,:3]=rotation(pose[3:]);T[:3,3]=pose[:3];return T

def pose(T):
 T=np.asarray(T,dtype=float).reshape(4,4)
 return np.r_[T[:3,3],quaternion(T[:3,:3])]

@dataclass(frozen=True)
class HeldReference:
    """Full immutable measured object-in-link6 transform, bound to one real grip.

    Create only after physically verified grasp. Never recapture to hide slip.
    On a real handoff create a new receiver reference plus preserve the sender's.
    """
    arm_tag: str
    object_label: str
    object_in_link6: tuple
    evidence_id: str

    @classmethod
    def measured(cls,arm_tag,object_label,link6_pose,object_pose,evidence_id):
        if arm_tag not in ('left_arm','right_arm') or not evidence_id:raise ValueError('Identity required')
        T=np.linalg.inv(transform(link6_pose))@transform(object_pose)
        return cls(arm_tag,object_label,tuple(T.reshape(-1)),evidence_id)

    def link6_goal(self,object_goal):
        T=transform(object_goal)@np.linalg.inv(np.asarray(self.object_in_link6).reshape(4,4))
        return pose(T)

def actual_link6(env,arm_tag):
    rm=env.robot_manager;robot=rm.get_robot_by_arm_name(arm_tag)
    return np.asarray(rm.get_real_endpose(robot,env_idx_list=[0],is_relative=False)[0],dtype=float)

def object_pose(env,label='mahjong9_0'):
    p,q=env.scene_manager.layout_manager.get_instance_pose(env_idx=0,label=label)
    def array(v):
        if hasattr(v,'detach'):v=v.detach().cpu().numpy()
        return np.asarray(v,dtype=float).reshape(-1)
    return np.r_[array(p),array(q)]

def grip_fraction(robot,joints):
    lo,hi=map(float,robot.gripper_scale)
    x=(float(np.mean(joints))-lo)/(hi-lo)
    if robot.gripper_move['sign']!=1:x=1-x
    return float(np.clip(x,0,1))

def joint_action(env,arm_targets=None,gripper_targets=None):
    """Native eval action: six actual radians per x5, one NORMALIZED [0,1] grip.

    Omitted arms hold the actually measured joints, not reset/retract defaults.
    Support arm omitted deliberately so its native task queue retains ownership.
    """
    rm=env.robot_manager;arm_targets=arm_targets or {};gripper_targets=gripper_targets or {};action={}
    for robot in rm.robot_list:
        if robot.type!='target':continue
        if robot.robot_type!='arm':raise ValueError('Only observed x5 arm schema supported')
        q=np.asarray(arm_targets.get(robot.arm_name,rm.get_joint(robot,[0])[0]),dtype=float).reshape(-1)
        if q.size!=len(robot.arm_joint_indices) or not np.isfinite(q).all():raise ValueError('Bad joint target')
        g=gripper_targets.get(robot.arm_name,grip_fraction(robot,rm.get_end_effector_real_val(robot,[0])[0]))
        if not np.isfinite(g) or not 0<=g<=1:raise ValueError('Grip must be normalized, never meters')
        action[rm.process_name(robot.arm_name)]=q.tolist()
        action[rm.process_name(robot.gripper_name)]=[float(g)]
    return action

def plan_link6_world(env,arm_tag,target_pose,*,validate_trajectory=None):
    """Reuse official cuRobo plan_path; expose limited collision model explicitly.

    Installed planner scene contains table only; success does not prove clearance
    from tiles, second arm or held object. This helper performs NO actuation.
    """
    rm=env.robot_manager;robot=rm.get_robot_by_arm_name(arm_tag)
    target_pose=np.asarray(target_pose,dtype=float).reshape(7);transform(target_pose)
    result=rm.planner[robot.robot_name].plan_path(
        curr_joint_pos=rm.get_joint(robot,[0])[0],target_ee_pose=target_pose.tolist(),
        real_robot_pose=list(robot.entity_origin_pose))
    if result.get('status')!='Success':raise RuntimeError('Official planner failed')
    q=np.asarray(result['position'],dtype=float)
    if q.ndim!=2 or q.shape[1]!=len(robot.arm_joint_indices) or not np.isfinite(q).all():raise ValueError('Bad planner shape')
    if validate_trajectory is not None and validate_trajectory(arm_tag,q,result) is not True:
        raise RuntimeError('Provided path check failed')
    result={**result,'clearance_model':'official_table_only; tiles/other_arm/held_object not certified',
            'full_scene_check_supplied':validate_trajectory is not None}
    return result

def object_goal_metrics(object_world_pose):
    """Descriptive metrics ONLY; do not call reward.step or mutate stage queues."""
    p=np.asarray(object_world_pose,dtype=float);q=p[3:]/np.linalg.norm(p[3:])
    middle=2*np.degrees(np.arccos(np.clip(abs(np.dot(q,MIDDLE_Q_WXYZ)),0,1)))
    yup=np.degrees(np.arccos(np.clip(rotation(q)[2,1],-1,1)))
    return {'middle_quat_angle_deg':float(middle),'local_Y_to_world_Z_angle_deg':float(yup),
            'final_xy_error_m':float(np.linalg.norm(p[:2]-FINAL_XY))}

def upright_candidate_from_verified_peer(peer_pose,xy):
    """Reuse SAME-ASSET observed stable peer's z+quat, no assumed table height.

    Candidate still needs support/occupancy/contact validation. The target area
    is occupied by mahjong4_0 in V003; never execute this candidate blindly.
    """
    p=np.asarray(peer_pose,dtype=float).copy().reshape(7);p[:2]=np.asarray(xy).reshape(2)
    m=object_goal_metrics(p)
    if m['local_Y_to_world_Z_angle_deg']>=7 or m['final_xy_error_m']>=.015:raise ValueError('Outside native final gates')
    return p
