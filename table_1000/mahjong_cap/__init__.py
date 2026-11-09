"""CAP Mahjong v1.0. One player row, +X right, archived right arm contact style."""
import numpy as np
from .rightmost_target import HAND_LABELS, arr, runtime_target as rightmost_target
from .style import angle, template
from .motion import Motion, OfficialSuccess

def read_pose(env, label):
    lm=env.scene_manager.layout_manager
    name=lm.get_instance_name(env_idx=0,label=label)
    obj=lm.get_scene_object(env_idx=0,inst_name=name)
    s=obj.get_current_dynamic_state()
    p=np.r_[arr(s.position).reshape(3),arr(s.orientation).reshape(4)]
    if not np.isfinite(p).all():
        raise ValueError("invalid pose: "+label)
    return p

def left_to_right(poses, targets):
    targets=list(targets)
    if len(set(targets))!=len(targets) or not set(targets)<=set(HAND_LABELS):
        raise ValueError("expected distinct player hand tiles")
    return sorted(targets,key=lambda name:poses[name][0])

def topple_tile(motion, label):
    """One standing tile. Caller prepares closed hand at clearance and installs guard."""
    p=read_pose(motion.env,label)
    if angle(p[3:],[0,1,0])>=7:
        raise ValueError("expected upright tile: "+label)
    poses,pre,forward=template(p)
    high=pre.copy()
    high[2]=max(motion.tool()[2],pre[2]+.04)
    motion.move(high,label+"_high",.012,6)
    motion.move(pre,label+"_lower",.008)
    motion.move(pre,label+"_precontact")
    for waypoint in poses:
        motion.move(waypoint,label+"_contact")
    rear=motion.tool().copy()
    rear[:3]+=-.05*forward+[0,0,.05]
    motion.move(rear,label+"_retreat",.012,6)
    if angle(read_pose(motion.env,label)[3:],[0,0,1])>=30:
        raise RuntimeError("tile not face up after retreat: "+label)

def reveal_tiles(env,emit_action,last_command,targets):
    """S1, expose 2 or 3 standing tiles from left to right. Raises on failed guard."""
    if len(targets) not in (2,3):
        raise ValueError("expected two or three matching targets")
    poses={name:read_pose(env,name) for name in HAND_LABELS}
    order=left_to_right(poses,targets)
    finished=set()
    active=None
    def guard():
        for name in HAND_LABELS:
            if name==active:
                continue
            q=read_pose(env,name)[3:]
            if name in finished:
                if angle(q,[0,0,1])>=30:
                    raise RuntimeError("revealed tile disturbed: "+name)
            elif angle(q,[0,1,0])>=7:
                raise RuntimeError("standing hand tile tilted: "+name)
    guard()
    motion=Motion(env,emit_action,last_command)
    motion.guard=guard
    _,_,forward=template(poses[order[0]])
    rear=motion.tool().copy()
    rear[:3]+=-.05*forward+[0,0,.05]
    motion.move(rear,"initial_rear_up",.012,6)
    grip=float(motion.cmd[motion.grip_key][0])
    for i in range(1,9):
        motion.cmd[motion.grip_key]=[grip*(1-i/8)]
        motion.send("close_at_clearance")
    for active in order:
        topple_tile(motion,active)
        finished.add(active)
    active=None
    for _ in range(3):
        motion.send("faceup_hold")
    return {"status":"revealed","order":order,"last_command":motion.cmd}

def expert(env,emit_action,last_command,kind,*,targets=None,job_output=None,bonus=None):
    """S2. For pong pass two resolved matching labels. Kong uses the task's env.kong."""
    if kind=="kong":
        if bonus is None and job_output is None:
            raise ValueError("kong requires native job_output")
        resolved=[str(row[0]) for row in env.kong]
    elif kind=="pong" and targets is not None and len(targets)==2:
        resolved=list(targets)
    else:
        raise ValueError("provide kong or pong with two matching labels")
    try:
        shown=reveal_tiles(env,emit_action,last_command,resolved)
    except OfficialSuccess:
        return {"status":"official_native_success"}
    if kind=="pong":
        return shown
    if bonus is not None:
        return bonus(env,emit_action,shown)
    from .bonus_suffix import run
    return run(env,emit_action,job_output)

from .place import run_place as place_rightmost
