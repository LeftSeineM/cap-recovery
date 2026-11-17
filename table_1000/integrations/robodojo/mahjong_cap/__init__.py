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
    # S2: continue a physically established contact, at most20mm along the
    # measured original forward direction. No virtual object-pose changes.
    measured=read_pose(motion.env,label)
    partial=7<angle(measured[3:],[0,1,0])<80 and 30<=angle(measured[3:],[0,0,1])<85
    origin=motion.tool().copy()
    if partial:
        for extension in range(10):
            before=read_pose(motion.env,label)
            if angle(before[3:],[0,0,1])<30:break
            desired=origin.copy()
            desired[:3]+=(extension+1)*.002*forward
            motion.move(desired,label+"_bounded_contact_extension",.0021,3)
            after=read_pose(motion.env,label)
            if audit:=__import__('os').environ.get('CAP_REVEAL_ARM_AUDIT'):
                row={'label':label,'native_step':int(motion.env.take_action_cnt[0]),'extension_index':extension+1,'planned_extension_m':(extension+1)*.002,'origin_link6':origin.tolist(),'desired_link6':desired.tolist(),'actual_link6':motion.tool().tolist(),'Yup':angle(after[3:],[0,1,0]),'Zup':angle(after[3:],[0,0,1]),'original_other_tile_guards_retained':True}
                with (__import__('pathlib').Path(audit).parent/'reveal_contact_extension.jsonl').open('a') as f:f.write(__import__('json').dumps(row)+'\n')
    rear=motion.tool().copy()
    rear[:3]+=-.05*forward+[0,0,.05]
    motion.move(rear,label+"_retreat",.012,6)
    if angle(read_pose(motion.env,label)[3:],[0,0,1])>=30:
        raise RuntimeError("tile not face up after retreat: "+label)

def reveal_entry_state(poses,targets):
    """Infer the existing reveal phase; retain original7deg/30deg guards.
    Matching targets may be in transition only within the bounded entry phase. No non-target is exempt.
    """
    finished=set();transition=[]
    for name,p in poses.items():
        upright=angle(p[3:],[0,1,0])<7
        faceup=angle(p[3:],[0,0,1])<30
        if name not in targets:
            if not upright:raise RuntimeError("standing hand tile tilted: "+name)
        elif faceup:finished.add(name)
        elif not upright:transition.append(name)
    # Pending targets are bounded together during entry; no arbitrary active target.
    return finished,(transition[0] if len(transition)==1 else None)

def entry_targets_resolved(poses,targets):
    return all(angle(poses[name][3:],[0,1,0])<7 or angle(poses[name][3:],[0,0,1])<30 for name in targets)

def entry_targets_settled(poses,twists,targets):
    if not entry_targets_resolved(poses,targets):return False
    for name in targets:
        v=np.asarray(twists[name],dtype=float).reshape(6)
        if not np.isfinite(v).all() or np.linalg.norm(v[:3])>=.02 or np.linalg.norm(v[3:])>=.20:return False
    return True

def reveal_tiles(env,emit_action,last_command,targets):
    """S1, finish revealing matching targets from the measured entry phase."""
    if len(targets) not in (2,3):
        raise ValueError("expected two or three matching targets")
    poses={name:read_pose(env,name) for name in HAND_LABELS}
    order=left_to_right(poses,targets)
    finished,active=reveal_entry_state(poses,set(targets))
    entry_active=active
    entry_finished=sorted(finished)
    entry_transition=sorted(name for name in targets if name not in finished and angle(poses[name][3:],[0,1,0])>=7)
    entry_pending=set(targets)-finished
    entry_phase=True
    def guard():
        for name in HAND_LABELS:
            if name==active or (entry_phase and name in entry_pending):
                continue
            q=read_pose(env,name)[3:]
            if name in finished:
                if angle(q,[0,0,1])>=30:
                    raise RuntimeError("revealed tile disturbed: "+name)
            elif angle(q,[0,1,0])>=7:
                raise RuntimeError("standing hand tile tilted: "+name)
    guard()
    remaining_for_arm=[name for name in order if name not in finished] or order
    selected_arm='left_arm' if np.mean([poses[name][0] for name in remaining_for_arm])<0 else 'right_arm'
    motion=Motion(env,emit_action,last_command,arm_tag=selected_arm)
    if job_arm_log:=__import__('os').environ.get('CAP_REVEAL_ARM_AUDIT'):
        __import__('pathlib').Path(job_arm_log).write_text(__import__('json').dumps({'selected_arm':selected_arm,'arm_selection_labels':remaining_for_arm,'target_world_positions':{k:poses[k].tolist() for k in targets},'world_contact_template_unchanged':True,'other_arm_original_command_retained':True},indent=2))
    motion.guard=guard
    # A partly toppled tile does not define a standing-contact coordinate frame.
    protected_reference=[name for name in HAND_LABELS if name not in targets and angle(poses[name][3:],[0,1,0])<7]
    upright_reference=min(protected_reference,key=lambda name:angle(poses[name][3:],[0,1,0]))
    _,_,forward=template(poses[upright_reference])
    rear=motion.tool().copy()
    rear[:3]+=-.05*forward+[0,0,.05]
    motion.move(rear,"initial_rear_up",.012,6)
    # Only pending matching targets may finish their source motion during entry.
    # Non-target upright and already-finished faceup guards remain checked per action.
    entry_wait_actions=0
    stable_streak=0
    for attempt in range(25):
        observed={name:read_pose(env,name) for name in targets}
        twists={}
        for name in targets:
            lm=env.scene_manager.layout_manager
            obj=lm.get_scene_object(env_idx=0,inst_name=lm.get_instance_name(env_idx=0,label=name))
            state=obj.get_current_dynamic_state()
            twists[name]=np.r_[arr(state.linear_velocity).reshape(3),arr(state.angular_velocity).reshape(3)]
        settled=entry_targets_settled(observed,twists,targets)
        stable_streak=stable_streak+1 if settled else 0
        if audit:=__import__('os').environ.get('CAP_REVEAL_ARM_AUDIT'):
            record={'native_step':int(env.take_action_cnt[0]),'attempt':attempt,'settled':settled,'stable_streak':stable_streak,'targets':{name:{'Yup':angle(observed[name][3:],[0,1,0]),'Zup':angle(observed[name][3:],[0,0,1]),'twist':twists[name].tolist()} for name in targets}}
            with (__import__('pathlib').Path(audit).parent/'reveal_entry_stability.jsonl').open('a') as f:f.write(__import__('json').dumps(record)+'\n')
        if stable_streak>=3:break
        if attempt==24:raise RuntimeError("entry targets not stable for3observations within24 bounded settle actions")
        motion.send("entry_target_motion_resolution")
        entry_wait_actions+=1
    for name in targets:
        if angle(read_pose(env,name)[3:],[0,0,1])<30:finished.add(name)
    active=None
    entry_phase=False
    guard()
    if audit_path:=__import__('os').environ.get('CAP_REVEAL_ARM_AUDIT'):
        audit_path=__import__('pathlib').Path(audit_path)
        (audit_path.parent/'reveal_entry_phase.json').write_text(__import__('json').dumps({'already_faceup':entry_finished,'entry_active':entry_active,'entry_transition_targets':entry_transition,'bounded_entry_wait_actions':entry_wait_actions,'entry_pending_targets':sorted(entry_pending),'after_withdrawal_faceup':sorted(finished),'native_after_withdrawal':int(env.take_action_cnt[0]),'upright_reference':upright_reference,'original_guard_thresholds_unchanged':True},indent=2))
    grip=float(motion.cmd[motion.grip_key][0])
    for i in range(1,9):
        motion.cmd[motion.grip_key]=[grip*(1-i/8)]
        motion.send("close_at_clearance")
    for active in order:
        if active in finished:continue
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
