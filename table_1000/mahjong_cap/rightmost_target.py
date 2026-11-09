"""UNRUN CPU/runtime target calculator only. No actions and no acceptance gates.

World +X is the rightward direction of the player's front hand row in this task.
Membership is semantic (14 original hand tiles), never inferred from current X/Y.
The bonus target must be recomputed immediately before planning the actual placement.
"""
from pathlib import Path
import itertools,json,re
import numpy as np

HAND_LABELS=tuple([f'mahjong{g}_{i}' for g in range(4) for i in range(3)]+['mahjong4_0','mahjong4_1'])
BONUS='mahjong9_0'
ROOT=Path(__file__).resolve().parent

def arr(x):
    if hasattr(x,'detach'):x=x.detach().cpu().numpy()
    return np.asarray(x,dtype=float)

def rotation(q):
    q=arr(q).reshape(4);n=np.linalg.norm(q)
    if not np.isfinite(q).all() or n<1e-12:raise ValueError('invalid pose quaternion')
    w,x,y,z=q/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def rotated_offsets(q,bounds):
    lo=arr(bounds['min']).reshape(3);hi=arr(bounds['max']).reshape(3)
    if not np.isfinite(np.r_[lo,hi]).all() or np.any(hi<lo):raise ValueError('invalid collision bounds')
    corners=np.array(list(itertools.product(*zip(lo,hi))))
    return corners@rotation(q).T

def compute_target(objects,bounds_by_label,gap_m=.008):
    """Pure function: measured pose dict + asset collision bounds -> same-table goal.

    objects[label] contains pose (xyz,wxyz), and optionally instance/category_idx.
    No coordinates are clipped toward the official XY target. Its distance is
    reported only to expose any user-target/native-criterion conflict.
    """
    gap=float(gap_m)
    if not np.isfinite(gap) or gap<0:raise ValueError('invalid placement spacing')
    entries=[]
    for label in HAND_LABELS:
        p=arr(objects[label]['pose']).reshape(7)
        if not np.isfinite(p).all():raise ValueError('nonfinite measured pose')
        offsets=rotated_offsets(p[3:],bounds_by_label[label])
        up=float(np.degrees(np.arccos(np.clip(rotation(p[3:])[2,1],-1,1))))
        entries.append({'label':label,'pose':p,'right_edge_x':float(p[0]+offsets[:,0].max()),
                        'Yup_deg':up,'bottom_z':float(p[2]+offsets[:,2].min())})
    right=max(entries,key=lambda v:v['right_edge_x'])
    # Upright peer selection supplies orientation/table height; it is not a
    # success gate. If none is upright, report that limitation and use least tilt.
    upright=[v for v in entries if v['Yup_deg']<7]
    peer=min(upright,key=lambda v:np.linalg.norm(v['pose'][:2]-right['pose'][:2])) if upright else min(entries,key=lambda v:v['Yup_deg'])
    q=peer['pose'][3:]/np.linalg.norm(peer['pose'][3:])
    bonus_offsets=rotated_offsets(q,bounds_by_label[BONUS])
    goal=np.r_[right['right_edge_x']+gap-float(bonus_offsets[:,0].min()),
               right['pose'][1],peer['bottom_z']-float(bonus_offsets[:,2].min()),q]
    actual_gap=float(goal[0]+bonus_offsets[:,0].min()-right['right_edge_x'])
    official_distance=float(np.linalg.norm(goal[:2]-np.array([.319,-.15])))
    return {'schema':'measured_hand_rightmost_target_v1','hand_member_labels':list(HAND_LABELS),'bonus_label':BONUS,
            'right_direction_world':[1,0,0],'rightmost_label':right['label'],'rightmost_pose':right['pose'].tolist(),
            'rightmost_actual_right_edge_x':right['right_edge_x'],'gap_m':gap,'computed_gap_m':actual_gap,
            'bonus_world_left_offset_m':float(bonus_offsets[:,0].min()),
            'bonus_world_x_half_extent_m':float((bonus_offsets[:,0].max()-bonus_offsets[:,0].min())/2),
            'goal_pose':goal.tolist(),'upright_peer_label':peer['label'],'peer_Yup_deg':peer['Yup_deg'],
            'upright_peer_available':bool(upright),'table_z_from_measured_peer':peer['bottom_z'],
            'Y_source':'actual rightmost hand tile center; not fixed -0.15',
            'Z_source':'same measured table bottom plus actual upright bonus collision extent',
            'official_final_xy_diagnostic':{'source':'unchanged deployed task make_kong.py final stage',
              'target':[.319,-.15],'distance_m':official_distance,'threshold_m':.015,
              'outside_threshold':bool(official_distance>=.015),'changes_computed_goal':False},
            'acceptance_gate_added':False,'moves_any_hand_tile':False,
            'member_edges':[{'label':v['label'],'right_edge_x':v['right_edge_x'],'Yup_deg':v['Yup_deg']} for v in entries]}

def runtime_target(env,gap_m=.008,bounds_by_category=None):
    """Read live label binding and poses; no cached recorded pose is used here.

    Identity/job archival remains the caller's responsibility. The result carries
    current native step and actual label/instance mapping for same-job evidence.
    Unknown asset bounds are an input error; they must be supplied, not fabricated.
    """
    if bounds_by_category is None:
        bounds_by_category=json.loads((ROOT/'enabled_collision_bounds.json').read_text(encoding='utf8'))
    lm=env.scene_manager.layout_manager;objects={};bounds={};binding={}
    for label in HAND_LABELS+(BONUS,):
        instance=str(lm.get_instance_name(env_idx=0,label=label))
        match=re.fullmatch(r'mahjong_(\d+)_(\d+)',instance)
        if match is None:raise ValueError('unrecognized runtime mahjong instance: '+instance)
        category=str(int(match.group(1)))
        entity=lm.get_scene_object(env_idx=0,inst_name=instance);state=entity.get_current_dynamic_state()
        objects[label]={'pose':np.r_[arr(state.position).reshape(3),arr(state.orientation).reshape(4)].tolist(),
                        'instance':instance,'category_idx':int(category)}
        bounds[label]=bounds_by_category[category]
        binding[label]={'instance':instance,'category_idx':int(category),'bounds_sha256':bounds[label].get('sha256')}
    result=compute_target(objects,bounds,gap_m)
    result['native_step_at_recompute']=int(env.take_action_cnt[0]);result['runtime_label_binding']=binding
    return result
