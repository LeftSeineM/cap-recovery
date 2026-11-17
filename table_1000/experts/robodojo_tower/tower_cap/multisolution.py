"""Review candidate only. Runs GPU IK ONLY when explicitly called by execution owner.
Integrate in NEW skill adapter after exact existing world/base/frame_bias conversion.
Do not modify official CuroboPlanner or create another solver instance.
"""
import numpy as np

def choose_continuous_candidate(planner, goal_tool_poses, current_state,
                                current_q, nearest_legal, limits,
                                diagnose=lambda row: None):
    current_q = np.asarray(current_q, dtype=float).reshape(-1)
    # Current implementation constructs 32 optimization seeds. Keeping <=32
    # avoids changing that optimization size; no constructor/warmup here.
    n = int(planner.ik_solver.config.num_seeds)
    if n != 32:
        raise ValueError('Re-audit runtime solver seed count before using adapter')
    result = planner.ik_solver.solve_pose(goal_tool_poses=goal_tool_poses,
                                          current_state=current_state,
                                          return_seeds=32)
    if result.js_solution is None or result.success is None:
        raise RuntimeError('IK failed: no candidate result')
    def cpu(value):
        return value.detach().cpu().numpy()
    values = cpu(result.js_solution.position)
    success = cpu(result.success).astype(bool)
    # Batch=1, candidates=32; retain candidate axis. No silent flattening.
    if values.ndim != 3 or values.shape[:2] != (1,32) or success.shape != (1,32):
        raise ValueError('Unexpected multi-solution shape; inspect runtime output')
    names = list(result.js_solution.joint_names or [])
    if len(names) != values.shape[-1] or len(set(names)) != len(names):
        raise ValueError('Missing/ambiguous solution joint names')
    indices = [names.index(name) for name in planner.active_joints_name]
    if len(indices) != len(current_q):
        raise ValueError('Current and active joint order/shape mismatch')
    feasible = getattr(result, 'feasible', None)
    if feasible is not None:
        feasible = cpu(feasible).astype(bool)
        if feasible.shape != success.shape:
            raise ValueError('Unexpected feasible mask shape')
        success &= feasible
    choices = []
    for i in range(32):
        if not success[0,i]:
            continue
        raw = values[0,i,indices]
        try:
            q, info = nearest_legal(raw, current_q, *limits)
        except ValueError as exc:
            diagnose(dict(candidate=i, accepted=False, reason=str(exc)))
            continue
        delta = q-current_q
        maximum = float(np.max(np.abs(delta)))
        diagnose(dict(candidate=i, accepted=maximum<=.20, raw=raw.tolist(),
                      normalized=q.tolist(), max_delta=maximum, detail=info))
        if maximum <= .20:
            choices.append((maximum, float(np.linalg.norm(delta)), i, q))
    if not choices:
        # Let owner integrate this result with bounded Cartesian subdivision.
        raise RuntimeError('joint discontinuity: no successful legal continuous candidate')
    choices.sort(key=lambda row: row[:3])
    return choices[0][3]

def continuous_world_ik(rm, robot, desired, current, limits, diagnose):
    from copy import deepcopy
    from utils.transformer import calculate_target_pose
    planner=rm.ik_solver[robot.robot_name]
    target=calculate_target_pose(deepcopy(robot.entity_origin_pose),planner.robot_origin_pose,deepcopy(desired.tolist()))
    p,q=planner._trans_from_world_to_base(np.array(planner.robot_origin_pose,dtype=np.float32),np.array(target,dtype=np.float32))
    p=np.asarray(p).copy();p+=np.asarray(planner.frame_bias)
    goal=planner._build_goal_pose(p,q);state=planner._build_joint_state(current)
    from .legal_joint_representative import nearest_legal
    return choose_continuous_candidate(planner,goal,state,current,nearest_legal,limits,diagnose)
