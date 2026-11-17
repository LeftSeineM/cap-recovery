"""Bounded object-pose feedback without resetting the hold-integrity baseline.

The original captured transforms stay authoritative for every hold guard.
Measured transforms are only short-horizon collision/control state estimates.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from .preflight_model import Model, transform, pose, digest, PreflightFailure
from .measured_geometry import measured_geometry_references, reference_digest
from .pair_control import control_references
from .feedback_target import bounded_target
from .self_clearance import check_path_self_margin
from .skill_contract import SkillInapplicable


def control_steps(request):
    translation=float(request.get('feedback_translation_step_m',.006));rotation=float(request.get('feedback_rotation_step_deg',2.))
    if translation not in (.006,.012) or rotation not in (2.,4.):raise SkillInapplicable('feedback step outside finite admitted family')
    return translation,rotation


def waypoint_complete(state,scene,table,request,actual,goal,stage,distance,angle):
    if len(request['arms'])==1:
        return bool(distance<=(.00075 if stage=='support_lower' else .001) and angle<=np.deg2rad(.5 if stage=='support_lower' else 2.))
    if stage=='held_vertical_clearance':
        # This waypoint establishes height. Fresh geometry checks and the next
        # lateral plan use the actual XY, including small tracking residuals.
        return bool(actual[2,3]>=goal[2,3]-.001 and angle<=np.deg2rad(2.))
    if stage=='held_transport':return bool(distance<=.002 and angle<=np.deg2rad(1.))
    if stage!='support_lower':raise SkillInapplicable('unknown feedback waypoint')
    from .support_release import support_report
    supports=[dict(id=k,pose=state['objects'][k]['pose'],vertices=scene[k]['vertices']) for k in request.get('protect',request.get('supports',[]))]
    report=support_report(state['objects'][request['object']]['pose'],scene[request['object']]['vertices'],supports,request.get('required_count',len(supports)),table if request.get('allow_table') else None)
    return bool(np.linalg.norm(actual[:2,3]-goal[:2,3])<=.002 and angle<=np.deg2rad(1.) and report['supported'])


def route_candidates(variants,single):
    # Keep the complete finite family, trying every nominal goal before
    # spending work on optional source-staging fallback routes.
    for yaw,goal,error in variants:
        for clearance in (.009,.012,.015,.003,.03,.045,.06):
            yield yaw,goal,error,0.,clearance
    if single:
        for yaw,goal,error in variants:
            # Preserve headroom over the protected structure before preferring
            # the shortest source-side staging distance. Every route is checked.
            for clearance in (.009,.015,.003):
                for offset in (.03,.06,.09,.12):
                    yield yaw,goal,error,offset,clearance


def plan_observed_transport(state,calibration,scene,table,request,source_sha,guard_references=None,include_lift=False,context=None):
    from .planner import task_goal_variants, rigid_move
    object_id=request['object'];arms=request['arms'];transport_spacing,transport_rotation=control_steps(request)
    if len(arms) not in (1,2) or len(set(arms))!=len(arms):raise SkillInapplicable('observed transport needs one or two distinct holding arms')
    original={a:np.asarray(h).copy() for a,h in (guard_references or {a:np.linalg.inv(transform(state['tools'][a]))@transform(state['objects'][object_id]['pose']) for a in arms}).items()}
    observed=measured_geometry_references(state,object_id,original);measured=control_references(observed,original)
    root=Model(state,calibration,scene,table,context=context);root.held=object_id;root.refs=measured;root.support_ids=tuple(request.get('protect',request.get('supports',[])))
    variants=task_goal_variants(request,scene)
    anchor=arms[0];variants.sort(key=lambda row:float(R.from_matrix(root.tools[anchor][:3,:3].T@(row[1]@np.linalg.inv(measured[anchor]))[:3,:3]).magnitude()))
    failures=[]
    for yaw,goal,error,offset,clearance in route_candidates(variants,len(arms)==1):
        m=root.fork()
        try:
            if include_lift:
                lifted=m.objects[object_id].copy();lifted[2,3]+=.03
                rigid_move(m,lifted,'physical_hold_lift',.004,3)
            now=m.tools[anchor]@m.refs[anchor]
            staged=now.copy()
            delta=goal[:2,3]-now[:2,3];distance=float(np.linalg.norm(delta))
            if offset and distance>1e-9:staged[:2,3]+=delta/distance*min(offset,distance)
            high=staged.copy();high[2,3]=max(now[2,3],goal[2,3]+clearance)
            above=goal.copy();above[2,3]=high[2,3]
            contact_approach=goal.copy();contact_approach[2,3]+=.001
            # The unchanged release rule permits a supported gap up to3mm.
            # Approach1mm above the ideal plane, then release/settle naturally.
            waypoints=([('held_clearance_escape',staged)] if offset else [])+[('held_vertical_clearance',high),('held_transport',above),('support_lower',contact_approach)]
            reserve=check_path_self_margin(m,.0005) if offset else None
            reserve_samples=reserve['checked_half_and_full_samples'] if reserve else 0
            for stage,target in waypoints:
                before=len(m.steps);initial_q={a:q.copy() for a,q in m.q.items()}
                rigid_move(m,target,stage,transport_spacing if stage!='support_lower' else .002,transport_rotation)
                if offset:
                    segment=check_path_self_margin(m,.0005,path=m.steps[before:],initial_q=initial_q)
                    reserve_samples+=segment['checked_half_and_full_samples']
            if offset:
                assert reserve_samples==2*sum(s['kind']=='move' for s in m.steps)
                reserve=dict(self_margin_m=.0005,checked_half_and_full_samples=reserve_samples,original_S0_unchanged=True)
            cert=m.certificate(request,source_sha)
            cert.update(object=object_id,arms=arms,guard_reference_sha256=reference_digest(original),measured_geometry_reference_sha256=reference_digest(observed),control_geometry_reference_sha256=reference_digest(measured),control_geometry_references={a:h.tolist() for a,h in measured.items()},control_mode='original_hand_pair_anchor_feedback',capture_reference_updated=False)
            return dict(certificate=cert,waypoints=[dict(stage=s,pose=pose(p).tolist()) for s,p in waypoints],selected_goal=pose(goal).tolist(),clearance_m=clearance,source_inward_staging_m=offset,self_clearance_reserve=reserve,goal_yaw_deg=yaw,goal_geometry_error_m=error,rejected=failures)
        except PreflightFailure as ex:failures.append(dict(yaw=yaw,clearance_m=clearance,source_inward_staging_m=offset,reason=str(ex)))
    raise SkillInapplicable('no checked observed-geometry transport route: '+str(failures))


def execute_observed_transport(rt,request,call):
    from .planner import rigid_move
    object_id=request['object'];arms=request['arms'];transport_spacing,transport_rotation=control_steps(request);original={a:h.copy() for a,h in rt.references.items()};baseline_hash=reference_digest(original)
    rt.guard();admission=plan_observed_transport(rt.snapshot(),rt.calibration,rt.scene,rt.table,request,rt.sha,original,False,rt.context)
    path=rt.out/('feedback_route_'+call.row['call_id']+'_'+str(int(rt.env.take_action_cnt[0]))+'.json')
    import json
    path.write_text(json.dumps(admission,indent=2));call.row['evidence'].append(str(path))
    start=int(rt.env.take_action_cnt[0]);iterations=0
    for waypoint in admission['waypoints']:
        goal=transform(waypoint['pose']);stage=waypoint['stage'];costs=[]
        while True:
            rt.guard()
            if reference_digest(rt.references)!=baseline_hash:raise RuntimeError('feedback changed the original capture reference')
            state=rt.snapshot();actual=transform(state['objects'][object_id]['pose'])
            distance=float(np.linalg.norm(actual[:3,3]-goal[:3,3]));angle=float(R.from_matrix(actual[:3,:3].T@goal[:3,:3]).magnitude())
            if waypoint_complete(state,rt.scene,rt.table,request,actual,goal,stage,distance,angle):break
            if iterations>=200 or int(rt.env.take_action_cnt[0])>=int(rt.env.step_lim):raise SkillInapplicable('observed transport exhausted bounded actions/native budget')
            costs.append(distance+.03*angle)
            if len(costs)>=7 and costs[-7]-min(costs[-3:])<.0001:raise SkillInapplicable('observed object approach made no measurable progress')
            observed=measured_geometry_references(state,object_id,original);measured=control_references(observed,original)
            target,lookahead=bounded_target(actual,goal,state['tools'],measured,transport_spacing if stage!='support_lower' else .002,transport_rotation)
            m=Model(state,rt.calibration,rt.scene,rt.table,context=rt.context);m.held=object_id;m.refs=measured;m.support_ids=tuple(request.get('protect',request.get('supports',[])))
            rigid_move(m,target,stage,transport_spacing if stage!='support_lower' else .002,transport_rotation)
            if admission['source_inward_staging_m']:check_path_self_margin(m,.0005)
            cert=m.certificate(request,rt.sha)
            lookahead['checked_model_substeps']=len(cert['path'])
            # Only execute one checked command, then re-observe before planning again.
            cert['path']=cert['path'][:1];cert['path_sha256']=digest(cert['path']);cert['actions']=len(cert['path'])
            cert.update(object=object_id,arms=arms,guard_reference_sha256=baseline_hash,measured_geometry_reference_sha256=reference_digest(observed),control_geometry_reference_sha256=reference_digest(measured),control_geometry_references={a:h.tolist() for a,h in measured.items()},control_mode='original_hand_pair_anchor_feedback',capture_reference_updated=False)
            if not cert['path']:raise SkillInapplicable('observed transport generated no command')
            before=int(rt.env.take_action_cnt[0]);rt.execute(cert,call);iterations+=1
            if int(rt.env.take_action_cnt[0])<=before:raise SkillInapplicable('observed transport S0 did not advance')
            rt.record(stage='observed_object_feedback',phase=stage,guard_reference_sha256=baseline_hash,capture_reference_updated=False,pre_action_position_error_m=distance,pre_action_angle_error_deg=float(np.rad2deg(angle)),short_target=lookahead)
    rt.guard()
    if reference_digest(rt.references)!=baseline_hash:raise RuntimeError('feedback capture baseline changed')
    return dict(selected_goal=admission['selected_goal'],feedback_iterations=iterations,actual_native_actions=int(rt.env.take_action_cnt[0])-start,guard_reference_sha256=baseline_hash,capture_reference_updated=False,clearance_m=admission['clearance_m'],source_inward_staging_m=admission['source_inward_staging_m'])
