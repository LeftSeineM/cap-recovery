"""Endpoint feasibility and hand envelopes for future pickups.

These are destination-selection hypotheses only. Ordinary S1 must still check
the full approach, closure, lift and transport using the actual later state.
"""
import copy
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from .planner import grasp_candidates
from .preflight_model import Model,PreflightFailure
from .skill_contract import SkillInapplicable


def future_pickups(state,calibration,scene,table,targets,template,relocated=()):
    future_state=copy.deepcopy(state)
    future_scene={k:v for k,v in scene.items() if k not in relocated}
    future_state['objects']={k:v for k,v in state['objects'].items() if k in future_scene}
    future_state['grips']={a:1. for a in state['qs']}
    future_state.pop('actual_gripper_joints',None)
    root=Model(future_state,calibration,future_scene,table)
    pickups=[];evidence=[]
    for target in targets:
        key=target['object'];arm=target['arms'][0];alternatives=[];accepted=[];rejected=[]
        for grasp in grasp_candidates(state['objects'][key]['pose'],scene[key]['vertices'],[arm],template):
            m=root.fork();contact=grasp['contacts'][arm];pre=contact.copy();pre[2,3]+=.045;endpoints=[]
            try:
                for stage,goal in (('pregrasp',pre),('contact',contact)):
                    local=np.linalg.inv(m.base[arm])@goal
                    def error(q):
                        fk=m.fk(q)
                        return np.r_[5*(fk[:3,3]-local[:3,3]),R.from_matrix(fk[:3,:3]@local[:3,:3].T).as_rotvec()]
                    result=least_squares(error,m.q[arm],bounds=([-10]*5+[-3.14],[10]*5+[3.14]),max_nfev=120,ftol=1e-9,xtol=1e-9,gtol=1e-9)
                    if np.linalg.norm(error(result.x))>.001:raise PreflightFailure(stage+' endpoint unreachable')
                    m.q[arm]=result.x;m.stage=stage;m.check(m.q,m.grips,key,stage=='contact')
                    endpoints.append(result.x.copy())
                meshes=[]
                for part in calibration['geometry']['gripper']['parts']:
                    meshes.append(np.vstack([m.parts(arm,q,opening)[part] for q in endpoints for opening in (1.,.34)]).tolist())
                alternatives.append(meshes);accepted.append(dict(yaw_deg=grasp['yaw'],height_m=grasp['grasp_height_m']))
            except PreflightFailure as ex:rejected.append(dict(yaw_deg=grasp['yaw'],height_m=grasp['grasp_height_m'],reason=str(ex)))
        if not alternatives:raise SkillInapplicable('no endpoint-feasible future pickup: '+str(dict(object=key,rejected=rejected)))
        pickups.append(dict(object=key,alternatives=alternatives))
        evidence.append(dict(object=key,accepted=accepted,rejected=rejected,relocated_source_objects=list(relocated),scope='Hypothetical empty-hand endpoint IK/collision checks and open-to-closing hand envelope; not a path or physical grasp certificate'))
    return pickups,evidence
