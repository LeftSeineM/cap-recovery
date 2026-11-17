"""Extra S1 clearance eligibility for optional source-staging routes.

Original Model.check remains authoritative and unmodified. This replay only
adds a conservative body-self clearance test to already checked paths.
"""
import numpy as np
from .preflight_model import overlap,PreflightFailure


def check_path_self_margin(model,margin,path=None,joint_bias=None,initial_q=None):
    if not 0<=float(margin)<=.001:raise ValueError('finite self-clearance reserve required')
    corners=np.array([[x,y,z] for x in (-margin,margin) for y in (-margin,margin) for z in (-margin,margin)])
    previous={a:np.array(q).copy() for a,q in (model.input['state']['qs'] if initial_q is None else initial_q).items()};samples=0
    if set(previous)!=set(model.q):raise ValueError('clearance replay arm identity mismatch')
    for index,step in enumerate(model.steps if path is None else path):
        if step['kind']!='move':continue
        goal={a:np.array(step['q'].get(a,q)) for a,q in previous.items()}
        for fraction in (.5,1.):
            for arm in goal:
                q=previous[arm]*(1-fraction)+goal[arm]*fraction
                if joint_bias is not None:q=q+np.array(joint_bias.get(arm,np.zeros_like(q)))
                parts=model.parts(arm,q,model.grips[arm])
                for i in range(1,7):
                    inflated=(parts['arm'+str(i)][:,None,:]+corners[None,:,:]).reshape(-1,3)
                    for j in range(i+2,7):
                        penetration=overlap(inflated,parts['arm'+str(j)])
                        if penetration>1e-6:raise PreflightFailure('S1 self-clearance reserve: '+str(dict(arm=arm,parts=[i,j],margin_m=margin,path_index=index,fraction=fraction,inflated_overlap_m=penetration)))
            samples+=1
        previous=goal
    return dict(self_margin_m=float(margin),checked_half_and_full_samples=samples,original_S0_unchanged=True)
