"""Choose a short target that fits every hand's existing motion limits."""
import numpy as np
from scipy.spatial.transform import Rotation as R,Slerp
from .preflight_model import transform


def bounded_target(actual,goal,tools,references,spacing,degrees):
    distance=float(np.linalg.norm(goal[:3,3]-actual[:3,3]))
    angle=float(R.from_matrix(actual[:3,:3].T@goal[:3,:3]).magnitude())
    limit=np.deg2rad(degrees)
    nominal=min(1.,spacing/max(distance,1e-12),limit/max(angle,1e-12))
    interpolate=Slerp([0,1],R.from_matrix([actual[:3,:3],goal[:3,:3]]))
    starts={a:transform(tools[a]) for a in references}
    inverses={a:np.linalg.inv(h) for a,h in references.items()}
    def at(fraction):
        p=np.eye(4);p[:3,3]=actual[:3,3]+fraction*(goal[:3,3]-actual[:3,3]);p[:3,:3]=interpolate(fraction).as_matrix()
        return p
    def ratio(target):
        values=[np.linalg.norm(target[:3,3]-actual[:3,3])/spacing,float(R.from_matrix(target[:3,:3]@actual[:3,:3].T).magnitude())/limit]
        for arm,inverse in inverses.items():
            end=target@inverse;start=starts[arm]
            values.extend([np.linalg.norm(end[:3,3]-start[:3,3])/spacing,float(R.from_matrix(end[:3,:3]@start[:3,:3].T).magnitude())/limit])
        return float(max(values))
    fraction=nominal;mode='full_short_target'
    if ratio(at(nominal))>1.+1e-12:
        if ratio(at(0.))>1.+1e-12:
            # Preserve the existing checked subdivision when recovery of the
            # original hand pair itself requires more than one short segment.
            mode='existing_subdivision_for_pair_recovery'
        else:
            low=0.;high=nominal
            for _ in range(26):
                middle=(low+high)/2
                if ratio(at(middle))<=1.+1e-12:low=middle
                else:high=middle
            fraction=low;mode='tool_limited_short_target'
    target=at(fraction)
    return target,dict(mode=mode,nominal_fraction=nominal,selected_fraction=fraction,maximum_motion_limit_ratio=ratio(target),translation_limit_m=spacing,rotation_limit_deg=degrees)
