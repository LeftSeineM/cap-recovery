"""Measured collision geometry is separate from the immutable hold certificate.

Never updates the Runtime capture reference or relaxes its original limits.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from .preflight_model import transform, digest, PreflightFailure


def reference_digest(references):
    return digest({a:np.asarray(h,dtype=float).tolist() for a,h in references.items()})


def observed_geometry_references(state,object_id,arms):
    obj=transform(state['objects'][object_id]['pose'])
    actual={a:np.linalg.inv(transform(state['tools'][a]))@obj for a in arms}
    if not all(np.isfinite(h).all() for h in actual.values()):raise PreflightFailure('nonfinite measured object geometry')
    return actual


def measured_geometry_references(state,object_id,guard_references):
    if not guard_references:raise PreflightFailure('held geometry requires original capture references')
    actual=observed_geometry_references(state,object_id,guard_references)
    for arm,baseline in guard_references.items():
        baseline=np.asarray(baseline,dtype=float).reshape(4,4)
        observed=actual[arm]
        if not np.isfinite(observed).all() or not np.isfinite(baseline).all():raise PreflightFailure('nonfinite held geometry/reference')
        dp=float(np.linalg.norm(observed[:3,3]-baseline[:3,3]))
        dq=float(np.rad2deg(R.from_matrix(observed[:3,:3]@baseline[:3,:3].T).magnitude()))
        if dp>.012 or dq>12:raise PreflightFailure('immutable held reference outside original limits; no recapture')
        actual[arm]=observed
    return actual
