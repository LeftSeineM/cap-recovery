"""Keep original inter-hand geometry while correcting a shared object target.

Observations estimate anchor-object error; they never redefine the hand span or
the independent original hold-integrity references.
"""
import numpy as np


def control_references(observed,original):
    if set(observed)!=set(original) or len(original) not in (1,2):raise RuntimeError('invalid observed/original holding arms')
    anchor=next(iter(original));result={anchor:np.asarray(observed[anchor]).copy()}
    for arm in original:
        if arm!=anchor:result[arm]=np.asarray(original[arm])@np.linalg.inv(original[anchor])@result[anchor]
    return result
