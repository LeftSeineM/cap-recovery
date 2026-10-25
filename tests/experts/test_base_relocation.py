"""A relocation after a failed approach must not accept the same stance again."""
from types import SimpleNamespace
import numpy as np
from table_1000.experts.skills.base import Base


def choosing(reachable=True):
    skill = Base()
    skill.model = SimpleNamespace(q=np.zeros(1), base=np.eye(4), arm_indices=lambda arm: np.array([0]))
    skill.ctx = SimpleNamespace(rng=np.random.default_rng(7), poses=lambda: {}, trace=None,
                                note=lambda *args, **kwargs: None)
    skill.arms, skill.held, skill.home, skill.solvers = ['arm'], {}, np.zeros(1), {}
    skill.base_pose = lambda: np.zeros(3)
    skill.put_base = lambda *args: None
    skill.in_reach = lambda *args: reachable
    skill.clear = lambda q: True
    skill.reached = lambda *args: (1, 1.)
    return skill


def test_relocation_excludes_a_stance_whose_endpoints_still_have_ik():
    current = choosing().base_for([np.eye(4)], need=1)
    moved = choosing().base_for([np.eye(4)], need=1, relocate=True)
    np.testing.assert_array_equal(current, np.zeros(3))
    assert np.linalg.norm(moved[:2] - current[:2]) >= .3


def test_no_alternative_does_not_fall_back_to_the_failed_current_stance():
    assert choosing(reachable=False).base_for([np.eye(4)], need=1, relocate=True) is None


def test_decision_record_explains_why_the_current_candidate_was_excluded(monkeypatch):
    events, candidate = [], None
    def mark(skill, stage, value=None, outcome=None, **values):
        nonlocal candidate
        if stage == 'candidate':
            candidate = value
        events.append((candidate, stage, outcome))
    monkeypatch.setattr('table_1000.experts.skills.base.mark', mark)
    choosing().base_for([np.eye(4)], need=1, relocate=True)
    assert (0, 'previous stance', 'rejected') in events
