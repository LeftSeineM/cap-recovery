"""An expired item cannot start another candidate or a full new native solve."""
from types import SimpleNamespace

import numpy as np
import pytest

from table_1000.experts import context
from table_1000.experts.context import WallBudgetExceeded
from table_1000.experts.audit.problems import Problems, decision
from table_1000.experts.solver import MotionSolver


@pytest.mark.parametrize('angle_probe', [False, True])
def test_budget_expires_inside_contact_candidate_screening(monkeypatch, angle_probe):
    from table_1000.experts.skills.hold import frames as grasps
    from tests.experts.test_grasp_geometry_cache import hand
    now = [0.]
    monkeypatch.setattr(context, 'time', SimpleNamespace(perf_counter=lambda: now[0]))
    ctx = SimpleNamespace(current='pick', wall_deadline=1.)
    screened = []

    def blocked(*args, **kwargs):
        screened.append(1)
        now[0] = 2.
        return None

    monkeypatch.setattr(grasps, '_placed', blocked)
    candidates = grasps.grasp_candidates(grasps.Box((.02, .02, .04)), np.eye(4), hand(), [],
                                        (0., 0., -1.), step=np.pi / 2,
                                        _first_angle=angle_probe,
                                        check_budget=lambda: context.remaining_wall(ctx))
    with pytest.raises(WallBudgetExceeded):
        next(candidates)
    assert screened == [1]


def test_budget_expires_during_occupancy_ranking_before_contact_queries(monkeypatch):
    from table_1000.experts.skills.hold import frames as grasps
    from tests.experts.test_grasp_geometry_cache import hand
    now = [0.]
    monkeypatch.setattr(context, 'time', SimpleNamespace(perf_counter=lambda: now[0]))
    ctx = SimpleNamespace(current='pick', wall_deadline=1.)
    ranked = []

    def occupancy(*args):
        ranked.append(1)
        now[0] = 2.
        return 0.

    monkeypatch.setattr(grasps, '_placed', lambda *a, **kw: pytest.fail('contact query after expiry'))
    candidates = grasps.grasp_candidates(grasps.Box((.02, .02, .04)), np.eye(4), hand(), [],
                                        (0., 0., -1.), step=np.pi / 2, occupancy_cost=occupancy,
                                        check_budget=lambda: context.remaining_wall(ctx))
    with pytest.raises(WallBudgetExceeded):
        next(candidates)
    assert ranked == [1]


def test_base_screening_stops_between_native_branches(monkeypatch):
    from table_1000.experts.skills.base import Base
    now = [0.]
    monkeypatch.setattr(context, 'time', SimpleNamespace(perf_counter=lambda: now[0]))
    skill = Base()
    skill.ctx = SimpleNamespace(current='base', wall_deadline=1.)
    solved = []

    def screw(*args, **kwargs):
        solved.append(1)
        now[0] = 2.
        return {'status': 'collision'}

    skill.model = SimpleNamespace(q=np.zeros(1), limits=np.array([[-2., 2.]]),
                                  arm_indices=lambda arm: [0],
                                  ik=lambda *a, **kw: ('Success', [np.zeros(1), np.zeros(1)]),
                                  plan_screw=screw)
    with pytest.raises(WallBudgetExceeded):
        skill.path_margin('left', [np.eye(4), np.eye(4)])
    assert solved == [1]


def test_placement_screening_stops_between_release_branches(monkeypatch):
    from table_1000.experts.skills.place.strict import Placing
    now = [0.]
    monkeypatch.setattr(context, 'time', SimpleNamespace(perf_counter=lambda: now[0]))
    skill = Placing()
    skill.ctx = SimpleNamespace(current='place', wall_deadline=1., trace=None)
    solved = []

    def ik(*args, **kwargs):
        solved.append(1)
        if len(solved) == 1:
            return 'Success', [np.zeros(1), np.zeros(1)]
        now[0] = 2.
        return 'IK Failed', None

    skill.model = SimpleNamespace(q=np.zeros(1), ik=ik)
    with pytest.raises(WallBudgetExceeded):
        skill.down_goals('left', np.eye(4), .08, True, 5)
    assert len(solved) == 2


def test_insertion_channel_expiry_does_not_cache_partial_geometry(monkeypatch):
    from table_1000.experts.skills.place import strict as placing
    now = [0.]
    monkeypatch.setattr(context, 'time', SimpleNamespace(perf_counter=lambda: now[0]))
    skill = placing.Placing()
    skill.ctx = SimpleNamespace(current='place', wall_deadline=1.)
    skill.hands = {'left': SimpleNamespace(frames=lambda gap: {'finger': np.eye(4)},
                                         shapes={'finger': [object(), object()]})}
    built = []

    def vertices(shape):
        built.append(shape)
        now[0] = 2.
        return np.array([[0., 0., 0.], [1., 1., 1.]])

    monkeypatch.setattr(placing, '_vertices', vertices)
    monkeypatch.setattr(placing, 'convex_keepout', lambda points: object())
    with pytest.raises(WallBudgetExceeded):
        skill.insertion_blockers('left', np.eye(4), .08, .03, [])
    assert len(built) == 1
    assert not skill._insertion_channels


def test_flat_opening_stops_between_fcl_queries_without_caching_a_verdict(monkeypatch):
    from table_1000.experts.skills.place import flat as flat_release, strict as placing
    from tests.experts.test_flat_release import box, hand, obstacle
    now = [0.]
    monkeypatch.setattr(context, 'time', SimpleNamespace(perf_counter=lambda: now[0]))
    skill = placing.Placing()
    skill.ctx = SimpleNamespace(current='place', wall_deadline=1.)
    wall = obstacle('wall', box((.2, .2, .2)))
    geometry = flat_release.FlatReleaseGeometry([wall, wall], .08)
    checked = []

    def collide(*args):
        checked.append(1)
        now[0] = 2.
        return SimpleNamespace(is_collision=lambda: False)

    monkeypatch.setattr(flat_release.fcl, 'collide', collide)
    with pytest.raises(WallBudgetExceeded):
        skill.flat_release_height(geometry, hand(), np.eye(4), np.eye(4), 0., .04, .08)
    assert checked == [1]
    assert not geometry.heights


@pytest.mark.parametrize('recorded', [False, True])
def test_expired_deadline_stops_before_the_next_candidate_and_closes_it(monkeypatch, recorded):
    now = [0.]
    monkeypatch.setattr(context, 'time', SimpleNamespace(perf_counter=lambda: now[0]))
    ctx = SimpleNamespace(current='pick', wall_deadline=1., spec=SimpleNamespace(name='fake'),
                          trace=SimpleNamespace(stack=[0], problems=None))
    if recorded:
        ctx.trace.problems = Problems(ctx)
    visited = []
    class Skill:
        def __init__(self):
            self.ctx = ctx
        @decision
        def candidates(self):
            try:
                for number in range(3):
                    visited.append(number)
                    yield number
            finally:
                visited.append('closed')
    iterator = Skill().candidates()
    assert next(iterator) == 0
    now[0] = 2.
    with pytest.raises(WallBudgetExceeded):
        next(iterator)
    assert visited == [0, 'closed']


def test_native_rrt_receives_only_the_remaining_item_budget(monkeypatch):
    monkeypatch.setattr(context, 'time', SimpleNamespace(perf_counter=lambda: 10.))
    solver = MotionSolver.__new__(MotionSolver)
    solver.ctx = SimpleNamespace(current='place', wall_deadline=10.25)
    solver.lock_lift = False
    solver.arm = SimpleNamespace(name='left')
    calls = []
    solver.model = SimpleNamespace(plan_rrt=lambda *a, **kw: calls.append(kw) or {'status': 'Success'})
    solver._validated = lambda result: result
    solver._plan = lambda kind, function, **kw: function()
    solver.check_geometry = None
    solver.follow = lambda result: None
    solver._rrt(None, 3., 3, False, [])
    assert calls[0]['planning_time'] == pytest.approx(.25)
    solver.ctx.wall_deadline = 9.
    with pytest.raises(WallBudgetExceeded):
        solver._rrt(None, 3., 3, False, [])
    assert len(calls) == 1
