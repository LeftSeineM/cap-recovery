"""Shortcuts preserve endpoints and only replace fully checked edges."""

from types import SimpleNamespace

import numpy as np

from table_1000.planning.shortcut import BoundedPlanner, shortcut


def assert_contact_path(path, pairs):
    """Use the execution validator as an independent contact-policy oracle."""
    from table_1000.robots.kinematics import RobotModel
    state = {}
    world = SimpleNamespace(
        set_qpos_all=lambda q: state.update(q=q.copy()),
        check_collision=lambda: [SimpleNamespace(link_name1=a, object_name2=b, link_name2=c)
                                 for a, b, c in pairs(state['q'])])
    model = SimpleNamespace(q=np.zeros(path.shape[1]), lift_indices=lambda: [],
                            planners={'arm': SimpleNamespace(planning_world=world,
                                       robot=SimpleNamespace(set_qpos=lambda *a: None))})
    _, collisions = RobotModel.check_path(model, 'arm', path, step=.001)
    assert not collisions


def test_shortcut_removes_a_detour_and_keeps_fixed_joints():
    path = np.array([[.2, 0., 0.], [.2, 1., 1.], [.2, 2., 0.]])
    checked = []
    result = shortcut(path, lambda q: checked.append(q.copy()) or True, budget=10.)
    np.testing.assert_array_equal(result, path[[0, -1]])
    ordered = sorted(checked, key=lambda q: q[1])
    assert max(np.max(np.abs(b - a)) for a, b in zip(ordered, ordered[1:])) <= .0100001
    assert all(q[0] == .2 for q in checked)


def test_shortcut_keeps_a_route_around_an_obstacle():
    path = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    result = shortcut(path, lambda q: not (.2 < q[0] < .8 and q[1] < .8), budget=10.)
    np.testing.assert_array_equal(result[[0, -1]], path[[0, -1]])
    for start, end in zip(result, result[1:]):
        for t in np.linspace(0., 1., 501):
            q = start + t * (end - start)
            assert not (.2 < q[0] < .8 and q[1] < .8)


def test_deadline_during_edge_does_not_accept_an_unchecked_shortcut():
    now = [0.]
    path = [[0., 0.], [0., 1.], [1., 1.]]

    def valid(q):
        now[0] += .02
        return True

    result = shortcut(path, valid, budget=.05, clock=lambda: now[0])
    np.testing.assert_array_equal(result, path)


def test_approximate_solution_never_checks_or_simplifies():
    calls = []
    path = np.zeros((3, 2))

    def plan(*args, **kwargs):
        calls.append(kwargs)
        return 'Approximate solution', path

    adapter = BoundedPlanner(SimpleNamespace(plan=plan), SimpleNamespace())
    status, result = adapter.plan([0., 0.], [[1., 1.]], simplify=True, time=3.)
    assert status == 'Approximate solution' and result is path
    assert calls == [{'simplify': False, 'time': 3.}]


def test_greedy_pass_removes_turn_after_random_budget():
    """Even when random attempts are disabled, the bounded greedy pass runs."""
    path = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    result = shortcut(path, lambda q: True, attempts=0, budget=10.)
    assert len(result) == 2
    np.testing.assert_array_equal(result[[0, -1]], path[[0, -1]])


def test_prepare_simplification_runs_after_exact_search_before_checks():
    events = []
    path = np.array([[0., 0.], [0., 1.], [1., 1.]])

    def plan(*args, **kwargs):
        events.append('solve')
        return 'Exact solution', path

    world = SimpleNamespace(set_qpos_all=lambda q: events.append('state'),
                            check_collision=lambda: [])
    adapter = BoundedPlanner(SimpleNamespace(plan=plan), world)
    adapter.before_simplify = lambda: events.append('bare world')
    status, result = adapter.plan()
    assert status == 'Exact solution'
    assert events[:3] == ['solve', 'bare world', 'state']
    np.testing.assert_array_equal(result, path[[0, -1]])


def test_expert_shortcuts_revoke_temporary_start_contacts():
    """The search can permit a starting contact; shortcuts cannot keep it."""
    from table_1000.experts.solver import MotionSolver
    state = dict(padding=None, allowed=False)
    checks = []
    native = SimpleNamespace(plan=lambda *a, **k: ('Exact solution',
                             np.array([[0., 0.], [0., 1.], [1., 1.]])))
    native_world = SimpleNamespace(set_qpos_all=lambda q: None,
                                  check_collision=lambda: checks.append(state.copy()) or [])
    adapter = BoundedPlanner(native, native_world)
    solver = MotionSolver.__new__(MotionSolver)
    solver.ctx = SimpleNamespace(current='move', steps=0)
    solver.arm = SimpleNamespace(name='arm')
    solver.model = SimpleNamespace(q=np.zeros(2), collisions=lambda *a: [('hand', 'table', 'table')])
    solver.planner = SimpleNamespace(robot=SimpleNamespace(get_name=lambda: 'robot'), planner=adapter)
    solver.world = SimpleNamespace(
        permissions=set(), objects={}, padding=.01, set_padding=lambda value: state.update(padding=value),
        allow=lambda a, b, value=True: state.update(allowed=value))
    solver.sync = lambda: None
    solver.timings, solver.log, solver.overlaps = {'planning_s': 0.}, [], []

    def function():
        assert state == dict(padding=.01, allowed=True)
        status, path = adapter.plan()
        return dict(status='Success', position=path, duration=1.)

    solver._plan('rrt', function)
    assert checks and all(check == dict(padding=0., allowed=False) for check in checks)
    assert adapter.before_simplify is None
    assert solver.log[0]['start_allowed'] == [('hand', 'table')]


def test_initial_contacts_can_be_left_once_but_never_reentered():
    path = np.array([[0., 0.], [0., 1.], [1., 1.]])
    contact = {('finger', 'table', 'table')}
    departing = lambda q: contact if q[0] < .1 else set()
    shortened = shortcut(path, lambda q: not departing(q), contact_pairs=departing,
                         budget=10., attempts=0)
    np.testing.assert_array_equal(shortened, path[[0, -1]])
    reentry = lambda q: contact if q[0] < .1 or (.4 < q[0] < .6 and q[1] < .8) else set()
    route = shortcut(path, lambda q: not reentry(q), contact_pairs=reentry,
                     budget=10., attempts=0)
    np.testing.assert_array_equal(route[[0, -1]], path[[0, -1]])
    assert_contact_path(route, reentry)


def test_departure_rejects_new_contacts_before_leaving():
    path = np.array([[0., 0.], [0., 1.], [1., 1.]])
    contact = {('finger', 'table', 'table')}
    added = {('elbow', 'table', 'table')}
    def pairs(q):
        if q[0] < .1:
            return contact
        return added if q[1] < .8 else set()
    result = shortcut(path, lambda q: not pairs(q), contact_pairs=pairs,
                      budget=10., attempts=0)
    np.testing.assert_array_equal(result[[0, -1]], path[[0, -1]])
    assert_contact_path(result, pairs)


def test_departure_permission_follows_a_retained_bend_until_first_clear_state():
    path = np.array([[0., 0.], [0., 1.], [1., 1.], [2., 1.], [3., 1.]])
    contact = {('finger', 'table', 'table')}
    obstacle = {('elbow', 'wall', 'wall')}

    def pairs(q):
        if .2 < q[0] < 2.1 and q[1] < .8:
            return obstacle
        return contact if q[0] < 2.5 else set()

    result = shortcut(path, lambda q: not pairs(q), contact_pairs=pairs,
                      budget=10., attempts=0)
    # The first bend cannot be removed, but the remaining contact prefix can.
    assert len(result) == 3
    np.testing.assert_array_equal(result[[0, -1]], path[[0, -1]])
    assert_contact_path(result, pairs)


def test_interior_departure_checks_for_leaving_between_prefix_vertices():
    path = np.array([[0., 0.], [0., 1.], [1., 1.], [2., 1.], [3., 1.]])
    contact = {('finger', 'table', 'table')}
    obstacle = {('elbow', 'wall', 'wall')}

    def pairs(q):
        if .2 < q[0] < 2.1 and q[1] < .8:
            return obstacle
        if .3 < q[1] < .7:
            return set()
        return contact if q[0] < 2.5 else set()

    result = shortcut(path, lambda q: not pairs(q), contact_pairs=pairs,
                      budget=10., attempts=0)
    # Equal endpoint contacts do not permit re-entry after a clear sample.
    np.testing.assert_array_equal(result, path)


def test_relaxation_shortens_a_bend_that_cannot_be_removed():
    path = np.array([[0., 0.], [0., 1.], [1., 1.]])
    valid = lambda q: not (.2 < q[0] < .8 and q[1] < .8)
    result = shortcut(path, valid, budget=10., attempts=0)
    np.testing.assert_array_equal(result[[0, -1]], path[[0, -1]])
    assert len(result) == 3
    assert np.linalg.norm(np.diff(result, axis=0), axis=1).sum() < 1.8
    for start, end in zip(result, result[1:]):
        assert all(valid(start + (end - start) * t) for t in np.linspace(0., 1., 1001))


def test_relaxation_rejects_a_shorter_path_with_worse_execution_cost():
    path = np.array([[0., 0.], [0., 1.], [1., 1.]])
    valid = lambda q: not (.2 < q[0] < .8 and q[1] < .8)
    result = shortcut(path, valid, budget=10., attempts=0,
                      cost=lambda rows: 1. + np.linalg.norm(rows[1] - path[1]))
    np.testing.assert_array_equal(result, path)


def test_relaxation_reduces_actual_topp_duration_around_an_obstacle():
    import mplib
    planner = SimpleNamespace(joint_vel_limits=np.ones(2), joint_acc_limits=np.ones(2))
    duration = lambda rows: mplib.Planner.TOPP(planner, rows, .04)[-1]
    path = np.array([[0., 0.], [0., 1.], [1., 1.]])
    result = shortcut(path, lambda q: not (.2 < q[0] < .8 and q[1] < .8),
                      budget=10., attempts=0, cost=duration)
    assert duration(result) < .98 * duration(path)


def test_one_initial_pair_cannot_reenter_while_another_still_touches():
    path = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    first = ('finger_left', 'drawer', 'front')
    second = ('finger_right', 'drawer', 'front')

    def pairs(q):
        result = {second} if q[0] < .7 else set()
        if q[0] < .1 or (.3 < q[0] < .5 and q[1] < .8):
            result.add(first)
        return result

    result = shortcut(path, lambda q: not pairs(q), contact_pairs=pairs,
                      budget=10., attempts=0)
    assert_contact_path(result, pairs)
    # Check independently of the whole-path oracle: each initial pair is
    # revoked as soon as it disappears, even while the other finger touches.
    remaining = pairs(result[0])
    for a, b in zip(result, result[1:]):
        for t in np.linspace(0., 1., 1001):
            contact = pairs(a + t * (b - a))
            assert contact <= remaining
            remaining &= contact
