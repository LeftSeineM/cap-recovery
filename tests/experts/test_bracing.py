"""Drawer stability is a path invariant, including other drawers and retreat."""

from table_1000.experts.skills.drawer import joints as drawer_joints

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from table_1000.experts.context import ExpertFailed
from table_1000.experts.skills import semantic
from table_1000.experts.skills.drawer.bracing import Stability, roof_contacts, stability


@pytest.fixture
def cabinet(monkeypatch):
    joints = {name: SimpleNamespace(name=name, parent='housing', limits=(0., .17))
              for name in ('top', 'middle', 'bottom')}
    pose, gaps, notes = np.eye(4), dict(top=0., middle=0., bottom=0.), []
    def fail(reason, **details):
        raise ExpertFailed(None, reason, **details)
    ctx = SimpleNamespace(pose=lambda *a: pose, observers=[], note=lambda *a, **kw: notes.append((a, kw)),
                          fail=fail, geometry={'cab': SimpleNamespace(joints={n: {'type': 'SLIDER'} for n in joints})})
    monkeypatch.setattr(drawer_joints.Joint, 'of', lambda g, n: joints[n])
    monkeypatch.setattr(drawer_joints, 'opening', lambda c, o, n: gaps[n])
    return ctx, joints, pose, gaps, notes


@pytest.mark.parametrize('kind', ['translation', 'rotation', 'other drawer'])
def test_transient_instability_fails_and_emits_peaks(cabinet, kind):
    ctx, joints, pose, gaps, notes = cabinet
    with pytest.raises(ExpertFailed, match='stability exceeded'):
        with stability(ctx, 'cab', joints['top']):
            if kind == 'translation':
                pose[0, 3] = .0201
            elif kind == 'rotation':
                pose[:3, :3] = Rotation.from_euler('z', 5.1, degrees=True).as_matrix()
            else:
                gaps['middle'] = .0201
            ctx.observers[0](ctx)
    assert ctx.observers == []
    assert notes[-1][0] == ('drawer stability',)
    assert notes[-1][1]['joint'] == 'top'


def test_active_drawer_is_excluded_and_failure_allows_cleanup(cabinet):
    ctx, joints, pose, gaps, notes = cabinet
    monitor = Stability(ctx, 'cab', joints['top'])
    gaps['top'] = .16
    monitor.sample(ctx)
    pose[0, 3] = .021
    with pytest.raises(ExpertFailed):
        monitor.sample(ctx)
    # Unwinding the failed action can now retract both hands.
    monitor.sample(ctx)
    pose[0, 3] = 0.
    monitor.sample(ctx)
    assert monitor.peak['translation_m'] == .021


def test_other_drawer_already_open_fails_at_entry(cabinet):
    ctx, joints, pose, gaps, notes = cabinet
    gaps['bottom'] = .021
    with pytest.raises(ExpertFailed, match='stability exceeded'):
        with stability(ctx, 'cab', joints['top']):
            pytest.fail('must fail before any motion')
    assert not ctx.observers


@pytest.mark.parametrize('operation', ['open', 'close'])
def test_bracing_with_a_busy_hand_fails_before_motion(cabinet, monkeypatch, operation):
    ctx, joints, pose, gaps, notes = cabinet
    monkeypatch.setattr(drawer_joints, 'slider', lambda *a: joints['top'])
    ctx.skills = lambda: SimpleNamespace(held={'left': object()}, arms=['left', 'right'])
    arguments = (.16,) if operation == 'open' else ()
    with pytest.raises(ValueError, match='place held objects first'):
        getattr(semantic, operation)(ctx, 'cab', 'top', *arguments, brace=True)
    assert not ctx.observers and not notes


def test_roof_contacts_use_real_collision_material_instead_of_housing_box():
    def box(x):
        return np.array([[a, b, z] for a in (x - .01, x + .01) for b in (-.1, .1) for z in (.19, .2)])
    geometry = SimpleNamespace(bodies=['housing'], shapes=[('housing', 'rim', np.eye(4), 'convex', box(x))
                                                           for x in (-.1, .1)])
    ctx = SimpleNamespace(geometry={'cab': geometry}, pose=lambda *a: np.eye(4))
    points = np.array(roof_contacts(ctx, 'cab', 'housing'))
    assert len(points) and np.all(np.abs(points[:, 0]) >= .09)
    np.testing.assert_allclose(points[:, 2], .2)


def test_semantic_brace_is_one_replayable_public_skill(monkeypatch):
    from table_1000.experts import skills
    calls = []
    monkeypatch.setattr(skills, 'braced_drawer', lambda *a: calls.append(a))
    context = object()
    semantic.open(context, 'cab', 'top', .12, brace=True)
    semantic.close(context, 'cab', 'top', brace=True)
    assert calls == [(context, 'cab', 'top', .12), (context, 'cab', 'top')]


@pytest.mark.parametrize('arms', [None, ['left'], ['right'], ['left', 'right']])
def test_support_hand_cannot_be_stowed_during_active_motion(arms):
    from table_1000.experts.skills.base import Base
    called = []
    s = SimpleNamespace(arms=['right'], held={}, braced=True, home=np.zeros(2),
                        model=SimpleNamespace(q=np.ones(2), arm_indices=lambda a: [0 if a == 'left' else 1]),
                        ctx=SimpleNamespace(note=lambda *a, **kw: None), ready=lambda a: called.extend(a))
    expected = ['right'] if arms is None or 'right' in arms else []
    assert Base.stow(s, arms) == bool(expected)
    assert called == expected


@pytest.mark.parametrize('outcome', ['completed', 'closed', 'failed'])
def test_braced_push_removes_only_its_observer_before_holding(monkeypatch, outcome):
    from table_1000.experts.skills.push.motion import Motion
    from table_1000.experts.solver import PlanningFailed

    joint = SimpleNamespace(child='drawer', kind='SLIDER', name='slide')
    monkeypatch.setattr(drawer_joints.Joint, 'of', lambda *a: joint)
    gap = [.04]
    monkeypatch.setattr(drawer_joints, 'opening', lambda *a: gap[0])
    events = []
    sentinel = lambda context: None
    observers = [sentinel]
    position = np.array([.3])

    def hold(action):
        assert observers == [sentinel]
        np.testing.assert_array_equal(action['right'], position[None])
        events.append('hold')

    motion = Motion()
    motion.ctx = SimpleNamespace(geometry={'cab': SimpleNamespace(joints={'slide': {}})},
                                 observers=observers, step=hold)
    motion.solvers = {'right': SimpleNamespace(qpos=lambda: position)}
    target = SimpleNamespace(body='cab:drawer', obj='cab')
    displacement = np.array([-.1, 0., 0.])
    failure = PlanningFailed('push', 'tracking')

    def move(arm, delta, drag):
        assert arm == 'right' and drag == ('cab', 'drawer')
        np.testing.assert_array_equal(delta, displacement)
        assert len(observers) == 2
        observers[-1](motion.ctx)  # Still open: do not stop before the measured threshold.
        if outcome == 'closed':
            gap[0] = drawer_joints.CLOSED / 2
            observers[-1](motion.ctx)
            pytest.fail('the measured closed end must interrupt motion')
        if outcome == 'failed':
            raise failure

    motion.move_by_displacement = move
    if outcome == 'failed':
        with pytest.raises(PlanningFailed) as caught:
            motion._push_braced('right', target, displacement)
        assert caught.value is failure
    else:
        motion._push_braced('right', target, displacement)
    assert observers == [sentinel]
    assert events == (['hold'] if outcome == 'closed' else [])


def test_moving_base_while_supported_is_rejected():
    from table_1000.experts.skills.base import Base
    def fail(reason, *, kind):
        raise ExpertFailed(None, reason)
    s = SimpleNamespace(braced=True, ctx=SimpleNamespace(fail=fail))
    with pytest.raises(ExpertFailed, match='release the brace'):
        Base.make_room(s, [])


@pytest.mark.parametrize('failure_kind', ['unreachable', 'planning'])
def test_assignment_falls_back_only_before_the_drawer_action(monkeypatch, failure_kind):
    from contextlib import contextmanager
    from table_1000.experts.context import Unreachable
    from table_1000.experts.solver import PlanningFailed
    from table_1000.experts.skills.drawer import bracing
    failure_class = Unreachable if failure_kind == 'unreachable' else PlanningFailed
    attempts = []
    @contextmanager
    def support(ctx, obj, joint, active, amount, **kwargs):
        attempts.append(active)
        if active == 'left':
            raise failure_class(None, 'no coupled reach')
        yield
    monkeypatch.setattr(bracing, '_brace', support)
    ctx = SimpleNamespace(skills=lambda: SimpleNamespace(arms=['left', 'right']), note=lambda *a, **kw: None)
    with bracing.brace(ctx, 'cab', None, 'left', .16) as active:
        assert active == 'right'
    assert attempts == ['left', 'right']
    attempts.clear()
    with pytest.raises(failure_class, match='action failed'):
        with bracing.brace(ctx, 'cab', None, 'right', .16):
            raise failure_class(None, 'action failed')
    assert attempts == ['right']


def test_both_hand_assignments_are_checked_before_relocating(monkeypatch):
    from contextlib import contextmanager
    from table_1000.experts.context import Unreachable
    from table_1000.experts.skills.drawer import bracing
    attempts = []
    @contextmanager
    def support(ctx, obj, joint, active, amount, *, relocate):
        attempts.append((active, relocate))
        if not relocate:
            raise Unreachable(None, 'not from this stance')
        yield
    monkeypatch.setattr(bracing, '_brace', support)
    ctx = SimpleNamespace(skills=lambda: SimpleNamespace(arms=['left', 'right']), note=lambda *a, **kw: None)
    with bracing.brace(ctx, 'cab', None, 'left', .16) as active:
        assert active == 'left'
    assert attempts == [('left', False), ('right', False), ('left', True)]


def test_oriented_release_detaches_overlapping_whole_and_carried_names_once():
    from table_1000.experts.skills.push.motion import Motion
    detached = []
    def detach(name):
        assert name not in detached
        detached.append(name)
    solver = SimpleNamespace(open_gripper=lambda _: None, detach=detach, sync=lambda: None,
                             tcp=lambda: np.eye(4), allow=lambda *a: None,
                             first_reachable=lambda *a: None)
    skill = Motion()
    skill.solvers = {'right': solver}
    skill.held = {'right': {'target': SimpleNamespace(names=['body', 'wall']), 'carried': ['wall'],
                            'riders': ['pen'], 'carry_permissions': [], 'supports': [],
                            'grasp': {'open_gap': .05}}}
    skill.release('right', upward=.1)
    assert detached == ['body', 'wall', 'pen']


def test_gravity_release_waits_and_reobserves_after_detachment_before_exit():
    from table_1000.experts.skills.push.motion import Motion
    events = []
    solver = SimpleNamespace(open_gripper=lambda _: events.append('open'),
        detach=lambda _: events.append('detach'), sync=lambda: events.append('observe'),
        tcp=lambda: np.eye(4), allow=lambda *args: None,
        first_reachable=lambda *args: events.append('withdraw'))
    skill = Motion()
    skill.ctx = SimpleNamespace(hold=lambda seconds: events.append(('settle', seconds)))
    skill.solvers = {'right': solver}
    skill.held = {'right': {'target': SimpleNamespace(names=['body']), 'carried': [],
        'riders': [], 'carry_permissions': [], 'supports': [], 'grasp': {'open_gap': .05}}}
    skill.release('right', settle=1.)
    assert events == ['open', 'detach', ('settle', 1.), 'observe', 'withdraw']


@pytest.mark.parametrize('withdrawal_fails', [False, True])
def test_release_revokes_finger_contacts_for_every_carried_part_and_payload(withdrawal_fails):
    from table_1000.experts.skills.push.motion import Motion
    from table_1000.experts.solver import PlanningFailed
    allowed = {'body', 'wall', 'pen', 'support'}
    def withdraw(*args):
        assert allowed == {'body', 'wall', 'pen', 'support'}
        if withdrawal_fails:
            raise PlanningFailed('release', 'no withdrawal')
    solver = SimpleNamespace(open_gripper=lambda _: None, detach=lambda _: None, sync=lambda: None,
                             tcp=lambda: np.eye(4), first_reachable=withdraw,
                             allow=lambda names, enabled: allowed.difference_update(names))
    skill = Motion()
    skill.solvers = {'right': solver}
    skill.held = {'right': {'target': SimpleNamespace(names=['body']), 'carried': ['wall'],
                            'riders': ['pen'], 'carry_permissions': [], 'supports': ['support'],
                            'grasp': {'open_gap': .05}}}
    if withdrawal_fails:
        with pytest.raises(PlanningFailed, match='no withdrawal'):
            skill.release('right')
    else:
        skill.release('right')
    assert not allowed


def test_minor_shift_is_recorded_and_retry_keeps_original_baseline(cabinet):
    ctx, joints, pose, gaps, notes = cabinet
    original = pose.copy()
    with stability(ctx, 'cab', joints['top'], original) as monitor:
        pose[0, 3] = .015
        pose[:3, :3] = Rotation.from_euler('z', 4., degrees=True).as_matrix()
        gaps['middle'] = .015
        monitor.sample(ctx)
    assert notes[-1][1]['translation_m'] == pytest.approx(.015)
    # A retry at the moved pose must not silently adopt it as its new origin.
    with pytest.raises(ExpertFailed, match='stability exceeded'):
        with stability(ctx, 'cab', joints['top'], original) as monitor:
            pose[0, 3] = .025
            monitor.sample(ctx)
    np.testing.assert_array_equal(original, np.eye(4))


@pytest.mark.parametrize('gap,closed', [(.0051, True), (.0199, True), (.0201, False)])
def test_closed_drawer_uses_two_centimetre_visible_gap(monkeypatch, gap, closed):
    monkeypatch.setattr(drawer_joints, 'opening', lambda *args: gap)
    assert drawer_joints.is_closed(None, 'cab', 'top') is closed
