"""Measured feedback, yaw wrap and finite failure of tabletop squaring."""
from table_1000.experts.world import geometry as world_geometry

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from table_1000.experts import skills
from table_1000.experts.skills.hold.frames import Box
from table_1000.experts.skills.push import squaring


def world(monkeypatch, *, heading=0., gain=.8, turn_gain=0.):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('z', heading).as_matrix()
    pose[2, 3] = .755
    points = Box((.3, .2, .1)).points() + [0., 0., .05]
    calls = []

    def fail(reason, **details):
        raise RuntimeError(reason)

    ctx = SimpleNamespace(
        geometry={'box': SimpleNamespace(joints={}, bodies=['body'])},
        scene=SimpleNamespace(TABLE={'height': .755}),
        pose=lambda obj: pose.copy(), poses=lambda: {'box:body': pose.copy()},
        skills=lambda: SimpleNamespace(held={}), note=lambda *a, **kw: None, fail=fail)
    monkeypatch.setattr(world_geometry, 'object_points',
                        lambda c, obj: points @ pose[:3, :3].T + pose[:3, 3])

    def push(c, obj, **options):
        direction = np.asarray(options['direction'])
        point = pose[:3, :3] @ options['offset'][:3, 3]
        torque = np.cross(point, direction)[2]
        angle = turn_gain * torque * options['distance']
        pose[:3, :3] = Rotation.from_euler('z', angle).as_matrix() @ pose[:3, :3]
        pose[:2, 3] += gain * direction[:2] * options['distance']
        calls.append(options)
        return {'moved': gain * options['distance']}

    monkeypatch.setattr(skills, 'push', push)
    return ctx, pose, calls


def test_translation_remeasures_slip_and_preserves_support(monkeypatch):
    ctx, pose, calls = world(monkeypatch)
    checked = []
    result = squaring.square_on_table(ctx, 'box', [.045, -.025], check=lambda c: checked.append(c.pose('box')))
    assert result['position_m'] <= .020
    assert len(calls) >= 2
    assert len(checked) >= len(calls) * 2
    assert pose[2, 3] == .755
    assert all(abs(c['direction'][2]) == 0 and c['distance'] <= .025 for c in calls)


def test_rotation_uses_a_real_offcentre_contact_then_corrects_translation(monkeypatch):
    ctx, pose, calls = world(monkeypatch, heading=-.15, gain=.12, turn_gain=80.)
    result = squaring.square_on_table(ctx, 'box', [0., -1e-8], 0., max_rounds=30)
    assert abs(result['yaw_rad']) <= np.radians(5.)
    assert result['position_m'] <= .020
    assert calls[0]['offset'][0, 3] > 0.
    np.testing.assert_array_equal(calls[0]['direction'], [0., 1., 0.])
    assert all(c['shape'].size == (1e-6, 1e-6, 1e-6) for c in calls)


def test_round_container_ignores_heading_and_needs_no_contact(monkeypatch):
    ctx, _, calls = world(monkeypatch, heading=2.)
    assert squaring.square_on_table(ctx, 'box', [0., 0.])['pushes'] == []
    assert calls == []


def test_position_match_does_not_accept_an_unsupported_object(monkeypatch):
    ctx, pose, calls = world(monkeypatch)
    pose[2, 3] += .1
    with pytest.raises(RuntimeError, match='table support'):
        squaring.square_on_table(ctx, 'box', [0., 0.])
    assert calls == []


def test_stall_and_round_limit_are_failures(monkeypatch):
    ctx, _, calls = world(monkeypatch, gain=0.)
    with pytest.raises(RuntimeError, match='did not converge'):
        squaring.square_on_table(ctx, 'box', [.05, 0.])
    assert len(calls) == 12
    ctx, _, calls = world(monkeypatch, gain=.01)
    with pytest.raises(RuntimeError, match='did not converge'):
        squaring.square_on_table(ctx, 'box', [.05, 0.], max_rounds=2)
    assert len(calls) == 2


def test_yaw_wrap_and_corner_ray():
    assert squaring.yaw_error(-np.pi + .01, np.pi - .01) == pytest.approx(.02)
    points = Box((.3, .2, .1)).points()
    np.testing.assert_allclose(squaring.push_contact(points, [0., 1.], -.08), [.08, -.1])
    with pytest.raises(ValueError, match='outside'):
        squaring.push_contact(points, [0., 1.], -.5)


def test_payload_failure_stops_before_another_push(monkeypatch):
    ctx, _, calls = world(monkeypatch)
    def check(c):
        if calls:
            raise RuntimeError('payload escaped')
    with pytest.raises(RuntimeError, match='payload escaped'):
        squaring.square_on_table(ctx, 'box', [.05, 0.], check=check)
    assert len(calls) == 1


def test_invalid_parameters_do_not_move_the_object_or_modify_direction(monkeypatch):
    ctx, _, calls = world(monkeypatch)
    direction = np.array([0., 2.])
    squaring.square_on_table(ctx, 'box', [0., 0.], yaw_direction=direction)
    np.testing.assert_array_equal(direction, [0., 2.])
    for options in ({'max_push': np.nan}, {'max_rounds': 0}, {'yaw_direction': [0., 0.]}):
        with pytest.raises(ValueError):
            squaring.square_on_table(ctx, 'box', [0., 0.], **options)
    assert calls == []


def test_only_carried_payload_contacts_are_allowed_and_restored_after_failure(monkeypatch):
    from table_1000.experts.skills.push.squaring import _payload_contacts
    objects = {name: SimpleNamespace(body=name, fcl=lambda pose, n=name: n)
               for name in ('pen', 'remote', 'glue')}
    original = frozenset(('pen', 'remote'))
    permissions = {original}
    touched = []
    def allow(first, second, value=True):
        pair = frozenset((*first, *second))
        touched.append((pair, value))
        permissions.add(pair) if value else permissions.discard(pair)
    world = SimpleNamespace(objects=objects, poses=dict.fromkeys(objects), permissions=permissions, allow=allow)
    s = SimpleNamespace(solvers={'left': SimpleNamespace(world=world)},
                        dragged=lambda *a: ([], list(objects), []))
    with pytest.raises(RuntimeError):
        with _payload_contacts(s, 'basket', 'body'):
            assert permissions == {original, frozenset(('remote', 'glue')), frozenset(('pen', 'glue'))}
            assert not any('table' in pair for pair in permissions)
            raise RuntimeError('planner failed')
    assert permissions == {original}
    assert len(touched) == 4


def test_payload_permissions_cover_cap_and_barrel_but_exclude_external_objects(monkeypatch):
    from table_1000.experts.skills.push.squaring import _payload_contacts
    names = ('a:cap', 'a:barrel', 'b:cap', 'b:barrel', 'c:cap')
    objects = {name: SimpleNamespace(body=name, fcl=lambda pose, n=name: n) for name in names}
    permissions = set()
    def allow(first, second, value=True):
        pair = frozenset((*first, *second))
        permissions.add(pair) if value else permissions.discard(pair)
    world = SimpleNamespace(objects=objects, poses=dict.fromkeys(names), permissions=permissions, allow=allow)
    s = SimpleNamespace(solvers={'left': SimpleNamespace(world=world)},
                        dragged=lambda *a: ([], list(names[:4]), ['c:cap']))
    with _payload_contacts(s, 'holder', 'body'):
        assert permissions == {frozenset((a, b)) for a in names[:2] for b in names[2:4]}
        assert not any('c:cap' in pair for pair in permissions)
    assert not permissions


def test_stalled_window_changes_contact_without_one_push_failure(monkeypatch):
    ctx, _, calls = world(monkeypatch, gain=0.)
    times = iter(range(0, 1000, 11))
    monkeypatch.setattr(squaring.time, 'monotonic', lambda: next(times))
    with pytest.raises(RuntimeError, match='did not converge'):
        squaring.square_on_table(ctx, 'box', [.05, .03], max_rounds=4)
    assert len(calls) == 4
    np.testing.assert_array_equal(calls[0]['direction'], [1., 0., 0.])
    assert calls[-1]['direction'][1] > 0.  # alternate contact plan after the window
