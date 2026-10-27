import numpy as np
from types import SimpleNamespace

from table_1000.experts.grasps import (Box, Cylinder, Hand, grasp_candidates, PAD_SPACING,
    MIN_PAD_AREA, HOLD_MARGIN, MIN_HOLD_AREA, OPEN_MARGIN)


def hand():
    result = Hand.__new__(Hand)
    result.pad_spacing, result.min_pad_area = PAD_SPACING, MIN_PAD_AREA
    result.hold_margin, result.min_hold_area = HOLD_MARGIN, MIN_HOLD_AREA
    result.opening_clearance, result.cost_width, result.drive = OPEN_MARGIN, 0., None
    result.side = SimpleNamespace(finger_links=('left', 'right'), finger_pivot=None)
    result.max_gap, result.geometry, result.collision_geometry = .08, {}, {}
    result.shapes = {'palm': [(np.eye(4), 'box', (.02, .06, .03), False)],
                     'left': [(np.eye(4), 'box', (.08, .01, .02), True)],
                     'right': [(np.eye(4), 'box', (.08, .01, .02), True)]}

    cached_frames = {}
    def frames(gap):
        gap = round(gap / .0005) * .0005
        if gap in cached_frames:
            return cached_frames[gap]
        poses = {name: np.eye(4) for name in result.shapes}
        poses['palm'][0, 3] = -.06
        poses['left'][1, 3], poses['right'][1, 3] = gap / 2 + .005, -gap / 2 - .005
        cached_frames[gap] = poses
        return poses

    result.frames = frames
    result.fingers = []
    for sign in (1., -1.):
        points = np.array([[x, -sign * .005, z] for x in np.arange(-.04, .04, .0005)
                           for z in np.arange(-.01, .01, .0005)])
        result.fingers.append((sign, points, np.ones(len(points), dtype=bool)))
    return result


def test_hand_reuses_shape_geometry_across_openings(monkeypatch):
    """Only pose wrappers vary with opening; the expensive shape conversion is shared."""
    from table_1000.planning import world
    calls = []
    original = world.geometry
    monkeypatch.setattr(world, 'geometry', lambda *args, **kwargs: (calls.append(args) or original(*args, **kwargs)))
    gripper = hand()
    gripper.fcl(np.eye(4), .01)
    first = len(calls)
    gripper.fcl(np.eye(4), .02)
    assert first and len(calls) == first


def test_cached_geometry_preserves_candidates_and_rejections():
    """Compare complete candidate streams with freshly rebuilt FCL shapes."""
    from table_1000.planning.world import box_object
    support = np.eye(4)
    support[2, 3] = -.07
    obj, transform = box_object('table', (.4, .4, .1), support)
    for shape in (Box((.02, .02, .04)), Cylinder(.012, .04)):
        cached, rebuilt = hand(), hand()
        original = rebuilt.fcl

        def fresh(*args, **kwargs):
            rebuilt.geometry.clear()
            rebuilt.collision_geometry.clear()
            return original(*args, **kwargs)

        rebuilt.fcl = fresh
        options = dict(obstacles=[(obj.fcl(transform), -.0001)], preferred=(0., 0., -1.), step=np.pi / 2)
        first = list(grasp_candidates(shape, np.eye(4), cached, **options))
        second = list(grasp_candidates(shape, np.eye(4), rebuilt, **options))
        assert len(first) == len(second) and first
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a.pose, b.pose)
            assert (a.open_gap, a.width, a.backoff, a.cost) == (b.open_gap, b.width, b.backoff, b.cost)


def test_angle_probe_reuses_placements_and_preserves_retry_order(monkeypatch):
    from table_1000.experts import grasps
    from table_1000.experts.grasps.checks import CollisionSurface

    calls, original = [], grasps._placed

    def placed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(grasps, '_placed', placed)
    shape = Box((.02, .02, .04))
    arguments = dict(obstacles=[], preferred=(0., 0., -1.), step=np.pi / 2,
                     contact_surface=CollisionSurface([(np.eye(4), 'convex', shape.points())]))
    plain = list(grasp_candidates(shape, np.eye(4), hand(), **arguments))
    plain_calls = len(calls)
    calls.clear()
    gripper, cache = hand(), {}
    probe = next(grasp_candidates(shape, np.eye(4), gripper, max_angle=np.radians(15.),
                                 _placement_cache=cache, **arguments), None)
    assert probe is not None and calls
    cached = list(grasp_candidates(shape, np.eye(4), gripper,
                                   _placement_cache=cache, **arguments))
    assert len(calls) == plain_calls
    assert len(cached) == len(plain)
    for first, second in zip(plain, cached):
        np.testing.assert_array_equal(first.pose, second.pose)
        assert (first.open_gap, first.width, first.backoff, first.cost, first.info) == (
            second.open_gap, second.width, second.backoff, second.cost, second.info)


def test_contact_math_scopes_blas_and_restores_after_failure(monkeypatch):
    import pytest
    from threadpoolctl import threadpool_limits
    from table_1000.experts import grasps
    from table_1000.experts.grasps import build_grasp_pose

    seen = []
    monkeypatch.setattr(grasps, '_collides', lambda *args: False)
    def close(*args, **kwargs):
        seen.extend(pool['num_threads'] for pool in grasps._CONTACT_MATH.info() if pool['user_api'] == 'blas')
        raise RuntimeError('contact query failed')
    monkeypatch.setattr(grasps, '_close', close)
    with threadpool_limits(limits=2, user_api='blas'):
        before = grasps._CONTACT_MATH.info()
        with pytest.raises(RuntimeError, match='contact query failed'):
            grasps._placed(Box((.02, .02, .04)), np.eye(4), None,
                build_grasp_pose((0., 0., -1.), (0., 1., 0.), (0., 0., 0.)),
                hand(), .05, [], .05)
        assert grasps._CONTACT_MATH.info() == before
    assert seen and set(seen) == {1}


def test_angle_probe_and_phase_stream_keep_geometry_ids_and_full_cost_order(monkeypatch):
    from table_1000.experts import grasps
    calls = []
    def placed(shape, transform, target, pose, *args):
        calls.append(pose.copy())
        # The closest approach is deliberately more expensive. The phase
        # needs its angle, rather than the angle of the cheapest full head.
        closest = pose[2, 0] < -.99
        contact = np.array([.1, 0., 0.]) if closest else np.zeros(3)
        return pose, 0., .02, contact, grasps.HOLD_AREA
    monkeypatch.setattr(grasps, '_placed', placed)
    args = Box((.02,) * 3), np.eye(4), hand(), [], (0., 0., -1.)
    full = list(grasps.grasp_candidates(*args, step=np.pi / 2))
    original_calls = len(calls)
    calls.clear()
    cache = {}
    kw = dict(step=np.pi / 2, _placement_cache=cache)
    probe = next(grasps.grasp_candidates(*args, **kw, _first_angle=True))
    assert probe.info['approach_deg'] == 0. and full[0].info['approach_deg'] > 0.
    phase = list(grasps.grasp_candidates(*args, **kw, max_angle=np.radians(15.)))
    expected = [g for g in full if g.info['approach_deg'] <= 15. + 1e-6]
    assert [g.choice for g in phase] == [g.choice for g in expected]
    for actual, original in zip(phase, expected):
        np.testing.assert_array_equal(actual.pose, original.pose)
        assert (actual.cost, actual.info) == (original.cost, original.info)
    movable = list(grasps.grasp_candidates(*args, **kw))
    assert [g.choice for g in movable] == [g.choice for g in full]
    assert len(calls) == original_calls  # each geometry is placed only once
