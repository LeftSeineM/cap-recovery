import numpy as np
import pytest

from table_1000.experts.skills.hold.frames import Box, Cylinder, build_grasp_pose, compute_grasp_info_by_obb, fitted, lengthwise


def test_grasp_pose_axes():
    pose = build_grasp_pose([0., 0., -1.], [0., 1., .3], [1., 2., 3.])
    assert np.allclose(pose[:3, 0], [0., 0., -1.])
    assert np.allclose(pose[:3, 1], [0., 1., 0.])        # made orthogonal to the approach
    assert np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3))
    assert np.allclose(pose[:3, 3], [1., 2., 3.])


def test_obb_grasp_closes_across_the_short_side_from_the_top():
    box = np.eye(4)
    box[:3, 3] = [0., 0., .05]
    info = compute_grasp_info_by_obb((box, np.array([.2, .04, .1])), depth=.02)
    assert np.allclose(np.abs(info['closing']), [0., 1., 0.])
    # The top face is at z = 0.10; the TCP goes 2 cm into the box.
    assert info['center'][2] == pytest.approx(.08)
    assert np.allclose(info['extents'], [.1, .04, .2])


def test_only_the_length_of_an_elongated_shape_is_not_squeezed():
    step = np.radians(15)
    assert lengthwise(list(Box((.02, .02, .179)).closings(step))) == .179        # a pen, not by its ends
    assert lengthwise(list(Cylinder(.01, .18).closings(step))) == .18
    assert lengthwise(list(Box((.12, .048, .015)).closings(step))) is None       # a flat handle, every way
    assert lengthwise(list(Cylinder(.04, .05).closings(step))) is None
    assert lengthwise(list(Box((.048, .01, .012)).closings(step))) is None       # a drawer handle, by its ends


def _prism(section, length=.15):
    """Vertices of a prism of the 2-D ``section`` points along Z."""
    section = np.asarray(section, dtype=float)
    return np.vstack([np.c_[section, np.full(len(section), z)] for z in (-length / 2, length / 2)])


def _rounded_square(side, corner, count=6):
    centres = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    return [(sx * (side / 2 - corner) + corner * np.cos(a), sy * (side / 2 - corner) + corner * np.sin(a))
            for k, (sx, sy) in enumerate(centres) for a in np.radians(np.linspace(90 * k, 90 * k + 90, count))]


def test_fitted_takes_a_polygonal_barrel_for_a_cylinder_and_a_rounded_square_for_a_box():
    turns = np.linspace(0., 2 * np.pi, 12, endpoint=False)
    shape, _ = fitted(_prism(np.c_[np.cos(turns), np.sin(turns)] * .009))
    assert isinstance(shape, Cylinder) and shape.radius == pytest.approx(.009)
    shape, _ = fitted(_prism(_rounded_square(.02, .006)))
    assert isinstance(shape, Box)


@pytest.mark.parametrize('limit', [30., 45., 90.])
def test_consumer_angle_limit_preserves_the_filtered_cost_order(monkeypatch, limit):
    pytest.importorskip('mplib')
    import types
    from table_1000.experts.skills.hold import frames as grasps
    preferred = np.array([0., 0., -1.])
    def placed(shape, transform, target, pose, hand, *args):
        angle = np.arccos(np.clip(pose[:3, 0] @ preferred, -1., 1.))
        if angle < np.radians(20.):
            return None
        return pose, abs(pose[0, 1]) * .01, .02, np.array([.001, .002, .003]), 30e-6
    monkeypatch.setattr(grasps, '_placed', placed)
    hand = types.SimpleNamespace(max_gap=.2, opening_clearance=.01, radius=lambda: .1,
        cost_width=0., pad_alignment=lambda gap: {'edge_only': False, 'pad_alignment_cost': 0.},
        moment_radius=lambda: .004)
    args = (Box((.04, .03, .02)), np.eye(4), hand, [], preferred)
    full = list(grasps.grasp_candidates(*args, step=np.radians(45.)))
    expected = [item for item in full if item.info['approach_deg'] <= limit + 1e-6]
    actual = list(grasps.grasp_candidates(*args, step=np.radians(45.), max_angle=np.radians(limit)))
    assert len(actual) == len(expected)
    for before, after in zip(expected, actual):
        np.testing.assert_array_equal(before.pose, after.pose)
        assert (before.cost, before.backoff, before.info) == (after.cost, after.backoff, after.info)


def test_wrist_rotation_reorders_existing_poses_without_changing_contact_tests(monkeypatch):
    pytest.importorskip('mplib')
    import types
    from table_1000.experts.skills.hold import frames as grasps
    calls = []
    def placed(shape, transform, target, pose, hand, *args):
        calls.append(pose.tobytes())
        return pose, 0., .02, np.zeros(3), grasps.HOLD_AREA
    monkeypatch.setattr(grasps, '_placed', placed)
    hand = types.SimpleNamespace(max_gap=.2, opening_clearance=.01, radius=lambda: .1,
        cost_width=0., pad_alignment=lambda gap: {'edge_only': False, 'pad_alignment_cost': 0.},
        moment_radius=lambda: .004)
    reference = build_grasp_pose([0., 0., -1.], [0., -1., 0.], [0., 0., 0.])[:3, :3]
    args = Box((.04, .03, .02)), np.eye(4), hand, [], [0., 0., -1.]
    before = list(grasps.grasp_candidates(*args, step=np.radians(45.)))
    raw_before, calls[:] = calls[:], []
    after = list(grasps.grasp_candidates(*args, step=np.radians(45.), reference_rotation=reference))
    assert sorted(calls) == sorted(raw_before)  # same contact inputs, a different lazy traversal
    assert {g.choice: g.pose.tobytes() for g in before} == {g.choice: g.pose.tobytes() for g in after}
    assert all(a.cost <= b.cost for a, b in zip(after, after[1:]))
    assert after[0].info['orientation_deg'] == pytest.approx(0.)
    np.testing.assert_allclose(after[0].pose[:3, :3], reference, atol=1e-14)
    assert [g.info['closest_deg'] for g in before] == [g.info['closest_deg'] for g in after]
    limited = list(grasps.grasp_candidates(*args, step=np.radians(45.), reference_rotation=reference,
                                          max_angle=np.radians(45.)))
    assert [g.choice for g in limited] == [g.choice for g in after if g.info['approach_deg'] <= 45. + 1e-6]


def test_reference_closing_parallel_to_requested_approach_is_finite(monkeypatch):
    pytest.importorskip('mplib')
    import types
    from table_1000.experts.skills.hold import frames as grasps
    monkeypatch.setattr(grasps, '_placed', lambda shape, transform, target, pose, *args:
                        (pose, 0., .02, np.zeros(3), grasps.HOLD_AREA))
    hand = types.SimpleNamespace(max_gap=.2, opening_clearance=.01, radius=lambda: .1,
        cost_width=0., pad_alignment=lambda gap: {'edge_only': False, 'pad_alignment_cost': 0.},
        moment_radius=lambda: .004)
    candidates = list(grasps.grasp_candidates(Box((.04, .03, .02)), np.eye(4), hand, [], [0., 1., 0.],
                      step=np.radians(90.), reference_rotation=np.eye(3)))
    assert candidates and all(np.isfinite(g.cost) for g in candidates)


def test_wrist_bound_does_not_place_expensive_flips_before_the_first_choice(monkeypatch):
    pytest.importorskip('mplib')
    import types
    from table_1000.experts.skills.hold import frames as grasps
    calls = []
    def placed(shape, transform, target, pose, *args):
        calls.append(pose[:3, :3])
        # The known offset contributes exactly one radian of cost.
        return pose, 0., .02, np.array([.025, 0., 0.]), grasps.HOLD_AREA
    monkeypatch.setattr(grasps, '_placed', placed)
    hand = types.SimpleNamespace(max_gap=.2, opening_clearance=.01, radius=lambda: .1,
        cost_width=0., pad_alignment=lambda gap: {'edge_only': False, 'pad_alignment_cost': 0.},
        moment_radius=lambda: .004)
    reference = build_grasp_pose([0., 0., -1.], [0., -1., 0.], [0., 0., 0.])[:3, :3]
    first = next(grasps.grasp_candidates(Box((.04, .03, .02)), np.eye(4), hand, [], [0., 0., -1.],
                 step=np.radians(45.), reference_rotation=reference))
    assert first.cost == pytest.approx(1.)
    angles = [np.arccos(np.clip((np.trace(reference.T @ frame) - 1.) / 2., -1., 1.)) for frame in calls]
    # Only the separate nearest-approach probe may exceed the first cost.
    assert sum(angle > first.cost for angle in angles) <= 1


def test_hand_collision_geometry_is_shared_without_sharing_opening_poses(monkeypatch):
    pytest.importorskip('mplib')
    import types
    from mplib.collision_detection import fcl
    from table_1000.experts.skills.hold.frames import Hand
    from table_1000.planning import world
    made, original = [], world.geometry
    def geometry(kind, data):
        made.append((kind, data))
        return original(kind, data)
    monkeypatch.setattr(world, 'geometry', geometry)
    hand = Hand.__new__(Hand)
    hand.max_gap, hand.geometry, hand.collision_geometry = .1, {}, {}
    hand.side = types.SimpleNamespace(finger_links=['finger'])
    hand.shapes = {link: [(np.eye(4), 'box', (.02,) * 3, False)] for link in ['palm', 'finger']}
    def frames(gap):
        palm, finger = np.eye(4), np.eye(4)
        palm[0, 3], finger[1, 3] = 1., gap
        return {'palm': palm, 'finger': finger}
    hand.frames = frames
    probe_pose = np.eye(4); probe_pose[1, 3] = .04
    probe = fcl.FCLObject('probe', world.pose(probe_pose), [fcl.CollisionObject(fcl.Box(.01, .01, .01))],
                          [world.pose(np.eye(4))])
    first, second = hand.fcl(np.eye(4), 0.), hand.fcl(np.eye(4), .04)
    assert fcl.distance(first, probe).min_distance > .02
    assert fcl.distance(second, probe).min_distance < 0.
    assert fcl.distance(hand.fcl(np.eye(4), .04, fingers=False), probe).min_distance > .9
    assert len(made) == 2  # one immutable shape per collider, across openings and palm-only queries
