import numpy as np
from types import SimpleNamespace

from table_1000.experts.grasps import Box, Cylinder, Hand, grasp_candidates


def hand():
    result = Hand.__new__(Hand)
    result.side = SimpleNamespace(finger_links=('left', 'right'))
    result.max_gap, result.geometry, result.collision_geometry = .08, {}, {}
    result.shapes = {'palm': [(np.eye(4), 'box', (.02, .06, .03), False)],
                     'left': [(np.eye(4), 'box', (.08, .01, .02), True)],
                     'right': [(np.eye(4), 'box', (.08, .01, .02), True)]}

    def frames(gap):
        gap = round(gap / .0005) * .0005
        poses = {name: np.eye(4) for name in result.shapes}
        poses['palm'][0, 3] = -.06
        poses['left'][1, 3], poses['right'][1, 3] = gap / 2 + .005, -gap / 2 - .005
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
