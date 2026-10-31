import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from table_1000.experts.skills.hold.checks import CollisionSurface
from table_1000.experts.skills.hold.frames import Box


def scalar_intervals(shapes, origins, direction):
    """Independent ray/halfspace oracle, one ray at a time."""
    result = []
    for origin in origins:
        best, bounds = np.inf, (np.inf, -np.inf)
        for transform, vertices in shapes:
            point = (origin - transform[:3, 3]) @ transform[:3, :3]
            vector = transform[:3, :3].T @ direction
            lo, hi = -np.inf, np.inf
            for normal in ConvexHull(vertices).equations:
                slope, distance = normal[:3] @ vector, normal[:3] @ point + normal[3]
                if abs(slope) < 1e-12:
                    if distance > 1e-10:
                        hi = -np.inf
                elif slope > 0:
                    hi = min(hi, -distance / slope)
                else:
                    lo = max(lo, -distance / slope)
            nearest = max(lo, -hi)
            if hi >= lo and nearest < best:
                best, bounds = nearest, (lo, hi)
        result.append(bounds)
    return np.asarray(result).T


@pytest.mark.parametrize('direction', [[0., 1., 0.], [1e-13, 1., 0.], [1e-12, 1., 0.],
                                       [1., 0., 0.], [0., 0., 1.], [.6, .8, 0.]])
def test_halfspace_queries_keep_parallel_boundary_and_nearest_piece_rules(direction):
    transforms = [np.eye(4), np.eye(4)]
    transforms[1][:3, :3] = Rotation.from_euler('xyz', [.2, .3, .4]).as_matrix()
    transforms[1][:3, 3] = [.04, -.01, .03]
    shapes = [(transform, Box((.02, .03, .04)).points()) for transform in transforms]
    surface = CollisionSurface([(transform, 'convex', points) for transform, points in shapes])
    origins = np.vstack([np.random.default_rng(9).uniform(-.08, .08, (80, 3)),
                         [[.01, 0., 0.], [.01 + 2e-10, 0., 0.], [0., .015, .02], [0., 0., 0.]]])
    expected = scalar_intervals(shapes, origins, np.asarray(direction))
    actual = surface.interval(origins, np.asarray(direction))
    # The scalar dot product may accumulate differently from batched einsum.
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_only_exactly_identical_hull_planes_are_removed():
    surface = CollisionSurface([(np.eye(4), 'convex', Box((.02, .03, .04)).points())])
    assert surface.pieces[0][2].shape == (6, 4)
    # The boundary and a point outside are still distinguished, without a
    # plane-rounding tolerance that could change thin contact layers.
    low, high = surface.interval(np.array([[.01, 0., 0.], [.010001, 0., 0.]]), np.array([0., 1., 0.]))
    np.testing.assert_allclose([low[0], high[0]], [-.015, .015])
    assert low[1] > high[1]


def test_shared_frames_and_parallel_misses_remain_correct_when_direction_changes():
    transform = np.eye(4)
    shapes = [(transform, Box((.02, .03, .004)).points() + [0., 0., z])
              for z in (-.08, -.04, 0., .04, .08)]
    surface = CollisionSurface([(frame, 'convex', vertices) for frame, vertices in shapes])
    origins = np.random.default_rng(12).uniform(-.1, .1, (120, 3))
    for direction in ([0., 1., 0.], [0., 0., 1.], [0., 1., 0.], [.6, .8, 0.]):
        direction = np.asarray(direction)
        expected = scalar_intervals(shapes, origins, direction)
        np.testing.assert_allclose(surface.interval(origins, direction), expected, rtol=1e-14, atol=1e-14)


def test_oblique_rays_culled_by_original_halfspace_subset_keep_full_intersections():
    angle = np.arange(24) * (2 * np.pi / 24)
    ring = np.c_[.012 * np.cos(angle), .012 * np.sin(angle)]
    vertices = np.vstack([np.c_[ring, np.full(24, z)] for z in (-.04, .04)])
    shapes = [(np.eye(4), vertices)]
    surface = CollisionSurface([(np.eye(4), 'convex', vertices)])
    origins = np.vstack([vertices, np.random.default_rng(17).uniform(-.1, .1, (100, 3))])
    for direction in ([.3, .8, .5], [-.4, .2, .7]):
        direction = np.asarray(direction)
        np.testing.assert_allclose(surface.interval(origins, direction),
            scalar_intervals(shapes, origins, direction), rtol=1e-14, atol=1e-14)
