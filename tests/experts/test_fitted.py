"""The grasp shape of a part from its collision vertices: a cylinder where it is round, else a box."""

import numpy as np

from table_1000.experts.skills.hold.frames import Box, Cylinder, fitted


def prism(sides, radius, length, axis):
    """Vertices of a regular prism along coordinate ``axis``, centred at (0.1, 0, 0)."""
    angles = np.arange(sides) * 2 * np.pi / sides + np.pi / sides
    ring = np.c_[np.cos(angles), np.sin(angles)] * radius
    points = np.vstack([np.c_[ring, np.full(sides, -length / 2)], np.c_[ring, np.full(sides, length / 2)]])
    return np.roll(points, axis + 1, axis=1) + [.1, 0., 0.]


def test_a_polygonal_barrel_is_a_cylinder_along_its_length():
    shape, offset = fitted(prism(8, .01, .18, axis=0))
    assert isinstance(shape, Cylinder) and np.isclose(shape.radius, .01) and np.isclose(shape.length, .18)
    assert np.allclose(offset[:3, 2], [1., 0., 0.]) and np.allclose(offset[:3, 3], [.1, 0., 0.])
    assert np.isclose(np.linalg.det(offset[:3, :3]), 1.)


def test_a_square_bar_stays_a_box():
    shape, offset = fitted(prism(4, .01, .18, axis=1))
    assert isinstance(shape, Box) and np.isclose(max(shape.size), .18)
    assert np.allclose(offset[:3, :3], np.eye(3))
