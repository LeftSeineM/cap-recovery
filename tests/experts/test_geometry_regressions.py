"""Round support floors and nondegenerate prisms encountered in scene 000100."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from table_1000.experts.skills.targets import oriented_shape
from table_1000.experts.grasps import Box, Cylinder, fitted, oriented_bounds
from table_1000.experts.skills import semantic
from table_1000.experts.skills.placing import Placing
from table_1000.experts.thin_objects import box_extents


def test_round_floor_does_not_keep_out_the_table_below_it():
    angle = np.arange(32) * 2 * np.pi / 32
    points = np.array([[.1 * np.cos(a), .1 * np.sin(a), z] for z in (.70, .71) for a in angle])
    floor = semantic.Shape('jar', 'jar', 'jar.floor', np.eye(4), points)
    table = {'size': (1., .8), 'height': .7, 'top_thickness': .03, 'center': (0., 0.)}
    region, blocked = semantic.support_region([semantic.table_shape(table), floor], 'jar',
                                               floor=True, headroom=.1)
    assert region[0][2, 3] == pytest.approx(.71)
    assert not any(np.allclose(low[:2], [-.5, -.4]) for _, low, _ in blocked)
    assert any(len(low) == 2 for _, low, _ in blocked)  # sampled gaps outside the circle
    target = SimpleNamespace(points=Box((.01, .01, .01)).points())
    dummy = SimpleNamespace(ctx=SimpleNamespace(trace=None))
    candidates = Placing.placements(dummy, target, np.eye(4), region, np.zeros(3), blocked, .02, .002)
    assert candidates
    for pose, _ in candidates:
        footprint = target.points @ pose[:3, :3].T + pose[:3, 3]
        assert np.linalg.norm(footprint[:, :2], axis=1).max() < .1 + semantic.SUPPORT_STEP


@pytest.mark.parametrize('points', [
    # Float32 Blender exports of a holder wall and a wrapper tail: both have
    # volume, but the unrepaired hull's face winding empties the silhouette.
    [[x, y, z] for x, top in [(.016, .080), (.030, .100)]
     for y in (-.040, -.038) for z in (.003, top)],
    [[x, y, z] for z in (.012, .016) for x, y in [(.015, 0.), (.030, -.002), (.026, -.010)]],
])
def test_oriented_bounds_repairs_prism_winding(points):
    points = np.asarray(points, dtype=np.float32).astype(float)
    with pytest.raises(ValueError, match='zero-size array'):
        trimesh.bounds.oriented_bounds(points)
    repaired = trimesh.bounds.oriented_bounds(trimesh.convex.convex_hull(points))
    actual = oriented_bounds(points)
    np.testing.assert_array_equal(actual[0], repaired[0])
    np.testing.assert_array_equal(actual[1], repaired[1])
    np.testing.assert_array_equal(box_extents(points), np.sort(repaired[1]))
    shape, transform = oriented_shape(points)
    local = (points - transform[:3, 3]) @ transform[:3, :3]
    assert np.all(np.ptp(local, axis=0) <= np.ptp(shape.points(), axis=0) + 1e-8)


def test_scope_probe_ground_lead_keeps_point_based_box(monkeypatch):
    # Collision piece 11 from the regression asset, in its authored frame.
    points = np.asarray(json.loads((Path(__file__).parent / 'fixtures' /
                                   'scope_probe_ground_lead.json').read_text()))
    expected_transform, expected_extents = trimesh.bounds.oriented_bounds(points)
    repaired_transform, _ = trimesh.bounds.oriented_bounds(trimesh.convex.convex_hull(points))
    repaired_local = points @ repaired_transform[:3, :3].T + repaired_transform[:3, 3]
    assert isinstance(fitted(repaired_local)[0], Cylinder)
    original_hull = trimesh.convex.convex_hull

    def only_unrepaired_hull(*args, **kwargs):
        assert kwargs.get('repair') is False, 'valid point-based bounds must not repair the hull'
        return original_hull(*args, **kwargs)

    monkeypatch.setattr(trimesh.convex, 'convex_hull', only_unrepaired_hull)
    actual_transform, actual_extents = oriented_bounds(points)
    np.testing.assert_array_equal(actual_transform, expected_transform)
    np.testing.assert_array_equal(actual_extents, expected_extents)
    np.testing.assert_array_equal(box_extents(points), np.sort(expected_extents))
    shape, transform = oriented_shape(points)
    assert isinstance(shape, Box)
    local = points @ expected_transform[:3, :3].T + expected_transform[:3, 3]
    expected_shape, offset = fitted(local)
    np.testing.assert_array_equal(shape.points(), expected_shape.points())
    np.testing.assert_array_equal(transform, np.linalg.inv(expected_transform) @ offset)
