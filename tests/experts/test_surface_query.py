"""Compiled collision queries retain the authored halfspace intersection rules."""
import numpy as np
import pytest
from table_1000.experts.grasps.checks import CollisionSurface, PARALLEL_PLANE_TOLERANCE
from table_1000.experts.grasps import Box


def reference(surface, origins, direction):
    low, high = np.full(len(origins), np.inf), np.full(len(origins), -np.inf)
    nearest = np.full(len(origins), np.inf)
    for transform, kind, data in surface.pieces:
        local = (origins - transform[:3, 3]) @ transform[:3, :3]
        if kind == 'sphere':
            b = local @ (transform[:3, :3].T @ direction)
            disc = b * b - (local * local).sum(1) + data ** 2
            root = np.sqrt(np.maximum(disc, 0.))
            lo, hi = np.where(disc >= 0., -b - root, np.inf), np.where(disc >= 0., -b + root, -np.inf)
        else:
            slope = data[:, :3] @ (transform[:3, :3].T @ direction)
            pos, neg = slope >= 1e-12, slope <= -1e-12
            distance = np.einsum('ij,kj->ik', local, data[:, :3]) + data[:, 3]
            lo = np.max(-distance[:, neg] / slope[neg], axis=1, initial=-np.inf)
            hi = np.min(-distance[:, pos] / slope[pos], axis=1, initial=np.inf)
            hi[np.any(distance[:, ~(pos | neg)] > PARALLEL_PLANE_TOLERANCE, axis=1)] = -np.inf
        hit = hi >= lo
        if hasattr(surface, '_closing_span'):
            axis, minimum, maximum = surface._closing_span
            indices = np.flatnonzero(hit)
            origin, slope = origins[indices] @ axis, direction @ axis
            first, last = origin + lo[indices] * slope, origin + hi[indices] * slope
            hit[indices] = (np.maximum(first, last) >= minimum) & (np.minimum(first, last) <= maximum)
        distance = np.maximum(lo, -hi)
        chosen = hit & (distance < nearest)
        low[chosen], high[chosen], nearest[chosen] = lo[chosen], hi[chosen], distance[chosen]
    return low, high


def test_compiled_halfspaces_match_numpy_for_rotated_compound_and_spheres():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(43)
    shapes = []
    for index in range(7):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.random(random_state=rng).as_matrix()
        pose[:3, 3] = rng.uniform(-.04, .04, 3)
        shapes.append((pose, 'convex', Box(tuple(rng.uniform(.005, .06, 3))).points()))
    shapes.append((np.eye(4), 'sphere', .017))
    surface = CollisionSurface(shapes)
    origins = rng.uniform(-.08, .08, (600, 3))
    for direction in [np.array([0., 1., 0.]), np.array([1e-13, 1., 1e-11]), *rng.normal(size=(8, 3))]:
        direction /= np.linalg.norm(direction)
        for query in (surface, surface.closing_span(direction, -.025, .025)):
            for actual, expected in zip(query.interval(origins, direction), reference(query, origins, direction)):
                np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-16)


def test_parallel_planes_contacts_and_empty_queries():
    surface = CollisionSurface([(np.eye(4), 'convex', Box((.02, .04, .06)).points())])
    origins = np.array([[x, 0., z] for x in (-.01, .01, .01 + 2e-10) for z in (-.03, .03, .031)])
    direction = np.array([0., 1., 0.])
    for actual, expected in zip(surface.interval(origins, direction), reference(surface, origins, direction)):
        np.testing.assert_array_equal(actual, expected)
    empty = surface.interval(np.zeros((0, 3)), direction)
    assert all(value.shape == (0,) for value in empty)
    empty = CollisionSurface([]).interval(origins, direction)
    assert np.isposinf(empty[0]).all() and np.isneginf(empty[1]).all()


def test_conservative_bounds_miss_skips_halfspace_plan(monkeypatch):
    surface = CollisionSurface([(np.eye(4), 'convex', Box((.02, .04, .06)).points())])
    monkeypatch.setattr(surface, '_query', lambda direction: pytest.fail('a bounds miss needs no ray plan'))
    origins = np.array([[.1, 0., 0.], [.11, .02, .01]])
    low, high = surface.interval(origins, np.array([0., 1., 0.]))
    assert np.isposinf(low).all() and np.isneginf(high).all()


def test_span_views_and_contact_subsets_keep_independent_query_semantics():
    pieces = []
    for y in (-.03, .03):
        transform = np.eye(4)
        transform[1, 3] = y
        pieces.append((transform, 'convex', Box((.02, .01, .04)).points()))
    surface = CollisionSurface(pieces)
    origins = np.array([[x, 0., z] for x in (0., .02) for z in (0., .03)])
    axis = np.array([0., 1., 0.])
    first = surface.closing_span(axis, -.04, -.02)
    second = surface.closing_span(axis, .02, .04)
    touched = surface.at_contacts(np.array([[0., -.025, 0.]]))
    # Alternate span, direction and piece set, then revisit the earlier view.
    # Neither a cached direction nor a shared span may leak into another query.
    for query, direction in [(first, axis), (second, axis),
                             (surface, np.array([1., 0., 0.])), (touched, axis),
                             (first, axis), (second, -axis)]:
        for actual, expected in zip(query.interval(origins, direction), reference(query, origins, direction)):
            np.testing.assert_array_equal(actual, expected)


def test_full_candidate_stream_matches_numpy_query(monkeypatch):
    from test_grasp_geometry_cache import hand
    from table_1000.experts.grasps import Cylinder, grasp_candidates

    for shape in (Box((.02, .02, .04)), Cylinder(.012, .04)):
        surface = CollisionSurface([(np.eye(4), 'convex', shape.points())])
        arguments = dict(obstacles=[], preferred=(0., 0., -1.), step=np.pi / 2,
                         contact_surface=surface)
        compiled = list(grasp_candidates(shape, np.eye(4), hand(), **arguments))
        with monkeypatch.context() as patch:
            patch.setattr(CollisionSurface, 'interval', reference)
            numpy = list(grasp_candidates(shape, np.eye(4), hand(), **arguments))
        assert len(compiled) == len(numpy) and compiled
        for first, second in zip(compiled, numpy):
            np.testing.assert_array_equal(first.pose, second.pose)
            assert (first.open_gap, first.width, first.backoff, first.cost, first.info) == (
                second.open_gap, second.width, second.backoff, second.cost, second.info)
