"""Physics merges preserve collision points and semantic part names in live frames."""

import json

import numpy as np
import pytest

from table_1000.planning.world import asset_objects
from table_1000.scenes.geometry import AssetGeometry, transform_points
from table_1000.scenes.layout import yaw_pose


def merged_model():
    root = yaw_pose(.2, -.1, .3, .6)
    cap = yaw_pose(.03, .02, .08, -.4)
    nodes = [{'name': 'pen', 'parent': None, 'matrix': root.tolist(), 'rigid': False}]
    for body, pose in [('pen.barrel', np.eye(4)), ('pen.cap', cap)]:
        nodes += [{'name': body, 'parent': 'pen', 'matrix': pose.tolist(), 'rigid': True},
                  {'name': body + '.grip', 'parent': body, 'matrix': np.eye(4).tolist(), 'rigid': False},
                  {'name': body + '.collision', 'parent': body + '.grip', 'matrix': np.eye(4).tolist(),
                   'rigid': False, 'role': 'collision', 'shape': 'SPHERE', 'radius': .01}]
    model = {'nodes': nodes, 'joints': [{'name': 'pen.snap', 'body1': 'pen.barrel', 'body2': 'pen.cap'}]}
    mapping = {body: {'body': 'pen', 'matrix': pose.tolist()}
               for body, pose in [('pen.barrel', np.eye(4)), ('pen.cap', cap)]}
    return model, mapping, root


def test_merge_preserves_parts_points_com_and_planning_frames():
    model, mapping, root = merged_model()
    original = AssetGeometry(model)
    geometry = AssetGeometry(model, {'pen': [.01, .02, .03]}, mapping)
    assert geometry.bodies == ['pen'] and geometry.joints == {}
    assert geometry.parts('pen.cap') == [('pen', 'pen.cap.grip')]
    assert np.allclose(geometry.rest['pen'], root)
    assert np.allclose(geometry.points(), original.points())
    assert np.allclose(geometry.center_of_mass, transform_points(root, [[.01, .02, .03]])[0])
    live = yaw_pose(-.3, .4, .8, -.7)
    objects = asset_objects(geometry, 'p:', lambda body, part: (part, body))
    assert {obj.body for obj in objects} == {'p:pen'}
    for obj in objects:
        part = obj.name.removeprefix('p:')
        transform, _, radius = obj.raw[0]
        assert np.allclose(transform_points(live @ transform, [[0, 0, radius]]),
                           transform_points(live, geometry.points(part, 'pen'))[[2]])


def test_old_merged_build_reports_the_rebuild_command(tmp_path):
    model, mapping, _ = merged_model()
    path = tmp_path / 'objects/pen/000000'
    path.mkdir(parents=True)
    (path / 'model.json').write_text(json.dumps(model))
    report = {'rigid_bodies': 1}
    (path / 'physics_build.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='lacks body_mapping.*build_physics.py.*--assets pen/000000'):
        AssetGeometry.of(tmp_path, 'objects/pen/000000')
    report['body_mapping'] = mapping
    (path / 'physics_build.json').write_text(json.dumps(report))
    assert AssetGeometry.of(tmp_path, 'objects/pen/000000').bodies == ['pen']
    # Articulated legacy builds need no merge migration.
    model['joints'] = []
    (path / 'model.json').write_text(json.dumps(model))
    (path / 'physics_build.json').write_text(json.dumps({'rigid_bodies': 2}))
    assert AssetGeometry.of(tmp_path, 'objects/pen/000000').bodies == ['pen.barrel', 'pen.cap']
