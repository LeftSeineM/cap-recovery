"""The moving-base scene keeps its targets on one table edge without pre-solving IK."""

import numpy as np
import pytest

pytest.importorskip('mplib')

from table_1000.scenes import ROOT, load
from table_1000.scenes.layout import Layout


def test_full_width_scene_preserves_the_local_approach_and_drawer_sweep():
    scene = load('000003')
    assets = ROOT / 'assets'
    if any(not (assets / obj['asset'] / 'model.json').exists() for obj in scene.OBJECTS.values()):
        pytest.skip('scene assets are not built')
    front = scene.TABLE['center'][1] - scene.TABLE['size'][1] / 2
    cabinet_x, pen_x, pen_yaw = [], [], []
    for seed in range(20):
        layout = Layout(scene, assets, np.random.default_rng([seed, 0]))
        scene.initialize(layout)
        assert layout.penetrations() == {}
        pen = layout.world_points('pen')
        assert np.all(pen[:, 1] >= front + .04 - 1e-8)
        assert np.all(pen[:, 1] <= front + .26 + 1e-8)
        assert layout.over_table('pen', margin=.04 - 1e-8)
        geometry = layout.geometry['cabinet']
        stroke = geometry.joints['cabinet.drawer3.slide']['limits'][1]
        sweep = np.vstack([h.vertices for q in (0., stroke)
                           for h in geometry.hulls(geometry.body_poses(layout.poses['cabinet'],
                                                                     {'cabinet.drawer3.slide': q}))])
        assert np.all(np.abs(sweep[:, :2] - scene.TABLE['center']) <= np.asarray(scene.TABLE['size']) / 2 - .01 + 1e-8)
        assert np.any(pen[:, :2].max(0) < sweep[:, :2].min(0) - .01) or np.any(
            sweep[:, :2].max(0) + .01 < pen[:, :2].min(0))
        cabinet = layout.poses['cabinet']
        assert abs(np.arctan2(cabinet[1, 0], cabinet[0, 0])) <= np.radians(30.)
        cabinet_x.append(cabinet[0, 3])
        pen_x.append((pen[:, 0].min() + pen[:, 0].max()) / 2)
        pen_yaw.append(np.arctan2(layout.poses['pen'][1, 0], layout.poses['pen'][0, 0]))
    # These deterministic samples exercise both ends and all heading quadrants,
    # rather than silently leaving the new scene near the old fixed layout.
    assert min(cabinet_x) < -.25 and max(cabinet_x) > .25
    assert min(pen_x) < -.40 and max(pen_x) > .40
    assert set(np.floor((np.asarray(pen_yaw) + np.pi) / (np.pi / 2)).astype(int)) == {0, 1, 2, 3}
