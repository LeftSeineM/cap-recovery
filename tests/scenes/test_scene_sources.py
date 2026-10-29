"""Every scene source initializes from its built assets without penetration, reproducibly."""

import numpy as np
import pytest

pytest.importorskip('mplib')

from table_1000.scenes import ROOT, load, scene_names
from table_1000.scenes.arrange import Drawer
from table_1000.scenes.layout import Layout

ASSETS = ROOT / 'assets'


@pytest.mark.parametrize('name', scene_names())
def test_scene_initializes(name):
    scene = load(name)
    missing = [spec['asset'] for spec in scene.OBJECTS.values() if not (ASSETS / spec['asset'] / 'model.json').is_file()]
    if missing:
        pytest.skip(f'assets not built: {missing}')
    if (ROOT / 'asset_sources' / 'scenes' / name / 'expert.py').is_file():
        assert hasattr(load(name, 'expert'), 'solve')
    for seed in range(3):
        layouts = []
        for _ in range(2):
            layout = Layout(scene, ASSETS, np.random.default_rng([seed, 0]))
            scene.initialize(layout)
            layouts.append(layout)
        assert set(layouts[0].poses) == set(scene.OBJECTS)
        assert layouts[0].penetrations() == {}
        for key in layouts[0].poses:
            assert np.array_equal(layouts[0].poses[key], layouts[1].poses[key])


def test_ready_to_work_stores_remaining_objects_on_closed_drawer_floors():
    scene = load('000100')
    if any(not (ASSETS / spec['asset'] / 'model.json').is_file() for spec in scene.OBJECTS.values()):
        pytest.skip('scene assets are not built')
    layout = Layout(scene, ASSETS, np.random.default_rng([0, 0]))
    scene.LAYOUTS[3]['arrange'](layout)
    assert set(layout.poses) == set(scene.OBJECTS)
    assert layout.penetrations() == {}
    assert all(value == 0. for value in layout.joints['cabinet'].values())
    # Both small jars and the remaining tools fit on the floors, rather than on the cabinet roof.
    contents = {1: [*scene.SMALL, 'knife', 'glue_stick', 'board_eraser'],
                2: ['remote', 'charger', 'snack', 'glue_bottle'],
                3: ['screwdriver', 'lidded_jar', 'dish']}
    for drawer, names in contents.items():
        floor = layout.poses['cabinet'][2, 3] + Drawer(layout, 'cabinet', drawer).floor
        for name in names:
            assert layout.supports[name] == 'cabinet'
            assert layout.world_points(name)[:, 2].min() == pytest.approx(floor, abs=.001)
