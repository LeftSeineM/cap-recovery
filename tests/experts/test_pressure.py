from types import SimpleNamespace

import numpy as np
import pytest

from table_1000.experts.skills.push.pressure import PressureLoad, correction_path, material_motion, same_pressure_contact
from table_1000.experts.solver import MotionSolver
from table_1000.sim.isaaclab.contacts import Contacts


def test_rotation_or_reverse_translation_is_a_new_material_state():
    points = np.array([[-.05, 0., 0.], [.05, 0., 0.]])
    start = np.eye(4)
    turned = np.eye(4)
    turned[:2, :2] = [[0., -1.], [1., 0.]]
    reversed_pose = np.eye(4)
    reversed_pose[0, 3] = -.002
    assert material_motion(points, start, turned) > .001  # COM did not move
    assert material_motion(points, start, reversed_pose) > .001  # negative progress
    old = dict(arm='right', flip=1., edge=np.array([1., 0., 0.]),
               origin=np.zeros(3), pose=start)
    args = ('right', 1., old['edge'], np.zeros(3))
    assert same_pressure_contact(old, *args, start, points, .001)
    assert not same_pressure_contact(old, *args, turned, points, .001)
    assert not same_pressure_contact(old, *args, reversed_pose, points, .001)
    assert not same_pressure_contact(old, 'right', -1., old['edge'], np.zeros(3), start, points, .001)
    assert not same_pressure_contact(old, 'right', 1., old['edge'], np.array([.002, 0., 0.]),
                                     start, points, .001)


def test_bearing_keeps_interval_load_and_substep_coverage_separate():
    load = PressureLoad('object:body', ['finger'], .004, .01, lambda: .036)
    pair = ('object:body.material', 'robot.finger')
    load.record(.004, pair, 10.)
    first = load.read()
    assert first['normal_impulse_ns'] == 10. and first['bearing']
    assert not first['continuous_substep_bearing']
    for step in range(10):
        load.record(step * .004, pair, .01 * .004 * 2)
    assert load.read()['bearing']
    assert not load.read()['bearing']  # old positive contacts do not survive a new control interval


def test_bearing_ignores_table_load_and_sums_pieces_in_the_same_substep():
    load = PressureLoad('object:body', ['finger'], .004, .01, lambda: .036)
    for step in range(10):
        load.record(step * .004, ('robot.finger', 'table'), 100.)
        load.record(step * .004, ('object:other.material', 'robot.finger'), 100.)
        for part in ('a', 'b'):
            load.record(step * .004, ('object:body.' + part, 'robot.finger'), .00003)
    sample = load.read()
    assert sample['bearing'] and sample['loaded_substeps'] == 10
    assert sample['normal_impulse_ns'] == pytest.approx(.0006)


def test_observing_impulses_does_not_consume_the_ordinary_record():
    contacts = Contacts.__new__(Contacts)
    contacts.time, contacts.impulses, contacts.impulse_observers = .004, {}, []
    observed = []
    pair = ('object:body', 'robot.finger')
    with contacts.observe_impulses(lambda *row: observed.append(row)):
        contacts.record_impulse(pair, .2)
    contacts.record_impulse(pair, .3)
    assert observed == [(.004, pair, .2)]
    assert contacts.take_impulses() == {pair: .5}


def test_pressure_anchor_excludes_unloaded_speculative_points():
    load = PressureLoad('object:body', ['finger'], .004, .01, lambda: .036)
    row = {'bodies': ['object:body.material', 'robot.finger'],
           'points': [[1., 2., 3.], [9., 8., 7.]],
           'impulses_ns': [[0., 0., .1], [0., 0., 0.]],
           'normals': [[0., 0., 1.], [0., 0., 1.]]}
    np.testing.assert_allclose(load.point([row]), [1., 2., 3.])


def test_load_monitor_uses_current_load_and_retains_the_interval_history():
    load = PressureLoad('object:body', ['finger'], .004, .01, lambda: .036)
    pair = ('object:body.material', 'robot.finger')
    for step in range(10):
        load.record(step * .004, pair, 0. if step == 5 else .001)
    sample = load.read()
    assert sample['bearing'] and not sample['continuous_substep_bearing'] and sample['bearing_now']
    for step in range(10):
        load.record(step * .004, pair, 0. if step == 9 else .001)
    assert not load.read()['bearing_now']


def test_bearing_now_drops_when_gpu_reports_stop_after_contact_is_lost():
    contacts = Contacts.__new__(Contacts)
    contacts.dt, contacts.time, contacts.impulses, contacts.impulse_observers = .004, 0., {}, []
    load = PressureLoad('object:body', ['finger'], .004, .01, lambda: contacts.time)
    pair = ('object:body.material', 'robot.finger')
    with contacts.observe_impulses(load.record):
        for step in range(10):
            contacts.time += contacts.dt
            if step == 0:
                contacts.record_impulse(pair, .001)  # later substeps report nothing
    sample = load.read()
    assert sample['bearing'] and not sample['bearing_now']


def test_sub_control_step_correction_reaches_its_endpoint_under_motion_limits():
    for displacement in (0., 1e-9, .001, .1):
        path = correction_path(np.zeros(2), np.array([displacement, -displacement]),
                               np.array([.2, .3]), np.array([.4, .5]))
        np.testing.assert_allclose(path['position'][-1], [displacement, -displacement])
        assert np.all(abs(path['velocity']) <= np.array([.2, .3]))
        assert np.all(abs(path['acceleration']) <= np.array([.4, .5]))


def test_interrupted_motion_never_executes_the_remaining_path_or_settles_at_its_end():
    solver = MotionSolver.__new__(MotionSolver)
    solver.arm = SimpleNamespace(name='right')
    actions = []
    solver.ctx = SimpleNamespace(step=actions.append)
    solver.qpos = lambda lift=False: actions[-1]['right'][0]
    solver.log, solver.timings = [{}], {'executing_s': 0.}
    path = {'position': np.arange(10.).reshape(5, 2)}
    assert not solver.follow(path, monitor=lambda: len(actions) < 2)
    assert len(actions) == 2 and solver.log[-1]['stopped_waypoint'] == 1
    assert np.array_equal(actions[-1]['right'][0], path['position'][1])
