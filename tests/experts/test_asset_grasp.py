"""Asset inventory, random placement, and the stability metric (no simulator)."""
import types

import numpy as np

from table_1000.experts.asset_grasp import displacement, inventory, summarize


def test_repeated_instances_run_once_and_exclusions_are_explicit():
    source = types.SimpleNamespace(OBJECTS={
        'a': {'asset': 'objects/a/000000'}, 'b': {'asset': 'objects/a/000000'},
        'fixture': {'asset': 'objects/f/000000'}})
    result = inventory(source, ('a', 'fixture'))
    assert result['objects/a/000000'] == {'instances': ['a', 'b'], 'excluded': False}
    assert result['objects/f/000000']['excluded']


def test_slip_includes_rotation_and_cannot_cancel_at_final_pose():
    reference = np.eye(4)
    turned = np.eye(4)
    turned[:2, :2] = [[0, -1], [1, 0]]
    points = np.array([[.1, 0, 0], [-.1, 0, 0]])
    assert np.isclose(displacement(reference, turned, points), np.sqrt(.02))
    assert displacement(reference, reference, points) == 0
    assert max(displacement(reference, pose, points) for pose in (reference, turned, reference)) > .005


def test_missing_and_failed_trials_stay_in_denominator():
    report = summarize([{'status': 'success'}, {'status': 'failed', 'failure': {'reason': 'no reachable grasp'}}], 3)
    assert report['rate'] == 1 / 3
    assert report['status'] == 'FAIL'
    assert report['failures'] == {'no reachable grasp': 1, 'missing report': 1}


def test_only_unreachable_transport_is_excluded_from_acceptance_denominator():
    report = summarize([{'status': 'success'}, {'status': 'not_evaluated',
        'failure': {'reason': 'transport motion unreachable'}}], 3)
    assert report['assessed'] == 2 and report['not_evaluated'] == 1
    assert report['rate'] == .5
    assert report['failures'] == {'missing report': 1}
    assert report['not_evaluated_reasons'] == {'transport motion unreachable': 1}
    empty = summarize([{'status': 'not_evaluated', 'failure': {'reason': 'transport motion unreachable'}}], 1)
    assert empty['rate'] is None and empty['status'] == 'FAIL'


def test_closure_table_uses_actual_measurement_and_separates_unknown_truth():
    from table_1000.experts.asset_grasp import closure_statistics
    def report(seed, predicted, status, **kwargs):
        return {'seed': seed, 'pick': {'closure_check': {'held': predicted, 'reasons': []}},
                'status': status, **kwargs}
    result = closure_statistics([
        report(0, False, 'success'), report(1, True, 'success'),
        report(2, True, 'failed', lift={'rise_m': 0.}),
        report(3, False, 'failed', lift={'rise_m': .1}, hold={'completed': True, 'held': False}),
        report(4, True, 'not_evaluated'), {'status': 'failed'}])
    assert set(result['table'].values()) == {1}
    assert result['unmeasured'] == 1 and result['no_check'] == 1
    assert [row['seed'] for row in result['disagreements']] == [0, 2]


def test_failed_closure_before_losing_reach_is_a_physical_failure():
    report = {'status': 'failed', 'failure': {'reason': 'no reachable grasp'},
              'events': [{'note': 'missed grasp'}]}
    assert summarize([report], 1)['failures'] == {'object not held': 1}
    assert report['failure']['reason'] == 'no reachable grasp'


def test_relocation_drop_is_measured_failure_but_invalid_motion_is_not():
    from table_1000.experts.asset_grasp import grasp_family
    dropped = {'seed': 0, 'status': 'failed', 'failure': {'reason': 'dropped'},
        'settled_pose': np.eye(4).tolist(),
        'pick': {'grasp': {'pose': np.eye(4).tolist()}, 'closure_check': {'held': True, 'reasons': []}},
        'lift': {'rise_m': .1}, 'hold': {'held': False, 'completed': False, 'dropped': True}}
    invalid = dict(dropped, seed=1, status='invalid_motion')
    summary = summarize([dropped, invalid], 2)
    assert summary['assessed'] == 1 and summary['invalid_motion'] == 1
    assert summary['failures'] == {'dropped': 1}
    assert summary['closure_check']['table']['predicted_held_actual_failed'] == 1
    assert summary['closure_check']['unmeasured'] == 1
    assert summary['preference_data']['by_grasp'][grasp_family(dropped)] == {
        'trials': 1, 'success': 0, 'rate': 0.}


def test_comparison_pairs_requested_seeds_with_a_larger_baseline(tmp_path):
    import json
    from table_1000.experts.asset_grasp.comparison import compare
    asset = 'objects/cup/000000'
    for name, seeds in [('old', range(3)), ('new', [1])]:
        root = tmp_path / name
        directory = root / asset / 'trials'
        directory.mkdir(parents=True)
        reports = []
        for seed in seeds:
            report = {'seed': seed, 'status': 'success' if seed == 1 else 'failed',
                'cpu_s': 3., 'wall_s': 4., 'sim_s': 1., 'trajectory': 'test.hdf5',
                'load': {'host': 'test', 'cores': 2,
                         'samples': [{'load_1m': 1., 'cpu_s': 3.}]}}
            (directory / f'{seed:04d}.json').write_text(json.dumps(report))
            reports.append(report)
        (root / asset / 'test.hdf5').touch()
        (root / 'summary.json').write_text(json.dumps({
            'commit': name, 'assets': {asset: summarize(reports, len(reports))}}))
    result = compare(tmp_path / 'old', tmp_path / 'new')
    row = result['rows'][0]
    assert row['old_expected'] == 1 and row['old_first'] == 1
    assert [pair['seed'] for pair in row['trial_changes']] == [1]
    assert row['timing']['baseline']['cpu_s']['mean'] == 3.


def test_hollow_assembly_tries_real_parts_before_filled_bounding_box():
    from table_1000.experts.asset_grasp import targets
    from table_1000.experts.grasps import Box
    names = ('ring', 'small', 'large')
    geometry = types.SimpleNamespace(
        shapes=[],
        points=lambda part, body: Box((.01 if part == 'small' else .03,) * 3).points(),
        parts=lambda: [('body', name) for name in names],
        solid=lambda name, narrow: name in ('small', 'large'))
    def target(obj, part=None):
        return types.SimpleNamespace(body='object:body', names=[part] if part else list(names),
                                     shape=Box((.01 if part == 'small' else .03,) * 3))
    skill = types.SimpleNamespace(target=target, ctx=types.SimpleNamespace(geometry={'object': geometry}))
    assert [t.names for t in targets(skill)] == [['large'], ['small'], list(names)]


def test_diagonal_thin_part_keeps_its_actual_width():
    from table_1000.experts.asset_grasp import oriented_shape
    from table_1000.experts.grasps import Box
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_euler('z', 30, degrees=True).as_matrix()
    points = Box((.04, .004, .002)).points() @ rotation.T
    shape, offset = oriented_shape(points)
    assert np.allclose(sorted(shape.size), [.002, .004, .04])
    local = (points - offset[:3, 3]) @ offset[:3, :3]
    assert np.all(np.abs(local) <= np.asarray(shape.size) / 2 + 1e-10)


def test_placement_uses_ten_cm_square_and_full_yaw(monkeypatch):
    from table_1000.experts import asset_grasp
    class Layout:
        def __init__(self, *args):
            pass
        def resting_pose(self, name, xy, yaw):
            out = np.eye(4)
            out[:2, 3] = xy
            return out
    monkeypatch.setattr(asset_grasp, 'Layout', Layout)
    samples = [asset_grasp.placement(None, None, seed, (.45, -.15))[1] for seed in range(100)]
    xy = np.array([v['xy_m'] for v in samples])
    assert (np.abs(xy - [.45, -.15]) <= .05).all()
    yaw = [v['yaw_rad'] for v in samples]
    assert min(yaw) < -3 and max(yaw) > 3
    assert samples[5] == asset_grasp.placement(None, None, 5, (.45, -.15))[1]


def test_grasp_family_does_not_change_with_world_yaw():
    from table_1000.experts.asset_grasp import grasp_family
    body = np.eye(4)
    pose = np.eye(4)
    report = {'pick': {'arm': 'right', 'grasp': {'pose': pose.tolist()}}, 'settled_pose': body.tolist()}
    family = grasp_family(report)
    body[:2, :2] = [[0, -1], [1, 0]]
    report['settled_pose'] = body.tolist()
    report['pick']['grasp']['pose'] = (body @ pose).tolist()
    assert grasp_family(report) == family
    summary = summarize([dict(report, status='success', lift={'rise_m': .1}, hold={'completed': True, 'held': True}),
                         dict(report, status='failed', lift={'rise_m': .1}, hold={'completed': True, 'held': False})], 2)
    assert summary['preference_data']['by_grasp'][family] == {'trials': 2, 'success': 1, 'rate': .5}


def test_unexecuted_transport_is_not_a_grasp_preference_failure():
    from table_1000.experts.asset_grasp import grasp_family
    grasp = {'pick': {'grasp': {'pose': np.eye(4).tolist()}}, 'settled_pose': np.eye(4).tolist()}
    reports = [dict(grasp, status='failed', lift={'rise_m': .1},
                    failure={'reason': 'transport motion unreachable'}),
               dict(grasp, status='failed', lift={'rise_m': 0.}),
               dict(grasp, status='success', lift={'rise_m': .1}, hold={'completed': True, 'held': True})]
    result = summarize(reports, 3)
    family = grasp_family(grasp)
    assert result['first_attempt_success'] == 1 and result['rate'] == 1 / 3
    assert result['preference_data']['by_grasp'][family] == {'trials': 2, 'success': 1, 'rate': .5}
    assert result['preference_data']['not_evaluated_by_grasp'][family] == 1


def test_shake_samples_transient_slip_and_executes_full_wrist_motion():
    from table_1000.experts.asset_grasp import Evaluator
    model = types.SimpleNamespace(q=np.zeros(1), limits=np.array([[-2., 2.]]), arm_indices=lambda arm: [0])
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.model = model
    ctx = types.SimpleNamespace(observers=[], steps=0,
                                env=types.SimpleNamespace(contacts=types.SimpleNamespace(phase='start')),
                                geometry={'object': types.SimpleNamespace(points=lambda **kw: np.array([[.1, 0, 0]]))})
    ctx.tcp = lambda arm: np.eye(4)
    def poses():
        pose = np.eye(4)
        if ctx.steps == 3:
            pose[0, 3] = .006  # transient slip, recovered before every segment ends
        return {'object:body': pose}
    ctx.poses = poses
    def step(action=None):
        ctx.steps += 1
        if action:
            model.q[:] = action['right'][0]
        for observer in ctx.observers:
            observer(ctx)
    ctx.step = step
    ctx.hold = lambda seconds: [step() for _ in range(round(seconds * 25))]
    result = evaluator.shake(ctx, types.SimpleNamespace(body='object:body'), 'right', {})
    assert result['completed'] and not result['held']
    assert np.isclose(result['max_slip_m'], .006)
    assert np.isclose(result['wrist_excursion_rad'], .8)
    assert not ctx.observers


def test_failed_lift_ends_acceptance_without_trying_next_target(monkeypatch):
    from table_1000.experts import asset_grasp
    from table_1000.experts.context import ExpertFailed
    from table_1000.experts.grasps import Box
    pose = np.eye(4)
    attempts, released = [], []
    candidates = [types.SimpleNamespace(shape=Box((.02,.03,.04)),
                  names=[name], body='object:body') for name in ('handle', 'head')]
    candidates[1].fallback = True
    monkeypatch.setattr(asset_grasp, 'targets', lambda skill: iter(candidates))
    def fail(reason, **details):
        raise ExpertFailed(None, reason, **details)
    ctx = types.SimpleNamespace(observers=[], poses=lambda: {'object:body': pose.copy()},
          env=types.SimpleNamespace(body_mass=lambda body: [.1], center_of_mass=lambda body: np.zeros((1,3)),
              contacts=types.SimpleNamespace(surface_contacts={}, take_impulses=lambda: {})),
          tcp=lambda arm: np.eye(4), hold=lambda seconds: None, fail=fail, note=lambda *a, **k: None)
    solver = types.SimpleNamespace(world=types.SimpleNamespace(objects={}),
             carry=lambda *args: None, release=lambda *args: None, finger_gap=lambda: .02)
    def grasp(targets, **kwargs):
        target = targets[0]
        skill.held = {"right": {"target": target}}
        if attempts:
            assert released == ['right']
        attempts.append(target.names[0])
        assert kwargs['closure_only_record'] is True
        return {'arm':'right','supports':[],'grasp':{},'closure_check': {'held': False}}
    def move(*args):
        if len(attempts)==2:
            pose[2,3] += .1
    skill = types.SimpleNamespace(ctx=ctx, hands={'right':types.SimpleNamespace()},
            grasp=grasp, solvers={'right':solver}, move_by_displacement=move, release=released.append)
    report = {}
    import pytest
    with pytest.raises(ExpertFailed, match='object not lifted'):
        asset_grasp.pick(skill, report)
    assert attempts == ['handle'] and released == ['right']
    assert report['target_attempts'][0]['lift']['rise_m'] == 0


def test_whole_envelope_is_only_a_planning_fallback():
    import pytest
    from table_1000.experts.asset_grasp import pick, target_groups
    from table_1000.experts.context import ExpertFailed
    from table_1000.experts.grasps import Box
    parts = [types.SimpleNamespace(shape=Box((.02,) * 3), names=[name], body='object:body', fallback=fallback)
             for name, fallback in [('whole', True), ('part_a', False), ('part_b', False)]]
    assert [[t.names[0] for t in group] for group in target_groups(parts)] == [['part_a', 'part_b'], ['whole']]
    calls = []
    def grasp(group, **kwargs):
        calls.append([t.names[0] for t in group])
        raise ExpertFailed(None, 'no reachable grasp' if len(calls) == 1 else 'tracking')
    report = {}
    with pytest.raises(ExpertFailed, match='tracking'):
        pick(types.SimpleNamespace(ctx=None, grasp=grasp), report, parts)
    assert calls == [['part_a', 'part_b'], ['whole']]
    assert [row['failure']['reason'] for row in report['target_attempts']] == ['no reachable grasp', 'tracking']


def test_solid_whole_envelope_also_follows_authored_parts(monkeypatch):
    from table_1000.experts import asset_grasp
    from table_1000.experts.grasps import Box
    from table_1000.experts.skills.common import Target
    box = Box((.02,) * 3)
    geometry = types.SimpleNamespace(solid=lambda part, **kw: True,
        parts=lambda: [('body', 'body.a'), ('body', 'body.b')], shapes=[], points=lambda *args: box.points())
    def target(obj, part=None, **kwargs):
        names = ['object:body.a', 'object:body.b'] if part is None else ['object:' + part]
        return Target(obj, names, 'object:body', box, np.eye(4), box.points())
    skill = types.SimpleNamespace(ctx=types.SimpleNamespace(geometry={'object': geometry}), target=target)
    monkeypatch.setattr(asset_grasp, 'oriented_shape', lambda points: (box, np.eye(4)))
    groups = asset_grasp.target_groups(list(asset_grasp.targets(skill)))
    assert [[t.names for t in group] for group in groups] == [
        [['object:body.a'], ['object:body.b']], [['object:body.a', 'object:body.b']]]


def test_shake_search_restarts_at_next_target_and_preserves_failures():
    from table_1000.experts.asset_grasp import Evaluator
    evaluator = Evaluator.__new__(Evaluator)
    calls = []
    def attempt(seed, center, size, video, excluded):
        calls.append((seed, center, size, list(excluded)))
        return {'status': 'failed' if not excluded else 'success', 'hold': {'held': bool(excluded)},
                'selected_target_index': 2, 'target_attempts': [{}], 'target_count': 3,
                'cpu_s': 1., 'wall_s': 2., 'sim_s': 3.}
    evaluator._attempt = attempt
    result = evaluator.trial(7, (.5, -.15))
    assert calls == [(7, (.5, -.15), .1, []), (7, (.5, -.15), .1, [2])]
    assert result['status'] == 'failed'
    assert [a['status'] for a in result['preference_attempts']] == ['success']
    assert result['cpu_s'] == 2 and result['wall_s'] == 4 and result['sim_s'] == 6
    summary = summarize([result], 1)
    assert summary['success'] == 0 and summary['first_attempt_success'] == 0
    assert summary['attempts'] == 2


def test_shake_search_stops_when_geometric_targets_are_exhausted():
    from table_1000.experts.asset_grasp import Evaluator
    evaluator = Evaluator.__new__(Evaluator)
    evaluator._attempt = lambda *args: {'status': 'failed', 'hold': {'held': False},
        'selected_target_index': 0, 'target_attempts': [{}], 'target_count': 1, 'cpu_s': 1., 'wall_s': 2., 'sim_s': 3.}
    result = evaluator.trial(0)
    assert result['status'] == 'failed' and len(result['preference_attempts']) == 0


def test_later_unreachable_target_does_not_count_as_initially_unreachable():
    from table_1000.experts.asset_grasp import Evaluator
    evaluator = Evaluator.__new__(Evaluator)
    def attempt(seed, center, size, video, excluded):
        report = {'status': 'failed', 'target_attempts': [{}], 'target_count': 2,
                  'cpu_s': 1., 'wall_s': 2., 'sim_s': 3.}
        if not excluded:
            report.update(selected_target_index=0, hold={'held': False}, failure={'reason': 'slipped during shake'})
        else:
            report['failure'] = {'reason': 'no reachable grasp'}
        return report
    evaluator._attempt = attempt
    result = evaluator.trial(0)
    assert result['failure']['reason'] == 'slipped during shake'
    assert result['preference_attempts'][0]['failure']['reason'] == 'no reachable grasp'
    assert summarize([result], 1)['initial_unreachable'] == 0


def test_comparison_self_uses_same_assessed_denominator_on_both_sides(tmp_path):
    import json
    from table_1000.experts.asset_grasp.comparison import compare
    asset = 'objects/keyboard/000000'
    directory = tmp_path / asset / 'trials'
    directory.mkdir(parents=True)
    reports = []
    for seed, status in enumerate(['success'] * 8 + ['invalid_motion', 'not_evaluated']):
        report = {'seed': seed, 'status': status, 'cpu_s': 3., 'wall_s': 4., 'sim_s': 1.,
                  'trajectory': 'test.hdf5', 'load': {'host': 'test', 'cores': 2,
                  'samples': [{'load_1m': 1., 'cpu_s': 3.}]}}
        (directory / f'{seed:04d}.json').write_text(json.dumps(report))
        reports.append(report)
    (tmp_path / asset / 'test.hdf5').touch()
    (tmp_path / 'summary.json').write_text(json.dumps({
        'commit': 'same', 'assets': {asset: summarize(reports, len(reports))}}))
    result = compare(tmp_path, tmp_path)
    for row in (result['rows'][0], result['totals']['non_thin']):
        assert row['old_first'] == row['new_first'] == 8
        assert row['old_assessed'] == row['new_assessed'] == 8
        assert row['old_expected'] == row['new_expected'] == 10
        assert row['old_not_evaluated'] == row['not_evaluated'] == 1
        assert row['old_invalid_motion'] == row['invalid_motion'] == 1
