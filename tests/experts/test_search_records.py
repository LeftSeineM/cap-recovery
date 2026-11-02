"""Search observations retain failed branches without changing solver inputs or criteria."""
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from table_1000.experts.audit.trace import CallTrace, decode, failure_state, frame, replay
from table_1000.experts.audit.search import check, measurement
from table_1000.experts.audit.records import classify, export_case, statistics, media_batch, search_costs, planning_rollbacks
from table_1000.experts.context import ExpertFailed, Step, run_steps


def context():
    model = SimpleNamespace(q=np.zeros(2), base=np.eye(4))
    ctx = SimpleNamespace(steps=0, current='move', _skills=None, solvers={}, model=model,
                          rng=np.random.default_rng(3), scene=SimpleNamespace(OBJECTS={}))
    ctx.env = SimpleNamespace(audit_state=lambda: {'rigid_object': {'pen': {'root_pose': np.array([[ctx.steps, 0.]])}}},
                              targets=lambda: {}, contacts=SimpleNamespace(phase=None))
    ctx.skills = lambda: SimpleNamespace(held={})
    ctx.trace = CallTrace(ctx, 3, 0)
    return ctx


def write_batch(path, ctx, status='success'):
    path.mkdir(exist_ok=True)
    (path / 'episodes').mkdir(exist_ok=True)
    report = dict(seed=3, attempt=0, repeat=0, status=status, control_steps=ctx.steps,
                  call_trace='calls/3.json.gz', retry_archives=[], step_names=['move'],
                  start={'qpos': [0., 0.], 'base_pose': [0., 0., 0.]}, failure={'reason': 'failed'})
    ctx.trace.save(path / report['call_trace'])
    (path / 'episodes/3.json').write_text(json.dumps(report))
    (path / 'summary.json').write_text(json.dumps(dict(scene='000002', robot='x1pro_rm001', expert='0000',
        scope={'0': None}, layout=0, outside_scope=False, physics='isaac-cpu', commit='source')))
    return report


def planning_error(ctx, reason='IK Failed'):
    error = ExpertFailed('move', reason, kind='planning')
    if ctx.trace.search.failures:
        error.search_failure_id = ctx.trace.search.failures[-1]['id']
    raise error


def test_planning_event_replays_recorded_native_arguments_without_trajectory(tmp_path):
    ctx = context()
    calls = []
    def ik(arm, goal, attempts=7):
        calls.append((arm, goal.copy(), attempts))
        return 'IK Failed', None
    ctx.model.ik = ctx.trace.search.observe('ik', ik)
    with frame(ctx, 'step', 'move', {'index': 1}):
        with pytest.raises(ExpertFailed):
            with frame(ctx, 'skill', 'pick', {'obj': 'pen'}, concrete=True):
                assert ctx.model.ik('left', np.eye(4), attempts=9)[0] == 'IK Failed'
                planning_error(ctx, 'no reachable grasp')
    report = write_batch(tmp_path, ctx)
    state, snapshot = failure_state(tmp_path, report, 1)
    assert state['rigid_object']['pen']['root_pose'].tolist() == [[0., 0.]]
    assert snapshot['operation']['name'] == 'ik'
    assert not list(tmp_path.glob('*.hdf5'))
    restored = context()
    restored.model.ik = ik
    result = replay(restored, snapshot, SimpleNamespace(STEPS=[]))
    assert result['same_status'] and result['status'] == 'IK Failed'
    assert len(calls) == 2 and calls[0][2] == calls[1][2] == 9
    np.testing.assert_array_equal(calls[0][1], calls[1][1])
    export_case([tmp_path], 3, tmp_path / 'case', 1)
    canonical = json.loads((tmp_path / 'case/summary.json').read_text())['case']
    assert canonical['version'] == 1 and canonical['category'] == 'b'
    assert (tmp_path / 'case' / canonical['root']['checkpoint']).is_file()


def test_segment_preserves_initial_and_last_state_and_html_uses_relative_media(tmp_path):
    import h5py
    from table_1000.experts.audit.view import export_view
    ctx = context()
    with frame(ctx, 'step', 'move', {'index': 1}):
        with frame(ctx, 'skill', 'pick', {'obj': 'pen'}, concrete=True):
            ctx.steps = 2
            failure = check(ctx, 'Grasping.check_closure', False, kind='check')
            ctx.steps = 3
    report = write_batch(tmp_path, ctx)
    assert classify(ctx.trace.document())[0]['recovery']['kind'] == 'step_passed'
    with h5py.File(tmp_path / 'trajectories.hdf5', 'w') as file:
        data = file.create_group('data')
        data.attrs['env_args'] = json.dumps(dict(scene='000002', robot='x1pro_rm001'))
        group = data.create_group('demo_3')
        group.attrs['num_samples'] = 3
        group.create_dataset('initial_state/x', data=[[10]])
        group.create_dataset('states/x', data=[[11], [12], [13]])
    view = export_view(tmp_path, 3, tmp_path / 'view', media=True)
    assert view['nodes'] and any(e['category'] == 'a2' for e in view['edges'])
    page = (tmp_path / 'view/index.html').read_text()
    assert 'https://' not in page and 'http://' not in page.replace('http://www.w3.org/2000/svg', '')
    assert 'media/videos/' in page and 'media/stills/' in page
    media_batch(tmp_path, report, [dict(start=1, end=3)], tmp_path / 'segment')
    with h5py.File(tmp_path / 'segment/trajectories.hdf5') as file:
        group = file['data/demo_0']
        assert group['initial_state/x'][()].tolist() == [[11]]
        assert group['states/x'][()].tolist() == [[12], [13], [13]]


def test_motion_validation_rejection_retains_existing_wrapper_for_replay(tmp_path):
    from table_1000.experts.audit.search import observed_motion
    from table_1000.experts.solver import PlanningFailed
    ctx = context()
    class Solver:
        def __init__(self):
            self.ctx, self.model, self.arm = ctx, ctx.model, SimpleNamespace(name='left')
        @observed_motion
        def plan(self, kind, function, goals=(), problem=None):
            function()
            raise PlanningFailed('move', 'path collision', collision=('finger', 'table'))
    with frame(ctx, 'step', 'move', {'index': 1}):
        with frame(ctx, 'skill', 'pick', {'obj': 'pen'}, concrete=True):
            with pytest.raises(PlanningFailed) as error:
                Solver().plan('rrt', lambda: None, goals=[[1., 2.]], problem={'validation_step': .01})
            assert error.value.search_failure_id == 1
    failure = ctx.trace.search.failures[0]
    assert classify(ctx.trace.document()) == [] and failure['reason'] == 'path collision'
    assert failure['operation']['name'] == 'motion'
    assert decode(failure['operation']['arguments'])['problem']['validation_step'] == .01


def test_shared_encoding_preserves_independent_state_and_rejects_historical_schema(tmp_path):
    from table_1000.experts.audit.codec import load_document, save_document
    state = {'q': np.arange(100).tolist()}
    document = {'version': 2, 'failures': [{'state': state} for _ in range(100)]}
    path = tmp_path / 'calls.json.gz'
    save_document(path, document)
    assert load_document(path) == document
    raw = json.load(gzip.open(path, 'rt'))
    assert raw['values'] and path.stat().st_size < 1000
    with gzip.open(path, 'wt') as stream:
        json.dump({'version': 1}, stream)
    with pytest.raises(ValueError, match='convert'):
        load_document(path)


def test_terminal_case_starts_at_failed_step_instead_of_episode_start(tmp_path):
    ctx = context()
    with frame(ctx, 'step', 'first', {'index': 1}):
        ctx.steps = 2
    with frame(ctx, 'step', 'second', {'index': 2}):
        ctx.steps = 3
    report = write_batch(tmp_path, ctx, 'failed')
    report['step_names'] = ['first', 'second']
    (tmp_path / 'episodes/3.json').write_text(json.dumps(report))
    result = export_case([tmp_path], 3, tmp_path / 'case')
    assert result['steps_run'] == [2, 2]
    assert result['case']['root']['checkpoint'] == 'checkpoints/step1.hdf5'


def test_discarded_recording_is_copied_without_redirecting_main_recorder(tmp_path):
    from table_1000.experts.generate import Generator
    import torch
    class Episode:
        def __init__(self):
            self.data = {'states': {'x': [torch.tensor([1.]), torch.tensor([2.])]}}
        def pre_export(self):
            self.data['states']['x'] = torch.stack(self.data['states']['x'])
    class File:
        def create(self, path):
            self.path = path
        def add_env_args(self, args):
            assert args == {'scene': 'test'}
        def write_episode(self, episode, seed):
            assert seed == 3 and episode.success
            assert episode.data['states']['x'].tolist() == [[1.], [2.]]
            Path(self.path + '.hdf5').write_text('repeat only')
        def close(self):
            pass
    original = Episode()
    manager = SimpleNamespace(get_episode=lambda _: original, output='main.hdf5')
    generator = Generator.__new__(Generator)
    generator.env = SimpleNamespace(env=SimpleNamespace(recorder_manager=manager))
    generator.env_args = {'scene': 'test'}
    generator.recording = SimpleNamespace(TrajectoryFile=File)
    generator.export_recording(None, tmp_path / 'repeat-trajectories/3-1', 3, {'status': 'success'})
    assert manager.output == 'main.hdf5'
    assert isinstance(original.data['states']['x'], list)
    assert (tmp_path / 'repeat-trajectories/3-1.hdf5').is_file()


def test_planning_case_selects_owning_retry_archive_when_frame_number_is_reused(tmp_path):
    from copy import deepcopy
    from table_1000.experts.audit.codec import save_document
    ctx = context()
    ctx.model.ik = ctx.trace.search.observe('ik', lambda arm: ('IK Failed', None))
    with frame(ctx, 'step', 'move', {'index': 1}):
        with pytest.raises(ExpertFailed):
            with frame(ctx, 'skill', 'pick', {'obj': 'pen'}, concrete=True):
                ctx.model.ik('left')
                planning_error(ctx)
    report = write_batch(tmp_path, ctx)
    archived = tmp_path / 'attempts/3/0'
    archived.mkdir(parents=True)
    write_batch(archived, ctx, 'failed')
    active = deepcopy(ctx.trace.document())
    for item in active['frames']:
        item['branch'] = 'main/0-1'
    save_document(tmp_path / report['call_trace'], active)
    report['retry_archives'] = [{'batch': 'attempts/3/0', 'failure_id': 1, 'branch': 'main'}]
    (tmp_path / 'episodes/3.json').write_text(json.dumps(report))
    result = export_case([tmp_path], 3, tmp_path / 'case', 1)
    assert result['case']['root']['tree'] == 'sources/0/attempts/3/0/calls/3.json.gz'
    assert (tmp_path / 'case' / result['case']['root']['tree']).is_file()
    from table_1000.experts.generate import read_checkpoints
    from table_1000.scenes import resolve_batch_ref
    state = read_checkpoints(tmp_path / 'case/checkpoints/step0.hdf5')[1][3][1]
    source = json.loads(state['_expert_frame'])['batch']
    assert not Path(source).is_absolute()
    assert resolve_batch_ref(source, tmp_path / 'case') == archived


def test_view_namespaces_repeat_trees_and_preserves_expert_search_costs(tmp_path):
    from table_1000.experts.audit.view import graph
    ctx = context()
    with frame(ctx, 'step', 'move', {'index': 1}):
        with pytest.raises(ExpertFailed):
            with frame(ctx, 'skill', 'pick', concrete=True):
                planning_error(ctx, 'zero candidates')
    report = write_batch(tmp_path, ctx)
    report.update(repeat=1, call_trace='calls/3-1.json.gz')
    (tmp_path / 'episodes/3-1.json').write_text(json.dumps(report))
    ctx.trace.save(tmp_path / report['call_trace'])
    result = graph(tmp_path, 3)
    assert len(result['nodes']) == 2
    assert sum(node['search_costs']['expert_rejections']['count'] for node in result['nodes']) == 2
    assert all([f['category'] for f in node['failures']] == ['b'] for node in result['nodes'])
    assert len({node['branch'] for node in result['nodes']}) == 2


def test_replay_command_uses_gpu_assigned_by_queue(tmp_path, monkeypatch):
    from scripts.experts.search_records import main
    import scripts.experts.search_records as command
    ctx = context()
    ctx.model.ik = ctx.trace.search.observe('ik', lambda arm: ('IK Failed', None))
    with frame(ctx, 'step', 'move', {'index': 1}):
        with pytest.raises(ExpertFailed):
            with frame(ctx, 'skill', 'pick', {'obj': 'pen'}, concrete=True):
                ctx.model.ik('left')
                planning_error(ctx)
    write_batch(tmp_path, ctx)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '6')
    calls = []
    monkeypatch.setattr(command.subprocess, 'run', lambda args, **kwargs: calls.append(args))
    main(['replay', str(tmp_path), '--seed', '3', '--failure', '1', '--output', str(tmp_path / 'replay'), '--run'])
    assert calls[0][-2:] == ['--gpus', '6']


def test_segments_selects_repeat_trajectory_report(tmp_path, monkeypatch):
    from scripts.experts.search_records import main
    from table_1000.experts.audit import records
    ctx = context()
    with frame(ctx, 'step', 'move', {'index': 1}):
        with frame(ctx, 'skill', 'pick', {'obj': 'pen'}, concrete=True):
            ctx.steps = 2
            failure = check(ctx, 'Grasping.check_closure', False, kind='check')
            ctx.steps = 3
    report = write_batch(tmp_path, ctx)
    report.update(repeat=1, call_trace='calls/3-1.json.gz', trajectory='repeat-trajectories/3-1.hdf5')
    (tmp_path / 'episodes/3-1.json').write_text(json.dumps(report))
    ctx.trace.save(tmp_path / report['call_trace'])
    selected = []
    monkeypatch.setattr(records, 'media_batch', lambda batch, own, spans, output: selected.append(own))
    main(['segments', str(tmp_path), '--output', str(tmp_path / 'segments')])
    assert sorted(r['repeat'] for r in selected) == [0, 1]
    assert next(r for r in selected if r['repeat'])['trajectory'] == 'repeat-trajectories/3-1.hdf5'


def test_physical_rollback_span_includes_motion_before_the_last_failed_solve():
    from table_1000.experts.audit.records import physical_start
    failure = dict(category='a3', at=30, start=30, action_start=30,
                   recovery={'kind': 'rollback', 'at': 4})
    assert physical_start(failure) == 4


def test_single_repeat_view_includes_the_actual_recovery_state(tmp_path):
    from table_1000.experts.audit.view import graph
    ctx = context()
    with frame(ctx, 'step', 'move', {'index': 1}):
        ctx.steps = 2
        with frame(ctx, 'skill', 'pick', {'obj': 'pen'}, concrete=True):
            ctx.steps = 3
            failure = check(ctx, 'Grasping.check_closure', False, kind='check')
            ctx.steps = 5
            ctx.steps = 7
    report = write_batch(tmp_path, ctx)
    report.update(repeat=1, call_trace='calls/3-1.json.gz')
    (tmp_path / 'episodes/3-1.json').write_text(json.dumps(report))
    ctx.trace.save(tmp_path / report['call_trace'])
    value = graph(tmp_path, 3, repeat=1)
    assert {n['at'] for n in value['nodes']} == {0, 2, 3, 7}
    assert len({n['branch'] for n in value['nodes']}) == 1
    assert any(e['name'] == 'move' and e['start'] == 0 and e['end'] == 2 for e in value['edges'])
    assert any(e['category'] == 'a2' and e['end'] == 7 for e in value['edges'])


def test_classify_covers_facts_outcomes_and_never_mutates_the_tree():
    from copy import deepcopy
    def event(number, kind, step=0, **values):
        return dict(id=number, kind=kind, step=step, frame=step, branch='main', at=3,
                    action_start=1, reason=f'criterion {number}', cpu_s=1., wall_s=2.,
                    cost_start=[0., 0.], **values)
    tree = dict(frames=[dict(id=i, parent=None, kind='step', control=False) for i in range(3)] +
                [dict(id=8, parent=0, kind='skill', control=False, exit=8),
                 dict(id=9, parent=2, kind='skill', control=False, exit=None)],
                failures=[event(1, 'planning'), event(2, 'check'),
                event(3, 'check'), event(4, 'check'), event(5, 'record'),
                event(6, 'check', step=1), event(7, 'check', step=2)], outcomes=[
        dict(id=0, kind='rollback', step=0, branch='main', at=4,
             discarded=[2, 3], failure_id=3, control=1, restore=dict(branch='main/1-1', at=1),
             cost_start=[0., 0.], observed_clock=[3., 6.]),
        dict(id=1, kind='step_passed', step=0, frame=0, branch='main/1-1', at=8, observed_clock=[5., 10.]),
        dict(id=2, kind='step_abandoned', step=1, frame=1, branch='main', at=6,
             reason='give up', failure_id=6)])
    tree['failures'][3]['frame'] = 8
    tree['failures'][6]['frame'] = 9
    original = deepcopy(tree)
    classified = classify(tree)
    assert tree == original
    assert [(f['id'], f['category']) for f in classified] == [
        (3, 'a3'), ('b2', 'b'), (4, 'a2'), (7, 'pending')]
    rollback = classified[0]
    assert [f['id'] for f in rollback['components']] == [2]
    assert rollback['cpu_s'] == 3. and rollback['wall_s'] == 6.
    assert [f['id'] for f in classified[1]['components']] == [6]
    assert classified[-2]['end'] == 8
    assert classified[-1]['end'] is None
    # A later outcome is a new fact, not a mutation or a promotion of the event.
    tree['outcomes'].append(dict(id=3, kind='step_passed', step=2, frame=2,
                                 branch='main', at=9, observed_clock=[7., 14.]))
    tree['frames'][-1]['exit'] = 9
    assert next(f for f in classify(tree) if f['id'] == 7)['category'] == 'a2'
    assert tree['failures'] == original['failures']


@pytest.mark.parametrize('ending', ['passed', 'abandoned', 'interrupted'])
def test_expert_check_exception_is_b_even_when_caught_and_repaired(tmp_path, ending):
    from table_1000.experts.context import Context
    ctx = context()
    ctx.only = None
    ctx.checkpoint = lambda *args: None
    def repair(ctx):
        try:
            with frame(ctx, 'skill', 'close', concrete=True):
                ctx.steps = 3
                Context.fail(ctx, 'other drawer moved', kind='check', measured=.021)
        except ExpertFailed:
            ctx.steps = 5
    def final(ctx):
        assert 'category' not in ctx.trace.search.failures[0]
        ctx.steps = 7
        if ending != 'passed':
            raise ExpertFailed('move', 'abandoned or interrupted')
    if ending == 'passed':
        run_steps(ctx, [Step('move', repair, final)])
    else:
        with pytest.raises(ExpertFailed):
            run_steps(ctx, [Step('move', repair, final)])
    if ending == 'abandoned':
        ctx.trace.search.terminal(dict(status='failed', failure=dict(reason='abandoned')))
    values = classify(ctx.trace.document())
    count = 2 if ending == 'abandoned' else 1
    assert [f['category'] for f in values] == ['b'] * count
    assert not any('outcome' in f for f in ctx.trace.frames)
    assert not any('category' in f for f in ctx.trace.search.failures)
    assert values[0]['details']['measured'] == .021
    write_batch(tmp_path, ctx, 'success' if ending == 'passed' else 'error')
    assert statistics([tmp_path])['categories']['b']['count'] == count


def test_internal_planning_rejection_after_motion_is_search_cost_and_record_is_not_a_failure():
    ctx = context()
    with frame(ctx, 'step', 'move', {'index': 1}):
        with frame(ctx, 'skill', 'pick', concrete=True):
            with ctx.trace.search.decision('lookahead', {}) as decision:
                ctx.steps = 3
                ctx.trace.search.mark('need', outcome='rejected')
            check(ctx, 'closure', False, kind='record')
    values = classify(ctx.trace.document())
    assert values == []
    assert search_costs(ctx.trace.document())[0]['layers']['candidate_rejections']['count'] == 1
    assert ctx.trace.search.failures[-1]['kind'] == 'record'
    assert ctx.trace.search.checks[-1]['kind'] == 'record'


def test_rollback_counts_trigger_once_and_links_exception_wrappers():
    from table_1000.experts.context import Context
    ctx = context()
    with frame(ctx, 'step', 'move', {'index': 1}):
        with frame(ctx, 'skill', 'retry') as control:
            control['control'] = True
            with pytest.raises(ExpertFailed) as caught:
                with frame(ctx, 'skill', 'pick', concrete=True):
                    ctx.steps = 2
                    check(ctx, 'secondary', False, kind='check')
                    Context.fail(ctx, 'not lifted', kind='check')
            outcome = ctx.trace.search.rollback(control['id'], 1, caught.value, 0)
            ctx.trace.search.next_branch(control['id'], 1)
            ctx.steps = 1
            ctx.steps = 4
    assert len(ctx.trace.search.failures) == 2
    assert outcome['failure_id'] == 2
    values = classify(ctx.trace.document())
    assert len(values) == 1 and values[0]['category'] == 'a3'
    assert values[0]['id'] == 2 and [f['id'] for f in values[0]['components']] == [1]


def test_planning_rollback_keeps_both_physical_nodes_without_counting_an_a3(tmp_path):
    from table_1000.experts.audit.view import graph
    ctx = context()
    with frame(ctx, 'step', 'move', {'index': 1}):
        with frame(ctx, 'skill', 'retry') as control:
            check(ctx, 'no route', False, kind='planning')
            error = ExpertFailed('move', 'no route')
            error.search_failure_id = 1
            ctx.trace.search.rollback(control['id'], 1, error, 0)
            ctx.trace.search.next_branch(control['id'], 1)
    write_batch(tmp_path, ctx)
    assert classify(ctx.trace.document()) == []
    assert len(planning_rollbacks(ctx.trace.document())) == 1
    value = graph(tmp_path, 3)
    restore = next(e for e in value['edges'] if e['category'] == 'rollback')
    assert restore['source'] == 'main@0' and restore['target'] == 'main/1-1@0'
    assert {restore['source'], restore['target']} <= {n['id'] for n in value['nodes']}


@pytest.mark.parametrize('moved', [False, True])
def test_planning_rollback_counts_only_discarded_motion_and_exports_its_segment(tmp_path, monkeypatch, moved):
    from scripts.experts.search_records import main
    from table_1000.experts.audit import records
    ctx = context()
    ctx.model.ik = ctx.trace.search.observe('ik', lambda arm: ('IK Failed', None))
    with frame(ctx, 'step', 'move', {'index': 1}):
        with frame(ctx, 'skill', 'retry') as control:
            control['control'] = True
            with pytest.raises(ExpertFailed) as caught:
                with frame(ctx, 'skill', 'pick', concrete=True):
                    ctx.steps = 3 if moved else 0
                    ctx.model.ik('left')
                    planning_error(ctx, 'no route')
            outcome = ctx.trace.search.rollback(control['id'], 1, caught.value, 0)
            ctx.trace.search.next_branch(control['id'], 1)
            ctx.steps = 0
            ctx.steps = 4
    tree = ctx.trace.document()
    values = classify(tree)
    assert [f['category'] for f in values] == (['a1'] if moved else [])
    assert planning_rollbacks(tree)[0]['moved'] is moved
    write_batch(tmp_path, ctx)
    archived = tmp_path / 'discarded'
    own = write_batch(archived, ctx, 'failed')
    report = json.loads((tmp_path / 'episodes/3.json').read_text())
    report['retry_archives'] = [dict(batch='discarded', failure_id=outcome['failure_id'], branch='main')]
    (tmp_path / 'episodes/3.json').write_text(json.dumps(report))
    stats = statistics([tmp_path])
    assert stats['categories']['a1']['count'] == int(moved)
    assert len(stats['clips']) == int(moved)
    selected = []
    monkeypatch.setattr(records, 'media_batch', lambda batch, report, spans, output: selected.append((batch, report, spans)))
    main(['segments', str(tmp_path), '--output', str(tmp_path / 'segments')])
    if moved:
        assert len(selected) == 1 and selected[0][0] == archived
        assert selected[0][2] == [dict(start=0, end=3)]
        assert selected[0][1]['archive']['failure_id'] == outcome['failure_id']
        assert values[0]['kind'] == 'planning' and values[0]['reason'] == 'IK Failed'
        result = export_case([tmp_path], 3, tmp_path / 'case', outcome['failure_id'])
        assert result['case']['category'] == 'a1'
    else:
        assert selected == []


def test_planning_trigger_owns_discarded_checks_as_a1_components():
    ctx = context()
    with frame(ctx, 'step', 'move', {'index': 1}):
        with frame(ctx, 'skill', 'retry') as control:
            ctx.steps = 2
            check(ctx, 'earlier check', False, kind='check')
            number = check(ctx, 'no route', False, kind='planning')
            error = ExpertFailed('move', 'no route', kind='planning')
            error.search_failure_id = number
            ctx.trace.search.rollback(control['id'], 1, error, 0)
            ctx.trace.search.next_branch(control['id'], 1)
            ctx.steps = 4
    [failure] = classify(ctx.trace.document())
    assert failure['category'] == 'a1' and failure['id'] == number
    assert failure['criterion_id'] == 'no route' and failure['kind'] == 'planning'
    assert [part['criterion_id'] for part in failure['components']] == ['earlier check']


def test_outer_rollback_owns_the_surviving_prefix_before_an_inner_retry():
    ctx = context()
    with frame(ctx, 'step', 'move', {'index': 1}):
        with frame(ctx, 'skill', 'outer retry') as outer:
            check(ctx, 'prefix check', False, kind='check')
            with frame(ctx, 'skill', 'inner retry') as inner:
                check(ctx, 'inner check', False, kind='check')
                error = ExpertFailed('move', 'inner check')
                error.search_failure_id = 2
                ctx.trace.search.rollback(inner['id'], 1, error, 1)
                ctx.trace.search.next_branch(inner['id'], 1)
            check(ctx, 'outer check', False, kind='check')
            error = ExpertFailed('move', 'outer check')
            error.search_failure_id = 3
            ctx.trace.search.rollback(outer['id'], 1, error, 0)
            ctx.trace.search.next_branch(outer['id'], 1)
    values = classify(ctx.trace.document())
    assert [(f['id'], f['category']) for f in values] == [(2, 'a3'), (3, 'a3')]
    assert [f['id'] for f in values[1]['components']] == [1]


def test_failure_cost_intervals_are_disjoint_and_abandonment_is_not_timed(tmp_path):
    from table_1000.experts.audit.codec import save_document
    ctx = context()
    with frame(ctx, 'step', 'move', {'index': 1}):
        ctx.steps = 5
    report = write_batch(tmp_path, ctx, 'failed')
    tree = ctx.trace.document()
    tree['frames'][0].update(cpu_s=10., wall_s=20.)
    tree['failures'] = [dict(id=1, frame=0, step=0, branch='main', kind='planning', reason='IK', at=0,
                            cpu_s=2., wall_s=4., cost_start=[1., 1.]),
                        dict(id=2, frame=0, step=0, branch='main', kind='check', reason='miss', at=5,
                            cpu_s=0., wall_s=0., cost_start=[0., 0.])]
    tree['outcomes'] = [dict(id=0, kind='rollback', step=0, frame=0, branch='main', at=5,
                            discarded=[1,2], failure_id=2, control=0, restore=dict(branch='main/0-1', at=0),
                            cost_start=[0.,0.], observed_clock=[5.,10.]),
                        dict(id=1, kind='step_abandoned', step=0, frame=0, branch='main/0-1', at=5,
                             reason='give up', failure_id=2)]
    save_document(tmp_path / report['call_trace'], tree)
    result = statistics([tmp_path])
    assert result['categories']['a1']['cpu_s'] == 0.
    assert result['categories']['a3']['cpu_s'] == 5.
    assert result['categories']['a3']['wall_s'] == 10.
    assert result['categories']['b']['count'] == 1 and result['categories']['b']['cpu_s'] == 0.


def test_a3_archive_uses_trigger_id_and_branch_instead_of_a_retained_prefix(tmp_path):
    from copy import deepcopy
    from table_1000.experts.audit.codec import save_document
    from table_1000.experts.audit.records import failures
    ctx = context()
    with frame(ctx, 'step', 'move', {'index': 1}):
        with frame(ctx, 'skill', 'retry') as control:
            ctx.steps = 2
            check(ctx, 'secondary closure', False, kind='check')
            number = check(ctx, 'closure', False, kind='check')
            error = ExpertFailed('move', 'closure')
            error.search_failure_id = number
            ctx.trace.search.rollback(control['id'], 1, error, 0)
            ctx.trace.search.next_branch(control['id'], 1)
            ctx.steps = 3
    report = write_batch(tmp_path, ctx)
    for name in ('correct', 'later-prefix'):
        write_batch(tmp_path / name, ctx, 'failed')
    prefix = deepcopy(ctx.trace.document())
    prefix['failures'] = prefix['failures'][:1]
    prefix['outcomes'] = []
    save_document(tmp_path / 'later-prefix/calls/3.json.gz', prefix)
    report['retry_archives'] = [dict(batch='correct', failure_id=number, branch='main'),
                               dict(batch='later-prefix', failure_id=number + 1, branch='main')]
    (tmp_path / 'episodes/3.json').write_text(json.dumps(report))
    result = failures([tmp_path])
    assert len(result) == 1 and result[0]['category'] == 'a3'
    assert Path(result[0]['batch']).name == 'correct'


@pytest.mark.parametrize('handled', [True, False])
def test_expert_caught_planning_exception_is_search_cost_and_propagation_is_b(handled):
    ctx = context()
    def attempt():
        with frame(ctx, 'skill', 'pick', concrete=True):
            ctx.steps = 3  # prior motion never changes the nature of failure
            planning_error(ctx, 'no reachable grasp')
    if handled:
        with frame(ctx, 'step', 'move', {'index': 1}):
            with pytest.raises(ExpertFailed):
                attempt()
            with frame(ctx, 'skill', 'push', concrete=True):
                ctx.steps = 5
    else:
        with pytest.raises(ExpertFailed):
            with frame(ctx, 'step', 'move', {'index': 1}):
                attempt()
        ctx.trace.search.terminal(dict(status='failed', failure=dict(reason='no reachable grasp')))
    failures = classify(ctx.trace.document())
    assert [f['category'] for f in failures] == ['b']
    assert failures[0]['skill'] == 'pick'
    costs = search_costs(ctx.trace.document())
    if handled:
        assert costs[0]['layers']['expert_rejections']['count'] == 1
        assert costs[0]['layers']['expert_rejections']['reasons'] == {'no reachable grasp': 1}
    else:
        assert costs == []


def test_internal_solver_candidates_and_decisions_are_node_costs_even_under_subtask(tmp_path):
    from table_1000.experts.audit.view import graph
    ctx = context()
    statuses = iter(['IK Failed', 'IK Failed', 'Success'])
    ctx.model.ik = ctx.trace.search.observe('ik', lambda arm: (next(statuses), None))
    with frame(ctx, 'step', 'move', {'index': 1}):
        with frame(ctx, 'skill', 'retry', concrete=False):
            with frame(ctx, 'subtask', 'internal recovery'):
                with pytest.raises(ExpertFailed):
                    with frame(ctx, 'skill', 'pick', concrete=True):
                        with ctx.trace.search.decision('grasps', {}) as decision:
                            for candidate in range(2):
                                ctx.model.ik('left')
                                ctx.trace.search.mark('IK', candidate, 'rejected', reason='IK Failed')
                            ctx.trace.search.finish(decision, [], SimpleNamespace())
                        planning_error(ctx)
                assert ctx.model.ik('left')[0] == 'Success'
    tree = ctx.trace.document()
    assert classify(tree) == []
    costs = search_costs(tree)[0]['layers']
    assert {k: v['count'] for k,v in costs.items()} == dict(solver_calls=3, candidate_rejections=2, decision_exhaustions=1)
    assert costs['solver_calls']['reasons'] == {'ik: IK Failed': 2, 'ik: Success': 1}
    write_batch(tmp_path, ctx)
    assert statistics([tmp_path])['search_costs']['solver_calls']['count'] == 3
    nodes = graph(tmp_path, 3)['nodes']
    assert sum(len(n['failures']) for n in nodes) == 0
    assert nodes[0]['search_costs']['solver_calls']['count'] == 3


def test_distinct_expert_failures_at_same_control_have_independent_ids():
    ctx = context()
    with frame(ctx, 'step', 'move', {'index': 1}):
        for _ in range(2):
            with pytest.raises(ExpertFailed):
                with frame(ctx, 'skill', 'pick', concrete=True):
                    raise ExpertFailed('move', 'zero candidates', kind='planning')
    assert [(f['id'], f['category']) for f in classify(ctx.trace.document())] == [(1, 'b'), (2, 'b')]
    assert [f['id'] for f in ctx.trace.search.failures] == [1, 2]
    assert search_costs(ctx.trace.document())[0]['layers']['expert_rejections']['count'] == 2



def test_successful_release_entry_is_exportable_when_only_the_step_check_fails(tmp_path):
    from table_1000.experts.audit.trace import snapshot_state
    ctx = context()
    ctx.steps = 2
    with pytest.raises(ExpertFailed):
        with frame(ctx, 'step', 'store', {'index': 1}) as step:
            with frame(ctx, 'skill', 'release', {'obj': 'pen'}, concrete=True):
                ctx.steps = 5
            step['phase'] = 'check'
            raise ExpertFailed('store', 'not stored', kind='check')
    report = write_batch(tmp_path, ctx, 'failed')
    state, snapshot = snapshot_state(tmp_path, report)
    assert state['rigid_object']['pen']['root_pose'].tolist() == [[2., 0.]]
    assert snapshot['frame']['name'] == 'release' and snapshot['check']
    assert snapshot['frame']['exit'] == 5


def test_segments_uses_actual_archive_report_for_nonzero_repeat(tmp_path, monkeypatch):
    from scripts.experts.search_records import main
    from table_1000.experts.audit import records
    ctx = context()
    with frame(ctx, 'step', 'store', {'index': 1}):
        with frame(ctx, 'skill', 'retry') as control:
            ctx.steps = 2
            number = check(ctx, 'stored', False, kind='check')
            error = ExpertFailed('store', 'not stored', kind='check')
            error.search_failure_id = number
            ctx.trace.search.rollback(control['id'], 1, error, 0)
            ctx.trace.search.next_branch(control['id'], 1)
            ctx.steps = 3
    report = write_batch(tmp_path, ctx)
    archived = tmp_path / 'attempts/3-1/0'
    archived.mkdir(parents=True)
    own = write_batch(archived, ctx, 'failed')
    own.update(repeat=1, trajectory='archived-trajectory.hdf5')
    (archived / 'episodes/3.json').write_text(json.dumps(own))
    report.update(repeat=1, retry_archives=[dict(batch='attempts/3-1/0', failure_id=number, branch='main')])
    (tmp_path / 'episodes/3.json').unlink()
    (tmp_path / 'episodes/3-1.json').write_text(json.dumps(report))
    selected = []
    monkeypatch.setattr(records, 'media_batch', lambda batch, own, spans, output: selected.append((batch,own)))
    main(['segments', str(tmp_path), '--output', str(tmp_path / 'segments')])
    assert len(selected) == 1 and selected[0][0] == archived
    assert selected[0][1]['trajectory'] == 'archived-trajectory.hdf5'
    assert selected[0][1]['repeat'] == 1


def test_expert_callback_control_is_transparent_but_skill_adaptation_is_not():
    from table_1000.experts.skills import retry
    ctx = context()
    def expert():
        with pytest.raises(ExpertFailed):
            with frame(ctx, 'skill', 'chain', concrete=False):
                with frame(ctx, 'subtask', 'pick'):
                    with frame(ctx, 'skill', 'pick', concrete=True):
                        raise ExpertFailed('pick', 'no reachable grasp', kind='planning')
        with frame(ctx, 'skill', 'push', concrete=True):
            ctx.steps = 2
    with frame(ctx, 'step', 'move', {'index': 1}):
        retry(ctx, expert, attempts=1)
    values = classify(ctx.trace.document())
    assert [(f['category'], f['skill']) for f in values] == [('b', 'chain')]
    assert ctx.trace.frames[1]['control']
    costs = search_costs(ctx.trace.document())[0]['layers']['expert_rejections']
    assert costs['count'] == 1 and costs['frames'] == [2]


def test_check_recovered_inside_the_expert_called_skill_remains_a2():
    from table_1000.experts.context import Context
    ctx = context()
    with frame(ctx, 'step', 'move', {'index': 1}):
        with frame(ctx, 'skill', 'chain', concrete=False):
            with pytest.raises(ExpertFailed):
                with frame(ctx, 'skill', 'pick', concrete=True):
                    ctx.steps = 2
                    Context.fail(ctx, 'not lifted', kind='check', rise_m=.001)
            ctx.steps = 5
    [failure] = classify(ctx.trace.document())
    assert failure['category'] == 'a2' and failure['reason'] == 'not lifted'
    assert failure['end'] == 5 and failure['details']['rise_m'] == .001


def test_expert_predicate_caught_without_a_skill_frame_is_b():
    from table_1000.experts.skills.semantic import require
    from table_1000.experts.context import Context
    ctx = context()
    ctx.fail = lambda *args, **kwargs: Context.fail(ctx, *args, **kwargs)
    with frame(ctx, 'step', 'store', {'index': 1}):
        with pytest.raises(ExpertFailed):
            require(ctx, False, 'not stored', kind='check')
        ctx.steps = 3
    [failure] = classify(ctx.trace.document())
    assert failure['category'] == 'b' and failure['task_outcome'] == 'expert_check'
    assert failure['reason'] == 'not stored'


@pytest.mark.parametrize('physical', [False, True])
def test_place_propagates_the_last_solve_or_physical_failure_without_relabeling_it(physical):
    from table_1000.experts.context import Context
    from table_1000.experts.solver import PlanningFailed
    from table_1000.experts.skills.place.strict import Placing
    ctx = context()
    ctx.spec = SimpleNamespace(base=None)
    ctx.poses = lambda: {'pen:body': np.eye(4)}
    ctx.note = lambda *args,**kwargs: None
    ctx.fail = lambda *args,**kwargs: Context.fail(ctx,*args,**kwargs)
    skill = Placing()
    target = SimpleNamespace(obj='pen',body='pen:body',points=np.array([[0.,0.,.2]]))
    skill.ctx, skill.held = ctx, {'left':{'target':target}}
    def move(*args,**kwargs):
        if physical:
            ctx.steps = 2
            Context.fail(ctx,'tracking',PlanningFailed,kind='check',position_error_m=.02)
        raise PlanningFailed('place','IK Failed',kind='planning')
    skill.solvers = {'left':SimpleNamespace(tcp=lambda:np.eye(4),move_to_pose_with_RRTConnect=move)}
    skill.placements = lambda *args,**kwargs: [(np.eye(4),np.zeros(3))]
    skill.down_goals = lambda *args: [np.zeros(2)]
    skill.stow = lambda: False
    with frame(ctx,'step','store',{'index':1}):
        with pytest.raises(ExpertFailed) as error:
            with frame(ctx,'skill','place',concrete=True):
                skill.place('pen',(np.eye(4),[-1.,-1.],[1.,1.]),arm='left')
        assert error.value.reason == 'no placement'
        assert error.value.details['last'] == ('tracking' if physical else 'IK Failed')
        assert error.value.kind == ('check' if physical else 'planning')
        with frame(ctx,'skill','release',concrete=True):
            ctx.steps += 1
    values = classify(ctx.trace.document())
    assert [f['category'] for f in values] == ['b']
    if physical:
        # The aggregate 'no placement' links to the existing physical failure.
        assert sum(f['kind']=='check' for f in ctx.trace.search.failures) == 1
        assert values[0]['origin_reason'] == 'tracking'
        assert values[0]['reason'] == 'no placement'
        assert values[0]['details']['position_error_m'] == .02
    else:
        assert search_costs(ctx.trace.document())[0]['layers']['expert_rejections']['count'] == 1


@pytest.mark.parametrize('abandoned', [False, True])
def test_raw_planning_event_replays_when_it_is_search_cost_or_step_abandonment(tmp_path, abandoned, capsys):
    from scripts.experts.search_records import main
    ctx = context()
    invocations = []
    def ik(arm, attempts=4):
        invocations.append((arm, attempts))
        return 'IK Failed', None
    ctx.model.ik = ctx.trace.search.observe('ik', ik)
    def execute():
        with frame(ctx, 'step', 'move', {'index': 1}):
            with frame(ctx, 'skill', 'pick', {'obj': 'pen'}, concrete=True):
                ctx.model.ik('left', attempts=8)
                if abandoned:
                    planning_error(ctx)
    if abandoned:
        with pytest.raises(ExpertFailed):
            execute()
    else:
        execute()
    if abandoned:
        ctx.trace.search.terminal(dict(status='failed', failure={'reason': 'IK Failed'}))
    assert [f['category'] for f in classify(ctx.trace.document())] == (['b'] if abandoned else [])
    report = write_batch(tmp_path, ctx, 'failed' if abandoned else 'success')
    _, snapshot = failure_state(tmp_path, report, 1)
    restored = context()
    restored.model.ik = ik
    assert replay(restored, snapshot, SimpleNamespace(STEPS=[]))['same_status']
    assert invocations == [('left', 8), ('left', 8)]
    main(['replay', str(tmp_path), '--seed', '3', '--failure', '1', '--output', str(tmp_path/'replay')])
    assert '--from' in capsys.readouterr().out
    case = json.loads((tmp_path/'replay/case/summary.json').read_text())['case']
    assert case['kind'] == 'planning' and case['category'] == ('b' if abandoned else None)
    assert case['failure_id'] == 1


def test_planning_replay_reports_nonplanning_missing_state_and_unknown_event(tmp_path):
    ctx = context()
    with frame(ctx, 'step', 'move', {'index': 1}):
        check(ctx, 'physical.result', False, kind='check')
    report = write_batch(tmp_path, ctx)
    with pytest.raises(ValueError, match='is check, not planning'):
        failure_state(tmp_path, report, 1)
    with pytest.raises(ValueError, match='event 99 not found'):
        export_case([tmp_path], 3, tmp_path/'unknown', 99)
    event = ctx.trace.search.failures[0]
    event['kind'] = 'planning'
    ctx.trace.save(tmp_path/report['call_trace'])
    with pytest.raises(ValueError, match='no replayable state'):
        failure_state(tmp_path, report, 1)
    event['state'] = ctx.trace.frames[0]['state']
    ctx.trace.save(tmp_path/report['call_trace'])
    with pytest.raises(ValueError, match='no replayable operation or callable skill'):
        failure_state(tmp_path, report, 1)


def test_repeat_trajectory_is_explicit_and_archive_is_its_own_recording(tmp_path):
    from table_1000.experts.audit.records import trajectory
    assert trajectory(tmp_path, dict(status='success', repeat=0)) == tmp_path/'trajectories.hdf5'
    assert trajectory(tmp_path, dict(status='failed', repeat=0)) == tmp_path/'trajectories_failed.hdf5'
    assert trajectory(tmp_path, dict(status='success', repeat=1)) is None
    assert trajectory(tmp_path, dict(status='failed', repeat=2, trajectory='repeat-trajectories/3-2.hdf5')) == tmp_path/'repeat-trajectories/3-2.hdf5'
    assert trajectory(tmp_path, dict(status='failed', repeat=1, from_retry=dict(control=3, retry=1))) == tmp_path/'trajectories_failed.hdf5'
