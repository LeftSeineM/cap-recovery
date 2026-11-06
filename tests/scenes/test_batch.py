from tests.common.batch_worker import manifest, publish, expert_job, expert_result
"""Batch runs and the three scene steps without a simulator: processes, seeds, initial-state files, merged batches,
the choice of renders."""

from pathlib import Path
import json
import sys

import h5py
import numpy as np
from PIL import Image
import pytest

from scripts.scenes import build_scene, render, run_expert
from table_1000.experts.generate import PROCESS_EXITED, summarize
from table_1000.common import batch as scene_batch
from table_1000.common.batch import launch, parse_seeds, plan, run_jobs, seed_range, split
from table_1000.scenes.initial import read_states, write_states
from table_1000.scenes.recordings import merge, video_path, videos

STATE = {'articulation': {'robot': {'joint_position': np.zeros((1, 3), np.float32)}},
         'rigid_object': {'pen': {'root_pose': np.ones((1, 7), np.float32)}}, 'deformable_object': {}}


@pytest.fixture(autouse=True)
def scene_sources(tmp_path):
    """Batch controllers read source identity before launching their mocked workers."""
    import shutil
    from table_1000.scenes import ROOT
    destination = tmp_path / 'checkout' / 'asset_sources' / 'scenes' / '000000'
    shutil.copytree(ROOT / 'asset_sources' / 'scenes' / '000000', destination)


def test_seeds_are_ranges_and_single_seeds(tmp_path):
    assert parse_seeds(['100:103', '7', '101']) == [7, 100, 101, 102]
    assert seed_range([0, 1, 2, 5, 7, 8]) == '0:3 5 7:9'
    assert parse_seeds(seed_range([3, 4, 9]).split()) == [3, 4, 9]
    assert split(list(range(10)), 3) == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert split([5, 6], 8) == [[5], [6]]
    assert plan(list(range(10)), lambda seed: None, None, 3) == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    from table_1000.common.batch_runtime import Unit, load_config
    from table_1000.common.batch_timing import Timings, record
    units = [Unit((asset, seed), {}, f'{asset}/{seed}.json', ('asset', asset, 'robot', 'cpu', str(seed)),
                  ('asset', asset, 'robot', 'cpu'))
             for asset, seeds in [('a', range(4)), ('b', range(2)), ('c', range(2))] for seed in seeds]
    keys = [u.key for u in units]
    empty = Timings(tmp_path, units)
    cold = plan(keys, lambda key: None, None, 3, timing=empty)
    assert len(cold) == 3 and sorted(item for part in cold for item in part) == sorted(keys)
    by_key = {u.key: u for u in units}
    for asset, seed, wall in [('a', 0, 100), ('a', 1, 300), ('b', 0, 20)]:
        record(tmp_path, tmp_path / 'batch', by_key[asset, seed], dict(wall_s=wall, cpu_s=wall,
               save_wall_s=10., save_cpu_s=5., status='success', app_start_s=10., asset_load_s=30.))
    timing = Timings(tmp_path, units)
    assert timing.coverage == (3, 8)
    assert timing.costs['a', 2] == 210 and timing.costs['b', 1] == 30 and timing.costs['c', 0] == 110
    parts = plan(keys, lambda key: None, None, 3, timing=timing)
    assert len(parts) == 3 and sorted(item for part in parts for item in part) == sorted(keys)
    assert list(map(timing.wall, parts)) == sorted(map(timing.wall, parts), reverse=True)
    assert any(len({asset for asset, _ in part}) > 1 for part in parts)
    assert len(plan(keys, lambda key: key[0], None, 5, timing=timing)) == 5
    assert load_config().jobs('auto', 12000., 200) == 20



def test_jobs_share_the_listed_gpus(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command[-1])
        return type('Done', (), {'wait': lambda self: int(command[0] == 'bad')})()

    monkeypatch.setattr(scene_batch.subprocess, 'Popen', run)
    jobs = [([name], tmp_path / name / 'run.log', name) for name in ('a', 'bad', 'c')]
    assert run_jobs(jobs, [4, 5]) == ['bad']
    assert len(calls) == 3 and set(calls) <= {'4', '5'}


def test_launcher_runs_every_part_with_its_gpu(tmp_path):
    """Each part runs through the launcher prefix without --gpu and reads the GPU the launcher set."""
    show = 'from table_1000.common.batch import worker_gpu; import sys; print(worker_gpu(None)); sys.exit(len(sys.argv) - 1)'
    jobs = [([sys.executable, '-c', show, *extra], tmp_path / name / 'run.log', name)
            for name, extra in (('a', []), ('b', ['fail']))]
    failed = launch(jobs, 'env CUDA_VISIBLE_DEVICES=5')
    assert failed == ['b']
    assert (tmp_path / 'a' / 'run.log').read_text().strip() == '5'


def test_initial_states_round_trip(tmp_path):
    write_states(tmp_path / 'states.hdf5', {12: (2, STATE), 3: (0, STATE)}, {'robot': 'x1pro'})
    attrs, states = read_states(tmp_path / 'states.hdf5')
    assert attrs['robot'] == 'x1pro' and list(states) == [3, 12]
    attempt, state = states[12]
    assert attempt == 2 and state['deformable_object'] == {}
    assert np.array_equal(state['rigid_object']['pen']['root_pose'], STATE['rigid_object']['pen']['root_pose'])


def trajectory_file(path, seeds):
    with h5py.File(path, 'w') as file:
        data = file.create_group('data')
        data.attrs.update(total=0, env_args=json.dumps({'scene': '000000', 'robot': 'x1pro'}))
        for seed in seeds:
            group = data.create_group(f'demo_{seed}')
            group.attrs['num_samples'] = 2
            group.create_dataset('actions', data=np.zeros((2, 1)))


def test_merge_orders_episodes_by_seed(tmp_path):
    trajectory_file(tmp_path / 'a.hdf5', [10, 2])
    trajectory_file(tmp_path / 'b.hdf5', [5])
    assert merge([tmp_path / 'a.hdf5', tmp_path / 'b.hdf5', tmp_path / 'missing.hdf5'], tmp_path / 'all.hdf5') == 3
    with h5py.File(tmp_path / 'all.hdf5', 'r') as file:
        assert list(file['data']) == ['demo_10', 'demo_2', 'demo_5']   # HDF5 lists names alphabetically
        assert file['data'].attrs['total'] == 6 and 'env_args' in file['data'].attrs


def test_merge_takes_attributes_from_a_file_with_episodes(tmp_path):
    with h5py.File(tmp_path / 'stopped.hdf5', 'w') as file:     # created, stopped before the first export
        file.create_group('data').attrs.update(total=0, env_args=json.dumps({'env_name': '', 'type': 2}))
    trajectory_file(tmp_path / 'b.hdf5', [7])
    merge([tmp_path / 'stopped.hdf5', tmp_path / 'b.hdf5'], tmp_path / 'all.hdf5')
    with h5py.File(tmp_path / 'all.hdf5', 'r') as file:
        assert list(file['data']) == ['demo_7']
        assert json.loads(file['data'].attrs['env_args'])['robot'] == 'x1pro'


def test_summary_counts_every_seed():
    reports = [{'seed': 0, 'status': 'success', 'duration_s': 20., 'wall_s': 60.},
               {'seed': 1, 'status': 'failed', 'failure': {'subtask': 'close the drawer', 'reason': 'handle not held'},
                'wall_s': 40.},
               {'seed': 3, 'status': 'error', 'failure': {'subtask': 'pick up the pen', 'reason': 'error'}, 'wall_s': 1.}]
    summary = summarize(reports, [0, 1, 2, 3])
    assert summary['seeds'] == {'range': '0:4', 'count': 4} and summary['success_rate'] == .25
    assert summary['failures'] == {PROCESS_EXITED: [2], 'close the drawer: handle not held': [1],
                                   'pick up the pen: error': [3]}


def test_render_chooses_episodes():
    succeeded, failed = [9, 2, 5], [7, 3]
    assert render.choose(succeeded, failed) == [2, 5, 9]
    assert render.choose(succeeded, failed, count=2) == [2, 5]
    assert render.choose(succeeded, failed, failures=True) == [3, 7]
    assert render.choose(succeeded, failed, failures=True, count=1) == [3]
    assert render.choose(succeeded, failed, seeds=[9, 3, 5], count=2) == [3, 5]
    with pytest.raises(ValueError) as error:
        render.choose(succeeded, failed, seeds=[2, 4])
    assert error.value.args[0] == [4]
    with pytest.raises(ValueError):
        render.choose(succeeded, failed, seeds=[2], failures=True)


@pytest.fixture
def batch(tmp_path):
    directory = tmp_path / 'data' / '000000' / 'x1pro' / '0000' / 'isaac-cpu'
    directory.mkdir(parents=True)
    trajectory_file(directory / 'trajectories.hdf5', [0, 1, 4])
    (directory / 'summary.json').write_text(json.dumps({'seeds': {'range': '0:5', 'count': 5}, 'videos': {}, 'fails': {},
                                                        'failures': {'close the drawer: x': [2, 3]}}))
    return directory


def test_render_keeps_other_videos(batch, tmp_path, monkeypatch):
    trajectory_file(batch / 'trajectories_failed.hdf5', [3, 2])
    video_path(batch, 4, 'wide').parent.mkdir(parents=True)
    assert video_path(batch, 4, 'wide') == batch / 'videos' / '4' / 'wide.mp4'
    video_path(batch, 4, 'wide').write_bytes(b'old')
    chosen = []

    def jobs(commands, gpus):
        assert gpus == [0, 1]
        for command, _, _ in commands:
            data = manifest(command)
            seeds = [unit['data']['seed'] for unit in data['units']]
            chosen.append(seeds)
            merge = data['options']['merge']
            for seed in seeds:
                target = video_path(batch, seed, None if merge else 'wide', failed=seed in (2, 3))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b'new')
                publish(data, next(u for u in data['units'] if u['data']['seed'] == seed), {'status': 'success'})
        return []

    monkeypatch.setattr(scene_batch, 'run_jobs', jobs)
    monkeypatch.setattr(render, 'check_cameras', lambda *args: None)
    common = ['000000', '--physics', 'isaac-cpu', '--gpus', '0', '1', '--jobs', '4',
              '--input', str(tmp_path / 'data'), '--root', str(tmp_path / 'checkout')]
    render.main([*common, '--cameras', 'wide', '--count', '2'])
    render.main([*common, '--cameras', 'wide', 'face_view', '--merge', '--failed'])
    assert chosen == [[0], [1], [2], [3]]
    assert videos(batch) == {'videos': {'wide': [0, 1, 4]}, 'fails': {'merge': [2, 3]}}
    assert (batch / 'fails' / '2' / 'merge.mp4').is_file()
    assert video_path(batch, 4, 'wide').read_bytes() == b'old'
    summary = json.loads((batch / 'summary.json').read_text())
    assert summary['videos'] == {'wide': '0:2 4'} and summary['fails'] == {'merge': '2:4'}


def test_run_expert_merges_parts(tmp_path, monkeypatch):
    initial = tmp_path / 'assets' / 'scenes' / '000000' / 'x1pro'
    initial.mkdir(parents=True)
    write_states(initial / 'initial_states.hdf5', {seed: (0, STATE) for seed in range(4)},
                 {'scene': '000000', 'robot': 'x1pro'})

    rounds = []

    def jobs(commands, gpus):
        rounds.append(len(commands))
        for command, _, _ in commands:
            data, work, seeds = expert_job(command)
            (work / 'episodes').mkdir(parents=True)
            # Seed 2 is cancelled during an HDF5 write, before its report.
            trajectory_file(work / 'trajectories.hdf5', [seed for seed in seeds if seed <= 2])
            if 2 in seeds:
                (work / 'trajectories.hdf5').write_bytes(b'\x89HDF\r\n\x1a\n')
                continue
            trajectory_file(work / 'trajectories.hdf5', seeds)
            for seed in seeds:
                expert_result(data, work, seed, {'seed': seed,
                    'status': 'error' if seed == 1 and data['attempt'] == 0 else 'success',
                    'failure': {'reason': 'interrupted result'}, 'duration_s': 1., 'wall_s': 2.})
        return []

    monkeypatch.setattr(scene_batch, 'run_jobs', jobs)
    output = tmp_path / 'data'
    with pytest.raises(SystemExit, match='no episode: seeds 2;'):
        run_expert.main(['000000', '--physics', 'isaac-cpu', '--gpus', '0', '1', '--jobs', '3', '--input', str(tmp_path / 'assets/scenes'),
                         '--output', str(output), '--root', str(tmp_path / 'checkout')])
    assert rounds == [3, 2]          # seeds 2 and 3 once more
    directory = output / '000000' / 'x1pro' / '0000' / 'isaac-cpu'
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['failures'] == {PROCESS_EXITED: [2]} and summary['succeeded'] == 3
    assert summary['missing_seeds'] == [2]
    assert sorted(path.name for path in (directory / 'episodes').iterdir()) == ['0.json', '1.json', '3.json']
    with h5py.File(directory / 'trajectories.hdf5', 'r') as file:
        assert sorted(file['data']) == ['demo_0', 'demo_1', 'demo_3']


@pytest.mark.parametrize('problems', ['failed', 'all', 'none'])
def test_run_expert_dev_runs_are_kept_apart(tmp_path, monkeypatch, problems, capfd):
    initial = tmp_path / 'assets' / 'scenes' / '000000' / 'x1pro'
    initial.mkdir(parents=True)
    write_states(initial / 'initial_states.hdf5', {seed: (0, STATE) for seed in range(4)},
                 {'scene': '000000', 'robot': 'x1pro'})

    def jobs(commands, gpus):
        for command, _, _ in commands:
            assert manifest(command)['options']['problems'] == problems
            data, work, seeds = expert_job(command)
            (work / 'episodes').mkdir(parents=True)
            trajectory_file(work / 'trajectories.hdf5', seeds)
            for seed in seeds:
                expert_result(data, work, seed, {'seed': seed, 'status': 'success', 'duration_s': 1., 'wall_s': 2.})
        return []

    monkeypatch.setattr(scene_batch, 'run_jobs', jobs)
    common = ['000000', '--physics', 'isaac-cpu', '--input', str(tmp_path / 'assets/scenes'),
              '--root', str(tmp_path / 'checkout')]
    with pytest.raises(SystemExit):     # some seeds are a development run
        run_expert.main([*common, '--output', str(tmp_path / 'data'), '--seeds', '1'])
    assert not (tmp_path / 'data').exists()
    run_expert.main([*common, '--dev', '--seeds', '1', '3', '--output', str(tmp_path / 'runs'), '--dry-run'])
    assert "BATCH ('000000', 'x1pro'): 2 jobs" in capfd.readouterr().out
    assert list((tmp_path / 'runs').rglob('plan.json'))
    run_expert.main([*common, '--dev', '--name', 'try', '--seeds', '1', '3', '--output', str(tmp_path / 'runs'),
                     *(['--problems', problems] if problems != 'none' else [])])
    [directory] = [p.parent for p in (tmp_path / 'runs').rglob('summary.json')]
    assert directory.name.endswith('-nocommit-x1pro-0000-layout0-try')
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['run'] == directory.name and '--dev' in summary['command'] and summary['succeeded'] == 2
    assert summary['problems'] == problems
    with h5py.File(directory / 'trajectories.hdf5', 'r') as file:
        assert sorted(file['data']) == ['demo_1', 'demo_3']


def test_run_names_carry_the_commit():
    from uuid import uuid4
    head = uuid4().hex[:7]
    assert run_expert.run_name(head + '+dirty', 'step2').endswith(f'-{head}-dirty-step2')
    assert run_expert.run_name(None).endswith('-nocommit')


def test_build_scene_combines_parts(tmp_path, monkeypatch):
    rounds = []
    assets = tmp_path / 'alternate-assets'
    assets.mkdir()
    monkeypatch.chdir(tmp_path / 'checkout')

    def jobs(commands, gpus):
        for command, _, _ in commands:
            data = manifest(command)
            assert Path(data['options']['assets']) == assets
            seeds = [u['data']['seed'] for u in data['units']]
            work = Path(data['work']) / str(data['units'][0]['data']['layout'])
            rounds.append(seeds)
            for name in ('states', 'images', 'records'):
                (work / name).mkdir(parents=True)
            for seed in seeds:
                if seed == 2 and data['attempt'] == 0 or seed == 3 and data['attempt'] == 1:
                    continue      # the process died here: earlier seeds keep their results
                if seed != 1:
                    write_states(work / 'states' / f'{seed}.hdf5', {seed: (0, STATE)}, {'scene': '000000', 'robot': 'x1pro'})
                    Image.new('RGB', (64, 36), 'red').save(work / 'images' / f'{seed}.jpg')
                record = {'seed': seed, 'attempt': None if seed == 1 else 0,
                          'attempts': [{'attempt': 0, 'settle': {'objects': {}, 'robot_contacts': []}}]}
                if seed == 3:
                    (work / 'states' / f'{seed}.hdf5').unlink()
                    record.update(attempt=None, error='RuntimeError: CUDA error')
                publish(data, next(u for u in data['units'] if u['data']['seed'] == seed), dict(
                    status='error' if 'error' in record else 'failed' if record['attempt'] is None else 'success',
                    failure={'reason': record.get('error')}, robot='x1pro', camera='face_view', record=record, work=str(work)))
        return []

    monkeypatch.setattr(scene_batch, 'run_jobs', jobs)
    root = tmp_path / 'checkout'
    output = tmp_path / 'custom'
    (output / '000000' / 'x1pro' / 'initial').mkdir(parents=True)
    with pytest.raises(SystemExit, match=r'errors: 3 \(see .*\); seeds without a valid state: 1$'):
        build_scene.main(['000000', '--jobs', '2', '--seeds', '0:5', '--gpus', '0', '1', '--grid', '2x1', '--output', str(output),
                          '--root', str(root), '--assets', '../alternate-assets'])
    assert sorted(seed for group in rounds[:2] for seed in group) == list(range(5))
    assert sorted(seed for group in rounds[2:] for seed in group) == [2, 3]  # seeds without a result or with an error run again
    directory = output / '000000' / 'x1pro'
    report = json.loads((directory / 'initial.json').read_text())
    assert report['seeds'] == {'range': '0:5', 'count': 5} and report['without_state'] == [1, 3]
    assert report['records']['3']['error'] == 'RuntimeError: CUDA error'
    attrs, states = read_states(directory / 'initial_states.hdf5')
    assert list(states) == [0, 2, 4] and attrs['scene'] == '000000' and attrs['robot'] == 'x1pro'
    assert Image.open(directory / 'initial_states.jpg').size == (2 * build_scene.TILE_WIDTH, 216 + build_scene.CAPTION)
    assert not (directory / 'initial').exists()            # replaced, and no --images this time
    assert not (directory / 'staging').exists()            # the processes' results, removed once merged
    assert (root / 'asset_sources/scenes/000000/thumbnail.jpg').read_bytes() == (Path(__file__).resolve().parents[2] / 'asset_sources/scenes/000000/thumbnail.jpg').read_bytes()           # a custom output keeps the thumbnail


def test_build_scene_layouts(tmp_path, monkeypatch):
    built = []

    def jobs(commands, gpus):
        for command, _, _ in commands:
            data = manifest(command)
            layout = data['units'][0]['data']['layout']
            seeds = [u['data']['seed'] for u in data['units']]
            work = Path(data['work']) / str(layout)
            for name in ('states', 'images', 'records'):
                (work / name).mkdir(parents=True)
            for seed in seeds:
                built.append((layout, seed))
                write_states(work / 'states' / f'{seed}.hdf5', {seed: (0, STATE)}, {})
                Image.new('RGB', (64, 36), 'red').save(work / 'images' / f'{seed}.jpg')
                record = {'seed': seed, 'attempt': 0, 'attempts': [{'attempt': 0, 'settle': None}]}
                publish(data, next(u for u in data['units'] if u['data']['seed'] == seed), dict(status='success', robot='x1pro', camera='top', record=record, work=str(work)))
        return []

    monkeypatch.setattr(scene_batch, 'run_jobs', jobs)
    root = tmp_path / 'checkout'
    (root / 'asset_sources' / 'scenes' / '000100').mkdir(parents=True)
    (root / 'asset_sources' / 'scenes' / '000100' / 'scene.py').write_text(
        "ROBOT = 'x1pro'\nROBOTS = {'x1pro': {'position': (0,0,0), 'yaw': 0, 'joints': {}}}\nLAYOUTS = {1: {'quality': 'good', 'description': 'tidy', 'arrange': None},\n"
        "           2: {'quality': 'bad', 'description': 'a heap', 'arrange': None}}\n")
    output = tmp_path / 'custom'
    with pytest.raises(SystemExit):
        build_scene.main(['000100', '--layouts', '3', '--seeds', '0', '--output', str(output), '--root', str(root)])
    build_scene.main(['000100', '--layouts', '0:3', '--seeds', '0', '1', '--jobs', '3', '--grid', '2x1',
                      '--output', str(output), '--root', str(root)])
    assert sorted(built) == [(layout, seed) for layout in range(3) for seed in range(2)]
    directory = output / '000100' / 'x1pro'
    assert list(read_states(directory / 'initial_states.hdf5')[1]) == [0, 1]
    report = json.loads((directory / 'layouts' / '2' / 'initial.json').read_text())
    assert report['layout'] == 2 and report['quality'] == 'bad' and report['valid'] == 2
    attrs, states = read_states(directory / 'layouts' / '1' / 'initial_states.hdf5')
    assert attrs['layout'] == 1 and list(states) == [0, 1]
    assert Image.open(directory / 'layouts.jpg').size == (2 * build_scene.TILE_WIDTH, 2 * (216 + build_scene.CAPTION))


def test_repeat_rates_count_every_run(tmp_path, monkeypatch):
    from table_1000.experts.generate import repeat_rates
    reports = [{'seed': 4, 'repeat': 0, 'status': 'success'}, {'seed': 4, 'repeat': 1, 'status': 'failed',
                                                               'failure': {'subtask': 'open', 'reason': 'no reachable grasp'}},
               {'seed': 72, 'repeat': 0, 'status': 'failed', 'failure': {'subtask': 'take', 'reason': 'no reachable grasp'}},
               {'seed': 72, 'repeat': 1, 'status': 'failed', 'failure': {'subtask': 'take', 'reason': 'no reachable grasp'}}]
    rates = repeat_rates(reports, [4, 72], 2)
    assert rates['succeeded'] == {'4': 1, '72': 0} and rates['success_rate'] == .25 and rates['short'] == {}
    assert repeat_rates(reports[:3], [4, 72], 2)['short'] == {'72': 1}
    assert rates['failures'] == {'take: no reachable grasp': {'72': 2}, 'open: no reachable grasp': {'4': 1}}
    from contextlib import ExitStack
    from types import SimpleNamespace
    from table_1000.common import batch_runtime as runtime
    from table_1000.common.batch_timing import Timings, timing_path
    from table_1000.experts.batch_program import program
    options = dict(units=[(('000003', 'robot', '0000', 0), 4)], layout=0, steps=None,
                   physics='isaac-cpu', repeats=3)
    unit = next(program.units(options))
    singles = next(program.units(options | dict(repeats=1)))
    assert unit.timing_key != singles.timing_key
    class Generator:
        load_s = 5.
        def run(self, seed, *state, repeat):
            return dict(status='success', wall_s=10. * (repeat + 1), cpu_s=8. * (repeat + 1))
    with ExitStack() as stack:
        context = runtime.Context(tmp_path, tmp_path, options, 0, tmp_path, stack,
            loaded=(Generator(), {4: (0, {}, None)}, 2.))
        result = program.run_unit(context, unit)
    assert result.report['wall_s'] == 10. and result.report['cpu_s'] == 8.
    assert result.timing == dict(wall_s=60., cpu_s=48.)
    from dataclasses import asdict
    manifest_path = tmp_path / 'units.json'
    manifest_path.write_text(json.dumps(dict(root=str(tmp_path), output=str(tmp_path), work=str(tmp_path),
        options=options, commit='submitted', attempt=0, program='fixture:program', units=[asdict(unit)])))
    monkeypatch.setattr(runtime, '_program', lambda _: SimpleNamespace(run_unit=lambda *args: result))
    runtime.work(manifest_path, gpu=0)
    report = json.loads((tmp_path / unit.result).read_text())
    assert report['wall_s'] == 10. and report['unit_timing']['wall_s'] == 60.
    timing = json.loads(timing_path(tmp_path, unit.timing_key).read_text())
    assert timing['wall_s'] == 60. and timing['cpu_s'] == 48.
    assert Timings(tmp_path, [unit]).costs[unit.key] == 60.


def test_render_rejects_seeds_without_episode(batch, tmp_path, capsys):
    with pytest.raises(SystemExit):
        render.main(['000000', '--physics', 'isaac-cpu', '--cameras', 'wide', '--seeds', '1:4',
                     '--input', str(tmp_path / 'data')])
    assert 'seeds 2:4 have no episode; the batch covers 0:5, failed: none' in capsys.readouterr().err


def test_render_rejects_unknown_cameras(batch, tmp_path, capsys):
    with pytest.raises(SystemExit):
        render.main(['000000', '--physics', 'isaac-cpu', '--cameras', 'wide', 'face_depth',
                     '--input', str(tmp_path / 'data')])
    assert 'face_depth is not an RGB camera of the robot' in capsys.readouterr().err


def test_run_expert_rejects_another_robot(tmp_path, monkeypatch):
    initial = tmp_path / '000000' / 'x1pro'
    initial.mkdir(parents=True)
    write_states(initial / 'initial_states.hdf5', {0: (0, STATE)}, {'scene': '000000', 'robot': 'other'})
    with pytest.raises(SystemExit):
        run_expert.main(['000000', '--physics', 'isaac-cpu', '--input', str(tmp_path), '--output', str(tmp_path / 'out')])
