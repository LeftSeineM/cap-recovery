"""Batch runs and the three scene steps without a simulator: processes, seeds, initial-state files, merged batches,
the choice of renders."""

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


def test_seeds_are_ranges_and_single_seeds():
    assert parse_seeds(['100:103', '7', '101']) == [7, 100, 101, 102]
    assert seed_range([0, 1, 2, 5, 7, 8]) == '0:3 5 7:9'
    assert parse_seeds(seed_range([3, 4, 9]).split()) == [3, 4, 9]
    assert split(list(range(10)), 3) == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert split([5, 6], 8) == [[5], [6]]
    assert plan(list(range(10)), lambda seed: None, None, 3) == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]


def test_jobs_share_the_listed_gpus(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command[-1])
        return type('Done', (), {'returncode': int(command[0] == 'bad')})

    monkeypatch.setattr(scene_batch.subprocess, 'run', run)
    jobs = [([name], tmp_path / name / 'run.log', name) for name in ('a', 'bad', 'c')]
    assert run_jobs(jobs, [4, 5]) == ['bad']
    assert len(calls) == 3 and set(calls) <= {'4', '5'}


def test_launcher_runs_every_part_with_its_gpu(tmp_path):
    """Each part runs through the launcher prefix without --gpu and reads the GPU the launcher set."""
    show = 'from table_1000.common.batch import worker_gpu; import sys; print(worker_gpu(None)); sys.exit(len(sys.argv) - 1)'
    jobs = [([sys.executable, '-c', show, *extra], tmp_path / name / 'run.log', name)
            for name, extra in (('a', []), ('b', ['fail']))]
    failed = launch(jobs, 'env CUDA_VISIBLE_DEVICES=5', 2)
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
    directory = tmp_path / 'data' / '000000' / 'isaac-cpu'
    directory.mkdir(parents=True)
    trajectory_file(directory / 'trajectories.hdf5', [0, 1, 4])
    (directory / 'summary.json').write_text(json.dumps({'seeds': {'range': '0:5', 'count': 5}, 'videos': {}, 'fails': {},
                                                        'failures': {'close the drawer: x': [2, 3]}}))
    return directory


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


def test_render_keeps_other_videos(batch, tmp_path, monkeypatch):
    trajectory_file(batch / 'trajectories_failed.hdf5', [3, 2])
    video_path(batch, 4, 'wide').parent.mkdir(parents=True)
    assert video_path(batch, 4, 'wide') == batch / 'videos' / '4' / 'wide.mp4'
    video_path(batch, 4, 'wide').write_bytes(b'old')
    chosen = []

    def jobs(commands, gpus):
        assert gpus == [0, 1, 0, 1][:len(commands)]
        for command, _, _ in commands:
            seeds = parse_seeds(command[command.index('--seeds') + 1:command.index('--batch')])
            chosen.append(seeds)
            merge = '--merge' in command
            for seed in seeds:
                target = video_path(batch, seed, None if merge else 'wide', failed=seed in (2, 3))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b'new')
        return []

    monkeypatch.setattr(scene_batch, 'run_jobs', jobs)
    monkeypatch.setattr(render, 'check_cameras', lambda *args: None)
    common = ['000000', '--physics', 'isaac-cpu', '--gpus', '0', '1', '--jobs', '4',
              '--input', str(tmp_path / 'data'), '--root', str(tmp_path)]
    render.main([*common, '--cameras', 'wide', '--count', '2'])
    render.main([*common, '--cameras', 'wide', 'face_view', '--merge', '--failed'])
    assert chosen == [[0], [1], [2], [3]]
    assert videos(batch) == {'videos': {'wide': [0, 1, 4]}, 'fails': {'merge': [2, 3]}}
    assert (batch / 'fails' / '2' / 'merge.mp4').is_file()
    assert video_path(batch, 4, 'wide').read_bytes() == b'old'
    summary = json.loads((batch / 'summary.json').read_text())
    assert summary['videos'] == {'wide': '0:2 4'} and summary['fails'] == {'merge': '2:4'}


def test_run_expert_merges_parts(tmp_path, monkeypatch):
    initial = tmp_path / 'assets' / 'scenes' / '000000'
    initial.mkdir(parents=True)
    write_states(initial / 'initial_states.hdf5', {seed: (0, STATE) for seed in range(4)},
                 {'scene': '000000', 'robot': 'x1pro'})

    rounds = []

    def jobs(commands, gpus):
        rounds.append(len(commands))
        for command, _, _ in commands:
            work = tmp_path / command[command.index('--work') + 1]
            seeds = parse_seeds(command[command.index('--seeds') + 1:command.index('--states')])
            (work / 'episodes').mkdir(parents=True)
            # Seed 2 kills its process in every attempt, after its episode is recorded, before its report.
            trajectory_file(work / 'trajectories.hdf5', [seed for seed in seeds if seed <= 2])
            if 2 in seeds:
                continue
            trajectory_file(work / 'trajectories.hdf5', seeds)
            for seed in seeds:
                (work / 'episodes' / f'{seed}.json').write_text(json.dumps(
                    {'seed': seed, 'status': 'success', 'duration_s': 1., 'wall_s': 2.}))
        return []

    monkeypatch.setattr(scene_batch, 'run_jobs', jobs)
    monkeypatch.setattr(run_expert, 'scene_robot', lambda scene, root: 'x1pro')
    output = tmp_path / 'data'
    with pytest.raises(SystemExit, match='no episode: seeds 2;'):
        run_expert.main(['000000', '--physics', 'isaac-cpu', '--gpus', '0', '1', '--input', str(tmp_path / 'assets/scenes'),
                         '--output', str(output), '--root', str(tmp_path / 'checkout')])
    assert rounds == [2, 2]          # seeds 2 and 3 once more
    directory = output / '000000' / 'isaac-cpu'
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['failures'] == {PROCESS_EXITED: [2]} and summary['succeeded'] == 3
    assert sorted(path.name for path in (directory / 'episodes').iterdir()) == ['0.json', '1.json', '3.json']
    with h5py.File(directory / 'trajectories.hdf5', 'r') as file:
        assert sorted(file['data']) == ['demo_0', 'demo_1', 'demo_3']


@pytest.mark.parametrize('problems', ['failed', 'all', 'none'])
def test_run_expert_dev_runs_are_kept_apart(tmp_path, monkeypatch, problems):
    initial = tmp_path / 'assets' / 'scenes' / '000000'
    initial.mkdir(parents=True)
    write_states(initial / 'initial_states.hdf5', {seed: (0, STATE) for seed in range(4)},
                 {'scene': '000000', 'robot': 'x1pro'})

    def jobs(commands, gpus):
        for command, _, _ in commands:
            assert command[command.index('--problems') + 1] == problems
            work = tmp_path / command[command.index('--work') + 1]
            seeds = parse_seeds(command[command.index('--seeds') + 1:command.index('--states')])
            (work / 'episodes').mkdir(parents=True)
            trajectory_file(work / 'trajectories.hdf5', seeds)
            for seed in seeds:
                (work / 'episodes' / f'{seed}.json').write_text(json.dumps(
                    {'seed': seed, 'status': 'success', 'duration_s': 1., 'wall_s': 2.}))
        return []

    monkeypatch.setattr(scene_batch, 'run_jobs', jobs)
    monkeypatch.setattr(run_expert, 'scene_robot', lambda scene, root: 'x1pro')
    common = ['000000', '--physics', 'isaac-cpu', '--input', str(tmp_path / 'assets/scenes'),
              '--root', str(tmp_path / 'checkout')]
    with pytest.raises(SystemExit):     # some seeds are a development run
        run_expert.main([*common, '--output', str(tmp_path / 'data'), '--seeds', '1'])
    assert not (tmp_path / 'data').exists()
    run_expert.main([*common, '--dev', '--name', 'try', '--seeds', '1', '3', '--output', str(tmp_path / 'runs'),
                     *(['--problems', problems] if problems != 'failed' else [])])
    [directory] = (tmp_path / 'runs' / '000000').iterdir()
    assert directory.name.endswith('-nocommit-try')
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['run'] == directory.name and '--dev' in summary['command'] and summary['succeeded'] == 2
    assert summary['problems'] == problems
    with h5py.File(directory / 'trajectories.hdf5', 'r') as file:
        assert sorted(file['data']) == ['demo_1', 'demo_3']


def test_run_names_carry_the_commit():
    assert run_expert.run_name('ab12cd3+dirty', 'step2').endswith('-ab12cd3-dirty-step2')
    assert run_expert.run_name(None).endswith('-nocommit')


def test_run_expert_rejects_another_robot(tmp_path, monkeypatch):
    monkeypatch.setattr(run_expert, 'scene_robot', lambda scene, root: 'x1pro')
    initial = tmp_path / '000000'
    initial.mkdir()
    write_states(initial / 'initial_states.hdf5', {0: (0, STATE)}, {'scene': '000000', 'robot': 'other'})
    with pytest.raises(SystemExit):
        run_expert.main(['000000', '--physics', 'isaac-cpu', '--input', str(tmp_path), '--output', str(tmp_path / 'out')])


def test_build_scene_combines_parts(tmp_path, monkeypatch):
    rounds = []

    def jobs(commands, gpus):
        for command, _, _ in commands:
            work = tmp_path / command[command.index('--work') + 1]
            seeds = parse_seeds(command[command.index('--seeds') + 1:command.index('--work')])
            rounds.append(seeds)
            for name in ('states', 'images', 'records'):
                (work / name).mkdir(parents=True)
            for seed in seeds:
                if seed == 2 and len(rounds) == 1 or seed == 3 and len(rounds) == 3:
                    break      # the process died here: earlier seeds keep their results
                if seed != 1:
                    write_states(work / 'states' / f'{seed}.hdf5', {seed: (0, STATE)}, {'scene': '000000', 'robot': 'x1pro'})
                    Image.new('RGB', (64, 36), 'red').save(work / 'images' / f'{seed}.jpg')
                record = {'seed': seed, 'attempt': None if seed == 1 else 0,
                          'attempts': [{'attempt': 0, 'settle': {'objects': {}, 'robot_contacts': []}}]}
                if len(rounds) == 2:
                    # seed 3 errs, then its rerun dies before it; seed 4 finds no valid state after the error
                    (work / 'states' / f'{seed}.hdf5').unlink()
                    record.update({'attempt': None, 'error': 'RuntimeError: CUDA error'} if seed == 3 else
                                  {'attempt': None, 'after_error': 3})
                (work / 'records' / f'{seed}.json').write_text(
                    json.dumps({'robot': 'x1pro', 'camera': 'face_view', 'record': record}))
        return []

    monkeypatch.setattr(scene_batch, 'run_jobs', jobs)
    root = tmp_path / 'checkout'
    output = tmp_path / 'custom'
    (output / '000000' / 'initial').mkdir(parents=True)
    with pytest.raises(SystemExit, match=r'errors: seeds 3 \(see .*\); seeds without a valid state: 1$'):
        build_scene.main(['000000', '--seeds', '0:5', '--gpus', '0', '1', '--grid', '2x1', '--output', str(output),
                          '--root', str(root)])
    assert rounds == [[0, 1, 2], [3, 4], [2, 3], [4]]  # seeds without a result or with an error run again
    directory = output / '000000'
    report = json.loads((directory / 'initial.json').read_text())
    assert report['seeds'] == {'range': '0:5', 'count': 5} and report['without_state'] == [1, 3]
    assert report['records']['3']['error'] == 'RuntimeError: CUDA error'
    attrs, states = read_states(directory / 'initial_states.hdf5')
    assert list(states) == [0, 2, 4] and attrs['scene'] == '000000' and attrs['robot'] == 'x1pro'
    assert Image.open(directory / 'initial_states.jpg').size == (2 * build_scene.TILE_WIDTH, 216 + build_scene.CAPTION)
    assert not (directory / 'initial').exists()            # replaced, and no --images this time
    assert not (directory / 'staging').exists()            # the processes' results, removed once merged
    assert not (root / 'asset_sources').exists()           # a custom output keeps the thumbnail


def test_build_scene_layouts(tmp_path, monkeypatch):
    built = []

    def jobs(commands, gpus):
        for command, _, _ in commands:
            work = tmp_path / command[command.index('--work') + 1]
            layout = int(command[command.index('--layout') + 1])
            seeds = parse_seeds(command[command.index('--seeds') + 1:command.index('--work')])
            for name in ('states', 'images', 'records'):
                (work / name).mkdir(parents=True)
            for seed in seeds:
                built.append((layout, seed))
                write_states(work / 'states' / f'{seed}.hdf5', {seed: (0, STATE)}, {})
                Image.new('RGB', (64, 36), 'red').save(work / 'images' / f'{seed}.jpg')
                record = {'seed': seed, 'attempt': 0, 'attempts': [{'attempt': 0, 'settle': None}]}
                (work / 'records' / f'{seed}.json').write_text(
                    json.dumps({'robot': 'x1pro', 'camera': 'top', 'record': record}))
        return []

    monkeypatch.setattr(scene_batch, 'run_jobs', jobs)
    root = tmp_path / 'checkout'
    (root / 'asset_sources' / 'scenes' / '000100').mkdir(parents=True)
    (root / 'asset_sources' / 'scenes' / '000100' / 'scene.py').write_text(
        "LAYOUTS = {1: {'quality': 'good', 'description': 'tidy', 'arrange': None},\n"
        "           2: {'quality': 'bad', 'description': 'a heap', 'arrange': None}}\n")
    output = tmp_path / 'custom'
    with pytest.raises(SystemExit):
        build_scene.main(['000100', '--layouts', '3', '--seeds', '0', '--output', str(output), '--root', str(root)])
    build_scene.main(['000100', '--layouts', '0:3', '--seeds', '0', '1', '--jobs', '3', '--grid', '2x1',
                      '--output', str(output), '--root', str(root)])
    assert sorted(built) == [(layout, seed) for layout in range(3) for seed in range(2)]
    directory = output / '000100'
    assert list(read_states(directory / 'initial_states.hdf5')[1]) == [0, 1]
    report = json.loads((directory / 'layouts' / '2' / 'initial.json').read_text())
    assert report['layout'] == 2 and report['quality'] == 'bad' and report['valid'] == 2
    attrs, states = read_states(directory / 'layouts' / '1' / 'initial_states.hdf5')
    assert attrs['layout'] == 1 and list(states) == [0, 1]
    assert Image.open(directory / 'layouts.jpg').size == (2 * build_scene.TILE_WIDTH, 2 * (216 + build_scene.CAPTION))


def test_repeat_rates_count_every_run():
    from table_1000.experts.generate import repeat_rates
    reports = [{'seed': 4, 'repeat': 0, 'status': 'success'}, {'seed': 4, 'repeat': 1, 'status': 'failed',
                                                               'failure': {'subtask': 'open', 'reason': 'no reachable grasp'}},
               {'seed': 72, 'repeat': 0, 'status': 'failed', 'failure': {'subtask': 'take', 'reason': 'no reachable grasp'}},
               {'seed': 72, 'repeat': 1, 'status': 'failed', 'failure': {'subtask': 'take', 'reason': 'no reachable grasp'}}]
    rates = repeat_rates(reports, [4, 72], 2)
    assert rates['succeeded'] == {'4': 1, '72': 0} and rates['success_rate'] == .25 and rates['short'] == {}
    assert repeat_rates(reports[:3], [4, 72], 2)['short'] == {'72': 1}
    assert rates['failures'] == {'take: no reachable grasp': {'72': 2}, 'open: no reachable grasp': {'4': 1}}
