"""Grasp output modes, source reconstruction, publication and recorded state replay."""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from table_1000.common import batch
from table_1000.experts.asset_grasp import results
from table_1000.experts.asset_grasp.recording import Recorder
from table_1000.sim.isaaclab import replay


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True).stdout


@pytest.fixture
def checkout(tmp_path):
    git(tmp_path, 'init', '-q')
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/test.py').write_text('TRIALS = 20\n')
    git(tmp_path, 'add', '.')
    git(tmp_path, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'test')
    return tmp_path


def test_run_name_shared_with_expert(monkeypatch):
    from scripts.scenes.run_expert import run_name
    monkeypatch.setattr(batch.time, 'strftime', lambda pattern: '20261025-010203')
    assert run_name('abcdef0+dirty', 'review') == '20261025-010203-abcdef0-dirty-review'
    assert batch.run_name('abcdef0') == '20261025-010203-abcdef0'


def test_modes_and_nonoverwriting_directories(checkout, monkeypatch):
    monkeypatch.setattr(batch.time, 'strftime', lambda pattern: '20261025-010203')
    head = batch.commit(checkout)
    name, dev = results.prepare(checkout, dev=True, label='review', checkout=head)
    assert dev == checkout / 'runs/grasp' / name
    assert name.endswith(head[:7] + '-review')
    with pytest.raises(FileExistsError):
        results.prepare(checkout, dev=True, label='review', checkout=head)
    name, formal = results.prepare(checkout, dev=False, label=None, checkout=head)
    assert formal == checkout / 'outputs/asset-grasp' / name
    assert results.acceptance_path(checkout / 'assets', 'objects/cup/000000', 'robot', 'isaac-cpu') == (
        checkout / 'assets/objects/cup/000000/grasp_test/robot/isaac-cpu')


@pytest.mark.parametrize('selection', [dict(asset=['objects/cup/000000']), dict(seeds=[0]), dict(video=True),
                                      dict(label='review'), dict(trials=2)])
def test_partial_runs_require_dev(selection):
    options = dict(dev=False, asset=None, seeds=None, video=False, label=None, trials=20, configured_trials=20)
    results.require_mode(**options)
    options.update(selection)
    with pytest.raises(ValueError, match='--dev'):
        results.require_mode(**options)
    results.require_mode(**{**options, 'dev': True})


def test_formal_requires_clean_commit(checkout):
    (checkout / 'notes.txt').write_text('local note')
    with pytest.raises(ValueError, match='干净'):
        results.prepare(checkout, dev=False, label=None, checkout=batch.commit(checkout))


def test_publish_replaces_whole_backend_and_keeps_model(tmp_path):
    assets = tmp_path / 'assets'
    target = results.acceptance_path(assets, 'objects/cup/000000', 'robot', 'isaac-cpu')
    target.mkdir(parents=True)
    (target / 'obsolete.json').write_text('{}')
    model = assets / 'objects/cup/000000/object.usdz'
    model.write_bytes(b'model')
    source = tmp_path / 'stage'
    (source / 'trials').mkdir(parents=True)
    (source / 'summary.json').write_text('{}')
    (source / 'trials/0000.json').write_text('{}')
    assert results.publish(source, assets, 'objects/cup/000000', 'robot', 'isaac-cpu') == target
    assert sorted(path.name for path in target.iterdir()) == ['summary.json', 'trials']
    assert model.read_bytes() == b'model' and not source.exists()


def test_video_first_failure_otherwise_seed_zero():
    reports = [{'seed': seed, 'status': status} for seed, status in [(2, 'failed'), (0, 'success'), (1, 'failed')]]
    assert results.video_seed(reports) == 1
    assert results.video_seed([{**report, 'status': 'success'} for report in reports]) == 0


def test_recorded_states_survive_mutation_and_search_reset(tmp_path):
    state = {'articulation': {'robot': {'joint_position': torch.zeros((1, 2))}},
             'rigid_object': {'object': {'root_pose': torch.ones((1, 7))}}}
    env = SimpleNamespace(env=SimpleNamespace(scene=SimpleNamespace(get_state=lambda **kwargs: state)))
    recorder = Recorder(env)
    for step, position in enumerate([1., 2., -1.]):  # Last frame is a new search attempt's reset.
        state['rigid_object']['object']['root_pose'][0, 0] = position
        state['articulation']['robot']['joint_position'][0, 0] = step
        recorder.capture(f'attempt {step}')
    path = tmp_path / '0007.hdf5'
    recorder.save(path, {'seed': 7, 'robot': 'test'})
    file, episodes = replay.episodes(path)
    try:
        episode = episodes['demo_0']
        assert episode.length == 3 and episode.attrs['sampling'] == 'post_control_step'
        assert episode.attrs['captions'] == ['attempt 0', 'attempt 1', 'attempt 2']
        for t, position in enumerate([1., 2., -1.]):
            restored = episode.state(t + 1)
            assert restored['rigid_object']['object']['root_pose'][0, 0] == position
            assert restored['articulation']['robot']['joint_position'][0, 0] == t
    finally:
        file.close()


@pytest.mark.parametrize('dev', [False, True])
def test_cli_complete_pipeline_uses_only_its_mode_destination(checkout, monkeypatch, dev):
    from scripts.experts import eval_asset_grasp as cli
    from table_1000.experts.asset_grasp import runner
    config = checkout / 'configs/test.py'
    config.write_text("SCENE = '000103'\nROBOT = 'x1pro_rm001'\nTRIALS = 2\nASSETS = ['objects/computer_mouse/000000']\n")
    git(checkout, 'add', '.')
    git(checkout, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'config')
    (checkout / '.git/info/exclude').write_text('runs/\noutputs/\nassets/\n')
    monkeypatch.setattr(cli, 'ROOT', checkout)
    monkeypatch.setattr(runner, 'version', lambda package: 'test')
    calls = []

    def run(units, **kwargs):
        for index, asset in enumerate(sorted({unit[0] if isinstance(unit, tuple) else unit for unit in units})):
            group = [unit for unit in units if (unit[0] if isinstance(unit, tuple) else unit) == asset]
            command, log, _ = kwargs['command'](group, index, 0)
            calls.append(command)
            output = Path(command[command.index('--output') + 1])
            start = command.index('--seeds') + 1
            stop = next((i for i in range(start, len(command)) if command[i].startswith('--')), len(command))
            seeds = list(map(int, command[start:stop]))
            for seed in seeds:
                if '--render-worker' in command:
                    target = output / asset / 'videos' / f'{seed:04d}.mp4'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b'video')
                else:
                    trajectory = output / asset / 'trajectories' / f'{seed:04d}.hdf5'
                    trajectory.parent.mkdir(parents=True, exist_ok=True)
                    trajectory.write_bytes(b'trajectory')
                    runner.write_json(runner.trial_path(output, asset, seed), {
                        'seed': seed, 'status': 'success' if seed == 0 else 'failed',
                        'failure': {'reason': 'object not held'},
                        'trajectory': f'trajectories/{seed:04d}.hdf5'})
        return batch.BatchResult([], batch.commit(checkout))

    monkeypatch.setattr(runner, 'run_batch', run)
    with pytest.raises(SystemExit) as exit:
        cli.main(['--config', str(config), '--assets', str(checkout / 'assets'), *(['--dev'] if dev else [])])
    assert exit.value.code == 1  # A valid grasp failure still publishes the full result.
    directories = list((checkout / ('runs/grasp' if dev else 'outputs/asset-grasp')).iterdir())
    assert len(directories) == 1
    output = directories[0]
    asset = 'objects/computer_mouse/000000'
    acceptance = results.acceptance_path(checkout / 'assets', asset, 'x1pro_rm001', 'isaac-cpu')
    result = output / asset if dev else acceptance
    assert acceptance.exists() != dev
    summary = json.loads((result / 'summary.json').read_text())
    assert summary['commit'] == batch.commit(checkout)
    assert summary['run_name'] == output.name and summary['config_text'] == config.read_text()
    assert len(list((result / 'trials').iterdir())) == 2
    assert len(list((result / 'trajectories').iterdir())) == 2
    if dev:
        assert not (result / 'videos').exists() and len(calls) == 1
    else:
        assert summary['video']['seed'] == 1
        assert list((result / 'videos').iterdir()) == [result / 'videos/0001.mp4']
        assert not (output / asset).exists()


def test_replay_cannot_write_acceptance_directory(tmp_path):
    from scripts.experts.eval_asset_grasp import render_main
    result = tmp_path / 'result'
    result.mkdir()
    (result / 'summary.json').write_text('{}')
    assets = tmp_path / 'assets'
    with pytest.raises(ValueError, match='正式运行'):
        render_main(['--result', str(result), '--seed', '0', '--assets', str(assets),
                     '--output', str(assets / 'test.mp4')])


@pytest.mark.parametrize('argv, message', [
    ([], '需要 --config'),
    (['--dev', '--asset', 'objects/computer_mouse/000000'], '需要 --robot'),
    (['--dev', '--asset', 'objects/missing/000000', '--robot', 'x1pro_rm001'], 'assets not built'),
])
def test_selection_requires_config_or_explicit_assets(tmp_path, capsys, argv, message):
    from scripts.experts import eval_asset_grasp as cli
    with pytest.raises(SystemExit) as exit:
        cli.main([*argv, '--assets', str(tmp_path)])
    assert exit.value.code == 2 and message in capsys.readouterr().err


@pytest.mark.parametrize('advance_commit', [False, True])
def test_formal_publication_keeps_frozen_source_after_workspace_changes(checkout, monkeypatch, advance_commit):
    """Actual CLI/batch/worker/publication in a temporary Git repo, without Isaac."""
    from scripts.experts import eval_asset_grasp as cli
    from table_1000.experts.asset_grasp import runner
    import sys
    config = checkout / 'configs/test.py'
    config.write_text("SCENE = '000103'\nROBOT = 'x1pro_rm001'\nTRIALS = 1\nASSETS = ['objects/computer_mouse/000000']\n")
    (checkout / '.gitignore').write_text('.local/\nassets/\ndata/\nruns/\noutputs/\n')
    script = checkout / 'worker.py'
    script.write_text('''\
import json
from pathlib import Path
import subprocess
import sys
asset, output, original, render, advance = sys.argv[1:6]
root = Path(__file__).resolve().parent
record = {'commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--short', 'HEAD'], text=True).strip(),
          'source': (root / 'source.txt').read_text()}
result = Path(output) / asset
if render == 'True':
    video = result / 'videos/0000.mp4'
    video.parent.mkdir(parents=True)
    video.write_text(json.dumps(record))
else:
    trajectory = result / 'trajectories/0000.hdf5'
    trajectory.parent.mkdir(parents=True)
    trajectory.write_bytes(b'trajectory')
    (result / 'trials').mkdir()
    (result / 'trials/0000.json').write_text(json.dumps(dict(record, seed=0, status='success', trajectory='trajectories/0000.hdf5')))
    original = Path(original)
    (original / 'source.txt').write_text('next task')
    (original / 'new_work.py').write_text('NEXT = True\\n')
    if advance == 'True':
        subprocess.run(['git', '-C', str(original), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(original), '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'next task'], check=True)
''')
    (checkout / 'source.txt').write_text('submitted')
    git(checkout, 'add', '.')
    git(checkout, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'workers')
    head = batch.commit(checkout)
    monkeypatch.setattr(cli, 'ROOT', checkout)
    monkeypatch.setattr(runner, 'version', lambda package: 'test')
    def worker_command(root, assets, robot, physics, work, asset, seeds, **paths):
        return [sys.executable, str(script), asset, str(paths['output']), str(checkout / '.local/..'),
                str(seeds == [0] and 'render' in work.name), str(advance_commit), '--']
    monkeypatch.setattr(runner, 'worker_command', worker_command)
    cli.main(['--config', str(config), '--assets', str(checkout / 'assets')])
    assert batch.commit(checkout) != head
    acceptance = results.acceptance_path(checkout / 'assets', 'objects/computer_mouse/000000', 'x1pro_rm001', 'isaac-cpu')
    summary = json.loads((acceptance / 'summary.json').read_text())
    report = json.loads((acceptance / 'trials/0000.json').read_text())
    video = json.loads((acceptance / 'videos/0000.mp4').read_text())
    assert summary['commit'] == report['commit'] == video['commit'] == head
    assert report['source'] == video['source'] == 'submitted'
    assert 'dirty_patch' not in summary
