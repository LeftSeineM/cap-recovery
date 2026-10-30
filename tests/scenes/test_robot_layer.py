"""Robot/expert identity, declared scope and paths."""

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest

from table_1000.scenes import (robot_config, load_expert, parse_scope, outside_scope,
                               initial_dir, data_dir, case_dir)
from table_1000.experts.identity import check_identity
from scripts.scenes.run_expert import resume_steps, check_checkpoint_name, main


def sources(root, declaration="ROBOT = 'a'\nSCOPE = {0: None, 2: [1, 3]}\n"):
    directory = root / 'asset_sources/scenes/demo'
    directory.mkdir(parents=True)
    (directory / 'scene.py').write_text("ROBOT = 'a'\nROBOTS = {'a': {'position': (0,0,0), 'yaw': 0, 'joints': {}}, 'b': {}}\n")
    (directory / 'experts/a').mkdir(parents=True)
    (directory / 'experts/a/0002.py').write_text(declaration + 'def solve(ctx): pass\n')
    (directory / 'experts/a/0005.py').write_text(declaration + 'def solve(ctx): pass\n')
    return directory


def test_robot_defaults_and_explicit_selection():
    scene = SimpleNamespace(ROBOT='a', ROBOTS={'a': {'joints': {}, 'yaw': 1, 'position': (1,2,3)}, 'b': {}})
    assert robot_config(scene) == {**scene.ROBOTS['a'], 'asset': 'robots/a'}
    assert robot_config(scene, 'b')['asset'] == 'robots/b'
    with pytest.raises(ValueError, match='not in ROBOTS'):
        robot_config(scene, 'unknown')
    scene.ROBOT = 'missing'
    with pytest.raises(ValueError, match='default robot'):
        robot_config(scene, 'a')


def test_numbered_expert_loading(tmp_path):
    sources(tmp_path)
    assert load_expert('demo', root=tmp_path).NUMBER == '0002'
    assert load_expert('demo', expert='0005', root=tmp_path).NUMBER == '0005'
    with pytest.raises(ValueError, match='available'):
        load_expert('demo', expert='0000', root=tmp_path)


def test_steps_only_expert_can_share_scene_helpers(tmp_path):
    directory = sources(tmp_path)
    (directory / 'experts/common.py').write_text('DESCRIPTION = "shared helper"\n')
    (directory / 'experts/a/0002.py').write_text(
        "from ..common import DESCRIPTION\nROBOT = 'a'\nSCOPE = {0: None}\nSTEPS = []\n")
    expert = load_expert('demo', root=tmp_path)
    assert expert.DESCRIPTION == 'shared helper' and callable(expert.solve)


def test_expert_robot_must_match_directory(tmp_path):
    sources(tmp_path, "ROBOT = 'b'\nSCOPE = {0: None}\n")
    with pytest.raises(ValueError, match='differs from directory'):
        load_expert('demo', root=tmp_path)


@pytest.mark.parametrize('scope', [{}, {'0': None}, {0: 'all'}, {0: []}, {0: [1,1]}, {-1: None}])
def test_invalid_scope(scope):
    with pytest.raises(ValueError, match='SCOPE'):
        parse_scope(scope)


def test_scope_and_product_paths():
    scope = parse_scope({0: None, 2: [1, 3]})
    assert not outside_scope(scope, 0, [99])
    assert not outside_scope(scope, 2, [3])
    assert outside_scope(scope, 2, [2]) and outside_scope(scope, 1, [1])
    assert initial_dir('/assets/scenes', 's', 'r', 0) == Path('/assets/scenes/s/r')
    assert initial_dir('/assets/scenes', 's', 'r', 2) == Path('/assets/scenes/s/r/layouts/2')
    assert data_dir('/data/scenes', 's', 'r', '0002', 'cpu') == Path('/data/scenes/s/r/0002/cpu')
    assert case_dir('/data/cases', 's', 'r', '0002', 'stuck') == Path('/data/cases/s/r/0002/stuck')


def test_checkpoint_next_step_and_name(tmp_path):
    directory = tmp_path / 'checkpoints'
    directory.mkdir()
    (directory / 'step2.hdf5').touch()
    assert resume_steps(tmp_path, None) == (3, 3)
    assert resume_steps(tmp_path, (1, 2)) == (1, 2)
    (directory / 'step3.hdf5').touch()
    with pytest.raises(ValueError, match='give --steps'):
        resume_steps(tmp_path, None)
    steps = [SimpleNamespace(name='open'), SimpleNamespace(name='pick')]
    check_checkpoint_name({'step': 0, 'name': 'initial state'}, steps)
    check_checkpoint_name({'step': 1, 'name': 'open'}, steps)
    with pytest.raises(ValueError, match='current steps with that name: 2'):
        check_checkpoint_name({'step': 1, 'name': 'pick'}, steps)


@pytest.mark.parametrize('saved, message', [
    ({}, 'batch predates robot/expert tags; rerun it on the current code'),
    ({'robot': 'a'}, 'batch predates robot/expert tags; rerun it on the current code'),
    ({'robot': 'b', 'expert': '0000'}, 'robot'), ({'robot': 'a', 'expert': '0002'}, 'expert')])
def test_checkpoint_identity(saved, message):
    with pytest.raises(ValueError, match=message):
        check_identity(saved, 'a', '0000', 'batch')


def test_scope_cli_rejects_outside_and_formal_override(tmp_path):
    sources(tmp_path)
    directory = initial_dir(tmp_path / 'assets/scenes', 'demo', 'a', 2)
    directory.mkdir(parents=True)
    with h5py.File(directory / 'initial_states.hdf5', 'w') as file:
        file.attrs.update(scene='demo', robot='a')
        file.create_group('2')
    base = ['demo', '--physics', 'isaac-cpu', '--root', str(tmp_path), '--input', str(tmp_path / 'assets/scenes')]
    with pytest.raises(SystemExit):
        main([*base, '--dev', '--layout', '2', '--seeds', '2'])
    with pytest.raises(SystemExit):
        main([*base, '--outside-scope'])


@pytest.mark.parametrize('summary_changes, checkpoint_changes, message', [
    ({'expert': None}, {}, 'batch predates robot/expert tags; rerun it on the current code'),
    ({'robot': 'b'}, {}, "robot 'b' differs"),
    ({}, {'expert': '0005'}, "expert '0005' differs"),
    ({}, {'name': 'pick'}, 'current steps with that name: 2'),
])
def test_from_cli_checks_summary_checkpoint_and_step(tmp_path, capfd, summary_changes, checkpoint_changes, message):
    directory = sources(tmp_path)
    path = directory / 'experts/a/0002.py'
    path.write_text(path.read_text() + "from types import SimpleNamespace\nSTEPS = [SimpleNamespace(name='open'), SimpleNamespace(name='pick')]\n")
    batch = tmp_path / 'batch'
    (batch / 'checkpoints').mkdir(parents=True)
    summary = {'robot': 'a', 'expert': '0002', 'layout': 0, **summary_changes}
    (batch / 'summary.json').write_text(json.dumps(summary))
    with h5py.File(batch / 'checkpoints/step1.hdf5', 'w') as file:
        file.attrs.update(scene='demo', robot='a', expert='0002', physics='isaac-cpu', step=1, name='open')
        file.attrs.update(checkpoint_changes)
        file.create_group('0')
    with pytest.raises(SystemExit):
        main(['demo', '--physics', 'isaac-cpu', '--root', str(tmp_path), '--dev', '--from', str(batch)])
    assert message in capfd.readouterr().err


def test_scope_defaults_run_every_layout_with_its_own_seeds(tmp_path, monkeypatch):
    sources(tmp_path)
    from table_1000.common import batch
    from table_1000.scenes.initial import write_states
    for layout, seeds in [(0, [0, 1, 2]), (2, [0, 1, 2, 3])]:
        directory = initial_dir(tmp_path / 'assets/scenes', 'demo', 'a', layout)
        directory.mkdir(parents=True)
        write_states(directory / 'initial_states.hdf5', {seed: (0, {}) for seed in seeds},
                     {'scene': 'demo', 'robot': 'a', 'layout': layout})
    selected = []

    def jobs(commands, gpus):
        for command, _, _ in commands:
            layout = int(command[command.index('--layout') + 1])
            seeds = batch.parse_seeds(command[command.index('--seeds') + 1:command.index('--states')])
            selected.append((layout, seeds))
            work = Path(command[command.index('--work') + 1])
            (work / 'episodes').mkdir(parents=True)
            for seed in seeds:
                (work / f'episodes/{seed}.json').write_text(json.dumps(
                    {'seed': seed, 'status': 'success', 'duration_s': 1., 'wall_s': 1.}))
        return []

    monkeypatch.setattr(batch, 'run_jobs', jobs)
    main(['demo', '--physics', 'isaac-cpu', '--root', str(tmp_path),
          '--input', str(tmp_path / 'assets/scenes'), '--output', str(tmp_path / 'data/scenes')])
    assert selected == [(0, [0, 1, 2]), (2, [1, 3])]
    for layout in [0, 2]:
        directory = data_dir(tmp_path / 'data/scenes', 'demo', 'a', '0002', 'isaac-cpu', layout)
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['layout'] == layout and summary['expert'] == '0002'


def scoped_input(root, seeds, *, checkpoint):
    directory = sources(root, "ROBOT = 'a'\nSCOPE = {0: [0, 1]}\n")
    expert = directory / 'experts/a/0002.py'
    expert.write_text(expert.read_text() + "from types import SimpleNamespace\n"
                      "STEPS = [SimpleNamespace(name='open'), SimpleNamespace(name='pick')]\n")
    directory = root / 'batch' if checkpoint else initial_dir(root / 'assets/scenes', 'demo', 'a')
    path = directory / ('checkpoints/step1.hdf5' if checkpoint else 'initial_states.hdf5')
    path.parent.mkdir(parents=True)
    with h5py.File(path, 'w') as file:
        file.attrs.update(scene='demo', robot='a', expert='0002', physics='isaac-cpu')
        if checkpoint:
            file.attrs.update(step=1, name='open')
        for seed in seeds:
            file.create_group(str(seed))
    if checkpoint:
        (directory / 'summary.json').write_text(json.dumps({'scene': 'demo', 'robot': 'a', 'expert': '0002', 'layout': 0}))
    return ['demo', '--physics', 'isaac-cpu', '--root', str(root),
            *(['--from', str(directory), '--steps', '2'] if checkpoint else
              ['--input', str(root / 'assets/scenes')])]


@pytest.mark.parametrize('checkpoint', [True, False])
@pytest.mark.parametrize('explicit', [False, True])
def test_partial_scope_development_runs_and_records_available_seeds(tmp_path, monkeypatch, checkpoint, explicit):
    from table_1000.common import batch
    args = scoped_input(tmp_path, [0, 2], checkpoint=checkpoint)
    scheduled = []

    def jobs(commands, gpus):
        for command, _, _ in commands:
            seeds = batch.parse_seeds(command[command.index('--seeds') + 1:command.index('--states')])
            scheduled.extend(seeds)
            work = Path(command[command.index('--work') + 1])
            (work / 'episodes').mkdir(parents=True)
            for seed in seeds:
                (work / f'episodes/{seed}.json').write_text(json.dumps(
                    {'seed': seed, 'status': 'success', 'duration_s': 1., 'wall_s': 1.}))
        return []

    monkeypatch.setattr(batch, 'run_jobs', jobs)
    main([*args, '--dev', *(['--seeds', '0'] if explicit else [])])
    assert scheduled == [0]
    (summary,) = (tmp_path / 'runs/expert/demo').glob('*/summary.json')
    assert json.loads(summary.read_text())['seeds'] == {'range': '0', 'count': 1}


@pytest.mark.parametrize('checkpoint', [True, False])
def test_partial_scope_empty_intersection_is_rejected(tmp_path, capfd, checkpoint):
    args = scoped_input(tmp_path, [2], checkpoint=checkpoint)
    with pytest.raises(SystemExit):
        main([*args, '--dev'])
    assert 'no states within SCOPE' in capfd.readouterr().err


def test_formal_initial_states_require_every_declared_seed(tmp_path, capfd):
    args = scoped_input(tmp_path, [0], checkpoint=False)
    with pytest.raises(SystemExit):
        main(args)
    assert 'holds no states of SCOPE seeds 1' in capfd.readouterr().err


def test_explicit_resume_seed_still_checks_scope(tmp_path, capfd):
    args = scoped_input(tmp_path, [0, 2], checkpoint=True)
    with pytest.raises(SystemExit):
        main([*args, '--dev', '--seeds', '2'])
    assert 'outside SCOPE' in capfd.readouterr().err
