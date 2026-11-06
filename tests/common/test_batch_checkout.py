"""Real worker processes: submitted sources/imports, editable environments, retries and cleanup."""

import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import venv

import pytest

from table_1000.common import batch, batch_runtime as runtime


FIXTURE = """
from pathlib import Path
import os, subprocess, sys, time
from table_1000.common.batch_runtime import Unit, UnitResult
class Program:
    def units(self, options):
        for item in options.get('units', [0, 1, 2]):
            yield Unit((item,), dict(item=item), f'unit-{item}.json', ('fixture', str(item)), ('fixture-load',))
    def pack_key(self, unit):
        return unit.key
    def run_unit(self, context, unit):
        if context.options.get('sleep'):
            ready = Path(context.options['sleep'])
            temporary = ready.with_suffix('.part')
            temporary.write_text(str(os.getpid()))
            temporary.replace(ready)
            time.sleep(60)
        item = unit.key[0]
        attempt = int(context.work.name.split('attempt')[1])
        subprocess.run([sys.executable, str(context.root / 'scripts/worker.py'),
            context.options['original'], str(context.root), str(item), str(attempt)], check=True)
        ok = (context.root / 'outputs' / f'done-{item}').exists()
        return UnitResult(dict(status='success' if ok else 'error', attempt=99))
program = Program()
"""


def execute(root, **options):
    return runtime.run('table_1000.fixture:program', dict(original=str(root / '.local/..'),
        jobs=1, gpus=[0], retries=1) | options, root=root, output=root / 'outputs')


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / 'repo with spaces'
    root.mkdir()
    git(root, 'init', '-q')
    git(root, 'config', 'user.name', 'test')
    git(root, 'config', 'user.email', 'test@example.com')
    (root / '.gitignore').write_text('.local/\n.venv/\n__pycache__/\nassets/\ndata/\nruns/\noutputs/\n')
    (root / 'table_1000').mkdir()
    (root / 'table_1000/__init__.py').write_text('')
    (root / 'table_1000/value.py').write_text("VALUE = 'submitted'\n")
    (root / 'asset_sources').mkdir()
    (root / 'asset_sources/source.txt').write_text('submitted')
    (root / 'scripts').mkdir()
    (root / 'scripts/worker.py').write_text('''\
import json
import os
from pathlib import Path
import subprocess
import sys
from table_1000 import value

original, source, item, attempt = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
root = Path(__file__).resolve().parents[1]
record = {'value': value.VALUE, 'module': value.__file__, 'script': __file__,
          'source': (source / 'asset_sources/source.txt').read_text(), 'python': sys.executable,
          'commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--short', 'HEAD'], text=True).strip()}
for name in ('assets', 'data', 'runs', 'outputs'):
    directory = root / name
    directory.mkdir(exist_ok=True)
    (directory / f'{item}-{attempt}.json').write_text(json.dumps(record))
if item == '0' and attempt == 0:
    (original / 'table_1000/value.py').write_text("VALUE = 'changed after submission'\\n")
    (original / 'scripts/worker.py').write_text('raise RuntimeError("mutable worker")\\n')
    (original / 'asset_sources/source.txt').write_text('changed after submission')
    subprocess.run(['git', '-C', str(original), 'commit', '-qam', 'advance source checkout'], check=True)
if item != '1' or attempt:
    (root / 'outputs' / f'done-{item}').touch()
''')
    source = Path(batch.__file__).parent
    (root / 'table_1000/common').mkdir()
    (root / 'table_1000/common/__init__.py').touch()
    for name in ('batch.py', 'batch_runtime.py', 'batch_timing.py', 'serialization.py'):
        shutil.copy2(source / name, root / 'table_1000/common' / name)
    (root / 'configs').mkdir()
    shutil.copy2(source.parents[1] / 'configs/batch.toml', root / 'configs/batch.toml')
    (root / 'table_1000/fixture.py').write_text(FIXTURE)
    import runpy
    implementation = runpy.run_path(str(root / 'table_1000/fixture.py'))['program']
    monkeypatch.setattr(runtime, '_program', lambda _: implementation)
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'submitted')
    return root


def registered(root):
    return [line for line in git(root, 'worktree', 'list', '--porcelain').splitlines()
            if line.startswith('worktree ')]


@pytest.mark.parametrize('launcher', [None, 'env'])
def test_workers_and_retries_keep_submitted_code_and_editable_imports(repo, monkeypatch, launcher):
    head = batch.commit(repo)
    # A .pth editable install points at the original checkout, as in the project's uv environment.
    environment = repo / '.venv'
    venv.EnvBuilder(with_pip=False).create(environment)
    site, = environment.glob('lib/python*/site-packages')
    (site / 'editable.pth').write_text(str(repo) + '\n')
    python = environment / 'bin/python'
    monkeypatch.setenv('PYTHONPATH', str(repo))
    assert subprocess.check_output([str(python), '-c', 'import table_1000; print(table_1000.__file__)'],
                                   cwd=repo, text=True).strip() == str(repo / 'table_1000/__init__.py')

    (repo / '.local').mkdir()
    monkeypatch.setattr(runtime.sys, 'executable', str(python))
    result = execute(repo, launcher=launcher)
    assert result == [] and result.commit == head
    report = json.loads((repo / 'outputs/unit-1.json').read_text())
    assert report['attempt'] == 99 and report['worker_attempt'] == 1
    assert batch.commit(repo) != head
    checkouts = set()
    for item, attempt in ((0, 0), (1, 0), (2, 0), (1, 1)):
        for name in batch.SHARED:
            record = json.loads((repo / name / f'{item}-{attempt}.json').read_text())
            assert record['value'] == record['source'] == 'submitted'
            assert record['commit'] == result.commit
            assert record['python'] == str(python)
            checkout = Path(record['module']).parents[1]
            assert checkout != repo and Path(record['script']).is_relative_to(checkout)
            assert checkout.is_relative_to(repo / '.local/batch-checkouts')
            assert not (checkout / '.venv').exists()
            checkouts.add(checkout)
    assert len(checkouts) == 1 and not next(iter(checkouts)).exists()
    assert registered(repo) == [f'worktree {repo}']


@pytest.mark.parametrize('launcher', [None, 'env'])
def test_dirty_workers_and_retries_keep_working_files_and_patch(repo, monkeypatch, launcher):
    head = batch.commit(repo)
    value = repo / 'table_1000/value.py'
    value.write_text("VALUE = 'staged'\n")
    git(repo, 'add', str(value))
    value.write_text("VALUE = 'dirty'\n")
    new = repo / 'table_1000/new.py'
    new.write_text("VALUE = 'untracked'\n")
    worker = repo / 'scripts/worker.py'
    # Keep the original fixture's queued work/retry mechanism, but mutate files without committing.
    worker.write_text(worker.read_text().replace('from table_1000 import value', 'from table_1000 import value, new')
                      .replace("'value': value.VALUE", "'new': new.VALUE, 'value': value.VALUE")
                      .replace("subprocess.run(['git', '-C', str(original), 'commit', '-qam', 'advance source checkout'], check=True)",
                               "(original / 'table_1000/new.py').unlink()"))
    original_files = {path: path.read_bytes() for path in (value, new, worker)}
    original_status = git(repo, 'status', '--porcelain')
    original_index = (repo / '.git/index').read_bytes()
    monkeypatch.setenv('PYTHONPATH', str(repo))
    (repo / '.local').mkdir()

    run_jobs = batch.run_jobs
    def checked_jobs(jobs, gpus):
        assert (repo / '.git/index').read_bytes() == original_index
        return run_jobs(jobs, gpus)
    monkeypatch.setattr(batch, 'run_jobs', checked_jobs)
    result = execute(repo, launcher=launcher)
    assert result == [] and result.commit == head + '+dirty'
    assert (repo / '.git/index').read_bytes() == original_index
    assert not new.exists() and 'changed after submission' in value.read_text()
    checkouts = set()
    for item, attempt in ((0, 0), (1, 0), (2, 0), (1, 1)):
        record = json.loads((repo / 'outputs' / f'{item}-{attempt}.json').read_text())
        assert record['value'] == 'dirty' and record['new'] == 'untracked'
        assert record['source'] == 'submitted' and record['commit'] == head
        checkout = Path(record['module']).parents[1]
        assert checkout != repo and '+dirty-' in checkout.name
        assert Path(record['script']).is_relative_to(checkout)
        checkouts.add(checkout)
    assert len(checkouts) == 1 and not next(iter(checkouts)).exists()
    assert registered(repo) == [f'worktree {repo}']
    directory = repo / 'runs/batch'
    assert result.record(directory) == {'commit': head + '+dirty', 'dirty_patch': 'dirty.patch'}
    restored = repo / '.local/restored'
    git(repo, 'worktree', 'add', '--detach', str(restored), head)
    try:
        git(restored, 'apply', str(directory / 'dirty.patch'))
        assert all((restored / path.relative_to(repo)).read_bytes() == data for path, data in original_files.items())
    finally:
        git(repo, 'worktree', 'remove', '--force', str(restored))


def test_dirty_patch_restores_binary_modes_symlinks_and_deletions(repo):
    (repo / 'asset_sources/source.txt').unlink()
    binary = repo / 'new binary.dat'
    binary.write_bytes(bytes(range(256)))
    binary.chmod(0o755)
    (repo / 'new link').symlink_to('new binary.dat')
    (repo / 'outputs').mkdir()
    (repo / 'outputs/ignored.py').write_text('ignored')
    index = (repo / '.git/index').read_bytes()
    with batch._checkout(repo) as (checkout, revision, patch):
        assert revision.endswith('+dirty')
        assert (checkout / binary.name).read_bytes() == binary.read_bytes()
        assert (checkout / binary.name).stat().st_mode & 0o111
        assert (checkout / 'new link').readlink() == Path(binary.name)
        assert not (checkout / 'asset_sources/source.txt').exists()
        assert b'ignored.py' not in patch
        assert (repo / '.git/index').read_bytes() == index
        # Apply independently to HEAD, checking binary additions, modes and symlinks too.
        restored = repo / '.local/restored'
        git(repo, 'worktree', 'add', '--detach', str(restored), revision.removesuffix('+dirty'))
        try:
            subprocess.run(['git', '-C', str(restored), 'apply'], input=patch, check=True)
            assert (restored / binary.name).read_bytes() == binary.read_bytes()
            assert (restored / binary.name).stat().st_mode & 0o111
            assert (restored / 'new link').readlink() == Path(binary.name)
            assert not (restored / 'asset_sources/source.txt').exists()
        finally:
            git(repo, 'worktree', 'remove', '--force', str(restored))


def test_non_git_sources_run_in_place(repo, monkeypatch):
    copy = repo / 'outputs/copy'
    shutil.copytree(repo / 'table_1000', copy / 'table_1000')
    with batch._checkout(copy) as (checkout, revision, patch):
        assert checkout == copy and revision is None and patch is None
    monkeypatch.chdir(copy)
    (copy / 'scripts').mkdir()
    shutil.copy2(repo / 'scripts/worker.py', copy / 'scripts/worker.py')
    # A dry run uses the same output without executing or reserving it.
    result = runtime.run('table_1000.fixture:program', dict(jobs=1, dry_run=True), root=copy, output=copy / 'outputs')
    assert result.commit is None and result.patch is None
    assert result.record(copy) == {'commit': None}
    assert (copy / 'outputs/plan.json').is_file()
    from table_1000.common.batch_runtime import Unit, load_config
    from table_1000.common.batch_timing import Timings, record, timing_path
    units = [Unit((seed,), {}, f'{seed}.json', ('asset', 'cup', 'robot', 'cpu', str(seed)),
                  ('asset', 'cup', 'robot', 'cpu')) for seed in (0, 1)]
    report = dict(status='failed', wall_s=110., cpu_s=95., save_wall_s=20., save_cpu_s=10.,
                  app_start_s=20., asset_load_s=30., commit='frozen')
    record(repo, copy, units[0], report)
    key = units[0].timing_key
    value = json.loads(timing_path(repo, key).read_text())
    assert value['commit'] == 'frozen' and value['batch'] == str(copy) and value['wall_s'] == 110.
    record(repo, copy, units[0], report | dict(wall_s=120., commit='new'))
    assert json.loads(timing_path(repo, key).read_text())['wall_s'] == 120.
    assert list(timing_path(repo, key).parent.iterdir()) == [timing_path(repo, key)]
    timing = Timings(repo, units)
    assert timing.coverage == (1, 2) and set(timing.costs.values()) == {140.}
    assert timing.wall([u.key for u in units]) == 330.
    machine = copy / 'machine.toml'
    machine.write_text('max_jobs = 10\n')
    assert load_config(path=machine).jobs('auto', 60000., 200) == 10
    assert load_config(dict(max_jobs=30), path=machine).jobs(30, 60000., 200) == 30
    config = load_config(path=copy / 'absent.toml')
    assert config.jobs('auto', 1500., 200) == 5
    assert config.jobs('auto', 12000., 200) == 20
    assert config.jobs('auto', 120000., 200) == 100


def test_pinned_command_redirects_bare_relative_files(repo, monkeypatch):
    monkeypatch.chdir(repo)
    (repo / 'README.md').write_text('readme')
    (repo / 'setup.py').write_text('setup')
    checkout = repo / '.local/frozen'
    argv = [sys.executable, 'setup.py', 'README.md', 'scripts/worker.py', '0', '0:2', 'label', '--name', 'test',
            'outputs/report.json', str(repo), str(repo / 'assets/file')]
    command = batch._pinned_command(argv, repo, checkout)
    assert command[4:] == [sys.executable, str(checkout / 'setup.py'), str(checkout / 'README.md'),
                           str(checkout / 'scripts/worker.py'), '0', '0:2', 'label', '--name', 'test',
                           'outputs/report.json', str(checkout), str(repo / 'assets/file')]


@pytest.mark.parametrize('dirty', [False, True])
@pytest.mark.parametrize('added', [False, True])
@pytest.mark.parametrize('interrupted', [False, True])
def test_failed_or_interrupted_worktree_add_cleans_directory(repo, monkeypatch, dirty, added, interrupted):
    if dirty:
        (repo / 'new.py').write_text('NEW = True\n')
    run = subprocess.run
    error = KeyboardInterrupt() if interrupted else subprocess.CalledProcessError(1, ['git', 'worktree', 'add'])

    def fail_add(argv, **kwargs):
        if 'worktree' in argv and 'add' in argv:
            if added:
                run(argv, **kwargs)
            raise error
        return run(argv, **kwargs)

    monkeypatch.setattr(batch.subprocess, 'run', fail_add)
    with pytest.raises(type(error)) as caught:
        with batch._checkout(repo):
            pytest.fail('add must fail')
    assert caught.value is error
    assert registered(repo) == [f'worktree {repo}']
    assert not list((repo / '.local/batch-checkouts').iterdir())


def test_checkout_cleanup_error_does_not_mask_add_interruption(repo, monkeypatch, capsys):
    run = subprocess.run
    error = KeyboardInterrupt()
    checkout = None

    def fail_add_and_remove(argv, **kwargs):
        nonlocal checkout
        if 'worktree' in argv:
            if 'add' in argv:
                run(argv, **kwargs)
                checkout = argv[-2]
                raise error
            if 'remove' in argv:
                raise OSError('cleanup unavailable')
        return run(argv, **kwargs)

    monkeypatch.setattr(batch.subprocess, 'run', fail_add_and_remove)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            with batch._checkout(repo):
                pytest.fail('add must be interrupted')
        assert caught.value is error
        assert 'cleanup unavailable' in capsys.readouterr().err
    finally:
        run(['git', '-C', str(repo), 'worktree', 'remove', '--force', checkout], check=True)


@pytest.mark.parametrize('dirty', [False, True])
def test_failed_and_exceptional_batches_remove_their_checkout(repo, monkeypatch, dirty):
    if dirty:
        (repo / 'new.py').write_text('NEW = True\n')
    monkeypatch.setattr(batch, 'run_jobs', lambda *args: ['failed'])
    result = execute(repo, units=[0], retries=0)
    assert result == [(0,)] and registered(repo) == [f'worktree {repo}']
    def broken(*args):
        raise RuntimeError('submission failed')
    monkeypatch.setattr(batch, 'run_jobs', broken)
    with pytest.raises(RuntimeError, match='submission failed'):
        execute(repo, units=[0])
    assert registered(repo) == [f'worktree {repo}']


@pytest.mark.parametrize('dirty', [False, True])
def test_concurrent_checkouts_have_separate_lifetimes(repo, dirty):
    if dirty:
        (repo / 'new.py').write_text('NEW = True\n')
    with batch._checkout(repo) as (first, revision, patch):
        with batch._checkout(repo) as (second, other, other_patch):
            assert revision == other and first != second
            assert patch == other_patch
            assert first.exists() and second.exists() and len(registered(repo)) == 3
        assert first.exists() and not second.exists() and len(registered(repo)) == 2
    assert not first.exists() and len(registered(repo)) == 1


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize('launcher', [None, 'env'])
@pytest.mark.parametrize('dirty', [False, True])
def test_interruption_stops_workers_and_removes_checkout(repo, tmp_path, signum, launcher, dirty):
    if dirty:
        (repo / 'new.py').write_text('NEW = True\n')
    ready = tmp_path / 'ready'
    script = tmp_path / 'driver.py'
    script.write_text(f'''\
import sys
from pathlib import Path
sys.path.insert(0, {str(Path(batch.__file__).resolve().parents[2])!r})
from table_1000.common.batch_runtime import run
import runpy
import table_1000.common.batch_runtime as runtime
root = Path({str(repo)!r})
implementation = runpy.run_path(str(root / 'table_1000/fixture.py'))['program']
runtime._program = lambda _: implementation
run('table_1000.fixture:program', dict(units=[0,1], jobs=1, gpus=[0], sleep={str(ready)!r},
    launcher={launcher!r}), root=root, output=root / 'outputs')
''')
    with subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, start_new_session=True) as process:
        try:
            deadline = time.monotonic() + 15
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            assert ready.exists()
            pid = int(ready.read_text())
            process.send_signal(signum)
            output, _ = process.communicate(timeout=15)
            assert process.returncode != 0, output
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    # A shell launcher can briefly leave its reaped-by-init child as a zombie in containers.
    status = Path(f'/proc/{pid}/status')
    assert not status.exists() or 'State:\tZ' in status.read_text()
    assert registered(repo) == [f'worktree {repo}']
    assert not list((repo / '.local/batch-checkouts').iterdir())


def test_later_phase_reuses_frozen_dirty_source_after_original_changes(repo):
    (repo / 'table_1000/value.py').write_text("VALUE = 'frozen dirty'\n")
    with batch._checkout(repo) as (_, commit, patch):
        result = batch.BatchResult([], commit, patch)
    assert result.patch
    (repo / 'table_1000/value.py').write_text("VALUE = 'next task'\n")
    (repo / 'new.py').write_text('NEXT = True\n')
    with batch._checkout(repo, source=result) as (frozen, revision, patch):
        assert revision == result.commit and patch == result.patch
        assert (frozen / 'table_1000/value.py').read_text() == "VALUE = 'frozen dirty'\n"
        assert not (frozen / 'new.py').exists()
    assert (repo / 'new.py').exists()
