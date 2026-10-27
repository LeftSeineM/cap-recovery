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

from table_1000.common import batch


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
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

    def command(group, index, attempt):
        item, = group
        # The original path is intentionally carried as a non-source payload to let worker 0 edit it.
        argv = [str(python), str(repo / 'scripts/worker.py'), str(repo / '.local' / '..'),
                str(repo), str(item), str(attempt)]
        return argv, repo / 'outputs' / f'{item}-{attempt}.log', str(item)

    (repo / '.local').mkdir()
    # Resolve '..' only inside the worker, so redirecting sources does not rewrite the edit target.
    result = batch.run_batch([0, 1, 2], command=command, missing=lambda item: not (repo / 'outputs' / f'done-{item}').exists(),
                             key=lambda item: item, count=1, gpus=[0], launcher=launcher, retries=1, root=repo)
    assert result == [] and result.commit == head
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

    def command(group, index, attempt):
        # Snapshot creation must not touch the source files or index before the first worker starts.
        if group == [0] and not attempt:
            assert all(path.read_bytes() == data for path, data in original_files.items())
            assert git(repo, 'status', '--porcelain') == original_status
        assert (repo / '.git/index').read_bytes() == original_index
        item, = group
        return ([sys.executable, str(worker), str(repo / '.local/..'), str(repo), str(item), str(attempt)],
                repo / 'outputs' / f'{item}-{attempt}.log', str(item))

    result = batch.run_batch([0, 1, 2], command=command, missing=lambda item: not (repo / 'outputs' / f'done-{item}').exists(),
                             key=lambda item: item, count=1, gpus=[0], launcher=launcher, retries=1, root=repo)
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
    result = batch.run_batch([0], root=copy, count=1, gpus=[0], missing=lambda _: False,
                             command=lambda *_: ([sys.executable, '-c', 'import pathlib; print(pathlib.Path.cwd())', '--'],
                                                 copy / 'worker.log', 'copy'))
    assert result.commit is None and result.patch is None
    assert result.record(copy) == {'commit': None}
    assert (copy / 'worker.log').read_text().strip() == str(copy)


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
def test_failed_and_exceptional_batches_remove_their_checkout(repo, dirty):
    if dirty:
        (repo / 'new.py').write_text('NEW = True\n')
    def command(group, index, attempt):
        return [sys.executable, '-c', 'raise SystemExit(1)', '--'], repo / 'outputs/failed.log', 'failed'

    result = batch.run_batch([0], command=command, missing=lambda _: True, count=1, gpus=[0], retries=0, root=repo)
    assert result == [0] and registered(repo) == [f'worktree {repo}']

    def broken(*args):
        raise RuntimeError('command failed')

    with pytest.raises(RuntimeError, match='command failed'):
        batch.run_batch([0], command=broken, missing=lambda _: True, count=1, gpus=[0], root=repo)
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
from table_1000.common.batch import run_batch
root = Path({str(repo)!r})
def command(group, index, attempt):
    code = "import os, pathlib, time; pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)"
    return [sys.executable, '-c', code, '--'], root / 'outputs/worker.log', 'sleep'
run_batch([0, 1], command=command, key=lambda item: item, missing=lambda _: True,
          count=1, gpus=[0], root=root, launcher={launcher!r})
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
    result = batch.run_batch([], command=lambda *args: None, missing=lambda _: False,
                             count=1, gpus=[0], root=repo)
    assert result.patch
    (repo / 'table_1000/value.py').write_text("VALUE = 'next task'\n")
    (repo / 'new.py').write_text('NEXT = True\n')
    with batch._checkout(repo, source=result) as (frozen, revision, patch):
        assert revision == result.commit and patch == result.patch
        assert (frozen / 'table_1000/value.py').read_text() == "VALUE = 'frozen dirty'\n"
        assert not (frozen / 'new.py').exists()
    assert (repo / 'new.py').exists()
