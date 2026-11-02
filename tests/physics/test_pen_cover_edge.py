"""The two edge-contact fixtures run in isolated Isaac processes on an assigned GPU."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('case', ['clip12', 'clip13'])
def test_pen_cover_edge_has_no_reverse_contact_or_launch(case, tmp_path):
    if not os.environ.get('GQ_GPUS') or importlib.util.find_spec('isaacsim') is None:
        pytest.skip('requires Isaac Sim and a gq-assigned GPU; run isaac_pen_cover_edge.py directly otherwise')
    # Path-configuration unit tests may leave SDK overrides in this process.
    # The isolated simulator must discover the SDK installed for this Python.
    environment = os.environ.copy()
    for name in ('ISAAC_PATH', 'EXP_PATH', 'CARB_APP_PATH'):
        environment.pop(name, None)
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('isaac_pen_cover_edge.py')),
                             '--case', case, '--output', str(tmp_path / f'{case}.json')],
                            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout[-5000:] + result.stderr[-5000:]
