"""The solid option preserves visuals and closes the original outer contour on all axes."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_solid_tube_collision_meshes():
    blender = shutil.which('blender')
    if blender is None:
        pytest.skip('requires Blender')
    script = Path(__file__).with_name('blender_tube_collision.py')
    result = subprocess.run([blender, '-b', '--python-exit-code', '1', '-P', str(script)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'TUBE_COLLISION_CHECK_COMPLETED' in result.stdout
