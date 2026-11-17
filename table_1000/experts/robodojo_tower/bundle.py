"""Materialize v031 with caller-supplied, hash-pinned runtime calibration.

Run in a RoboDojo-compatible NumPy/SciPy/threadpoolctl environment. The public
source snapshot does not redistribute deployment geometry or grasp data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).parent


def release_manifest():
    return json.loads((ROOT / 'release.json').read_text(encoding='utf8'))


def verify_files(directory, expected):
    for name, digest in expected.items():
        actual = hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError('Runtime bundle SHA256 mismatch: ' + name)


def materialize(calibration_directory, destination):
    """Write a fresh top-level tower_cap package with all 27 validated bytes."""
    manifest = release_manifest()
    verify_files(ROOT / 'tower_cap', manifest['python_files_sha256'])
    verify_files(calibration_directory, manifest['external_runtime_files_sha256'])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for source, names in ((ROOT / 'tower_cap', manifest['python_files_sha256']),
                          (Path(calibration_directory), manifest['external_runtime_files_sha256'])):
        for name in names:
            shutil.copyfile(source / name, destination / name)
    verify_files(destination, {**manifest['python_files_sha256'], **manifest['external_runtime_files_sha256']})
    return manifest['package_fingerprint_sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-directory', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    digest = materialize(args.calibration_directory, args.destination)
    print(json.dumps({'package': str(args.destination), 'package_fingerprint_sha256': digest}))


if __name__ == '__main__':
    main()
