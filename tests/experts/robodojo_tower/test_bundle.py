import ast
import hashlib
from pathlib import Path
import tempfile
import unittest

from table_1000.experts.robodojo_tower.bundle import ROOT, materialize, release_manifest, verify_files


class BundleTests(unittest.TestCase):
    def test_all_published_code_is_exact_native_validated_snapshot(self):
        manifest = release_manifest()
        expected = manifest['python_files_sha256']
        self.assertEqual(set(expected), {p.name for p in (ROOT / 'tower_cap').glob('*.py')})
        self.assertEqual(len(expected), 23)
        verify_files(ROOT / 'tower_cap', expected)
        for name in expected:
            ast.parse((ROOT / 'tower_cap' / name).read_text(encoding='utf8'))

    def test_missing_runtime_data_does_not_create_package(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'out'
            with self.assertRaises(FileNotFoundError):
                materialize(directory, destination)
            self.assertFalse(destination.exists())

    def test_changed_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'geometry.json'
            path.write_bytes(b'calibration')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            verify_files(directory, {'geometry.json': digest})
            path.write_bytes(b'different geometry')
            with self.assertRaisesRegex(ValueError, 'geometry.json'):
                verify_files(directory, {'geometry.json': digest})


if __name__ == '__main__':
    unittest.main()
