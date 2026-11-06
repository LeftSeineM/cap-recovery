"""Physics test runs: per-case results, batch scheduling and the per-asset summary."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from tests.common.batch_worker import manifest, publish

from table_1000.physics.test import PhysicalFailure, main, run_case, run_test, summary


class CaseResultsTests(unittest.TestCase):
    def test_definition_import_failure_closes_kit_with_nonzero_status(self):
        from table_1000.common import batch_runtime as runtime
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            units = [runtime.Unit((i,), dict(seed=i), f'{i}.json', ('physics', str(i)), ('load',)) for i in range(2)]
            from dataclasses import asdict
            path = root / 'units.json'
            path.write_text(json.dumps(dict(root=str(root), output=str(root), work=str(root), options={},
                commit='submitted', attempt=0, program='fake:program', units=[asdict(u) for u in units])))
            program = Mock()
            program.run_unit.side_effect = ImportError('broken definition')
            with patch.object(runtime, '_program', return_value=program):
                runtime.work(path, gpu=0)
            self.assertEqual(program.run_unit.call_count, 2)
            for i in range(2):
                result = json.loads((root / f'{i}.json').read_text())
                self.assertEqual(result['status'], 'error')
                self.assertIn('broken definition', result['failure']['reason'])

    def test_benchmark_failure_removes_old_case_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'physics_test'
            case = output / 'pull'
            (case / 'frames').mkdir(parents=True)
            (case / 'frames' / 'old.jpg').write_bytes(b'old frame')
            (case / 'result.json').write_text('{"execution_status":"completed"}')
            video = output / 'pull.mp4'
            video.write_bytes(b'old video')
            other = output / 'close.mp4'
            other.write_bytes(b'other result')
            source = root / 'physics_test.py'
            source.write_text('TESTS = {}\n')
            with patch('table_1000.physics.test.benchmark', side_effect=PhysicalFailure('benchmark failed')), \
                 patch('table_1000.physics.test.run_video') as render:
                result = run_case(root / 'object.usda', 'pull', None, output, None, 0, {}, standard=True)
                render.assert_not_called()
            self.assertFalse(video.exists())
            self.assertFalse((case / 'frames').exists())
            self.assertFalse((case / 'result.json').exists())
            self.assertEqual(result['execution_status'], 'failed')
            self.assertIn('benchmark failed', result['error'])
            self.assertEqual(other.read_bytes(), b'other result')
            self.assertEqual(source.read_text(), 'TESTS = {}\n')
            with self.assertRaisesRegex(PhysicalFailure, 'did not settle'):
                run_test(Mock(side_effect=AssertionError('did not settle')), None)
            from types import SimpleNamespace
            from table_1000.common import batch, batch_runtime as runtime
            from table_1000.physics.batch_program import program
            options = dict(source=str(root), units=[('cup', 'isaac-cpu', 'pull')],
                           num_envs=1, benchmark=False, jobs=1, retries=1, gpus=[0])
            rounds = []
            def jobs(commands, gpus):
                for command, _, _ in commands:
                    data = manifest(command)
                    rounds.append(data['attempt'])
                    context = SimpleNamespace(options=options, loaded=dict(tests={'pull': None},
                        app=None, startup=0., gpu_info={}, utilization=None))
                    unit = next(program.units(options))
                    if data['attempt'] == 0:
                        with patch('table_1000.physics.test.run_video', side_effect=KeyboardInterrupt):
                            with self.assertRaises(KeyboardInterrupt):
                                program.run_unit(context, unit)
                        self.assertFalse(Path(unit.result).exists())
                    else:
                        with patch('table_1000.physics.test.run_video', side_effect=PhysicalFailure('did not settle')):
                            result = program.run_unit(context, unit)
                        self.assertEqual(result.report['status'], 'failed')
                        self.assertFalse(Path(unit.result).exists())
                        with patch.object(runtime, '_program', return_value=Mock(run_unit=Mock(return_value=result))):
                            runtime.work(command[command.index('table_1000.common.batch_runtime') + 1], gpu=0)
                return []
            with patch.object(batch, 'run_jobs', side_effect=jobs):
                outcome = runtime.run('table_1000.physics.batch_program:program', options,
                    root=root, output=root / 'batch')
            self.assertEqual(rounds, [0, 1])
            self.assertEqual(outcome, [])
            self.assertFalse(runtime._pending(root / 'batch', next(program.units(options))))
            published = json.loads(Path(next(program.units(options)).result).read_text())
            self.assertEqual(published['status'], 'failed')
            self.assertEqual(published['execution_status'], 'failed')

    def test_default_run_skips_the_standard_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'physics_test'
            with patch('table_1000.physics.test.benchmark') as measure, \
                 patch('table_1000.physics.test.run_video', return_value={}) as render:
                run_case(Path(temporary) / 'object.usda', 'pull', None, output, None, 0, {})
            measure.assert_not_called()
            self.assertIsNone(render.call_args.args[6])

    def test_batch_runs_every_asset_backend_and_writes_summaries(self):
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as checkout:
            root = Path(temporary)
            for reference in ('pen/000002', 'cabinet/000000'):
                (root / reference).mkdir(parents=True)
                (root / reference / 'physics_test.py').write_text("TESTS = {'drop': None}\n")
            # The pen's definition was edited after the build.
            edited = Path(checkout) / 'asset_sources/objects/pen/000002/physics_test.py'
            edited.parent.mkdir(parents=True)
            edited.write_text("TESTS = {'drop': None, 'close': None}\n")
            def run(jobs, gpus):
                for command, _, _ in jobs:
                    data = manifest(command)
                    for unit in data['units']:
                        publish(data, unit, dict(status='success', execution_status='completed',
                            conditions={'isaac_version': ['5.1.0']}, environments={'count': 1}, metrics={},
                            performance={'no_render': {'RTF': 1}, 'video_end_to_end': {'RTF': 1}}))
                return []
            with patch('table_1000.common.batch.run_jobs', side_effect=run) as jobs, \
                 patch('table_1000.physics.test.ROOT', Path(checkout)), patch('builtins.print') as output:
                main(['--source', temporary, '--physics', 'isaac-cpu', 'isaac-gpu', '--gpus', '1', '2'])
            warnings = [call.args[0] for call in output.call_args_list if str(call.args[0]).startswith('WARNING')]
            self.assertEqual([line.split(':')[0] for line in warnings], ['WARNING pen/000002'])
            self.assertEqual(jobs.call_count, 1)
            commands = jobs.call_args.args[0]
            self.assertEqual(jobs.call_args.args[1], [1, 2])
            submitted = [tuple(unit['key']) for command, _, _ in commands for unit in manifest(command)['units']]
            self.assertEqual(sorted(submitted), sorted((asset, backend, 'drop') for asset in ('pen/000002', 'cabinet/000000')
                                                     for backend in ('isaac-cpu', 'isaac-gpu')))
            for reference in ('pen/000002', 'cabinet/000000'):
                self.assertTrue((root / reference / 'physics_test/summary.json').exists())

    def test_summary_lines_up_backends_per_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            asset = Path(temporary) / 'pen/000002'
            def write(backend, name, result):
                path = asset / 'physics_test' / backend / name / 'result.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(result))
            def completed(push, envs):
                return {'execution_status': 'completed', 'conditions': {'isaac_version': ['5.1.0']},
                        'environments': {'count': envs},
                        'metrics': {'seating_push': {'label': '合帽力', 'unit': 'N', 'value': push}},
                        'performance': {'no_render': {'RTF': .57123}, 'video_end_to_end': {'RTF': .34}}}
            write('isaac-cpu', 'close', completed(3.7412, 1))
            write('isaac-gpu', 'close', completed(2.14, 4))
            write('isaac-gpu', 'drop', {'execution_status': 'failed', 'error': 'Traceback\nAssertionError: pen did not settle\n'})
            write('isaac-cpu', 'removed', completed(1, 1))
            result = summary(asset, ['close', 'drop'])
        self.assertEqual(result['asset'], 'pen/000002')
        self.assertEqual(result['backends']['isaac-cpu'], {'completed': 1, 'failed': [], 'engine': 'Isaac Sim 5.1.0 CPU PhysX', 'environments': 1})
        self.assertEqual(result['backends']['isaac-gpu']['failed'], ['drop'])
        self.assertEqual(result['tests']['close']['metrics']['seating_push'],
                         {'label': '合帽力', 'unit': 'N', 'isaac-cpu': 3.741, 'isaac-gpu': 2.14})
        self.assertEqual(result['tests']['close']['rtf']['isaac-cpu'], {'physics': .571, 'video': .34})
        self.assertEqual(result['tests']['drop']['error'], {'isaac-gpu': 'AssertionError: pen did not settle'})
        self.assertEqual(list(result['tests']), ['close', 'drop'])

    def test_failed_backend_never_reports_old_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            asset = Path(temporary) / 'pen/000002'
            asset.mkdir(parents=True)
            (asset / 'physics_test.py').write_text("TESTS = {'close': None, 'drop': None}\n")
            for name in ('close', 'drop'):
                path = asset / 'physics_test/isaac-gpu' / name / 'result.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'execution_status': 'completed', 'conditions': {'isaac_version': ['5.1.0']},
                                            'environments': {'count': 4}, 'metrics': {},
                                            'performance': {'no_render': {'RTF': 1}, 'video_end_to_end': {'RTF': 1}}}))
            # The worker dies at startup and writes nothing.
            with patch('table_1000.common.batch.run_jobs', return_value=['pen/000002 isaac-gpu']):
                with self.assertRaises(SystemExit):
                    main(['--source', temporary, '--physics', 'isaac-gpu', '--tests', 'close'])
            result = json.loads((asset / 'physics_test/summary.json').read_text())
        self.assertEqual(result['tests']['close']['status'], {'isaac-gpu': 'failed'})
        self.assertIn('not run', result['tests']['close']['error']['isaac-gpu'])
        self.assertEqual(result['tests']['drop']['status'], {'isaac-gpu': 'completed'})
        self.assertEqual(result['backends']['isaac-gpu']['failed'], ['close'])

    def test_broken_definition_fails_only_its_asset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for reference, text in (('pen/000002', "TESTS = {'drop': None}\n"), ('cabinet/000000', "raise ImportError('broken')\n")):
                (root / reference).mkdir(parents=True)
                (root / reference / 'physics_test.py').write_text(text)
            with patch('table_1000.common.batch.run_jobs', return_value=[]) as jobs, \
                 patch('traceback.print_exc'):
                with self.assertRaisesRegex(SystemExit, 'cabinet/000000'):
                    main(['--source', temporary, '--physics', 'isaac-cpu'])
            self.assertEqual([tuple(u['key']) for c, _, _ in jobs.call_args.args[0] for u in manifest(c)['units']], [('pen/000002', 'isaac-cpu', 'drop')])
            broken = json.loads((root / 'cabinet/000000/physics_test/summary.json').read_text())
            self.assertIn('broken', broken['error'])
            self.assertTrue((root / 'pen/000002/physics_test/summary.json').exists())

    def test_plots_only_fails_on_a_broken_definition(self):
        with tempfile.TemporaryDirectory() as temporary:
            asset = Path(temporary) / 'bad/000000'
            asset.mkdir(parents=True)
            (asset / 'physics_test.py').write_text("raise ImportError('broken')\n")
            with patch('traceback.print_exc'):
                with self.assertRaisesRegex(SystemExit, 'bad/000000'):
                    main(['--source', temporary, '--assets', 'bad/000000', '--physics', 'isaac-cpu', '--plots-only'])
