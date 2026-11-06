"""Physics tests share one simulator process for one asset/backend partition."""
import time
import subprocess
from pathlib import Path

from table_1000.common.batch_runtime import Unit, UnitResult


class Program:
    @staticmethod
    def units(options):
        for reference, backend, name in options['units']:
            result = Path(options['source']) / reference / 'physics_test' / backend / name / 'result.json'
            yield Unit((reference, backend, name), dict(reference=reference, physics=backend, case=name),
                str(result), ('physics', reference, backend, name),
                ('physics-load', reference, backend), 320. if name == 'metrics' else 120.)

    @staticmethod
    def pack_key(unit):
        return unit.key[:2]

    @staticmethod
    def run_unit(context, unit):
        from threadpoolctl import threadpool_limits
        from table_1000.physics.test import BACKENDS, load_tests, run_case, entry
        from table_1000.physics.utilization import Utilization
        reference, backend, name = unit.key
        options = context.options
        if context.loaded is None:
            from table_1000.sim.isaaclab.app import start_app
            from table_1000.sim.isaaclab.render import RT2_ARGS
            started = time.perf_counter()
            gpu_info = dict(physical_index=context.gpu, before_process_start=subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,driver_version,memory.used,utilization.gpu', '--format=csv'],
                check=True, capture_output=True, text=True).stdout.strip())
            app = start_app(gpu=context.gpu, device='cpu' if BACKENDS[backend] == 'cpu' else 'cuda:0',
                            extra_args=(*RT2_ARGS, '--/rtx/post/motionblur/enabled=False'))
            context.stack.callback(app.close)
            startup = time.perf_counter() - started
            import carb
            from table_1000.physics.simulation import enable_scripting
            enable_scripting()
            for _ in range(5):
                app.update()
            carb.settings.get_settings().set('/persistent/physics/numThreads', 4)
            carb.settings.get_settings().set('/rtx/post/motionblur/enabled', False)
            utilization = Utilization(context.gpu)
            context.stack.callback(utilization.close)
            limits = threadpool_limits(limits=1, user_api='blas')
            limits.__enter__()
            context.stack.callback(limits.__exit__, None, None, None)
            scripts = context.root / options['scripts_relative'] if options['scripts_relative'] else Path(options['scripts'])
            tests = load_tests(scripts / reference / 'physics_test.py')
            context.loaded = dict(app=app, startup=startup, gpu_info=gpu_info, utilization=utilization, tests=tests)
        loaded = context.loaded
        asset = Path(options['source']) / reference
        started, cpu = time.perf_counter(), time.process_time()
        result = run_case(entry(asset), name, loaded['tests'][name], asset / 'physics_test' / backend,
            loaded['app'], loaded['startup'], loaded['gpu_info'], physics=BACKENDS[backend],
            num_envs=options['num_envs'] or (4 if backend == 'isaac-gpu' else 1),
            utilization=loaded['utilization'], standard=options['benchmark'])
        result['wall_s'] = time.perf_counter() - started
        result['cpu_s'] = time.process_time() - cpu
        result['app_start_s'] = loaded['startup']
        return UnitResult(result | dict(status='success' if result['execution_status'] == 'completed' else 'failed'))


program = Program()
