"""Render asset trajectories with one reusable evaluator per loaded asset."""
from pathlib import Path
import shutil
import time

from table_1000.common.batch_runtime import Unit, UnitResult


class Program:
    @staticmethod
    def units(options):
        for asset, seed, source, target in options['units']:
            yield Unit((asset, seed), dict(asset=asset, seed=seed, source=source, target=target),
                f'renders/{asset}/{seed}.json', ('asset-render', asset, options['robot'], options['physics'], str(seed)),
                ('asset-render-loads', asset, options['robot'], options['physics']))

    @staticmethod
    def pack_key(unit):
        return unit.data['asset']

    @staticmethod
    def run_unit(context, unit):
        from . import Evaluator
        from .recording import VIEWS, render
        if context.loaded is None:
            setup = time.perf_counter()
            from table_1000.sim.isaaclab.app import start_app
            from table_1000.sim.isaaclab.render import RT2_ARGS, configure
            device = 'cpu' if context.options['physics'] == 'isaac-cpu' else 'cuda:0'
            app = start_app(gpu=context.gpu, workers=4, device=device, cameras=True, extra_args=RT2_ARGS)
            context.stack.callback(app.close)
            app_start_s = time.perf_counter() - setup
            setup = time.perf_counter()
            configure()
            evaluator = Evaluator(unit.data['asset'], context.options['robot'], Path(context.options['assets']),
                context.work, context.root, device, VIEWS)
            context.stack.callback(evaluator.close)
            context.loaded = evaluator, app_start_s, time.perf_counter() - setup
        evaluator, app_start_s, setup_s = context.loaded
        wall, cpu = time.perf_counter(), time.process_time()
        frames = context.work / f'frames-{unit.data["seed"]}'
        render(evaluator, Path(unit.data['source']), Path(unit.data['target']), frames)
        shutil.rmtree(frames)
        return UnitResult(dict(status='success', app_start_s=app_start_s, scene_load_s=setup_s,
            wall_s=time.perf_counter() - wall, cpu_s=time.process_time() - cpu))


program = Program()
