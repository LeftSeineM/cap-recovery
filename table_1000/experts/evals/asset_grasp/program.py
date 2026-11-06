"""Asset units and per-process simulation reuse; reset remains part of Evaluator.trial."""
from pathlib import Path
import shutil
import time

from table_1000.common.batch import Load
from table_1000.common.batch_runtime import Unit, UnitResult
from table_1000.common.diagnostics import dump_stacks_every


class Program:
    @staticmethod
    def units(options):
        for asset, seed in options['units']:
            robot, backend = options['robot'], options['physics']
            yield Unit((asset, seed), dict(asset=asset, seed=seed, robot=robot, physics=backend, seeds_per_worker=options['seeds_per_worker']),
                f'{asset}/trials/{seed:04d}.json', ('asset', asset, robot, backend, str(seed)),
                ('asset', asset, robot, backend))

    @staticmethod
    def pack_key(unit):
        return (unit.data['asset'], unit.data['seed'] // unit.data['seeds_per_worker']) if unit.data['seeds_per_worker'] else (unit.data['robot'], unit.data['physics'])

    @staticmethod
    def run_unit(context, unit):
        from . import Evaluator
        from .recording import Recorder, VIEWS
        options = context.options
        if context.loaded is None:
            import psutil
            from table_1000.sim.isaaclab.app import start_app
            from table_1000.sim.isaaclab.render import RT2_ARGS, configure
            device = 'cpu' if options['physics'] == 'isaac-cpu' else 'cuda:0'
            app = start_app(gpu=context.gpu, workers=4, device=device, cameras=options['video'],
                            extra_args=RT2_ARGS if options['video'] else ())
            context.stack.callback(app.close)
            if options['video']:
                configure()
            context.loaded = dict(asset=None, evaluator=None, device=device,
                app_start_s=time.time() - psutil.Process().create_time(), app_cpu_s=time.process_time())
            context.stack.callback(lambda: context.loaded['evaluator'].close() if context.loaded['evaluator'] is not None else None)
        loaded = context.loaded
        asset, seed = unit.key
        if loaded['asset'] != asset:
            if loaded['evaluator'] is not None:
                loaded['evaluator'].close()
                loaded['evaluator'] = None
            wall, cpu = time.perf_counter(), time.process_time()
            loaded['evaluator'] = Evaluator(asset, options['robot'], Path(options['assets']), context.work,
                context.root, loaded['device'], VIEWS if options['video'] else {}, diagnostic_shake=options['diagnostic_shake'],
                collect_preferences=options['collect_preferences'])
            loaded.update(asset=asset, asset_load_s=time.perf_counter() - wall, asset_load_cpu_s=time.process_time() - cpu)
        evaluator = loaded['evaluator']
        dump_stacks_every()
        result = context.output / asset
        frames = context.work / f'frames-{seed:04d}' if options['video'] else None
        recorder = Recorder(evaluator.env, frames)
        with Load() as load:
            wall, cpu = time.perf_counter(), time.process_time()
            report = evaluator.trial(seed, options['center'], options['region_size'], recorder)
            wall_s, cpu_s = time.perf_counter() - wall, time.process_time() - cpu
        wall, cpu = time.perf_counter(), time.process_time()
        report.update(load=load.record(), robot=options['robot'], physics=options['physics'])
        trajectory = result / 'trajectories' / f'{seed:04d}.hdf5'
        recorder.save(trajectory, dict(asset=asset, seed=seed, robot=options['robot'], physics=options['physics']))
        report['trajectory'] = str(trajectory.relative_to(result))
        if options['video']:
            from table_1000.sim.isaaclab.video import encode
            video = result / 'videos' / f'{seed:04d}.mp4'
            video.parent.mkdir(parents=True, exist_ok=True)
            encode(frames, 25, video)
            report['video'] = str(video.relative_to(result))
            shutil.rmtree(frames)
        timing = dict(wall_s=wall_s, cpu_s=cpu_s, save_wall_s=time.perf_counter() - wall,
            save_cpu_s=time.process_time() - cpu,
            **{name: loaded[name] for name in ('asset_load_s', 'asset_load_cpu_s', 'app_start_s', 'app_cpu_s')})
        return UnitResult(report | timing)


program = Program()
