"""Standalone pick scenarios reuse one evaluator per robot/scenario."""
from argparse import Namespace
from pathlib import Path
import time

from table_1000.common.batch_runtime import Unit, UnitResult


class Program:
    @staticmethod
    def units(options):
        from scripts.experts.eval_pick import group_dir, trial_path
        for robot, scenario, seed, index in options['units']:
            path = trial_path(group_dir(Path('.'), robot, scenario, seed, bool(options['case'])), index)
            yield Unit((robot, scenario, seed, index), dict(robot=robot, scenario=scenario, seed=seed, index=index),
                str(path), ('pick', robot, scenario, str(index), str(seed)), ('pick-load', robot, scenario))

    @staticmethod
    def pack_key(unit):
        return unit.key[:2]

    @staticmethod
    def run_unit(context, unit):
        from scripts.experts.eval_pick import Recorder, WIDE, CLOSE, aim_close_view, group_dir, tasks
        from .pick_eval import Evaluator, scene
        options = context.options
        robot, scenario, seed, index = unit.key
        if context.loaded is None:
            setup = time.perf_counter()
            from table_1000.sim.isaaclab.app import start_app
            app = start_app(gpu=context.gpu, workers=4, cameras=options['video'])
            context.stack.callback(app.close)
            app_start_s = time.perf_counter() - setup
            setup = time.perf_counter()
            views = {'wide': WIDE, 'close': CLOSE} if options['video'] else {}
            evaluator = Evaluator(scene(options['gap'], options['lift'], robot=robot, scenario=scenario),
                Path(options['assets']), context.work, context.root, views, options['random_pushes'], profile=options['profile'])
            context.stack.callback(evaluator.close)
            context.loaded = evaluator, views, app_start_s, time.perf_counter() - setup
        evaluator, views, app_start_s, setup_s = context.loaded
        wall, cpu = time.perf_counter(), time.process_time()
        _, name, placement = next(t for t in tasks(Namespace(**options), scenario, seed) if t[0] == index)
        directory = group_dir(context.output, robot, scenario, seed, bool(options['case']))
        video = Recorder(evaluator.env, views, context.work / 'frames') if options['video'] else None
        if video:
            aim_close_view(evaluator.env, placement)
        report = evaluator.trial(name, placement, seed * 100000 + index, video)
        if video:
            from table_1000.sim.isaaclab.video import encode
            (directory / 'videos').mkdir(parents=True, exist_ok=True)
            encode(context.work / 'frames', 25, directory / 'videos' / f'{index:03d}_{name}_{report["status"]}.mp4')
        return UnitResult(report | dict(wall_s=time.perf_counter() - wall,
            cpu_s=time.process_time() - cpu, app_start_s=app_start_s, scene_load_s=setup_s))


program = Program()
