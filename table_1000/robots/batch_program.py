"""Robot checks and their renders expose units to the shared batch runtime."""
from pathlib import Path
import time

from table_1000.common.batch_runtime import Unit, UnitResult


class Checks:
    @staticmethod
    def units(options):
        from scripts.robots.check_robot import COST, GPU_COST, unit_dir
        for robot, backend, case, trial in options['units']:
            path = unit_dir(Path(options['directories'][robot]) / backend, case, trial) / 'report.json'
            yield Unit((robot, backend, case, trial), dict(robot=robot, physics=backend, case=case, trial=trial),
                str(path), ('robot-check', robot, backend, case, str(trial)), ('robot-load', robot, backend),
                COST.get(case, 30) * (GPU_COST if backend == 'isaac-gpu' else 1),
                priority=-1 if case == 'parallel' else 0)

    @staticmethod
    def pack_key(unit):
        return unit.key[:2]

    @staticmethod
    def run_unit(context, unit):
        from . import load_robot_test, load_spec
        from .checks import Checker
        from .parallel import check_parallel
        from table_1000.sim.isaaclab.replay import write_episode
        robot, backend, case, trial = unit.key
        if context.loaded is None:
            setup = time.perf_counter()
            from table_1000.sim.isaaclab.app import start_app
            from table_1000.sim.isaaclab.render import RT2_ARGS, configure
            device = 'cpu' if backend == 'isaac-cpu' else 'cuda:0'
            app = start_app(gpu=context.gpu, workers=4, device=device, extra_args=RT2_ARGS)
            context.stack.callback(app.close)
            app_start_s = time.perf_counter() - setup
            setup = time.perf_counter()
            configure()
            context.loaded = dict(spec=load_spec(robot, context.root), settings=load_robot_test(robot, context.root),
                checker=None, device=device, app_start_s=app_start_s, setup_s=time.perf_counter() - setup)
            context.stack.callback(lambda: context.loaded['checker'].env.close() if context.loaded['checker'] else None)
        loaded = context.loaded
        directory = Path(unit.result).parent
        directory.mkdir(parents=True, exist_ok=True)
        started, cpu = time.perf_counter(), time.process_time()
        if case == 'parallel':
            result = check_parallel(loaded['spec'], loaded['settings'], Path(context.options['assets']), context.root)
        else:
            if loaded['checker'] is None:
                loaded['checker'] = Checker(loaded['spec'], loaded['settings'], context.work,
                    Path(context.options['assets']), context.root, device=loaded['device'])
                loaded['setup_s'] += time.perf_counter() - started
                started, cpu = time.perf_counter(), time.process_time()
            checker = loaded['checker']
            checker.output, checker.trials, checker.states = directory, (trial,), []
            result = checker.run(case)
            if checker.states and len(checker.states) > 1:
                write_episode(directory / 'trajectory.hdf5', checker.states,
                    dict(robot=robot, physics=backend, case=case, trial=-1 if trial is None else trial))
            checker.states = None
        result['wall_s'] = time.perf_counter() - started
        status = result.get('status', 'PASS' if case == 'metrics' else 'FAIL')
        return UnitResult(dict(result=result, status=status, log=str(context.work / 'run.log'),
                               wall_s=result['wall_s'], cpu_s=time.process_time() - cpu,
                               app_start_s=loaded['app_start_s'], scene_load_s=loaded['setup_s']))


class Renders(Checks):
    @staticmethod
    def units(options):
        for raw, length in zip(options['units'], options['lengths']):
            robot, backend, case, trial = raw
            yield Unit(tuple(raw), dict(robot=robot, physics=backend, case=case, trial=trial),
                f'reports/{robot}/{backend}/{case}-{trial}.json',
                ('robot-render', robot, backend, case, str(trial)), ('robot-render-load', robot, backend), length / 25)

    @staticmethod
    def run_unit(context, unit):
        from . import load_robot_test, load_spec
        from .checks import VIEWS, check_scene
        from table_1000.sim.isaaclab import render, replay
        from table_1000.sim.isaaclab.video import caption, encode, save_frame
        from table_1000.sim import CONTROL_HZ
        from scripts.robots.check_robot import unit_dir
        from scripts.robots.render_check import FIRST_FRAME_RENDERS
        from PIL import Image
        import shutil
        robot, backend, case, trial = unit.key
        options = context.options
        if context.loaded is None:
            setup = time.perf_counter()
            from table_1000.sim.isaaclab.app import start_app
            from table_1000.sim.isaaclab.render import RT2_ARGS
            app = start_app(gpu=context.gpu, workers=4, extra_args=RT2_ARGS)
            context.stack.callback(app.close)
            app_start_s = time.perf_counter() - setup
            setup = time.perf_counter()
            from table_1000.sim.isaaclab.env import RobotEnv
            render.configure()
            spec = load_spec(robot, context.root)
            env = RobotEnv(check_scene(spec, load_robot_test(robot, context.root)), spec, Path(options['assets']),
                cameras={n: spec.cameras[n] for n in options['views'] if n not in VIEWS},
                views={n: VIEWS[n] for n in options['views'] if n in VIEWS}, root=context.root)
            context.stack.callback(env.close)
            context.loaded = env, app_start_s, time.perf_counter() - setup
        env, app_start_s, setup_s = context.loaded
        wall, cpu = time.perf_counter(), time.process_time()
        directory = unit_dir(Path(options['directories'][robot]) / backend, case, trial)
        file, episodes = replay.episodes(directory / 'trajectory.hdf5')
        episode = next(iter(episodes.values()))
        frames = context.work / 'frames'
        for view in options['views']:
            (frames / view).mkdir(parents=True, exist_ok=True)
        title = case if trial is None else f'{case}  trial {trial}'
        for t in range(episode.length):
            replay.set_state(env, episode.state(t))
            for _ in range(FIRST_FRAME_RENDERS - 1 if t == 0 else 0):
                env.render([])
            rendered = env.render(options['views'])
            for view in options['views']:
                save_frame(caption(Image.fromarray(rendered[view]['rgb'][0]), title, f'{t / CONTROL_HZ:.2f} s'),
                    frames / view / f'{t:05d}.jpg')
        file.close()
        for view in options['views']:
            encode(frames / view, CONTROL_HZ, directory / f'{view}.mp4')
        shutil.rmtree(frames)
        return UnitResult(dict(status='success', wall_s=time.perf_counter() - wall,
                               cpu_s=time.process_time() - cpu, app_start_s=app_start_s, scene_load_s=setup_s))


checks = Checks()
renders = Renders()
