"""Trajectory renders reuse one loaded scene inside each shared batch worker."""
from pathlib import Path
import shutil
import time
from table_1000.common.batch_runtime import Unit, UnitResult


class Program:
    @staticmethod
    def units(options):
        for seed, length in options['units']:
            yield Unit((seed,), dict(seed=seed), f'reports/{seed}.json',
                ('scene-render', options['scene'], options['robot'], options['batch_key'], str(seed)),
                ('scene-render-load', options['scene'], options['robot']), length / 25)

    @staticmethod
    def pack_key(unit):
        return None

    @staticmethod
    def run_unit(context, unit):
        from PIL import Image
        from scripts.scenes.render import FIRST_FRAME_RENDERS, camera_sets
        from table_1000.scenes.recordings import MERGED, demo_name, video_path
        from table_1000.sim import CONTROL_HZ
        from table_1000.sim.isaaclab import replay
        from table_1000.sim.isaaclab.video import caption, encode, save_frame, tile
        options = context.options
        if context.loaded is None:
            setup = time.perf_counter()
            from table_1000.sim.isaaclab.app import start_app
            from table_1000.sim.isaaclab.render import RT2_ARGS
            app = start_app(gpu=context.gpu, workers=4, extra_args=RT2_ARGS)
            context.stack.callback(app.close)
            app_start_s = time.perf_counter() - setup
            setup = time.perf_counter()
            from table_1000.robots import load_spec
            from table_1000.scenes import load
            from table_1000.sim.isaaclab import render
            from table_1000.sim.isaaclab.env import RobotEnv
            render.configure()
            directory, work = Path(options['batch']), context.work
            file, episodes = replay.episodes(directory / 'trajectories.hdf5')
            context.stack.callback(file.close)
            meta = replay.env_args(file)
            failures = {}
            if (directory / 'trajectories_failed.hdf5').is_file():
                failed_file, failures = replay.episodes(directory / 'trajectories_failed.hdf5')
                context.stack.callback(failed_file.close)
                episodes.update(failures)
            scene = load(meta['scene'], 'scene', context.root)
            spec = load_spec(meta['robot'], context.root)
            cameras, views = camera_sets(scene, spec, options['cameras'])
            env = RobotEnv(scene, spec, Path(options['assets']).resolve(), cameras=cameras, views=views, root=context.root)

            context.stack.callback(env.close)
            context.loaded = env, episodes, failures, app_start_s, time.perf_counter() - setup
        env, episodes, failures, app_start_s, setup_s = context.loaded
        wall, cpu = time.perf_counter(), time.process_time()
        directory, work = Path(options['batch']), context.work
        outputs = [MERGED] if options['merge'] else list(options['cameras'])
        cell_height = None
        seed = unit.data['seed']
        started = time.perf_counter()
        name = demo_name(seed)
        episode = episodes[name]
        failure = episode.attrs.get('failure')
        note = f'    FAILED {failure["subtask"]}: {failure["reason"]}' if failure else ''
        frames = work / name
        shutil.rmtree(frames, ignore_errors=True)
        for output in outputs:
            (frames / output).mkdir(parents=True)
        for t in range(episode.length):
            replay.set_state(env, episode.state(t))
            for _ in range(FIRST_FRAME_RENDERS - 1 if t == 0 else 0):
                env.render([])
            rendered = env.render(options['cameras'])
            images = {camera: Image.fromarray(rendered[camera]['rgb'][0]) for camera in options['cameras']}
            cell_height = cell_height or min(image.height for image in images.values())
            left, right = f'{episode.subtask(t) or ""}{note}', f'{t / CONTROL_HZ:.2f} s'
            for output in outputs:
                image = tile([images[c] for c in options['cameras']], cell_height) if output == MERGED else images[output]
                save_frame(caption(image, left, right) if options['captions'] else image, frames / output / f'{t:05d}.jpg')
        for output in outputs:
            target = video_path(directory, seed, None if output == MERGED else output, failed=name in failures)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(encode(frames / output, CONTROL_HZ, frames / f'{output}.mp4'), target)
        shutil.rmtree(frames)
        print(name, episode.length, f'{time.perf_counter() - started:.0f} s', flush=True)
        return UnitResult(dict(status='success', wall_s=time.perf_counter() - wall,
                           cpu_s=time.process_time() - cpu, app_start_s=app_start_s, scene_load_s=setup_s))


program = Program()
