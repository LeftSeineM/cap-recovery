"""Sample scene entries one seed at a time in the shared batch worker."""
from pathlib import Path
import time

from table_1000.common.batch_runtime import Unit, UnitResult


class Program:
    @staticmethod
    def units(options):
        for layout, seed in options['units']:
            yield Unit((layout, seed), dict(layout=layout, seed=seed), f'records/{layout}/{seed}.json',
                ('scene-build', options['scene'], options['robot'], str(layout), str(seed)),
                ('scene-build-load', options['scene'], options['robot']))

    @staticmethod
    def pack_key(unit):
        return unit.data['layout']

    @staticmethod
    def run_unit(context, unit):
        from .initial import InitializationFailed, initial_state, write_states
        from scripts.scenes.build_scene import main_camera, to_numpy
        options = context.options
        if context.loaded is None:
            setup = time.perf_counter()
            from table_1000.sim.isaaclab.app import start_app
            from table_1000.sim.isaaclab.render import RT2_ARGS
            render = options['images'] or options['first'] > 0
            app = start_app(gpu=context.gpu, workers=4, cameras=render, extra_args=RT2_ARGS if render else ())
            context.stack.callback(app.close)
            app_start_s = time.perf_counter() - setup
            setup = time.perf_counter()
            from . import load, robot_config
            from table_1000.robots import load_spec
            from table_1000.sim.isaaclab.env import RobotEnv
            scene = load(options['scene'], root=context.root)
            spec = load_spec(Path(robot_config(scene, options['robot'])['asset']).name, context.root)
            kwargs, camera = {}, None
            if render:
                from table_1000.sim.isaaclab.render import configure
                configure()
                camera = options['camera'] or getattr(scene, 'IMAGE_VIEW', None) or main_camera(spec)
                kwargs['views' if camera in scene.VIEWS else 'cameras'] = {
                    camera: scene.VIEWS[camera] if camera in scene.VIEWS else spec.cameras[camera]}
            env = RobotEnv(scene, spec, Path(options['assets']), root=context.root, **kwargs)
            context.stack.callback(env.close)
            context.loaded = scene, spec, env, camera, [0], app_start_s, time.perf_counter() - setup
        scene, spec, env, camera, pictured, app_start_s, setup_s = context.loaded
        wall, cpu = time.perf_counter(), time.process_time()
        layout, seed = unit.key
        work = context.work / str(layout)
        for name in ('states', 'images'):
            (work / name).mkdir(parents=True, exist_ok=True)
        arrange = scene.LAYOUTS[layout]['arrange'] if layout else None
        try:
            _, record = initial_state(env, scene, Path(options['assets']), seed, initialize=arrange)
            write_states(work / 'states' / f'{seed}.hdf5',
                {seed: (record['attempt'], to_numpy(env.env.scene.get_state(is_relative=True)))}, {})
            if options['images'] or pictured[0] < options['first']:
                from PIL import Image
                env.render([camera])
                Image.fromarray(env.render([camera])[camera]['rgb'][0]).save(work / 'images' / f'{seed}.jpg', quality=90)
                pictured[0] += 1
            status = 'success'
        except InitializationFailed as error:
            record = dict(seed=seed, attempt=None, attempts=error.attempts)
            status = 'failed'
        return UnitResult(dict(status=status, robot=spec.name, camera=camera, record=record, work=str(work),
            wall_s=time.perf_counter() - wall, cpu_s=time.process_time() - cpu, app_start_s=app_start_s, scene_load_s=setup_s))


program = Program()
