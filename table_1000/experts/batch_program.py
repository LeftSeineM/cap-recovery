"""Expert episodes use the common batch worker; Generator owns resetting each entry."""
from pathlib import Path
import time

from table_1000.common.batch_runtime import Unit, UnitResult


class Program:
    @staticmethod
    def units(options):
        for identity, seed in options['units']:
            identity = tuple(identity)
            yield Unit((identity, seed), dict(scene=identity[0], robot=identity[1], expert=identity[2],
                layout=options['layout'], steps_run=options['steps'], seed=seed, physics=options['physics']),
                f'episodes/{seed}.json', ('expert', *identity, options['physics'],
                    f"repeats-{options['repeats']}", str(seed)),
                ('expert-loads', identity[0], identity[1], options['physics']))

    @staticmethod
    def pack_key(unit):
        return unit.data['scene'], unit.data['robot']

    @staticmethod
    def run_unit(context, unit):
        options = context.options
        if context.loaded is None:
            import psutil
            from table_1000.sim.isaaclab.app import start_app
            app = start_app(gpu=context.gpu, workers=4, cameras=False)
            context.stack.callback(app.close)
            app_start_s = time.time() - psutil.Process().create_time()
            from table_1000.experts.generate import Generator, read_checkpoints
            from table_1000.scenes.initial import read_states
            source = Path(options['states'])
            if options['checkpoints']:
                attrs, states = read_checkpoints(source)
                if attrs['step'] == 0:
                    states = {seed: (attempt, state, start if '_expert_frame' in state else None)
                              for seed, (attempt, state, start) in states.items()}
            else:
                states = {seed: (attempt, state, None) for seed, (attempt, state) in read_states(source)[1].items()}
            generator = Generator(options['scene'], context.work, Path(options['assets']), context.root,
                physics=options['physics'], steps=options['steps'], source=str(source) if options['checkpoints'] else None,
                problems=options['problems'], skill_attempts=options['skill_attempts'], profile=options['profile'],
                robot=options['robot'], expert=options['expert'], layout=options['layout'],
                outside_scope=options['outside_scope'], record_root=context.output)
            context.stack.callback(generator.close)
            context.loaded = generator, states, app_start_s
        generator, states, app_start_s = context.loaded
        seed = unit.data['seed']
        timing = dict(wall_s=0., cpu_s=0.)
        for repeat in range(options['repeats']):
            report = generator.run(seed, *states[seed], repeat=repeat)
            for name in timing:
                timing[name] += report[name]
            if not repeat:
                first = report
        first.update(physics=options['physics'], app_start_s=app_start_s, scene_load_s=generator.load_s)
        return UnitResult(first, timing=timing)


program = Program()
