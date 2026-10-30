"""Generate a numbered robot expert's trajectories from stored scene initial states.

``run_expert.py SCENE --physics isaac-cpu [--robot NAME] [--expert NNNN]``
``run_expert.py SCENE --physics isaac-cpu --dev [--seeds 0:20] [--steps k[-m] --from BATCH]``

Defaults to the scene's ROBOT and that robot's smallest expert number. Without
--layout, run every layout declared by SCOPE; without --seeds, run its declared
seeds that have stored states (None: all stored valid states). Formal data from
initial states requires every explicitly declared seed to be present. Initial states are under
assets/scenes/<scene>/<robot>/, with layouts/<n>/ for nonzero layouts. Requests
outside SCOPE require --outside-scope --dev and are marked in every report.

Formal data replaces data/scenes/<scene>/<robot>/<expert>/<backend>/; a nonzero
layout uses its layouts/<n>/ subdirectory. --dev keeps each batch under
runs/expert/<scene>/<date>-<time>-<commit>-<robot>-<expert>-layout<n>[-<name>]/.
Partial seeds/steps, repeats, named runs and --from require --dev.

--steps k[-m] runs only those steps; k>1 requires --from BATCH with step k-1's
checkpoint. A source with one numbered checkpoint infers the next single step.
Summary and checkpoint robot/expert identity, backend and completed step name
must match.

Each batch holds trajectory HDF5 files, episodes/<seed>.json, calls/, problems/,
checkpoints/step<k>.hdf5 and summary.json (scope, identity, status, timings and
machine load). --repeats K adds reports <seed>-<repeat>.json with independent
random streams after the first run, without extra trajectories/checkpoints.
Jobs freeze the submitted source and run through --gpus or --launcher, retrying
seeds without an episode; logs and partial products are kept in the batch's
parts/ (--dev) or outputs/scenes/<scene>/<robot>/<expert>/<backend>/<layout>/.
Episode failures are results; a nonzero exit means some seed has no episode.
"""

import argparse
import itertools
import json
from pathlib import Path
import shlex
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from table_1000.common.diagnostics import dump_stacks_every

from table_1000.common.batch import run_name

# Simulation backends, as for asset physics tests (only CPU PhysX is validated for scenes yet).
BACKENDS = ('isaac-cpu',)


def step_range(text):
    first, _, last = text.partition('-')
    first, last = int(first), int(last or first)
    if not 1 <= first <= last:
        raise argparse.ArgumentTypeError(f'steps {text}: k[-m] with 1 <= k <= m')
    return first, last


def resume_steps(source, steps):
    """Infer just the next step from a sole numbered checkpoint, or keep an explicit range."""
    if steps is not None:
        return steps
    paths = [path for path in (source / 'checkpoints').glob('step*.hdf5')
             if path.stem[4:].isdigit()]
    if len(paths) != 1:
        raise ValueError(f'--from {source}: give --steps; found {len(paths)} numbered checkpoints, not one')
    first = int(paths[0].stem[4:]) + 1
    return first, first


def check_checkpoint_name(attrs, expert_steps):
    """Reject a checkpoint whose completed step was moved or renamed in this expert."""
    from table_1000.experts.generate import INITIAL_STATE
    index, saved = int(attrs['step']), attrs['name']
    expected = INITIAL_STATE if index == 0 else (expert_steps[index - 1].name
                                                if index <= len(expert_steps) else None)
    if saved != expected:
        matches = [str(k) for k, step in enumerate(expert_steps, 1) if step.name == saved]
        hint = f'current steps with that name: {", ".join(matches)}' if matches else 'no current step has that name'
        raise ValueError(f'checkpoint step {index} name {saved!r} does not match current name {expected!r}; {hint}')


def worker(args):
    """One part: the given seeds' episodes in this process, into ``args.work``."""
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    from table_1000.common.batch import worker_gpu
    from table_1000.sim.isaaclab.app import start_app
    app = start_app(gpu=worker_gpu(args.gpu), workers=4, cameras=False)
    failed = False
    generator = None
    try:
        from table_1000.experts.generate import Generator, read_checkpoints
        from table_1000.common.batch import parse_seeds
        from table_1000.scenes.initial import read_states
        if args.checkpoints:
            attrs, states = read_checkpoints(args.states)
            if attrs['step'] == 0:      # the initial states: run as from them
                states = {seed: (attempt, state, start if '_expert_frame' in state else None)
                          for seed, (attempt, state, start) in states.items()}
        else:
            states = {seed: (attempt, state, None) for seed, (attempt, state) in read_states(args.states)[1].items()}
        generator = Generator(args.scene, work, args.assets.resolve(), args.root.resolve(), physics=args.physics,
                              steps=args.steps, source=str(args.states) if args.checkpoints else None,
                              problems=args.problems, skill_attempts=args.skill_attempts, profile=args.profile, robot=args.robot, expert=args.expert, layout=args.layout,
                              outside_scope=args.outside_scope)
        for seed, repeat in itertools.product(parse_seeds(args.seeds), range(args.repeats)):
            report = generator.run(seed, *states[seed], repeat=repeat)
            print(seed, *([f'repeat {repeat}'] if repeat else []), report['status'],
                  report.get('failure', {}).get('reason', ''), f'{report["wall_s"]:.0f} s',
                  flush=True)
    except Exception:
        traceback.print_exc()
        failed = True
    if generator is not None:
        generator.close()
    app.close(exit_code=int(failed))


def main(argv=None):
    dump_stacks_every()
    from table_1000.common.batch import add_arguments, commit, job_count, load_record, parse_seeds, process_name, run_batch, seed_range
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scene')
    parser.add_argument('--robot', help='default: scene.ROBOT')
    parser.add_argument('--expert', help='four-digit expert number; default: smallest available')
    parser.add_argument('--layout', type=int, help='one layout from SCOPE; default: all declared layouts')
    parser.add_argument('--outside-scope', action='store_true', help='allow seeds/layouts outside SCOPE (--dev only)')
    parser.add_argument('--physics', choices=BACKENDS, required=True, help='simulation backend')
    add_arguments(parser)
    parser.add_argument('--input', type=Path, default=ROOT / 'assets' / 'scenes',
                        help='initial states from <input>/<scene>/<robot>/initial_states.hdf5')
    parser.add_argument('--output', type=Path,
                        help='the batch goes to <output>/<scene>/<robot>/<expert>/<backend>/ (replaced; default data/scenes), with '
                             '--dev to <output>/<scene>/<run>/ (default runs/expert)')
    parser.add_argument('--profile', action='store_true', help='record nested wall/CPU times and available instructions')
    parser.add_argument('--dev', action='store_true', help='a development run, kept apart from the data')
    parser.add_argument('--name', help="a label at the end of a development run's name")
    parser.add_argument('--root', type=Path, default=ROOT, help='repository root holding the scene and robot sources')
    parser.add_argument('--assets', type=Path, default=ROOT / 'assets')
    parser.add_argument('--steps', type=step_range, help="k[-m]: run steps k to m of the expert (see run_steps)")
    parser.add_argument('--from', dest='source', type=Path,
                        help='a batch directory whose checkpoints of step k - 1 the steps start from (step 0: its '
                             'initial states)')
    parser.add_argument('--repeats', type=int, default=1,
                        help='runs of each seed from its state, with other random streams after the first '
                             '(a development run)')
    parser.add_argument('--problems', choices=('failed', 'all', 'none'), default='failed',
                        help='save numerical solve problems for failed/error runs (default), all runs, or none; '
                             'none also disables recording and extra IK diagnostics')
    parser.add_argument('--skill-attempts', type=int, default=3,
                        help='total tries per retry callback, restoring entry before later random tries (default 3)')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--seeds', nargs='+', help='only these seeds of the initial states or checkpoints: 12 27, 0:20')
    parser.add_argument('--states', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--checkpoints', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--work', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args)
        return
    import h5py
    from table_1000.experts.generate import checkpoint_path, merge_checkpoints, repeat_rates, step_rates, summarize
    from table_1000.scenes.recordings import clear_batch, merge
    jobs = job_count(parser, args)
    root = args.root.resolve()
    from table_1000.scenes import load_expert, initial_dir, data_dir, outside_scope
    try:
        selected = load_expert(args.scene, args.robot, args.expert, root)
    except ValueError as error:
        parser.error(str(error))
    args.robot, args.expert = selected.ROBOT, selected.NUMBER
    if args.outside_scope and not args.dev:
        parser.error('--outside-scope requires --dev')
    if args.layout is None:
        if args.source:
            args.layout = json.loads((args.source / 'summary.json').read_text())['layout']
        elif len(selected.SCOPE) == 1:
            args.layout = next(iter(selected.SCOPE))
        else:
            original = list(sys.argv[1:] if argv is None else argv)
            for layout in selected.SCOPE:
                main([*original, '--layout', str(layout)])
            return
    if args.source:
        try:
            args.steps = resume_steps(args.source, args.steps)
        except ValueError as error:
            parser.error(str(error))
    first = args.steps[0] if args.steps else 1
    if args.skill_attempts < 1:
        parser.error('--skill-attempts: at least 1')
    if args.repeats < 1:
        parser.error('--repeats: at least 1')
    if not args.dev and (args.steps or args.source or args.seeds or args.name or args.repeats > 1):
        parser.error('--steps, --from, --seeds, --name and --repeats make a development run, not the scene\'s data: '
                     'give --dev')
    if args.layout not in selected.SCOPE and not args.outside_scope:
        parser.error(f'layout {args.layout} outside SCOPE {selected.SCOPE}; use --outside-scope --dev')
    if args.steps:
        from table_1000.scenes import load
        expert_steps = getattr(selected, 'STEPS', None)
        if expert_steps is None:
            parser.error(f'--steps: the expert of {args.scene} has no STEPS (solve = run_steps(ctx, STEPS))')
        if args.steps[1] > len(expert_steps):
            parser.error(f'--steps {args.steps[0]}-{args.steps[1]}: the expert has {len(expert_steps)} steps')
    parent = None
    if args.source:
        states = checkpoint_path(args.source, first - 1).resolve()
        if not states.is_file():
            parser.error(f'no checkpoints of step {first - 1} in {args.source}')
        parent = json.loads((args.source / 'summary.json').read_text())
    else:
        if first > 1:
            parser.error(f'steps {first}-: give --from <batch directory> with checkpoints of step {first - 1}')
        states = (initial_dir(args.input, args.scene, args.robot, args.layout) / 'initial_states.hdf5').resolve()
        if not states.is_file():
            parser.error(f'no initial states {states}; run build_scene.py first')
    with h5py.File(states, 'r') as file:
        attrs = dict(file.attrs)
        seeds = sorted(int(name) for name in file)
    robot = args.robot
    if args.source:
        from table_1000.experts.identity import check_identity
        try:
            check_identity(parent, robot, args.expert, args.source / 'summary.json')
            check_identity(attrs, robot, args.expert, states)
            check_checkpoint_name(attrs, expert_steps)
        except ValueError as error:
            parser.error(str(error))
    if attrs.get('scene') != args.scene or attrs.get('robot') != robot:
        parser.error(f'{states} holds scene {attrs.get("scene")} with robot {attrs.get("robot")}, the scene has '
                     f'{robot}; run build_scene.py again')
    if args.source and attrs.get('physics') != args.physics:
        parser.error(f'{states} was made on {attrs.get("physics")}, not {args.physics}')
    if not args.seeds and args.layout in selected.SCOPE and selected.SCOPE[args.layout] is not None:
        declared = set(selected.SCOPE[args.layout])
        if not args.dev and not args.source and (absent := sorted(declared - set(seeds))):
            parser.error(f'{states} holds no states of SCOPE seeds {seed_range(absent)}')
        seeds = sorted(declared.intersection(seeds))
        if not seeds:
            parser.error(f'{states} holds no states within SCOPE for layout {args.layout}')
    if args.seeds:
        chosen = parse_seeds(args.seeds)
        if absent := sorted(set(chosen) - set(seeds)):
            parser.error(f'{states} holds no states of seeds {seed_range(absent)}')
        seeds = chosen
    if not seeds:
        parser.error(f'{states} holds no initial states')
    if outside_scope(selected.SCOPE, args.layout, seeds) and not args.outside_scope:
        parser.error(f'layout {args.layout}, seeds {seeds} outside SCOPE {selected.SCOPE}; use --outside-scope --dev')
    checkout = commit(root)
    if args.dev:
        run = run_name(checkout, '-'.join([robot, args.expert, f'layout{args.layout}', *([args.name] if args.name else [])]))
        directory = ((args.output or root / 'runs' / 'expert') / args.scene / run).resolve()
        try:        # claims the name: another run started in the same second gets an error
            directory.mkdir(parents=True)
        except FileExistsError:
            parser.error(f'{directory} exists; give another --name')
    else:
        run = None
        directory = data_dir(args.output or root / 'data' / 'scenes', args.scene, robot, args.expert, args.physics, args.layout).resolve()
    staging = directory / 'parts' if args.dev else root / 'outputs' / 'scenes' / args.scene / robot / args.expert / args.physics / str(args.layout)
    clear_batch(directory, keep=('layouts',) if not args.dev and args.layout == 0 else ())
    clear_batch(staging)
    works = {}

    def command(part, index, attempt):
        name = process_name(f'part{index}', attempt)
        works.update(dict.fromkeys(part, staging / name))
        argv = [sys.executable, str(Path(__file__).resolve()), args.scene, '--worker', '--physics', args.physics,
                '--robot', robot, '--expert', args.expert, '--layout', str(args.layout),
                '--seeds', *seed_range(part).split(), '--states', str(states), '--work', str(staging / name),
                '--root', str(root), '--assets', str(args.assets.resolve()), '--repeats', str(args.repeats),
                '--problems', args.problems, '--skill-attempts', str(args.skill_attempts),
                *(['--steps', '-'.join(map(str, args.steps))] if args.steps else []),
                *(['--checkpoints'] if args.source else []), *(['--profile'] if args.profile else []),
                *(['--outside-scope'] if args.outside_scope else [])]
        return argv, staging / f'{name}.log', f'{args.scene} {args.physics} {name} seeds {seed_range(part)}'

    started = time.perf_counter()
    missing = run_batch(seeds, command=command, count=jobs, gpus=args.gpus, launcher=args.launcher, retries=args.retries,
                        root=root,
                        # A seed's first run is its episode (trajectory, checkpoints); a process stopped in
                        # its later runs leaves it fewer of them (see repeat_rates), not to run again.
                        missing=lambda seed: not (works[seed] / 'episodes' / f'{seed}.json').is_file())
    checkout = missing.commit
    parts = sorted(staging.glob('part*/'))
    (directory / 'episodes').mkdir(parents=True)
    for name in ('trajectories', 'trajectories_failed'):
        # A stopped process may have recorded an episode without its report, which its rerun records again.
        merge([part / f'{name}.hdf5' for part in parts], directory / f'{name}.hdf5',
              keep=lambda source, seed: (source.parent / 'episodes' / f'{seed}.json').is_file())
        for part in parts:
            (part / f'{name}.hdf5').unlink(missing_ok=True)
    # A stopped process may have left a checkpoint without the episode's report, run again elsewhere.
    sources = [path for path in staging.glob('part*/checkpoints/*.hdf5')
               if (path.parent.parent / 'episodes' / f'{path.stem}.json').is_file()]
    merge_checkpoints(sources, directory, {'scene': args.scene, 'robot': robot, 'expert': args.expert, 'layout': args.layout, 'physics': args.physics,
                                           'commit': checkout or 'nocommit'})
    for path in staging.glob('part*/checkpoints/*.hdf5'):
        path.unlink()
    for path in sorted(staging.glob('part*/attempts/*')):
        target = directory / 'attempts' / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.move(path, target)
    runs = []
    for kind in ('calls', 'problems'):
        (directory / kind).mkdir(exist_ok=True)
        for path in sorted(staging.glob(f'part*/{kind}/*.json.gz')):
            if (path.parent.parent / 'episodes' / f'{path.name.removesuffix(".json.gz")}.json').is_file():
                shutil.move(path, directory / kind / path.name)
    for path in sorted(staging.glob('part*/episodes/*.json')):
        runs.append(json.loads(path.read_text()))
        shutil.move(path, directory / 'episodes' / path.name)
    reports = [report for report in runs if not report.get('repeat')]
    summary = {'run': run, 'command': shlex.join(sys.argv if argv is None else ['run_expert.py', *argv]),
               'scene': args.scene, 'robot': robot, 'expert': args.expert, 'scope': selected.SCOPE,
               'layout': args.layout, 'outside_scope': args.outside_scope, 'physics': args.physics, **missing.record(directory),
               'steps_run': args.steps, 'from': str(args.source.resolve()) if args.source else None,
               'problems': args.problems, 'skill_attempts': args.skill_attempts,
               **summarize(reports, seeds), 'steps': step_rates(reports, seeds, parent),
               'repeats': repeat_rates(runs, seeds, args.repeats) if args.repeats > 1 else None,
               'load': load_record(runs), 'total_s': time.perf_counter() - started, 'videos': {}, 'fails': {}}
    (directory / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float) + '\n')
    print('SUMMARY', directory / 'summary.json',
          json.dumps({key: summary[key] for key in ('succeeded', 'success_rate', 'failures')}, default=float), flush=True)
    for row in (summary['steps'] or {}).get('rows', []):
        print(f'STEP {row["step"]} {row["completed"]}/{row["started"]} progressive {row["progressive_rate"]:.2f}: '
              f'{row["name"]}', flush=True)
    if missing:
        raise SystemExit(f'no episode: seeds {seed_range(missing)}; see {staging}')


if __name__ == '__main__':
    main()
