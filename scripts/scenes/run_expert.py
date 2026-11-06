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
runs/expert/<scene>/<robot>/<expert>/<date>-<time>-<commit>-<robot>-<expert>-layout<n>[-<name>]/.
Partial seeds/steps, repeats, named runs and --from require --dev.

--steps k[-m] runs only those steps; k>1 requires --from BATCH with step k-1's
checkpoint. A source with one numbered checkpoint infers the next single step.
Summary and checkpoint robot/expert identity, backend and completed step name
must match.

Each batch holds trajectory HDF5 files, episodes/<seed>.json, calls/, problems/,
checkpoints/step<k>.hdf5 and summary.json (scope, identity, status, timings and
machine load). --repeats K adds reports <seed>-<repeat>.json with independent
random streams after the first run; recovery trajectories are retained for a2/a3 clips, without extra checkpoints.
Jobs freeze the submitted source and run through --gpus or --launcher, retrying
seeds without an episode; logs and partial products are kept in the batch's
workers/ (grouped units, retry attempts and process logs).
Episode failures are results; a nonzero exit means some seed has no episode.
"""

import argparse
import json
from pathlib import Path
import shlex
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from table_1000.common.serialization import dumps
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


def main(argv=None):
    dump_stacks_every()
    from table_1000.common.batch import add_arguments, commit, job_count, load_record, parse_seeds, seed_range
    from table_1000.common.batch_runtime import run as run_program
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
                             '--dev to <output>/<scene>/<robot>/<expert>/<run>/ (default runs/expert)')
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
    parser.add_argument('--problems', choices=('failed', 'all', 'none'), default='none',
                        help='save full numerical problems for failed/error runs or all runs; none (default) keeps lightweight calls only; '
                             'none disables full problem recording and extra IK diagnostics')
    parser.add_argument('--skill-attempts', type=int, default=3,
                        help='total tries per retry callback, restoring entry before later random tries (default 3)')
    parser.add_argument('--seeds', nargs='+', help='only these seeds of the initial states or checkpoints: 12 27, 0:20')
    args = parser.parse_args(argv)
    import h5py
    from table_1000.experts.generate import checkpoint_path, merge_checkpoints, repeat_rates, step_rates, summarize
    from table_1000.scenes.recordings import clear_batch, merge
    job_count(parser, args)
    root = args.root.resolve()
    from table_1000.scenes import load_expert, initial_dir, data_dir, dev_dir, batch_ref, outside_scope
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
        directory = dev_dir(args.output or root / 'runs' / 'expert', args.scene, robot, args.expert, run).resolve()
        if directory.exists() and (directory / 'summary.json').exists():
            parser.error(f'{directory} exists; give another --name')
    else:
        run = None
        directory = data_dir(args.output or root / 'data' / 'scenes', args.scene, robot, args.expert, args.physics, args.layout).resolve()
    identity = (args.scene, robot, args.expert, str(args.layout),
                '-'.join(map(str, args.steps)) if args.steps else 'all')
    options = vars(args) | dict(units=[(identity, seed) for seed in seeds],
        states=str(states), checkpoints=bool(args.source), assets=str(args.assets.resolve()))
    if args.dry_run:
        run_program('table_1000.experts.batch_program:program', options, root=root, output=directory)
        return
    clear_batch(directory, keep=('plan.json', 'layouts') if not args.dev and args.layout == 0 else ('plan.json',))
    started = time.perf_counter()
    missing = run_program('table_1000.experts.batch_program:program', options, root=root, output=directory)
    checkout = missing.commit
    staging = directory / 'workers'
    reports = [json.loads(path.read_text()) for path in sorted((directory / 'episodes').glob('*.json'))]
    owners = {report['seed']: report['job'] for report in reports}
    parts = [part for part in sorted(staging.glob('*/'))
             if any((part / 'episodes' / f'{seed}.json').is_file()
                    for seed, owner in owners.items() if part.name == owner)]
    for name in ('trajectories', 'trajectories_failed'):
        merge([part / f'{name}.hdf5' for part in parts], directory / f'{name}.hdf5',
              keep=lambda source, seed: owners.get(int(str(seed).split('-')[0])) == source.parent.name)
        for part in parts:
            (part / f'{name}.hdf5').unlink(missing_ok=True)
    sources = [path for part in parts for path in (part / 'checkpoints').glob('*.hdf5')
               if owners.get(int(path.stem)) == part.name]
    merge_checkpoints(sources, directory, dict(scene=args.scene, robot=robot, expert=args.expert,
        layout=args.layout, physics=args.physics, commit=checkout or 'nocommit'))
    for path in sources:
        path.unlink()
    runs = reports.copy()
    for part in parts:
        for path in sorted((part / 'episodes').glob('*-*.json')):
            report = json.loads(path.read_text())
            if owners.get(report['seed']) == part.name:
                runs.append(report)
                shutil.move(path, directory / 'episodes' / path.name)
        for kind in ('attempts', 'calls', 'problems', 'repeat-trajectories'):
            for path in sorted((part / kind).glob('*')):
                seed = int(path.name.split('-')[0].split('.')[0])
                if owners.get(seed) == part.name:
                    target = directory / kind / path.name
                    target.parent.mkdir(exist_ok=True)
                    shutil.move(path, target)
    summary = {'run': run, 'command': shlex.join(sys.argv if argv is None else ['run_expert.py', *argv]),
               'scene': args.scene, 'robot': robot, 'expert': args.expert, 'scope': selected.SCOPE,
               'layout': args.layout, 'outside_scope': args.outside_scope, 'physics': args.physics, **missing.record(directory),
               'missing_seeds': [key[-1] for key in missing], 'steps_run': args.steps, 'from': batch_ref(args.source, directory, root) if args.source else None,
               'problems': args.problems, 'skill_attempts': args.skill_attempts,
               **summarize(reports, seeds), 'steps': step_rates(reports, seeds, parent),
               'repeats': repeat_rates(runs, seeds, args.repeats) if args.repeats > 1 else None,
               'load': load_record(runs), 'total_s': time.perf_counter() - started, 'videos': {}, 'fails': {}}
    (directory / 'summary.json').write_text(dumps(summary, indent=1, ensure_ascii=False) + '\n')
    print('SUMMARY', directory / 'summary.json',
          dumps({key: summary[key] for key in ('succeeded', 'success_rate', 'failures')}), flush=True)
    for row in (summary['steps'] or {}).get('rows', []):
        print(f'STEP {row["step"]} {row["completed"]}/{row["started"]} progressive {row["progressive_rate"]:.2f}: '
              f'{row["name"]}', flush=True)
    if missing:
        raise SystemExit(f'no episode: seeds {seed_range(key[-1] for key in missing)}; see {staging}')


if __name__ == '__main__':
    main()
