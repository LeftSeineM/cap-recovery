"""Expert trajectories of a scene: ``expert.solve(ctx)`` from a stored initial state, recording and a report per seed.

An expert made of steps (:func:`~table_1000.experts.context.run_steps`) leaves a
checkpoint after each step that passed its check: the scene state, in
``checkpoints/<seed>.hdf5`` (merged by step into ``checkpoints/step<k>.hdf5``,
:func:`merge_checkpoints`); an episode from the initial state keeps that too, as step 0. With ``steps`` ``(first, last)`` only those steps run,
from the initial states (``first`` 1) or from the checkpoints of step ``first - 1``;
the robot's home stays where the seed's episode began.

One :class:`Generator` builds the environment once and runs seeds one after
another, each from its initial state of ``initial_states.hdf5``
(:mod:`table_1000.scenes.initial`). An expert failure is recorded as a failed
episode (``trajectories_failed.hdf5``) with the subtask and reason, so the
success rate is a metric, not a requirement. Each seed also gets
``episodes/<seed>.json``: subtasks, failure, planning log, the parts the fingers
touched, the objects' final positions and joints, and timings.
"""

import json
from pathlib import Path
import time
import traceback

import numpy as np

from table_1000.experts.context import Context, ExpertFailed, Unreachable
from table_1000.scenes import INSTRUCTION, load
from table_1000.scenes.initial import read_tree, write_tree
from table_1000.sim import CONTROL_HZ, PHYSICS_DT

# A seed without a report: its process stopped before finishing it.
PROCESS_EXITED = '-: process exited'
RESUME_S = .2       # s of holding still after resuming from a checkpoint
INITIAL_STATE = 'initial state'     # the name of step 0's checkpoints


def checkpoint_path(directory, step):
    """The checkpoints of step ``step`` in a batch directory."""
    return Path(directory) / 'checkpoints' / f'step{step}.hdf5'


def write_checkpoints(path, seed, attempt, start, checkpoints):
    """One seed's checkpoints (:attr:`Context.checkpoints`) and the robot where its episode began."""
    import h5py
    with h5py.File(path, 'w') as file:
        file.attrs.update(seed=seed, attempt=attempt)
        write_tree(file.create_group('start'), start)
        for entry in checkpoints:
            group = file.create_group(str(entry['step']))
            group.attrs.update(name=entry['name'], at=entry['at'])
            write_tree(group.create_group('state'), entry['state'])


def merge_checkpoints(sources, directory, attrs):
    """Gather the seeds' checkpoint files ``sources`` by step into ``checkpoint_path(directory, k)``:
    one group per seed (attributes ``seed``, ``attempt``, ``at``; ``state`` and ``start``), file
    attributes ``attrs`` and the step's ``step`` and ``name``. Returns the steps written."""
    import h5py
    files = [h5py.File(path, 'r') for path in sorted(sources)]
    try:
        steps = sorted({int(key) for file in files for key in file if key != 'start'})
        for step in steps:
            path = checkpoint_path(directory, step)
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, 'w') as out:
                out.attrs.update(attrs, step=step)
                for file in sorted(files, key=lambda file: int(file.attrs['seed'])):
                    if str(step) not in file:
                        continue
                    out.attrs['name'] = file[str(step)].attrs['name']
                    group = out.create_group(str(int(file.attrs['seed'])))
                    group.attrs.update(seed=file.attrs['seed'], attempt=file.attrs['attempt'],
                                       at=file[str(step)].attrs['at'])
                    file.copy(file[str(step)]['state'], group, 'state')
                    file.copy(file['start'], group, 'start')
        return steps
    finally:
        for file in files:
            file.close()


def read_checkpoints(path):
    """``(attrs, {seed: (attempt, state, start)})`` of a step's checkpoint file."""
    import h5py
    with h5py.File(path, 'r') as file:
        return dict(file.attrs), {int(name): (int(group.attrs['attempt']), read_tree(group['state']),
                                              read_tree(group['start']))
                                  for name, group in sorted(file.items(), key=lambda item: int(item[0]))}


def finger_contacts(intervals, spec):
    """What the fingers touched outside the robot, ``{pair: {'max_penetration_m', 'seconds', 'subtasks'}}``."""
    fingers = {f'robot.{link}' for arm in spec.arms.values() for link in arm.finger_links}
    summary = {}
    for entry in intervals:
        a, b = entry['pair']
        if not ((a in fingers and not b.startswith('robot')) or (b in fingers and not a.startswith('robot'))):
            continue
        item = summary.setdefault(' / '.join(entry['pair']), {'max_penetration_m': 0., 'seconds': 0., 'subtasks': []})
        item['max_penetration_m'] = max(item['max_penetration_m'], entry['max_penetration_m'])
        if entry.get('end') is not None:
            item['seconds'] += entry['end'] - entry['start']
        if entry.get('phase') not in item['subtasks']:
            item['subtasks'].append(entry.get('phase'))
    return summary


class Generator:
    """Runs scene ``scene_name`` (import after ``start_app``) on ``physics`` (``'isaac-cpu'``)."""

    DEVICES = {'isaac-cpu': 'cpu'}

    def __init__(self, scene_name, output, assets, root, physics='isaac-cpu', steps=None, source=None,
                 problems='failed', skill_attempts=3, **env_kwargs):
        """``steps``: ``(first, last)`` of the expert's steps to run (all when None); ``source``: where
        the states come from (a checkpoint file), for the reports; ``problems`` saves numerical
        problems of failed/error runs, all runs, or none; ``env_kwargs`` (cameras, views) go to
        the :class:`~table_1000.sim.isaaclab.env.RobotEnv`."""
        from table_1000.robots import load_spec
        from table_1000.robots.kinematics import RobotModel
        from table_1000.sim.isaaclab import recording
        from table_1000.sim.isaaclab.env import RobotEnv
        self.recording = recording
        started = time.perf_counter()
        self.name, self.output, self.assets = scene_name, Path(output), Path(assets)
        self.steps, self.source, self.problems = steps, source, problems
        self.skill_attempts = skill_attempts
        self.scene, self.expert = load(scene_name, 'scene', root), load(scene_name, 'expert', root)
        if steps and not 1 <= steps[0] <= steps[1] <= len(getattr(self.expert, 'STEPS', ())):
            raise ValueError(f'steps {steps[0]}-{steps[1]}: the expert of {scene_name} has '
                             f'{len(getattr(self.expert, "STEPS", ()))} STEPS')
        self.spec = load_spec(Path(self.scene.ROBOT['asset']).name, root)
        self.env = RobotEnv(self.scene, self.spec, self.assets, root=root, device=self.DEVICES[physics],
                            recorders=recording.recorder_cfg(self.output), **env_kwargs)
        self.model = RobotModel(self.spec, self.output / 'planning', root)
        for name in ('episodes', 'checkpoints'):
            (self.output / name).mkdir(parents=True, exist_ok=True)
        self.env_args = {'scene': scene_name, 'robot': self.spec.name, 'physics': physics, 'instruction': INSTRUCTION,
                         'control_hz': CONTROL_HZ, 'physics_dt': PHYSICS_DT, 'joints': self.env.joints,
                         'action_layout': self.env.action_layout}
        self.load_s = time.perf_counter() - started

    def archive_retry(self, ctx, control, retry, failure, repeat):
        """Retain one failed branch as a complete batch before rolling back its state."""
        from copy import deepcopy
        import gzip
        from table_1000.experts.retrying import copy_recorded
        trace = ctx.trace
        name = f'{trace.seed}-{repeat}' if repeat else str(trace.seed)
        branch = Path('attempts') / name / f'{len(ctx.retry_archives):04d}-{control}-{retry}'
        directory = self.output / branch
        directory.mkdir(parents=True)
        branch_trace = trace.failed_branch(failure, control)
        calls = f'calls/{trace.seed}.json.gz'
        (directory / 'calls').mkdir()
        with gzip.open(directory / calls, 'wt') as file:
            json.dump(branch_trace, file, separators=(',', ':'))
        report = dict(scene=self.name, seed=trace.seed, attempt=trace.attempt, repeat=repeat, status='failed',
                      steps_run=self.steps, from_retry=dict(control=control, retry=retry), step_names=ctx.step_names,
                      failure=dict(subtask=failure.subtask, reason=failure.reason,
                                   unreachable=isinstance(failure, Unreachable), details=failure.details),
                      steps=[{key: e[key] for key in ('step','name','at','wall_s','cpu_s')} for e in ctx.checkpoints],
                      call_trace=calls, failure_frame=branch_trace['failure_frame'], solve_trace=None,
                      control_steps=ctx.steps, events=deepcopy(ctx.events))
        if trace.problems:
            report['solve_trace'] = f'problems/{trace.seed}.json.gz'
            trace.problems.save(directory / report['solve_trace'])
        episode = type(self.env.env.recorder_manager.get_episode(0))()
        episode.data = copy_recorded(self.env.env.recorder_manager.get_episode(0).data)
        episode.pre_export()
        episode.success = False
        episode.attrs = dict(scene=self.name, seed=trace.seed, status='failed', failure=report['failure'])
        handler = self.recording.TrajectoryFile()
        handler.create(str(directory / 'trajectories_failed'))
        try:
            handler.add_env_args(self.env_args)
            handler.write_episode(episode, trace.seed)
        finally:
            handler.close()
        first = (self.steps or (1,))[0]
        checkpoints = [dict(step=first-1, name='retry episode entry', at=0, state=ctx.initial_state), *ctx.checkpoints]
        checkpoint = directory / 'checkpoints' / f'{trace.seed}.hdf5'
        checkpoint.parent.mkdir()
        write_checkpoints(checkpoint, trace.seed, trace.attempt, ctx.start, checkpoints)
        merge_checkpoints([checkpoint], directory, dict(scene=self.name, robot=self.spec.name,
                                                       physics=self.env_args['physics']))
        checkpoint.unlink()
        (directory / 'episodes').mkdir()
        (directory / 'episodes' / f'{trace.seed}.json').write_text(json.dumps(report, indent=1, default=float)+'\n')
        summary = dict(scene=self.name, robot=self.spec.name, physics=self.env_args['physics'],
                       steps_run=self.steps, retry_parent='../../..', **{'from':None}, seeds=[trace.seed], succeeded=0, success_rate=0.)
        (directory / 'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
        archived = dict(batch=str(branch), control=control, retry=retry, frame=branch_trace['failure_frame'], reason=failure.reason)
        ctx.retry_archives.append(archived)
        return archived

    def run(self, seed, attempt, state, start=None, repeat=0):
        """Episode ``seed`` from its initial ``state`` (made by sub-seed ``attempt``), or from a
        checkpoint's with ``start`` the robot where the seed's episode began; its report has the
        machine's load meanwhile (``load``, see :class:`~table_1000.common.batch.Load`) and the CPU
        seconds the process used (``cpu_s``). ``repeat`` r > 0 runs it again from the same state
        with other random streams, for how its outcome varies: report ``episodes/<seed>-<r>.json``,
        no trajectory and no checkpoints."""
        from table_1000.common.batch import Load
        with Load() as load:
            report, started = self.episode(seed, attempt, state, start, repeat)
        report.update(load=load.record(), cpu_s=sum(sample['cpu_s'] for sample in load.samples))
        return self.save(report, started)

    def episode(self, seed, attempt, state, start, repeat):
        started = time.perf_counter()
        first = self.steps[0] if self.steps else 1
        report = {'scene': self.name, 'seed': seed, 'attempt': attempt, 'steps_run': self.steps, 'from': self.source,
                  'repeat': repeat, 'skill_attempts': self.skill_attempts}
        snapshot = json.loads(state['_expert_frame']) if '_expert_frame' in state else None
        state = {key: value for key, value in state.items() if key != '_expert_frame'}
        self.env.reset_to_state(state)
        init_s = time.perf_counter() - started
        self.recording.begin(self.env)
        # MPlib's IK restarts and RRTConnect draw from its own generator; a later step, or a repeat,
        # from its own.
        import mplib.pymp
        streams = [seed, attempt, 1] + ([first] if first > 1 or repeat else []) + ([repeat] if repeat else [])
        mplib.pymp.set_global_seed(int(np.random.SeedSequence(streams).generate_state(1)[0]) | 1 if repeat
                                   else seed + 1 + 1_000_003 * (first - 1))   # OMPL rejects 0
        ctx = Context(self.env, self.model, self.scene, self.assets, np.random.default_rng(streams), start=start,
                      only=self.steps)
        from table_1000.experts.trace import CallTrace, replay, restore, encode
        ctx.trace = CallTrace(ctx, seed, attempt, repeat)
        ctx.initial_state = state
        ctx.skill_attempts = self.skill_attempts
        ctx.archive_retry = lambda control, retry, failure: self.archive_retry(ctx, control, retry, failure, repeat)
        from table_1000.experts.problems import Problems
        if self.problems != 'none' and not getattr(self.expert, 'human', False):
            ctx.trace.problems = Problems(ctx)
            ctx.trace.problems.install()
        try:
            if snapshot:
                if getattr(self.expert, 'human', False):
                    restore(ctx, snapshot['frame'])
                    self.expert.solve(ctx)
                else:
                    ctx.step_names = [step.name for step in self.expert.STEPS]
                    report['result'] = encode(replay(ctx, snapshot, self.expert))
            else:
                if start is not None:
                    ctx.hold(RESUME_S)
                self.expert.solve(ctx)
            report['status'] = 'success'
        except ExpertFailed as error:
            report.update(status='failed', failure={'subtask': error.subtask, 'reason': error.reason,
                                                    'unreachable': isinstance(error, Unreachable),
                                                    'details': error.details})
        except Exception:   # an error in the expert's code is reported with the episode
            report.update(status='error', failure={'subtask': ctx.current, 'reason': 'error',
                                                   'traceback': traceback.format_exc()})
        finally:
            if ctx.trace.problems:
                ctx.trace.problems.close()
            for solver in ctx.solvers.values():
                solver.world.clear()
        name = f'{seed}-{repeat}' if repeat else str(seed)
        report['retry_archives'] = ctx.retry_archives
        report['call_trace'] = f'calls/{name}.json.gz'
        report['failure_frame'] = ctx.trace.failure_frame
        ctx.trace.save(self.output / report['call_trace'])
        report['solve_trace'] = None
        if ctx.trace.problems and (self.problems == 'all' or report['status'] != 'success'):
            report['solve_trace'] = f'problems/{name}.json.gz'
            ctx.trace.problems.save(self.output / report['solve_trace'])
        if snapshot:
            failed = ctx.trace.frames[ctx.trace.failure_frame] if ctx.trace.failure_frame is not None else None
            original = snapshot['failure']
            report['replay'] = {'source_batch': snapshot.get('batch'), 'source_frame': snapshot['frame']['id'],
                                'source_enter': snapshot['frame']['enter'], 'name': snapshot['frame']['name'],
                                'check': snapshot['check'], 'original_failure': original,
                                'same_frame': bool(failed and (failed.get('source_id') == snapshot['failed_frame_id'])),
                                'same_reason': report.get('failure', {}).get('reason') == original.get('reason')}
            if snapshot['check'] and failed and failed.get('phase') == 'check':
                report['replay']['same_frame'] = True
        report.update(
            step_names=ctx.step_names, steps=[{key: entry[key] for key in ('step', 'name', 'at', 'wall_s', 'cpu_s')}
                                              for entry in ctx.checkpoints],
            control_steps=ctx.steps, duration_s=ctx.steps / CONTROL_HZ, subtasks=ctx.subtasks, events=ctx.events,
            expert_wall_s=time.perf_counter() - ctx.started, subtask_wall_s=ctx.subtask_wall_s,
            plans=[entry for solver in ctx.solvers.values() for entry in solver.log],
            finger_contacts=finger_contacts(self.env.contacts.intervals(), self.spec),
            final={'objects': {name: self.env.object_pose(name)[0][:3, 3].tolist() for name in self.scene.OBJECTS},
                   'joints': {name: {joint: float(value[0]) for joint, value in joints.items()}
                              for name, joints in self.env.privileged()['joints'].items()}},
            timings={'init_s': init_s, **{key: sum(s.timings[key] for s in ctx.solvers.values())
                                          for key in ('planning_s', 'executing_s')}})
        if repeat:
            return report, started
        checkpoints = ctx.checkpoints
        if first == 1 and start is None and hasattr(self.expert, 'STEPS'):
            checkpoints = [{'step': 0, 'name': INITIAL_STATE, 'at': 0, 'state': state}, *checkpoints]
        if checkpoints:
            write_checkpoints(self.output / 'checkpoints' / f'{seed}.hdf5', seed, attempt, ctx.start, checkpoints)
        attrs = {'scene': self.name, 'seed': seed, 'attempt': attempt,
                 'instruction': INSTRUCTION, 'subtasks': ctx.subtasks, 'status': report['status'],
                 'steps': [{key: entry[key] for key in ('step', 'name', 'at')} for entry in report['steps']]}
        if self.source:
            attrs['from'] = self.source
        if 'failure' in report:
            attrs['failure'] = {key: report['failure'][key] for key in ('subtask', 'reason')}
        self.recording.finish(self.env, seed, report['status'] == 'success', attrs, self.env_args)
        return report, started

    def save(self, report, started):
        report['wall_s'] = time.perf_counter() - started
        name = f'{report["seed"]}-{report["repeat"]}' if report['repeat'] else str(report['seed'])
        (self.output / 'episodes' / f'{name}.json').write_text(json.dumps(report, indent=1, default=float))
        return report

    def close(self):
        self.env.env.recorder_manager.close()
        self.env.close()


def failure_key(report):
    failure = report.get('failure', {})
    return f'{failure.get("subtask")}: {failure.get("reason")}'


def step_rates(reports, seeds, parent=None):
    """Per step of an expert made of steps: ``{'step', 'name', 'started', 'completed', 'success_rate',
    'progressive_rate'}``. A step starts in the seeds that completed the one before (all the seeds
    for the first run here); the progressive rate counts the seeds that completed it and every
    step before, out of the seeds the first step started in: those of the ``parent`` summary (the
    batch whose checkpoints this one resumed from), whose rows come first. ``None`` for an expert
    without steps."""
    reports = [report for report in reports if report.get('step_names')]
    if not reports:
        return None
    names = reports[0]['step_names']
    first, last = reports[0].get('steps_run') or (1, len(names))
    base = parent['steps']['seeds'] if parent and parent.get('steps') else len(seeds)
    rows = [row for row in (parent or {}).get('steps', {}).get('rows', []) if row['step'] < first]
    done = {report['seed']: {entry['step'] for entry in report['steps']} for report in reports}
    started = len(seeds)
    for step in range(first, last + 1):
        completed = sum(step in steps for steps in done.values())
        rows.append({'step': step, 'name': names[step - 1], 'started': started, 'completed': completed,
                     'success_rate': completed / started if started else None,
                     'progressive_rate': completed / base if base else None})
        started = completed
    return {'seeds': base, 'rows': rows}


def repeat_rates(reports, seeds, count):
    """Of a batch run ``count`` times per seed (``--repeats``), from all its reports: each seed's
    successes (out of ``count``), the failures by ``"subtask: reason"`` and seed with their number,
    the success rate over all the runs, and the seeds with fewer runs (a process stopped) with
    their number of runs (``short``)."""
    succeeded, failures = dict.fromkeys(map(str, seeds), 0), {}
    done = dict.fromkeys(map(str, seeds), 0)
    for report in reports:
        done[str(report['seed'])] += 1
        if report['status'] == 'success':
            succeeded[str(report['seed'])] += 1
        else:
            seeds_failed = failures.setdefault(failure_key(report), {})
            seeds_failed[str(report['seed'])] = seeds_failed.get(str(report['seed']), 0) + 1
    return {'count': count, 'succeeded': succeeded,
            'success_rate': sum(succeeded.values()) / sum(done.values()) if any(done.values()) else None,
            'failures': dict(sorted(failures.items(), key=lambda item: -sum(item[1].values()))),
            'short': {seed: runs for seed, runs in done.items() if runs < count}}


def summarize(reports, seeds):
    """A batch over ``seeds`` from its episode reports: success rate of all the seeds, the failed
    seeds by ``"subtask: reason"`` (seeds without a report under :data:`PROCESS_EXITED`), means."""
    from table_1000.common.batch import seeds_record
    reports = {report['seed']: report for report in reports}
    succeeded = [report for report in reports.values() if report['status'] == 'success']
    failures = {}
    for seed in seeds:
        report = reports.get(seed)
        if report is None or report['status'] != 'success':
            failures.setdefault(PROCESS_EXITED if report is None else failure_key(report), []).append(seed)
    unreachable = sorted(seed for seed, report in reports.items() if report.get('failure', {}).get('unreachable'))
    return {'seeds': seeds_record(seeds), 'succeeded': len(succeeded),
            'success_rate': len(succeeded) / len(seeds) if seeds else None,
            # Nothing in reach: a high share points at the planning (parameters, collisions) or at the
            # state the step before left.
            'unreachable': unreachable, 'unreachable_rate': len(unreachable) / len(seeds) if seeds else None,
            'failures': dict(sorted(failures.items(), key=lambda item: (-len(item[1]), item[0]))),
            'mean_duration_s': float(np.mean([report['duration_s'] for report in succeeded])) if succeeded else None,
            'mean_wall_s': float(np.mean([report['wall_s'] for report in reports.values()])) if reports else None}
