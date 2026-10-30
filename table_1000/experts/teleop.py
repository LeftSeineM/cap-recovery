"""Teleoperating one step of an expert from a case: a seed's state before a step its expert failed.

A **case** is ``(batch, step, seed)`` of a development batch (``runs/expert/<scene>/<run>/``):
the step the seed's episode failed in (:func:`cases`). It starts from the checkpoint of the step
before, or, where the failed run was recorded, from the state at the moment it failed (the last of
its trajectory's states) or any moment of the failed step before that (:func:`trajectory_state`). Batches run with ``--repeats K`` give each seed's success rate over its K
runs; the cases come sorted by it, lowest first.

The operator does the step by hand (:class:`HumanExpert`): the expert's other steps stay as
they are, the case's step runs :func:`operate` instead, a loop at the control rate that turns
the operator's input into the same joint targets an expert sends (TCP targets solved by IK,
base speeds, lift and gripper targets). The step's check runs when the operator asks and only
shows its verdict; the operator ends the step as done or not done. A done step is kept as the
step's checkpoint, so later steps can run from it (``run_expert.py --from``).

Input (:class:`Operator`, fed by the page of ``scripts/experts/teleop.py``):

=====================  =============================================================
left drag on a view    the TCP target in the view's image plane (through the target)
Shift + wheel          the TCP target toward / away from the view's camera
right drag, wheel      (the outside view, :class:`Orbit`) turn about its target, zoom
middle / Shift+right   (the outside view) pan
Z / X, C / V, B / N    TCP target roll, pitch, yaw (+ / -), about the base's axes
I / K, J / L, U / O    TCP target along the base's x, y, z (+ / -)
W / S, A / D           base forward / back, turn left / right
R / F                  torso lift up / down
arrows                 head yaw / pitch
Tab, or the buttons    switch the controlled arm
Space, or the buttons  open / close a gripper
P                      pause / resume the simulation
Shift                  faster
=====================  =============================================================
"""

import json
from pathlib import Path
import queue
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from table_1000.experts.context import ExpertFailed, Step, run_steps
from table_1000.robots.teleop import FAST, Keys

LINEAR, ROTATION = .10, .6       # TCP target m/s, rad/s from keys
BASE_SPEED, BASE_TURN = .3, .6   # m/s, rad/s
LIFT_SPEED, HEAD_SPEED = .1, .6  # m/s, rad/s
WHEEL = .01                      # m per wheel notch
ORBIT, ZOOM = .005, 1.1          # rad per pixel an outside view turns, distance factor per wheel notch
JOINT_SPEED = 1.5                # rad/s an arm joint target moves at most
GRIPPER_TIME = .6                # s for the gripper to open or close fully
APERTURE = 20.955                # mm, a view camera's horizontal aperture (see sim.isaaclab.cameras)


# -- cases ---------------------------------------------------------------------------------
def failed_step(report):
    """The step ``report``'s episode failed in (the first of its steps it did not complete), None
    for a success or an episode of an expert without steps."""
    if report.get('status') == 'success' or not report.get('step_names'):
        return None
    first, last = report.get('steps_run') or (1, len(report['step_names']))
    done = {entry['step'] for entry in report.get('steps') or []}
    return next((step for step in range(first, last + 1) if step not in done), None)


def start_checkpoint(batch, summary, step):
    """The checkpoint file of step ``step - 1`` a case of ``batch`` starts from: the batch's own, or,
    for the batch's first step, that of the batch it ran from (``from``)."""
    own = Path(batch) / 'checkpoints' / f'step{step - 1}.hdf5'
    first = (summary.get('steps_run') or [1])[0]
    if step == first and summary.get('from'):
        return Path(summary['from']) / 'checkpoints' / f'step{step - 1}.hdf5'
    return own


def failed_trajectories(batch):
    """``{seed: samples}`` of the failed episodes recorded in a batch (``trajectories_failed.hdf5``)."""
    import h5py
    path = Path(batch) / 'trajectories_failed.hdf5'
    if not path.is_file():
        return {}
    with h5py.File(path, 'r') as file:
        return {int(name.split('_')[1]): int(group.attrs['num_samples']) for name, group in file['data'].items()}


def trajectory_state(path, seed, index):
    """The scene state after control step ``index + 1`` of episode ``seed`` in a trajectory file (its
    ``states``, environment axis kept), as a checkpoint holds it: with ``targets``, the action of
    that step by group (``env_args['action_layout']``), which the action terms were holding."""
    import h5py

    def take(group):
        return {key: take(value) if isinstance(value, h5py.Group) else value[index:index + 1]
                for key, value in group.items()}
    with h5py.File(path, 'r') as file:
        episode = file['data'][f'demo_{seed}']
        state = take(episode['states'])
        layout = json.loads(file['data'].attrs['env_args'])['action_layout']
        action = episode['actions'][index:index + 1]
        return {**state, 'targets': {name: action[:, start:stop] for name, (start, stop) in layout.items()}}


def trajectory_start(path, seed):
    """``{'qpos', 'base_pose'}`` of the robot where a recorded episode began."""
    import h5py
    with h5py.File(path, 'r') as file:
        obs = file['data'][f'demo_{seed}']['obs']
        return {'qpos': obs['qpos'][0:1], 'base_pose': obs['base_pose'][0:1]}


def batch_cases(batch):
    """The cases of one batch: ``{'batch', 'run', 'scene', 'step', 'step_name', 'seed', 'successes',
    'runs', 'rate', 'failure', 'start'}`` per seed that failed in any of its runs (``rate`` None
    without ``--repeats``). A case whose failed run was recorded (the first run of a seed) also has
    ``trajectory`` (its file), ``failure_at`` (its length in control steps: the state at the
    failure is the last) and ``step_at`` (where the failed step began in it)."""
    batch = Path(batch)
    summary = json.loads((batch / 'summary.json').read_text())
    recorded = failed_trajectories(batch)
    repeats = summary.get('repeats') or {}
    reports = {}
    for path in sorted((batch / 'episodes').glob('*.json')):
        seed, _, repeat = path.stem.partition('-')
        reports.setdefault(int(seed), []).append((int(repeat or 0), json.loads(path.read_text())))
    cases = []
    for seed, runs in sorted(reports.items()):
        failed = [(repeat, report) for repeat, report in sorted(runs, key=lambda run: run[0]) if failed_step(report)]
        if not failed:
            continue
        repeat, report = failed[0]
        step = failed_step(report)
        successes = repeats.get('succeeded', {}).get(str(seed)) if repeats else None
        count = repeats.get('count') if repeats else None
        cases.append({'batch': str(batch.resolve()), 'run': summary.get('run') or '/'.join(batch.resolve().parts[-3:]),
                      'scene': summary.get('scene'), **{key: summary[key] for key in ('robot', 'expert', 'scope', 'layout', 'outside_scope')}, 'step': step, 'step_name': report['step_names'][step - 1],
                      'seed': seed, 'attempt': report.get('attempt', 0), 'successes': successes, 'runs': count,
                      'rate': successes / count if count else None,
                      'failure': (report.get('failure') or {}).get('reason'),
                      'start': str(start_checkpoint(batch, summary, step))})
        if repeat == 0 and seed in recorded:
            cases[-1].update(trajectory=str((batch / 'trajectories_failed.hdf5').resolve()), failure_at=recorded[seed],
                             step_at=next((entry['at'] for entry in report.get('steps') or []
                                           if entry['step'] == step - 1), 0))
    return cases


def cases(runs, scene=None, batches=()):
    """The cases of every batch under ``runs`` (``runs/expert``), of ``scene`` only when given, and of
    ``batches``: lowest success rate first, cases without a rate last."""
    found = []
    for summary in sorted(Path(runs).glob(f'{scene or "*"}/*/summary.json')):
        found += batch_cases(summary.parent)
    for batch in batches:
        found += batch_cases(batch)
    return sorted(found, key=lambda case: (case['rate'] is None, case['rate'] or 0., case['run'], case['seed']))


# -- views -----------------------------------------------------------------------------------
def view_camera(view):
    """``(cam2world rotation, position, fx in pixels)`` of a fixed view, camera axes x right, y down,
    z forward."""
    position = np.asarray(view['position'], dtype=float)
    forward = np.asarray(view['target'], dtype=float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(view.get('up', (0., 0., 1.)), dtype=float))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.c_[right, down, forward], position, view['focal_length_mm'] / APERTURE * view['resolution'][0]


def drag(camera, point, dx, dy, wheel=0.):
    """World displacement of ``point`` dragged ``(dx, dy)`` pixels in a camera's image (``camera``
    as :func:`view_camera`), and moved ``wheel`` notches toward the camera."""
    rotation, position, fx = camera
    depth = max(float((np.asarray(point) - position) @ rotation[:, 2]), .05)
    return rotation @ np.array([dx * depth / fx, dy * depth / fx, -wheel * WHEEL])


class Orbit:
    """An outside view (a scene view ``{'position', 'target', ...}``) the operator moves: turned about
    its target, panned, zoomed (:meth:`move`)."""

    def __init__(self, view):
        self.view = dict(view)

    def move(self, kind, dx, dy):
        """``orbit`` or ``pan`` by ``(dx, dy)`` pixels of a drag, ``zoom`` by ``dy`` wheel notches (+: closer)."""
        position, target = np.asarray(self.view['position'], float), np.asarray(self.view['target'], float)
        offset = position - target
        distance = np.linalg.norm(offset)
        if kind == 'orbit':
            yaw = np.arctan2(offset[1], offset[0]) - dx * ORBIT
            pitch = np.clip(np.arcsin(offset[2] / distance) + dy * ORBIT, -1.45, 1.45)
            offset = distance * np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        elif kind == 'pan':
            rotation, _, fx = view_camera(self.view)
            target = target + rotation @ np.array([-dx, -dy, 0.]) * distance / fx
        elif kind == 'zoom':
            offset = offset * ZOOM ** -dy
        else:
            raise ValueError(f'unknown view move {kind}')
        self.view.update(position=tuple(target + offset), target=tuple(target))

    def camera(self):
        return view_camera(self.view)


# -- operating ---------------------------------------------------------------------------------
class Operator:
    """The operator's input, shared between the page's thread and the simulation loop: held keys,
    drags, moves of the outside views and commands (``check``, ``done``, ``not done``, ``arm``,
    ``gripper``, ``load`` of a case)."""

    def __init__(self):
        self.keys = Keys()
        self.drags, self.views, self.commands = queue.Queue(), queue.Queue(), queue.Queue()
        self.lock = threading.Lock()
        self.status = {}

    def drag(self, view, dx, dy, wheel=0.):
        self.drags.put((view, float(dx), float(dy), float(wheel)))

    def move_view(self, view, kind, dx, dy):
        self.views.put((view, kind, float(dx), float(dy)))

    def command(self, name, **values):
        self.commands.put({'command': name, **values})

    @staticmethod
    def drain(source):
        while not source.empty():
            yield source.get()


def check(ctx, step):
    """``{'passed', 'reason'}`` of ``step``'s check (a :class:`Step`) on the current state."""
    if step.check is None:
        return {'passed': True, 'reason': None}
    current = ctx.current
    ctx.current = step.name
    try:
        step.check(ctx)
        return {'passed': True, 'reason': None}
    except ExpertFailed as error:
        return {'passed': False, 'reason': error.reason}
    finally:
        ctx.current = current


def operate(ctx, step, operator, show, cameras, log):
    """Do ``step`` by the operator's input until they end it, at the control rate in real time.
    ``show(ctx, status, targets)`` renders and publishes now and then; ``cameras`` are the views a
    drag may come from (:func:`view_camera`, or a callable giving one for a robot camera); ``log``
    (a dict) receives the checks asked for, the operator's verdict (``done``, ``not done``, or
    ``left`` for another command, kept as ``next``) and the check when it ended. A step not ended
    as done fails."""
    from table_1000.experts.solver import OPEN_GAP
    from table_1000.sim import CONTROL_HZ
    dt, spec, model = 1 / CONTROL_HZ, ctx.spec, ctx.model
    arms = list(spec.arms)
    arm = spec.primary_arm
    base = ctx.base_transform()
    targets = {name: np.linalg.inv(base) @ ctx.tcp(name) for name in arms}       # in the base frame
    commands = {name: ctx.env.qpos(spec.arms[name].joints)[0].copy() for name in arms}
    # The commanded gaps, not the measured: a gripper holding something keeps squeezing it.
    held = ctx.env.targets()
    gaps = {name: float(held[f'{name}_gripper'][0, 0]) for name in arms}
    wanted = dict(gaps)
    lift = ctx.env.qpos(spec.lift)[0].copy() if spec.lift else None
    head = ctx.env.qpos(spec.head)[0].copy() if spec.head else None
    lift_limits = model.limits[model.lift_indices()] if spec.lift else None
    paused, reached, last_check = False, {name: True for name in arms}, None
    log.setdefault('checks', [])
    keys = operator.keys
    list(keys.presses()), list(Operator.drain(operator.drags))      # input from before the step
    with ctx.subtask(step.name):
        while True:
            started = time.perf_counter()
            base = ctx.base_transform()
            for command in Operator.drain(operator.commands):
                if command['command'] == 'check':
                    last_check = {**check(ctx, step), 'at': ctx.steps}
                    log['checks'].append(last_check)
                elif command['command'] == 'arm':
                    arm = command['arm']
                elif command['command'] == 'gripper':
                    name = command.get('arm', arm)
                    wanted[name] = OPEN_GAP if wanted[name] < OPEN_GAP / 2 else 0.
                elif command['command'] in ('done', 'not done'):
                    log['verdict'] = command['command']
                    log['final_check'] = {**check(ctx, step), 'at': ctx.steps}
                    if command['command'] == 'not done':
                        ctx.fail('the operator ended the step as not done')
                    return
                else:       # another case: this one ends as not done, the command waits in the log
                    log.update(verdict='left', final_check={**check(ctx, step), 'at': ctx.steps}, next=command)
                    ctx.fail('the operator left the step')
            for key in keys.presses():
                if key == 'Tab':
                    arm = arms[(arms.index(arm) + 1) % len(arms)]
                elif key == 'Space':
                    wanted[arm] = OPEN_GAP if wanted[arm] < OPEN_GAP / 2 else 0.
                elif key == 'KeyP':
                    paused = not paused
            k = keys.fast()
            target = targets[arm]
            for view, dx, dy, wheel in Operator.drain(operator.drags):
                camera = cameras.get(view)
                camera = camera(ctx) if callable(camera) else camera
                if camera is not None:
                    point = base @ target
                    target[:3, 3] += (np.linalg.inv(base)[:3, :3] @ drag(camera, point[:3, 3], dx, dy, wheel))
            move = np.array([keys.axis('KeyI', 'KeyK'), keys.axis('KeyJ', 'KeyL'), keys.axis('KeyU', 'KeyO')])
            turn = np.array([keys.axis('KeyZ', 'KeyX'), keys.axis('KeyC', 'KeyV'), keys.axis('KeyB', 'KeyN')])
            target[:3, 3] += move * LINEAR * k * dt
            if turn.any():
                target[:3, :3] = Rotation.from_rotvec(turn * ROTATION * k * dt).as_matrix() @ target[:3, :3]
            if paused:
                show(ctx, status(ctx, arm, paused, reached, last_check, wanted), targets, arm)
                time.sleep(dt)
                continue
            action = {}
            for name in arms:
                status_, q = model.ik(name, base @ targets[name], world=True, attempts=1, check_collision=False)
                reached[name] = q is not None
                if q is not None:
                    goal = np.asarray(q)[model.arm_indices(name)]
                    commands[name] += np.clip(goal - commands[name], -JOINT_SPEED * dt, JOINT_SPEED * dt)
                action[name] = commands[name][None]
                gaps[name] += np.clip(wanted[name] - gaps[name], -OPEN_GAP / GRIPPER_TIME * dt, OPEN_GAP / GRIPPER_TIME * dt)
                action[f'{name}_gripper'] = [[gaps[name]]]
            if spec.base:
                action['base'] = [[BASE_SPEED * k * keys.axis('KeyW', 'KeyS'), BASE_TURN * k * keys.axis('KeyA', 'KeyD')]]
            if lift is not None:
                lift = np.clip(lift + LIFT_SPEED * k * keys.axis('KeyR', 'KeyF') * dt, lift_limits[:, 0], lift_limits[:, 1])
                action['lift'] = lift[None]
            if head is not None:
                head = head + HEAD_SPEED * k * dt * np.array([keys.axis('ArrowLeft', 'ArrowRight'),
                                                              keys.axis('ArrowDown', 'ArrowUp')])[:len(head)]
                action['head'] = head[None]
            ctx.step(action)
            show(ctx, status(ctx, arm, paused, reached, last_check, wanted), targets, arm)
            spare = dt - (time.perf_counter() - started)
            if spare > 0:
                time.sleep(spare)


def status(ctx, arm, paused, reached, last_check, wanted):
    """The status the page shows."""
    from table_1000.sim import CONTROL_HZ
    tcp = ctx.tcp(arm)[:3, 3]
    grippers = {name: {'gap': round(float(spec.gap(ctx.env.qpos(spec.fingers)[0])), 3),
                       'closing': bool(wanted[name] < 1e-6)} for name, spec in ctx.spec.arms.items()}
    return {'arm': arm, 'arms': list(ctx.spec.arms), 'grippers': grippers, 'paused': paused,
            'time_s': ctx.steps / CONTROL_HZ,
            'steps': ctx.steps, 'tcp': [round(float(v), 3) for v in tcp],
            'unreachable': [name for name, ok in reached.items() if not ok], 'check': last_check}


class HumanExpert:
    human = True
    """The expert ``expert`` (a module with ``STEPS``) with its step ``step`` done by
    :func:`operate` (``operate(ctx, step)``), unchecked: the operator decides when it is done."""

    def __init__(self, expert, step, operate_step):
        self.STEPS = [Step(entry.name, (lambda ctx, entry=entry: operate_step(ctx, entry)), None) if index == step
                      else entry for index, entry in enumerate(expert.STEPS, 1)]

    def solve(self, ctx):
        run_steps(ctx, self.STEPS)
