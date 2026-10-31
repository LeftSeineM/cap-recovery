"""Serializable expert call frames and replay of an evaluated atomic skill."""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import fields, is_dataclass
from functools import wraps
import gzip
import inspect
import json
from pathlib import Path

import numpy as np


def encode(value):
    if isinstance(value, np.ndarray):
        return {'$': 'array', 'dtype': str(value.dtype), 'value': value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {'$': 'bytes', 'value': value.hex()}
    if isinstance(value, tuple):
        return {'$': 'tuple', 'value': [encode(item) for item in value]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if type(value).__name__ == 'InHand':
        return {'$': 'InHand', 'value': encode(value.recipe)}
    if type(value).__name__ == 'FlatReleaseGeometry':
        # These caches contain native objects and hand identity keys. Replay
        # rebuilds them from the same live hulls and logical release inputs.
        return {'$': 'FlatReleaseGeometry', 'value': encode(dict(
            obstacles=value.obstacles, wall_height=value.wall_height,
            released=value.released, backoff=value.backoff, ceiling=value.ceiling,
            blocked_by=sorted(value.blocked_by)))}
    if type(value).__name__ == 'FCLObject':
        shapes = []
        for shape, pose in zip(value.shapes, value.shape_poses):
            geometry = shape.get_collision_geometry()
            if type(geometry).__name__ != 'Convex':
                raise TypeError(f'cannot record FCLObject geometry {type(geometry).__name__}')
            faces = np.asarray(geometry.get_faces(), dtype=np.int32).reshape(-1, 4)
            if np.any(faces[:, 0] != 3):
                raise TypeError('cannot record FCLObject with nontriangular convex faces')
            shapes.append(dict(vertices=np.asarray(geometry.get_vertices()),
                               faces=faces[:, 1:],
                               pose=pose.to_transformation_matrix()))
        return {'$': 'FCLObject', 'value': encode(dict(name=value.name,
            pose=value.pose.to_transformation_matrix(), shapes=shapes))}
    if is_dataclass(value):
        return {'$': type(value).__name__, 'value': {f.name: encode(getattr(value, f.name)) for f in fields(value)}}
    if callable(value):
        return {'$': 'callback', 'name': value.__qualname__}
    if type(value).__name__ in ('Pick', 'Place', 'Handover'):
        return {'$': type(value).__name__, 'value': encode(vars(value))}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'cannot record {type(value).__name__}')


def decode(value, ctx=None):
    if isinstance(value, list):
        return [decode(item, ctx) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get('$')
    if not kind:
        return {key: decode(item, ctx) for key, item in value.items()}
    if kind == 'array':
        return np.asarray(value['value'], dtype=value['dtype'])
    if kind == 'bytes':
        return bytes.fromhex(value['value'])
    if kind == 'tuple':
        return tuple(decode(item, ctx) for item in value['value'])
    if kind == 'InHand':
        return ctx.skills().restore_need(decode(value['value'], ctx))
    if kind == 'FlatReleaseGeometry':
        from table_1000.experts.skills.place.flat import FlatReleaseGeometry
        arguments = decode(value['value'], ctx)
        arguments['blocked_by'] = set(arguments['blocked_by'])
        return FlatReleaseGeometry(**arguments)
    if kind == 'FCLObject':
        from mplib.collision_detection import fcl
        from table_1000.planning.world import pose
        arguments = decode(value['value'], ctx)
        shapes = [fcl.CollisionObject(fcl.Convex(shape['vertices'], shape['faces']))
                  for shape in arguments['shapes']]
        return fcl.FCLObject(arguments['name'], pose(arguments['pose']), shapes,
                             [pose(shape['pose']) for shape in arguments['shapes']])
    from table_1000.experts.skills.hold.frames import Box, Cylinder
    from table_1000.experts.skills.common import Target
    return {'Box': Box, 'Cylinder': Cylinder, 'Target': Target}[kind](**decode(value['value'], ctx))


def planner_state(ctx):
    return {arm: {'attached': encode(solver.attachments),
                  'poses': encode(solver.world.poses), 'frame': encode(solver.world.frame),
                  'qpos': encode(solver.planner.robot.get_qpos()),
                  'permissions': [sorted(pair) for pair in sorted(solver.world.local_permissions, key=lambda p: sorted(p))],
                  'shared_permissions': [sorted(pair) for pair in sorted(solver.world.shared_permissions, key=lambda p: sorted(p))],
                  'padding': solver.world.padding, 'object_padding': solver.world.object_padding.copy(),
                  'extra_objects': [dict(name=obj.name, body=obj.body, raw=encode(obj.raw), padded=obj.padded,
                                         pose=encode(solver.world.poses[name]))
                                    for name, obj in solver.world.objects.items()
                                    if not obj.body or obj.body.split(':')[0] not in ctx.scene.OBJECTS]}
            for arm, solver in ctx.solvers.items()}


class CallTrace:
    def __init__(self, ctx, seed, attempt, repeat=0):
        self.ctx, self.seed, self.attempt, self.repeat = ctx, seed, attempt, repeat
        self.frames, self.stack = [], []
        self.failure_frame = None
        self.replay_entry = None
        self.problems = None
        self.retry_path = []

    @contextmanager
    def frame(self, kind, name, arguments=None, concrete=False):
        ctx = self.ctx
        number = len(self.frames)
        entry = {'id': number, 'parent': self.stack[-1] if self.stack else None, 'kind': kind, 'name': name,
                 'arguments': encode(arguments or {}), 'enter': ctx.steps, 'exit': None, 'current': ctx.current, 'result': None,
                 'held': encode(ctx._skills.held if ctx._skills else {}),
                 'reaches': encode(ctx._skills.reaches if ctx._skills else {}), 'planners': planner_state(ctx),
                 'rng': encode(ctx.rng.bit_generator.state), 'targets': encode(ctx.env.targets()), 'concrete': concrete}
        if kind == 'skill':
            import mplib.pymp
            entry['mplib_seed'] = int(np.random.SeedSequence([self.seed, self.attempt, number, self.repeat, *self.retry_path]).generate_state(1)[0]) | 1
            if self.retry_path:
                entry['retry_path'] = list(self.retry_path)
            if self.replay_entry is not None and concrete:
                saved = self.replay_entry
                self.replay_entry = None
                if not self.repeat:
                    ctx.rng.bit_generator.state = decode(saved['rng'])
                    entry['mplib_seed'] = saved['mplib_seed']
                entry['rng'] = encode(ctx.rng.bit_generator.state)
                entry['source_id'] = saved['id']
            mplib.pymp.set_global_seed(entry['mplib_seed'])
            if self.problems:
                self.problems.events.append({'seed': entry['mplib_seed'], 'frame': number})
        self.frames.append(entry)
        self.stack.append(number)
        try:
            yield entry
        except BaseException as error:
            from table_1000.experts.context import Unreachable
            entry['exception'] = {'type': type(error).__name__, 'reason': getattr(error, 'reason', str(error)),
                                  'details': encode(getattr(error, 'details', {})), 'unreachable': isinstance(error, Unreachable)}
            if not hasattr(error, 'skill_frame_id'):
                error.skill_frame_id = number
            self.failure_frame = error.skill_frame_id
            raise
        finally:
            entry['exit'] = ctx.steps
            self.stack.pop()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wt') as file:
            json.dump({'version': 1, 'failure_frame': self.failure_frame, 'frames': self.frames}, file, separators=(',', ':'))

    def failed_branch(self, error, control):
        """Freeze a caught failure before retry unwinds its still-active control frame.

        Successful cleanup may have cleared ``failure_frame``. The exception retains
        its originating frame; a failure in the callback itself belongs to the control
        frame, whose concrete children remain available for skill-only replay.
        """
        from table_1000.experts.context import Unreachable
        frames = deepcopy(self.frames)
        failed = getattr(error, 'skill_frame_id', None)
        if failed is None or not 0 <= failed < len(frames):
            failed = control
        frames[failed]['exception'] = dict(type=type(error).__name__, reason=getattr(error, 'reason', str(error)),
                                            details=encode(getattr(error, 'details', {})),
                                            unreachable=isinstance(error, Unreachable))
        for entry in frames:
            if entry['exit'] is None:
                entry['exit'] = self.ctx.steps
        return dict(version=1, failure_frame=failed, frames=frames)


def frame(ctx, kind, name, arguments=None, concrete=False):
    trace = getattr(ctx, 'trace', None)
    return trace.frame(kind, name, arguments, concrete) if trace else nullcontext({})


def traced(concrete=True, action=False):
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def call(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            ctx = arguments.pop('ctx')
            instance = arguments.pop('self', None)
            if instance is not None:
                arguments['action'] = instance
            if not getattr(ctx, 'trace', None):
                return function(*args, **kwargs)
            with frame(ctx, 'skill', type(instance).__name__ if action else function.__name__, arguments, concrete) as entry:
                result = function(*args, **kwargs)
                entry['result'] = encode(result)
                # A caught failed attempt is not the failure path of a successful parent.
                ctx.trace.failure_frame = None
                return result
        return call
    return decorate


def read_trace(batch, report):
    with gzip.open(Path(batch) / report['call_trace'], 'rt') as file:
        trace = json.load(file)
    offset = report.get('call_trace_offset', 0)
    for entry in trace['frames']:
        entry['enter'] += offset
        if entry['exit'] is not None:
            entry['exit'] += offset
    return trace


def select_frame(trace, selected=None):
    """Innermost evaluated skill on the failed path; a check uses its last skill."""
    frames = trace['frames']
    root = trace['failure_frame'] if selected is None else selected
    ancestors = set()
    cursor = root
    while cursor is not None:
        ancestors.add(cursor)
        cursor = frames[cursor]['parent']
    if selected is None:
        candidates = [f for f in frames if f['id'] in ancestors and f['concrete']]
        if candidates:
            return candidates[-1], False
    descendants = {root}
    for f in frames:
        if f['parent'] in descendants:
            descendants.add(f['id'])
    candidates = [f for f in frames if f['id'] in descendants and f['concrete']]
    failed = trace['failure_frame']
    check = frames[root].get('phase') == 'check' or (
        failed in ancestors and frames[failed].get('phase') == 'check')
    return candidates[-1], check


def restore(ctx, entry, restore_rng=True):
    from table_1000.planning.world import PlanningObject
    skills = ctx.skills()
    skills.held = decode(entry['held'], ctx)
    skills.reaches = decode(entry['reaches'], ctx)
    for arm, state in entry['planners'].items():
        solver = ctx.solver(arm)
        for obj in state['extra_objects']:
            solver.world.add(PlanningObject(obj['name'], obj['body'], decode(obj['raw']), obj['padded']), decode(obj['pose']))
        solver.world.set_shared_permissions(())
        for pair in list(solver.world.permissions):
            names = sorted(pair)
            solver.world.allow(names[:1], names[1:] or names[:1], False)
        # IK can run before sync(): preserve the exact planning world seen by the
        # original call, including obstacles last updated by an earlier plan.
        solver.world.frame = decode(state['frame'])
        solver.world.set_padding(state['padding'])
        for name, pose in decode(state['poses']).items():
            solver.world.set_pose(name, pose)
        for name, padding in state['object_padding'].items():
            default = solver.world.padding
            solver.world.padding = padding
            solver.world.set_pose(name, solver.world.poses[name])
            solver.world.padding = default
        for name, relative in decode(state['attached']).items():
            solver.attach_at(name, relative)
        shared = {frozenset(pair) for pair in state['shared_permissions']}
        for pair in state['permissions']:
            solver.world.allow(pair[:1], pair[1:] or pair[:1])
        solver.world.set_shared_permissions(shared)
        solver.planner.robot.set_qpos(decode(state['qpos']), True)
    ctx.current = entry['current']
    ctx.env.contacts.phase = ctx.current
    if restore_rng:
        import mplib.pymp
        ctx.rng.bit_generator.state = decode(entry['rng'])
        mplib.pymp.set_global_seed(entry['mplib_seed'])


def replay(ctx, snapshot, expert):
    from table_1000.experts import skills
    entry = snapshot['frame']
    restore(ctx, entry, restore_rng=not ctx.trace.repeat)
    ctx.trace.replay_entry = entry
    step = expert.STEPS[snapshot['step'] - 1]
    with frame(ctx, 'step', step.name, {'index': snapshot['step']}) as step_frame:
        result = getattr(skills, entry['name'])(ctx, **decode(entry['arguments'], ctx))
        if snapshot['check']:
            step_frame['phase'] = 'check'
            ctx.current = step.name
            step.check(ctx)
    return result


def snapshot_state(batch, report, selected=None):
    """Scene and evaluated skill at entry, including commands and the optional step check."""
    import h5py
    from table_1000.experts.teleop import trajectory_state
    from table_1000.scenes.initial import read_tree
    batch = Path(batch)
    trace = read_trace(batch, report)
    entry, check = select_frame(trace, selected)
    path = batch / 'trajectories_failed.hdf5'
    if report.get('call_trace_initial_state') and entry['enter'] == report['call_trace_offset']:
        with h5py.File(batch / report['call_trace_initial_state'], 'r') as file:
            state = read_tree(file['initial_state'])
    elif entry['enter']:
        state = trajectory_state(path, report['seed'], entry['enter'] - 1)
    else:
        with h5py.File(path, 'r') as file:
            state = read_tree(file[f"data/demo_{report['seed']}/initial_state"])
    state['targets'] = decode(entry['targets'])
    owner = entry
    while owner['kind'] != 'step':
        owner = trace['frames'][owner['parent']]
    snapshot = {'batch': str(batch), 'frame': entry, 'check': check, 'step': owner['arguments']['index'],
                'failed_frame_id': trace['failure_frame'], 'failure': report['failure']}
    state['_expert_frame'] = np.bytes_(json.dumps(snapshot))
    return state, snapshot
