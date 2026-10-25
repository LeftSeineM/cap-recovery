"""Recorded numerical solve problems; usable with MPlib without a simulator."""

from contextlib import contextmanager
from functools import wraps
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from table_1000.experts.trace import decode, encode


def packed(value):
    if isinstance(value, set):
        return [packed(item) for item in sorted(value, key=repr)]
    if isinstance(value, dict):
        return {str(key): packed(item) for key, item in value.items()}
    return encode(value)


def summary(result):
    if isinstance(result, tuple):
        return {'status': result[0], 'q': encode(result[1])}
    kept = {key: encode(value) for key, value in result.items()
            if key not in ('position', 'velocity', 'acceleration', 'time')}
    if result['status'] == 'Success':
        path = np.asarray(result['position'])
        kept.update(waypoints=len(path), path_length=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()))
    return kept


class Problems:
    """One episode's compressed, shared world snapshots and chronological solver calls."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.objects, self.worlds, self.operations, self.decisions, self.events = {}, {}, [], [], []
        self.stack, self.solving = [], []
        self._objects, self._methods, self._diagnostics = {}, {}, {}
        self.document = {'version': 2, 'robot': ctx.spec.name, 'objects': self.objects, 'worlds': self.worlds,
                         'operations': self.operations, 'decisions': self.decisions, 'events': self.events}
        self.document['table'] = packed(getattr(getattr(ctx, 'scene', None), 'TABLE', {}))

    def install(self):
        for name in ('ik', 'plan_rrt', 'plan_screw'):
            function = getattr(self.ctx.model, name)
            self._methods[name] = function
            setattr(self.ctx.model, name, self.observe(name, function))

    def close(self):
        for name, function in self._methods.items():
            setattr(self.ctx.model, name, function)

    def intern(self, table, value):
        key = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]
        table.setdefault(key, value)
        return key

    def world(self, arm):
        solver = self.ctx.solvers[arm]
        world = solver.world
        objects = {}
        for name, obj in world.objects.items():
            if obj not in self._objects:
                self._objects[obj] = self.intern(self.objects, dict(name=name, body=obj.body, raw=encode(obj.raw), padded=obj.padded))
            objects[name] = self._objects[obj]
        return self.intern(self.worlds, dict(objects=objects, poses=encode(world.poses), frame=encode(world.frame),
                                             padding=world.padding, attached=encode(solver.attachments),
                                             object_padding=world.object_padding.copy(),
                                             permissions=sorted(sorted(pair) for pair in world.permissions),
                                             robot_q=encode(solver.planner.robot.get_qpos())))

    @contextmanager
    def operation(self, kind, arm, arguments):
        model, trace = self.ctx.model, self.ctx.trace
        owner = self.decisions[self.stack[-1]] if self.stack else {}
        entry = dict(id=len(self.operations), parent=self.solving[-1] if self.solving else None, kind=kind, arm=arm,
                     frame=trace.stack[-1] if trace.stack else None, decision=owner.get('id'),
                     candidate=owner.get('candidate'), stage=owner.get('stage'), step=self.ctx.steps,
                     arguments=encode(arguments), q=encode(model.q), base=encode(model.base), world=self.world(arm))
        self.operations.append(entry)
        if not self.solving:
            self.events.append({'operation': entry['id']})
        self.solving.append(entry['id'])
        started = time.perf_counter()
        try:
            yield entry
        except BaseException as error:
            entry['exception'] = dict(type=type(error).__name__, reason=getattr(error, 'reason', str(error)),
                                      details=packed(getattr(error, 'details', {})))
            raise
        finally:
            entry['wall_s'] = time.perf_counter() - started
            self.solving.pop()

    def observe(self, name, function):
        signature = inspect.signature(function)
        @wraps(function)
        def call(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            arm = arguments.pop('arm')
            arguments.pop('check', None)
            with self.operation(name, arm, arguments) as entry:
                result = function(*args, **kwargs)
                entry['result'] = summary(result)
                if name == 'ik' and result[0] != 'Success' and arguments['check_collision']:
                    self.diagnose(entry, function, arguments)
                return result
        return call

    def diagnose(self, entry, ik, arguments):
        """A bounded collision-free CLIK probe uses its own NumPy stream, preserving MPlib's.

        RobotModel's check_collision=False branch is deterministic and restores model.q;
        preserve the actual planner pose too, since an IK call may have left another pose.
        """
        owner = entry['decision']
        while owner is not None and self.decisions[owner]['name'].split('.')[-1] in ('lower', 'in_reach_down', 'down_goals', 'why'):
            owner = self.decisions[owner]['parent']
        key = (owner, entry['stage'])
        count = self._diagnostics.get(key, 0)
        self._diagnostics[key] = count + 1
        if count >= 5:
            return
        planner = self.ctx.model.planners[entry['arm']]
        previous = planner.robot.get_qpos().copy()
        try:
            status, q = ik(entry['arm'], **dict(arguments, check_collision=False))
            entry['diagnostic'] = dict(without_collision=status)
            if status == 'Success':
                entry['diagnostic']['collisions'] = self.ctx.model.collisions(entry['arm'], q)
        finally:
            planner.robot.set_qpos(previous, True)

    @contextmanager
    def decision(self, name, arguments):
        parent = self.decisions[self.stack[-1]] if self.stack else {}
        entry = dict(id=len(self.decisions), parent=self.stack[-1] if self.stack else None,
                     frame=self.ctx.trace.stack[-1] if self.ctx.trace.stack else None,
                     name=name, arguments=packed(arguments), events=[], candidate=parent.get('candidate'), stage=None)
        if hasattr(self.ctx, 'model'):
            entry['state'] = dict(q=encode(self.ctx.model.q), base=encode(self.ctx.model.base),
                                  worlds={arm: self.world(arm) for arm in self.ctx.solvers},
                                  poses=encode(self.ctx.poses()), rng=encode(self.ctx.rng.bit_generator.state))
            skill = self.ctx._skills
            if skill is not None:
                entry['state'].update(held=encode(skill.held), home=encode(skill.home), start_base=encode(skill.start_base),
                                      reaches=encode(skill.reaches))
        self.decisions.append(entry)
        self.events.append({'decision': entry['id']})
        with self.active(entry):
            try:
                yield entry
            except BaseException as error:
                entry['exception'] = dict(type=type(error).__name__, reason=getattr(error, 'reason', str(error)),
                                          details=packed(getattr(error, 'details', {})))
                raise

    @contextmanager
    def active(self, entry):
        self.stack.append(entry['id'])
        try:
            yield entry
        finally:
            self.stack.pop()

    def mark(self, stage, candidate=None, outcome=None, **values):
        if not self.stack:
            return
        entry = self.decisions[self.stack[-1]]
        entry['stage'] = stage
        if candidate is not None:
            entry['candidate'] = str(candidate)
        entry['events'].append(dict(stage=stage, candidate=entry['candidate'], outcome=outcome, **packed(values)))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.document['ik_diagnostics'] = [dict(decision=owner, stage=stage, failed=count, examples=min(5, count))
                                            for (owner, stage), count in self._diagnostics.items()]
        with gzip.open(path, 'wt', compresslevel=6) as file:
            json.dump(self.document, file, separators=(',', ':'))


def recorder(skill):
    return getattr(getattr(skill.ctx, 'trace', None), 'problems', None)


def mark(skill, stage, candidate=None, outcome=None, **values):
    record = recorder(skill)
    if record:
        record.mark(stage, candidate, outcome, **values)


def decision(function):
    """Observe a decision without changing candidate generation, budgets or random draws."""
    signature = inspect.signature(function)
    def arguments(args, kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop('self')
        return values
    if inspect.isgeneratorfunction(function):
        @wraps(function)
        def iterate(self, *args, **kwargs):
            record = recorder(self)
            if not record:
                yield from function(self, *args, **kwargs)
                return
            with record.decision(function.__qualname__, arguments((self, *args), kwargs)) as entry:
                generator = function(self, *args, **kwargs)
            try:
                while True:
                    with record.active(entry):
                        try:
                            value = next(generator)
                        except StopIteration:
                            return
                    yield value
            finally:
                generator.close()
        return iterate
    @wraps(function)
    def call(self, *args, **kwargs):
        record = recorder(self)
        if not record:
            return function(self, *args, **kwargs)
        with record.decision(function.__qualname__, arguments((self, *args), kwargs)) as entry:
            result = function(self, *args, **kwargs)
            entry['result'] = packed(result)
            return result
    return call


def load(path):
    with gzip.open(path, 'rt') as file:
        return json.load(file)


class Offline:
    """Reconstruct numerical planning worlds and solve recorded calls with the production model."""

    def __init__(self, document, output, root=None):
        from table_1000.robots import ROOT, load_spec
        from table_1000.robots.kinematics import RobotModel
        self.document = document
        self.model = RobotModel(load_spec(document['robot'], root or ROOT), Path(output), root or ROOT)
        self.solvers = {}
        self.objects = {}

    def restore(self, entry):
        from table_1000.experts.solver import MotionSolver
        from table_1000.planning.world import PlanningObject, pose
        arm, model = entry['arm'], self.model
        if arm not in self.solvers:
            ctx = SimpleNamespace(model=model, spec=model.spec)
            self.solvers[arm] = MotionSolver(ctx, arm)
        solver = self.solvers[arm]
        for pair in list(solver.world.permissions):
            values = sorted(pair)
            solver.world.allow(values[:1], values[-1:], False)
        solver.world.clear()
        solver.attachments.clear()
        model.q, model.base = decode(entry['q']), decode(entry['base'])
        world = self.document['worlds'][entry['world']]
        solver.world.frame = decode(world['frame'])
        solver.world.padding = world['padding']
        poses = decode(world['poses'])
        for name, key in world['objects'].items():
            obj = self.document['objects'][key]
            if key not in self.objects:
                self.objects[key] = PlanningObject(name, obj['body'], decode(obj['raw']), obj['padded'])
            solver.world.padding = world['object_padding'][name]
            solver.world.add(self.objects[key], poses[name])
        solver.world.padding = world['padding']
        for name, relative in decode(world['attached']).items():
            solver.attach_at(name, relative)
        for pair in world['permissions']:
            solver.world.allow(pair[:1], pair[-1:])
        solver.planner.robot.set_qpos(decode(world['robot_q']), True)
        return solver

    def solve(self, entry):
        solver = self.restore(entry)
        arguments = decode(entry['arguments'])
        if entry['kind'] == 'motion':
            solver.overlaps = arguments.pop('start_allowed')
            kind = arguments.pop('kind')
            validation_step = arguments.pop('validation_step', None)
            for key in ('attempt', 'attempts'):
                arguments.pop(key, None)
            if kind == 'rrt':
                arguments['check'] = lambda rows: solver.check_geometry(rows, validation_step)
            result = getattr(self.model, f'plan_{kind}')(entry['arm'], **arguments)
            if kind == 'rrt':
                result = solver._validated(result, validation_step)
        else:
            if entry['kind'] == 'plan_rrt':
                arguments['check'] = solver.check_geometry
            result = getattr(self.model, entry['kind'])(entry['arm'], **arguments)
        return summary(result)

    def replay(self, selected=None, seed=None):
        """Replay the original random stream prefix, or use a new seed for each selected problem.

        Each solve starts at its recorded measured state. This checks the recorded candidate
        graph; it does not invent unrecorded downstream tests after a rejected candidate passes.
        """
        import mplib.pymp
        selected = set(range(len(self.document['operations']))) if selected is None else set(selected)
        results = []
        for event in self.document['events']:
            if 'decision' in event:
                continue
            if 'seed' in event:
                if seed is None:
                    mplib.pymp.set_global_seed(event['seed'])
                continue
            entry = self.document['operations'][event['operation']]
            if seed is not None and entry['id'] not in selected:
                continue
            if seed is not None:
                mplib.pymp.set_global_seed(int(np.random.SeedSequence([seed, entry['id']]).generate_state(1)[0]) | 1)
            result = self.solve(entry)
            if entry['id'] in selected:
                results.append(dict(operation=entry['id'], frame=entry['frame'], decision=entry['decision'],
                                    candidate=entry['candidate'], stage=entry['stage'], original=entry['result'],
                                    result=result, same_status=result['status'] == entry['result']['status'],
                                    same_reason=result.get('reason', result['status']) ==
                                    entry['result'].get('reason', entry['result']['status'])))
            if selected and entry['id'] >= max(selected):
                break
        return results

    def decide(self, number, seed=None):
        """Run an evaluated decision again, with ideal joint following instead of simulation.

        Candidate generation, IK, requirements and planning use the expert's own methods.
        Contact dynamics, tracking and checks of physical outcomes are outside this solve.
        """
        import mplib.pymp
        from table_1000.experts.context import ExpertFailed, DT
        from table_1000.experts.grasps import Hand
        from table_1000.experts.skills import Skills
        from scipy.spatial.transform import Rotation
        entry = self.document['decisions'][number]
        if seed is None:
            for event in self.document['events']:
                if event.get('decision') == number:
                    break
                if 'seed' in event:
                    mplib.pymp.set_global_seed(event['seed'])
                elif 'operation' in event:
                    self.solve(self.document['operations'][event['operation']])
        else:
            mplib_seed = int(np.random.SeedSequence(seed).generate_state(1)[0]) | 1
            mplib.pymp.set_global_seed(mplib_seed)
        state, model = entry['state'], self.model
        for arm, world in state['worlds'].items():
            self.restore(dict(arm=arm, world=world, q=state['q'], base=state['base']))
        poses = decode(state['poses'])
        ctx = SimpleNamespace(model=model, spec=model.spec, solvers=self.solvers, steps=0, current=entry['name'],
                              scene=SimpleNamespace(TABLE=decode(self.document['table'])), rng=np.random.default_rng(),
                              trace=SimpleNamespace(stack=[entry['frame']]), events=[])
        ctx.rng.bit_generator.state = decode(state['rng'])
        if seed is not None:
            ctx.rng = np.random.default_rng(seed)
        def base_pose():
            return np.r_[model.base[:2, 3], Rotation.from_matrix(model.base[:3, :3]).as_euler('xyz')[2]]
        ctx.env = SimpleNamespace(qpos=lambda names: np.array([[model.q[model.index[name]] for name in names]]),
                                  obs={'agent': {'base_pose': base_pose()[None]}})
        def step(action=None):
            values = {}
            for name, target in (action or {}).items():
                target = np.asarray(target)[0]
                if name in model.spec.arms:
                    values.update(zip(model.spec.arms[name].joints, target))
                elif name.endswith('_gripper'):
                    values.update(model.spec.arms[name.removesuffix('_gripper')].finger_targets(float(target[0])))
                elif name == 'lift':
                    values.update(zip(model.spec.lift, target))
                elif name == 'base':
                    x, y, yaw = base_pose()
                    model.base[:2, 3] = [x + target[0] * np.cos(yaw) * DT, y + target[0] * np.sin(yaw) * DT]
                    model.base[:3, :3] = Rotation.from_euler('z', yaw + target[1] * DT).as_matrix()
            model.set_state(values)
            ctx.env.obs['agent']['base_pose'] = base_pose()[None]
            for arm, solver in self.solvers.items():
                for name, relative in solver.attachments.items():
                    body = solver.world.objects[name].body
                    if body:
                        poses[body] = model.tcp(arm, world=True) @ relative
            ctx.steps += 1
        ctx.step, ctx.poses = step, lambda: poses
        ctx.hold = lambda seconds: None
        ctx.note = lambda name, **values: ctx.events.append(dict(name=name, **values))
        def fail(reason, failure=ExpertFailed, **details):
            raise failure(ctx.current, reason, **details)
        ctx.fail = fail
        skill = object.__new__(Skills)
        skill.ctx, skill.model, skill.solvers = ctx, model, self.solvers
        skill.arms = list(model.spec.arms)
        skill.hands = {arm: Hand(model, arm) for arm in skill.arms}
        skill.held, skill.home = decode(state['held']), decode(state['home'])
        skill.start_base, skill.reaches = decode(state['start_base']), decode(state['reaches'])
        skill.log, skill.rejected = [], {}
        ctx._skills, ctx.skills = skill, lambda: skill
        for solver in self.solvers.values():
            solver.ctx = ctx
        arguments = decode(entry['arguments'], ctx)
        if 'skip' in arguments:
            arguments['skip'] = set(map(tuple, arguments['skip']))
        name = entry['name'].split('.')[-1]
        if name == 'options':
            arguments['tried'] = None
        record = Problems(ctx)
        ctx.trace.problems = record
        record.install()
        try:
            result = getattr(skill, name)(**arguments)
            if name in ('options', 'handoffs'):
                result = next(result, None)
            success = (bool(result.get('done')) if name == 'put_down' else
                       bool(result) if isinstance(result, (bool, np.bool_)) else result is not None)
            outcome = dict(success=success, result=packed(result), rejected=packed(skill.rejected))
        except ExpertFailed as error:
            outcome = dict(success=False, reason=error.reason, details=packed(error.details))
        finally:
            record.close()
        outcome.update(seed=seed, mplib_seed=mplib_seed if seed is not None else None)
        outcome['problems'] = record.document
        return outcome
