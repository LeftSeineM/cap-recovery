"""What a scene's ``solve(ctx)`` gets: the environment, motion solvers, geometry queries and subtasks.

The expert is plain blocking code: every motion steps the environment until it
is done. ``ctx`` offers

- **motion** -- :meth:`Context.solver` per arm (:class:`~table_1000.experts.solver.MotionSolver`),
  :meth:`Context.step`, :meth:`Context.hold`; the atomic skills of
  :mod:`table_1000.experts.skills` (grasp, pick, place, ...) on top of them;
- **state and geometry queries** -- live poses of object bodies, object joint
  values and centres of mass, each object's
  :class:`~table_1000.scenes.geometry.AssetGeometry`, the TCP;
- **subtasks** -- ``with ctx.subtask('open the bottom drawer of the cabinet'):``
  marks the steps of one English instruction; :class:`Step` groups subtasks into steps
  that start and end with the hands empty, each with a check of the state it leaves
  (:func:`run_steps`); after each checked step the scene state is kept as a checkpoint
  a later run resumes from (``ctx.only`` runs some of the steps);
- ``ctx.rng`` -- the episode's random generator (from the seed), for choices that
  vary the trajectory.

Failures raise :class:`ExpertFailed` (:class:`Unreachable` where nothing is in
reach) with the subtask and a reason; nothing is retried unless the expert does it.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from table_1000.scenes import robot_config

import time
from typing import Callable

from table_1000.common.timing import WallProfile, timed

import numpy as np
from scipy.spatial.transform import Rotation

from table_1000.experts.trace import frame
from table_1000.scenes.geometry import AssetGeometry
from table_1000.sim import CONTROL_HZ

DT = 1 / CONTROL_HZ


class ExpertFailed(RuntimeError):
    def __init__(self, subtask, reason, **details):
        super().__init__(f'{subtask}: {reason}')
        self.subtask, self.reason, self.details = subtask, reason, details


class WallBudgetExceeded(ExpertFailed):
    """A caller's cooperative wall deadline expired between planning operations."""


def remaining_wall(ctx):
    """Seconds left in an optional ``ctx.wall_deadline``; fail when it expires."""
    deadline = getattr(ctx, 'wall_deadline', None)
    if deadline is None:
        return None
    remaining = deadline - time.perf_counter()
    if remaining <= 0.:
        raise WallBudgetExceeded(getattr(ctx, 'current', None), 'wall budget exceeded')
    return remaining


@dataclass
class Step:
    """A step of an expert, between two states in which the hands hold nothing: ``run(ctx)`` does
    its subtasks (:meth:`Context.subtask`); ``check(ctx)`` fails it (:meth:`Context.fail`) unless
    the state it leaves is fit for the next step."""
    name: str
    run: Callable
    check: Callable = None


def run_steps(ctx, steps):
    """Run ``steps`` (:class:`Step`) in order, each checked when done and then kept as a checkpoint
    (:meth:`Context.checkpoint`); only steps ``ctx.only`` (``(first, last)``, from 1) when set."""
    first, last = ctx.only or (1, len(steps))
    if not 1 <= first <= last <= len(steps):
        raise ValueError(f'steps {first}-{last} of an expert with {len(steps)}')
    ctx.step_names = [step.name for step in steps]
    for index, step in enumerate(steps[first - 1:last], first):
        with frame(ctx, 'step', step.name, {'index': index}) as entry:
            step.run(ctx)
            if step.check:
                entry['phase'] = 'check'
                ctx.current = step.name
                try:
                    step.check(ctx)
                finally:
                    ctx.current = None
        ctx.checkpoint(index, step.name)


class Unreachable(ExpertFailed):
    """Nothing the step could do is in reach (no grasp, push or placement has a motion)."""


def steps(seconds):
    return max(1, round(seconds / DT))


class Context:
    def __init__(self, env, model, scene, assets, rng, start=None, only=None):
        self.env, self.model, self.scene, self.assets, self.rng = env, model, scene, Path(assets), rng
        self.spec = env.spec
        self.geometry = {name: AssetGeometry.of(assets, spec['asset']) for name, spec in scene.OBJECTS.items()}
        self.subtasks, self.current = [], None
        self.solvers = {}
        self._skills = None
        self.trace = None
        self.events = []        # (step, note) along the way, for the episode report
        self.origin = env.steps  # the episode starts here, after settling
        self.started, self.cpu_started = time.perf_counter(), time.process_time()
        self.subtask_wall_s = []     # wall seconds each subtask took, in the order of subtasks
        self.profile = WallProfile()
        self.subtask_profiles = []
        self.observers = []      # called after every step (e.g. video capture)
        # The robot where the seed's episode began ({'qpos', 'base_pose'}): its arms' home and its
        # base's start, also when this run resumes from a checkpoint.
        self.start = start or self.robot_state()
        self.only = only            # (first, last) steps to run (see run_steps)
        self.step_names = None      # the expert's steps, once run_steps runs
        self.requirements = []    # live, reconstructible InHand requirements
        self.retry_archives = []  # all discarded branches, including nested retries
        self.checkpoints = []       # {'step', 'name', 'at', 'wall_s', 'cpu_s', 'state'} after each checked step
        self.sync_model()

    # -- stepping ------------------------------------------------------------------
    @property
    def steps(self):
        """Control steps since the episode started."""
        return self.env.steps - self.origin

    def robot_state(self):
        return {'qpos': self.env.obs['agent']['qpos'][0].copy(), 'base_pose': self.env.obs['agent']['base_pose'][0].copy()}

    def base_transform(self):
        x, y, yaw = self.env.obs['agent']['base_pose'][0]
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
        transform[:2, 3] = x, y
        if not self.spec.base:
            transform[2, 3] = robot_config(self.scene, self.spec.name)['position'][2]
        return transform

    def sync_model(self):
        """The planning model follows the measured joints, as on the real robot."""
        self.model.set_state(dict(zip(self.env.joints, self.env.obs['agent']['qpos'][0])), self.base_transform())

    @timed('control_step')
    def step(self, action=None):
        remaining_wall(self)
        obs = self.env.step(action or {})
        self.sync_model()
        for observer in self.observers:
            observer(self)
        return obs

    def hold(self, seconds):
        for _ in range(steps(seconds)):
            self.step()

    def solver(self, arm=None):
        from table_1000.experts.solver import MotionSolver
        arm = arm or self.spec.primary_arm
        if arm not in self.solvers:
            self.solvers[arm] = MotionSolver(self, arm)
        return self.solvers[arm]

    def skills(self):
        """The :class:`~table_1000.experts.skills.Skills` of every arm (made on first use)."""
        from table_1000.experts.skills import Skills
        if self._skills is None:
            self._skills = Skills(self)
        return self._skills

    # -- subtasks ------------------------------------------------------------------
    @contextmanager
    def subtask(self, instruction):
        """Steps taken inside belong to ``instruction``; contacts are labelled with it."""
        entry = {'instruction': instruction, 'start': self.steps, 'end': None}
        self.subtasks.append(entry)
        before = self.profile.snapshot()
        started = time.perf_counter()
        self.current = instruction
        self.env.contacts.phase = instruction
        try:
            with frame(self, 'subtask', instruction):
                yield entry
        finally:
            entry['end'] = self.steps
            self.subtask_wall_s.append(time.perf_counter() - started)
            self.subtask_profiles.append(self.profile.since(before))
            self.current = None
            self.env.contacts.phase = None

    def checkpoint(self, index, name):
        """Keep the scene state after step ``index`` (``name``) with the action targets
        (``state['targets']``), at control step :attr:`steps`, ``wall_s`` seconds since the context
        was made and ``cpu_s`` CPU seconds of the process since."""
        from table_1000.sim.isaaclab.replay import numpy_state
        state = {**numpy_state(self.env), 'targets': self.env.targets()}
        self.checkpoints.append({'step': index, 'name': name, 'at': self.steps,
                                 'wall_s': time.perf_counter() - self.started,
                                 'cpu_s': time.process_time() - self.cpu_started, 'state': state})
        self.checkpoints[-1]['wall_profile'] = self.profile.snapshot()

    def note(self, text, **values):
        self.events.append({'step': self.steps, 'subtask': self.current, 'note': text, **values})

    def fail(self, reason, failure=ExpertFailed, **details):
        raise failure(self.current, reason, **details)

    # -- state and geometry queries --------------------------------------------------
    def poses(self):
        """``{'<object>:<body>': world 4x4}`` of every object body (environment 0)."""
        return {key: value[0] for key, value in self.env.privileged()['poses'].items()}

    def pose(self, name, body=None):
        """World pose of object ``name``'s frame, or of its body ``body``."""
        if body is None:
            return self.env.object_pose(name)[0]
        return self.poses()[f'{name}:{body}']

    def joint(self, name, joint):
        return float(self.env.privileged()['joints'][name][joint][0])

    def center_of_mass(self, name, body):
        return self.env.center_of_mass(f'{name}:{body}')[0]

    def model_path(self, name):
        """The built ``model.json`` of object ``name`` (its collision geometry)."""
        return self.assets / self.scene.OBJECTS[name]['asset'] / 'model.json'

    def part_box(self, name, part, body):
        """``(T_world_box, size)`` of ``part``'s collision box, axis-aligned in ``body``'s frame, at
        the body's live pose."""
        low, high = self.geometry[name].bounds(part, body)
        box = np.eye(4)
        box[:3, 3] = (low + high) / 2
        return self.pose(name, body) @ box, high - low

    def tcp(self, arm=None):
        return self.env.obs['agent']['tcp'][arm or self.spec.primary_arm][0]
