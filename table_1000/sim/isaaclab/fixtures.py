"""Native six-axis fixtures controlled through PhysX tensors, in world frame.

A collision-free chain of world X/Y/Z prismatic joints and one spherical joint
provides the six coordinates of a held body. The spherical coordinates are a
world-frame rotation vector with three orthogonal motion axes, so no
orientation loses a rotational degree of freedom, unlike serial Euler joints.
Each virtual link has 10 microgram mass and 1e-11 kg m² inertia; gravity is
disabled. The held body is the terminal link, avoiding a mass-ratio-sensitive
external joint. Joints are never deleted and dynamic body poses never written.

:class:`Fixture` is the mechanism: a static rigid lock (equal limits), native
drives, or released (no limits or drives). :class:`Grip` tracks timed world
trajectories with those drives: position and velocity targets from the
trajectory, plus inverse-dynamics feedforward for gravity and acceleration of
the held tree, so a held body moves continuously rather than step by step.
Equal-limit locks cannot do that: moving them leaves the body at rest after
every step and, with written velocities, injects impulses into drawers.
"""

import numpy as np
from scipy.spatial.transform import Rotation

LINK_MASS = 1e-8
LINK_INERTIA = 1e-11
# Default grip stiffness k = M (STEPS/dt)², critical damping. The
# implicit drives stay stable; stiffness well above the held mass keeps light
# parts in place against contact loads, and velocity targets keep tracking
# continuous. Measured effective stiffness exceeds k on CPU. A stiffer angular
# drive makes GPU PhysX resolve joint friction downstream of the grip poorly
# (cabinet drawers slide early under a tilt), so rotation is kept softer.
LINEAR_STEPS, ANGULAR_STEPS = 10., 3.
DOFS = ('axis_0', 'axis_1', 'axis_2', 'rotation:0', 'rotation:1', 'rotation:2')


def quaternion_rotation(quaternion):
    return Rotation.from_quat(np.asarray(quaternion, dtype=float)[[1, 2, 3, 0]])


def rotation_quaternion(rotation):
    return rotation.as_quat()[[3, 0, 1, 2]]


class Fixture:
    """Fixture coordinates of one body: world translation, then world rotation vector.

    Coordinates are zero at the body's declared pose. The spherical joint
    reports positions as the principal rotation vector and velocities in the
    child (joint) frame.
    """

    def __init__(self, ctx, body, *, name, locked, position=None, rotation=None):
        from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
        assert not ctx.initialized, 'Declare fixtures before the first simulation step'
        self.ctx, self.body, self.name, self.locked, self.released = ctx, body, name, locked, False
        self.origin, declared = ctx.body_pose(body)
        native_position, native_quaternion = ctx.usd_body_pose(body)
        native = quaternion_rotation(native_quaternion)
        # Child joint frame in body coordinates: world-aligned at the declared
        # origin while the body has its native (authored) pose.
        self.local_rotation = native.inv()
        self.local_position = native.inv().apply(self.origin - native_position)
        self.coordinates = self.pose_coordinates(
            self.origin if position is None else position, declared if rotation is None else rotation)
        self.velocity_targets, self.actuation = np.zeros(6), np.zeros(6)
        self.stiffness, self.damping = np.zeros(6), np.zeros(6)
        self.max_forces = np.full(6, np.inf)
        self.path = ctx.session.environment(0) + '/Fixtures/' + name
        stage = ctx.session.stage
        # Changing the articulation root must retain the asset's solver and
        # collision settings, rather than silently taking PhysX defaults.
        source = stage.GetPrimAtPath(body.instance.articulation_root or body.path)
        source_api = (PhysxSchema.PhysxArticulationAPI(source) if body.instance.articulation_root
                      else PhysxSchema.PhysxRigidBodyAPI(source))
        position_iterations = source_api.GetSolverPositionIterationCountAttr().Get()
        velocity_iterations = source_api.GetSolverVelocityIterationCountAttr().Get()
        self_collisions = (source_api.GetEnabledSelfCollisionsAttr().Get()
                           if body.instance.articulation_root else False)
        UsdGeom.Xform.Define(stage, self.path)
        previous = None
        for i in range(4):
            path = f'{self.path}/link_{i}'
            link = UsdGeom.Xform.Define(stage, path)
            link.AddTranslateOp().Set(Gf.Vec3d(*self.origin))
            UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
            mass = UsdPhysics.MassAPI.Apply(link.GetPrim())
            mass.CreateMassAttr(LINK_MASS)
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(LINK_INERTIA))
            PhysxSchema.PhysxRigidBodyAPI.Apply(link.GetPrim()).CreateDisableGravityAttr(True)
            if i == 0:
                joint = UsdPhysics.FixedJoint.Define(stage, self.path+'/root_joint')
                joint.CreateBody1Rel().SetTargets([path])
                joint.CreateLocalPos0Attr(Gf.Vec3f(*self.origin))
                UsdPhysics.ArticulationRootAPI.Apply(joint.GetPrim())
                api = PhysxSchema.PhysxArticulationAPI.Apply(joint.GetPrim())
                api.CreateEnabledSelfCollisionsAttr(self_collisions)
                api.CreateSolverPositionIterationCountAttr(position_iterations)
                api.CreateSolverVelocityIterationCountAttr(velocity_iterations)
            else:
                joint = UsdPhysics.PrismaticJoint.Define(stage, f'{self.path}/axis_{i-1}')
                joint.CreateAxisAttr('XYZ'[i-1])
                joint.CreateBody0Rel().SetTargets([previous])
                joint.CreateBody1Rel().SetTargets([path])
                joint.CreateLowerLimitAttr(-1e10)
                joint.CreateUpperLimitAttr(1e10)
                drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
                drive.CreateTypeAttr('force')
            previous = path
        # Put the tested body in the reduced-coordinate tree: an external
        # fixed joint to a near-massless proxy would be ill-conditioned.
        joint = UsdPhysics.Joint.Define(stage, self.path+'/rotation')
        joint.CreateBody0Rel().SetTargets([previous])
        joint.CreateBody1Rel().SetTargets([body.path])
        for axis in ('transX', 'transY', 'transZ'):
            limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            limit.CreateLowAttr(1.)
            limit.CreateHighAttr(-1.)
        for axis in ('rotX', 'rotY', 'rotZ'):
            # PhysX accepts spherical limits only within [-pi, pi]; authoring
            # them selects the limited motion needed for a tensor lock.
            limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            limit.CreateLowAttr(-180.)
            limit.CreateHighAttr(180.)
            UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis).CreateTypeAttr('force')
        q = self.local_rotation.as_quat()
        joint.CreateLocalRot1Attr(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*map(float, self.local_position)))
        instance = body.instance
        if not hasattr(instance, "fixture_bodies"):
            instance.fixture_bodies = set()
        instance.fixture_bodies.add(body.name)
        if instance.articulation_root:
            stage.GetPrimAtPath(instance.articulation_root).RemoveAPI(UsdPhysics.ArticulationRootAPI)
            instance.fixture_root = self.path+'/root_joint'
        else:
            # The pen's radial joints stay maximal, with their original drives.
            for name in instance.joint_names:
                UsdPhysics.Joint(instance.joint(name)).CreateExcludeFromArticulationAttr(True)
        # Bodies whose mass the fixture carries: the whole reduced-coordinate
        # tree, or just the body when its other joints stay maximal.
        self.tree = list(instance.body_names) if instance.articulation_root else [body.name]
        self.view = None
        ctx.fixtures.append(self)

    # -- coordinates ---------------------------------------------------------------------
    def pose_coordinates(self, position, quaternion, reference=None):
        """Fixture coordinates placing the body origin at a world pose.

        The rotation vector is the equivalent (angle ± 2π) nearest to
        ``reference``, so a drive target never jumps across the ±π wrap.
        """
        rotation = quaternion_rotation(quaternion)
        vector = (rotation*self.local_rotation).as_rotvec()
        if reference is not None:
            angle = np.linalg.norm(vector)
            if angle > 1e-9:
                vector = min((vector*(1+2*np.pi*k/angle) for k in (-1, 0, 1)),
                             key=lambda v: np.linalg.norm(v-reference))
        translation = np.asarray(position, dtype=float) + rotation.apply(self.local_position) - self.origin
        return np.r_[translation, vector]

    def velocity_coordinates(self, quaternion, linear_velocity, angular_velocity):
        """Fixture velocities for the body origin moving at world velocities."""
        rotation = quaternion_rotation(quaternion)
        omega = np.asarray(angular_velocity, dtype=float)
        linear = np.asarray(linear_velocity, dtype=float) + np.cross(omega, rotation.apply(self.local_position))
        return np.r_[linear, (rotation*self.local_rotation).inv().apply(omega)]

    def joint_point(self):
        """World position of the spherical joint, the point the fixture holds."""
        return self.origin + self.positions()[:3]

    def joint_rotation(self):
        return Rotation.from_rotvec(self.positions()[3:])

    def positions(self):
        """Coordinates in environment 0."""
        return self.environment_positions()[0]

    def environment_positions(self):
        """Coordinates in every environment, one row each."""
        return self.view.get_dof_positions()[:, self.order].astype(float)

    # -- tensors -------------------------------------------------------------------------
    def initialize(self):
        from isaacsim.core.simulation_manager import SimulationManager
        from table_1000.sim.isaaclab.asset_views import TensorView
        session = self.ctx.session
        # One fixture articulation per environment, all at the same local coordinates.
        self.view = TensorView(SimulationManager.get_physics_sim_view().create_articulation_view(
            [session.replica_path(self.path+'/root_joint', i) for i in range(session.envs)]),
            SimulationManager.get_physics_sim_device())
        self.indices = np.arange(session.envs, dtype=np.int32)
        names = self.view.shared_metatype.dof_names
        self.order = [names.index(name) for name in DOFS]
        self.link = self.view.shared_metatype.link_names.index(self.body.path.rsplit('/', 1)[-1])
        if not hasattr(self, 'free_limits'):
            self.free_limits = self.view.get_dof_limits().copy()
        self.apply()
        # Start at the declared coordinates. Recovering a large initial limit
        # violation converges to a different orientation on CPU PhysX.
        positions = self.view.get_dof_positions()
        positions[:, self.order] = self.coordinates
        self.view.set_dof_positions(positions, self.indices)

    def apply(self):
        """Write targets, gains and limits; each value is per coordinate or per environment row."""
        if self.view is None:
            return
        active = not (self.released or self.locked)
        values = {'targets': self.view.get_dof_position_targets(),
                  'velocity_targets': self.view.get_dof_velocity_targets(),
                  'stiffness': self.view.get_dof_stiffnesses(), 'damping': self.view.get_dof_dampings(),
                  'actuation': self.view.get_dof_actuation_forces(),
                  'max_forces': self.view.get_dof_max_forces()}
        values['targets'][:, self.order] = self.coordinates
        values['velocity_targets'][:, self.order] = self.velocity_targets if active else 0
        values['stiffness'][:, self.order] = self.stiffness if active else 0
        values['damping'][:, self.order] = self.damping if active else 0
        values['actuation'][:, self.order] = self.actuation if active else 0
        values['max_forces'][:, self.order] = np.minimum(self.max_forces, 3.4e38)
        self.view.set_dof_position_targets(values['targets'], self.indices)
        self.view.set_dof_velocity_targets(values['velocity_targets'], self.indices)
        self.view.set_dof_stiffnesses(values['stiffness'], self.indices)
        self.view.set_dof_dampings(values['damping'], self.indices)
        self.view.set_dof_actuation_forces(values['actuation'], self.indices)
        self.view.set_dof_max_forces(values['max_forces'], self.indices)
        limits = self.free_limits.copy()
        if self.locked and not self.released:
            limits[:, self.order] = np.asarray(self.coordinates)[..., None]
        self.view.set_dof_limits(limits, self.indices)

    def release(self, released=True):
        self.released = released
        self.apply()


class Motion:
    """A timed world trajectory of a body origin, evaluated with derivatives."""

    def __init__(self, path, duration, start):
        self.path, self.duration, self.start = path, duration, start

    @property
    def end(self):
        return self.start + self.duration

    def __call__(self, t):
        """Pose, world velocities and accelerations at absolute time ``t``."""
        tau = min(max(t - self.start, 0.), self.duration)
        h = min(1e-4, self.duration/4)
        times = np.clip([tau-h, tau, tau+h], 0, self.duration)
        samples = [self.path(s) for s in times]
        positions = np.array([np.asarray(p, dtype=float) for p, _ in samples])
        rotations = [quaternion_rotation(q) for _, q in samples]
        span = times[2] - times[0]
        velocity = (positions[2]-positions[0]) / span
        omega = (rotations[2]*rotations[0].inv()).as_rotvec() / span
        if times[0] < tau < times[2]:
            acceleration = (positions[2]-2*positions[1]+positions[0]) / h**2
            alpha = ((rotations[2]*rotations[1].inv()).as_rotvec()-(rotations[1]*rotations[0].inv()).as_rotvec()) / h**2
        else:
            acceleration, alpha = np.zeros(3), np.zeros(3)
        return positions[1], rotation_quaternion(rotations[1]), velocity, omega, acceleration, alpha


def smooth(u):
    """Quintic time scaling: zero velocity and acceleration at both ends."""
    return u*u*u*(10 - 15*u + 6*u*u)


class Grip:
    """Ideal six-axis grip of one body, tracking timed world trajectories.

    Attached through joints rather than contact: no geometry, no collision, no
    slip. Declared before initialization, it initially holds the body at its
    declared pose. Motions start where the previous one ends; ``wait=False``
    queues a motion so several grips can move together (see ``ctx.wait``).
    Gains follow from the held tree's mass and inertia; drives receive the
    trajectory velocity, and feedforward joint forces carry gravity and the
    trajectory acceleration of the held tree. ``limit`` bounds the force and
    torque magnitudes, turning position control into force control.
    """

    def __init__(self, ctx, body, *, name):
        self.ctx, self.body, self.name = ctx, body, name
        self.fixture = Fixture(ctx, body, name=name, locked=False)
        position, quaternion = ctx.body_pose(body)
        self.motions = []
        self.linear_axes, self.angular = np.ones(3), True
        self.force_limit = self.torque_limit = np.inf
        self.planned = (position.copy(), quaternion.copy())
        self.target = (position.copy(), quaternion.copy(), np.zeros(3), np.zeros(3))
        self.gains = None
        ctx.grips.append(self)
        for field, unit in [('target.position', 'm'), ('force', 'N'), ('torque', 'N·m')]:
            for i, axis in enumerate('xyz'):
                ctx.observe(f'{name}.{field}.{axis}', lambda f=field, i=i: self.values()[f][i], unit=unit)
        ctx.observe(f'{name}.error.position', lambda: self.values()['error.position'], unit='m')
        ctx.observe(f'{name}.error.angle', lambda: self.values()['error.angle'], unit='rad')

    @property
    def released(self):
        return self.fixture.released

    @property
    def position(self):
        """World position where the last commanded motion ends; the next motion starts here."""
        return self.planned[0].copy()

    @property
    def quaternion(self):
        """World orientation ``(w, x, y, z)`` where the last commanded motion ends."""
        return self.planned[1].copy()

    # -- commands ------------------------------------------------------------------------
    def move_to(self, *, duration, position=None, rotation=None, wait=True):
        """Straight line and constant-axis rotation to a world pose, quintic in time.

        Starts where the previous motion ends; zero velocity and acceleration at both ends.
        """
        start_position, start_quaternion = self.planned
        end_position = start_position if position is None else np.asarray(position, dtype=float)
        start = quaternion_rotation(start_quaternion)
        end = start if rotation is None else quaternion_rotation(rotation)
        turn = (end*start.inv()).as_rotvec()

        def path(t):
            s = smooth(t/duration)
            return start_position + (end_position-start_position)*s, rotation_quaternion(Rotation.from_rotvec(turn*s)*start)
        return self.follow(path, duration=duration, wait=wait)

    def follow(self, path, *, duration, wait=True):
        """Track ``path(t) -> (position, quaternion)`` for ``t`` in ``[0, duration]``.

        The path must start at the pose where the previous motion ends; ``duration`` is a
        positive multiple of the physics step. ``wait=True`` steps until the motion ends,
        ``wait=False`` queues it and returns the :class:`Motion`.
        """
        steps = duration / self.ctx.dt
        if duration <= 0 or abs(steps-round(steps)) > 1e-7:
            raise ValueError('A motion duration must be a positive multiple of the physics step')
        if self.released:
            raise RuntimeError('Grasp the body before moving the grip')
        position, quaternion = path(0.)
        planned_position, planned_quaternion = self.planned
        angle = (quaternion_rotation(quaternion)*quaternion_rotation(planned_quaternion).inv()).magnitude()
        if np.linalg.norm(np.asarray(position)-planned_position) > 1e-6 or angle > 1e-6:
            raise ValueError('A path must start where the previous motion of the grip ends')
        start = max(self.motions[-1].end if self.motions else 0., self.ctx.session.time)
        motion = Motion(path, duration, start)
        self.motions.append(motion)
        end_position, end_quaternion = path(duration)
        self.planned = (np.asarray(end_position, dtype=float), np.asarray(end_quaternion, dtype=float))
        if wait:
            self.ctx.wait(motion)
        return motion

    def track(self, position, quaternion, linear_velocity=(0, 0, 0), angular_velocity=(0, 0, 0)):
        """Hold a world pose (and velocity) from the next step; replaces queued motions."""
        self.motions.clear()
        self.planned = (np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float))
        self.target = (*self.planned, np.asarray(linear_velocity, dtype=float), np.asarray(angular_velocity, dtype=float))

    def axes(self, linear=(1, 1, 1), angular=True):
        """Enable world translation axes and rotation; a disabled axis exerts nothing."""
        self.linear_axes, self.angular = np.asarray(linear, dtype=float), bool(angular)

    def limit(self, force=np.inf, torque=np.inf):
        """Bound the magnitude of the applied force (N) and torque (N·m).

        Each limit is a number or a function of simulation time; ``inf``
        removes it. While a limit binds, the grip pushes toward its target with
        that magnitude and the body lags its trajectory.
        """
        self.force_limit, self.torque_limit = force, torque

    def release(self):
        """Let go: the body is free, queued motions are dropped."""
        self.motions.clear()
        self.fixture.release()

    def grasp(self):
        """Hold the body where it is now, at rest, without a jump."""
        state = self.body.state
        self.track(state['position'], state['quaternion'])
        self.fixture.release(False)

    # -- control -------------------------------------------------------------------------
    def initialize(self):
        tree = self.tree_state(0, self.fixture.positions())
        linear, angular = LINEAR_STEPS / self.ctx.dt, ANGULAR_STEPS / self.ctx.dt
        inertia = np.linalg.eigvalsh(tree['inertia']).max()
        self.gains = {'linear_stiffness_N_per_m': tree['mass']*linear**2, 'linear_damping_N_s_per_m': 2*tree['mass']*linear,
                      'angular_stiffness_N_m_per_rad': inertia*angular**2, 'angular_damping_N_m_s_per_rad': 2*inertia*angular,
                      'held_mass_kg': tree['mass'], 'held_inertia_kg_m2': inertia,
                      'linear_natural_frequency_rad_s': linear, 'angular_natural_frequency_rad_s': angular}

    def tree_state(self, environment=0, positions=None):
        """Mass, center of mass and inertia (about the joint point) of the held tree.

        Environment-local coordinates; ``positions`` are that environment's
        fixture coordinates (default: the declared joint point).
        """
        instance = self.body.instance
        point = self.fixture.origin + (positions[:3] if positions is not None else 0)
        _, rotations, centers = instance.environment_states()
        mass, moment, inertia = 0., np.zeros(3), np.zeros((3, 3))
        for name in self.fixture.tree:
            i = instance.indices[name]
            m, r, rotation = float(instance.mass[i]), centers[environment, i] - point, rotations[environment, i]
            mass += m
            moment += m*centers[environment, i]
            inertia += rotation @ instance.local_inertia[i] @ rotation.T + m*(r@r*np.eye(3)-np.outer(r, r))
        return {'mass': mass, 'com': moment/mass, 'inertia': inertia}

    def update(self, t, dt):
        """Drive targets for the step ending at ``t + dt``, in every environment.

        Each environment's grip follows the same trajectory and limits with
        feedback and feedforward from its own state.
        """
        if self.released:
            return
        # Motions are aligned with the step: the step ending at t+dt belongs to
        # the first motion that has not ended before it.
        while len(self.motions) > 1 and self.motions[0].end < t + dt - 1e-9:
            self.motions.pop(0)
        acceleration, alpha = np.zeros(3), np.zeros(3)
        if self.motions:
            motion = self.motions[0]
            position, quaternion, velocity, omega, acceleration, alpha = motion(t + dt)
            self.target = (position, quaternion, velocity, omega)
        position, quaternion, velocity, omega = self.target
        if self.motions and self.motions[0].end <= t + dt + 1e-9:
            self.motions.pop(0)
            if not self.motions:
                # Hold the end pose at rest from the next step.
                self.target = (position, quaternion, np.zeros(3), np.zeros(3))
        fixture, gains = self.fixture, self.gains
        positions = fixture.environment_positions()
        fixture.coordinates = np.array([fixture.pose_coordinates(position, quaternion, reference=row[3:])
                                        for row in positions])
        fixture.velocity_targets = fixture.velocity_coordinates(quaternion, velocity, omega)
        linear = self.linear_axes
        stiffness = np.r_[linear*gains['linear_stiffness_N_per_m'], [gains['angular_stiffness_N_m_per_rad']*self.angular]*3]
        damping = np.r_[linear*gains['linear_damping_N_s_per_m'], [gains['angular_damping_N_m_s_per_rad']*self.angular]*3]
        # Inverse dynamics of the held tree about the joint point: carry its
        # weight and supply the trajectory's acceleration.
        gravity = np.asarray(self.ctx.gravity, dtype=float)
        feedforward = []
        for environment, row in enumerate(positions):
            tree = self.tree_state(environment, row)
            force = tree['mass']*(acceleration - gravity)
            torque = tree['inertia'] @ alpha + np.cross(tree['com'] - fixture.origin - row[:3], force)
            torque = Rotation.from_rotvec(row[3:]).inv().apply(torque) if self.angular else np.zeros(3)
            feedforward.append(np.r_[force*linear, torque])
        feedforward = np.array(feedforward)
        # Limits bound the magnitude of the force and of the torque. A group
        # whose estimated wrench (drive plus feedforward) would exceed its limit
        # pushes with exactly the limit along that estimate instead: the
        # trajectory feedforward assumes tracking, which no longer holds.
        # Magnitudes do not depend on the frame, so the joint-frame torque
        # is bounded as the world torque is.
        estimate = (stiffness*(fixture.coordinates-positions) +
                    damping*(fixture.velocity_targets-self.measured_velocities(positions)) + feedforward)
        fixture.stiffness, fixture.damping = np.tile(stiffness, (len(positions), 1)), np.tile(damping, (len(positions), 1))
        fixture.actuation, fixture.max_forces = feedforward.copy(), np.full(feedforward.shape, np.inf)
        for group, limit in ((slice(0, 3), self.force_limit), (slice(3, 6), self.torque_limit)):
            limit = self.limit_value(limit, t)
            if np.isinf(limit):
                continue
            size = np.linalg.norm(estimate[:, group], axis=1, keepdims=True)
            saturated = size > limit
            push = estimate[:, group]*limit/np.maximum(size, 1e-30)
            carried = np.linalg.norm(feedforward[:, group], axis=1, keepdims=True)
            held = feedforward[:, group]*np.minimum(1, limit/np.maximum(carried, 1e-30))
            fixture.stiffness[:, group] = np.where(saturated, 0, stiffness[group])
            fixture.damping[:, group] = np.where(saturated, 0, damping[group])
            fixture.actuation[:, group] = np.where(saturated, push, held)
            # Tracking within the limit: the drive may add up to what is left.
            fixture.max_forces[:, group] = np.where(saturated, np.inf, np.maximum(limit - np.minimum(carried, limit), 0))
        self.previous_positions = positions
        fixture.apply()

    def measured_velocities(self, positions):
        """Fixture velocities over the last step from coordinate differences, one row per environment.

        CPU TGS can report a nonzero end-of-step velocity for a body at rest.
        """
        previous = getattr(self, 'previous_positions', None)
        if previous is None:
            return self.fixture.view.get_dof_velocities()[:, self.fixture.order].astype(float)
        velocities = []
        for now, before in zip(positions, previous):
            rotation = Rotation.from_rotvec(now[3:])
            omega = (rotation*Rotation.from_rotvec(before[3:]).inv()).as_rotvec() / self.ctx.dt
            velocities.append(np.r_[(now[:3]-before[:3])/self.ctx.dt, rotation.inv().apply(omega)])
        return np.array(velocities)

    @staticmethod
    def limit_value(limit, t):
        value = float(limit(t) if callable(limit) else limit)
        if value < 0:
            raise ValueError('A grip limit must be non-negative')
        return value

    def values(self):
        """Recorded world target, tracking error and the wrench applied to the held tree."""
        key = (self.ctx.session.steps, self.ctx.session.time)
        if getattr(self, '_values', (None,))[0] != key:
            self._values = (key, self._compute_values())
        return self._values[1]

    def _compute_values(self):
        position, quaternion, _, _ = self.target
        state = self.body.state
        if self.fixture.view is None:
            wrench = np.zeros(6)
        else:
            wrench = self.wrench()
        error = (quaternion_rotation(state['quaternion'])*quaternion_rotation(quaternion).inv()).magnitude()
        return {'target.position': position, 'force': wrench[:3], 'torque': wrench[3:],
                'error.position': float(np.linalg.norm(state['position']-position)), 'error.angle': float(error)}

    def wrench(self):
        """World force and torque (about the joint point) the grip applies to the held tree.

        CPU PhysX reports the full incoming joint force of the held link; GPU
        PhysX leaves drive forces out of it, so the drive law gives the force.
        The GPU spherical drive is effectively stiffer than its nominal gains,
        so GPU torque reads low (about 5% at the default gains); GPU force is
        exact.
        """
        fixture = self.fixture
        if fixture.released:
            return np.zeros(6)
        positions = fixture.positions()
        if self.ctx.physics == 'cpu':
            incoming = fixture.view.get_link_incoming_joint_force()[0, fixture.link].astype(float)
            frame = fixture.joint_rotation()
            return np.r_[frame.apply(incoming[:3]), frame.apply(incoming[3:])]
        first = lambda value: np.broadcast_to(value, (len(self.fixture.indices), 6))[0]
        velocities = self.measured_velocities(fixture.environment_positions())[0]
        drive = (first(fixture.stiffness)*(first(fixture.coordinates)-positions) +
                 first(fixture.damping)*(first(fixture.velocity_targets)-velocities))
        generalized = np.clip(drive, -first(fixture.max_forces), first(fixture.max_forces)) + first(fixture.actuation)
        return np.r_[generalized[:3], fixture.joint_rotation().apply(generalized[3:])]
