"""Atomic skills for experts: grasp, pick, move by a displacement, push, release, place.

Named after RoboTwin 2.0's task API (``grasp_actor``, ``move_by_displacement``,
``place_actor``, ``open_gripper``/``close_gripper``) but blocking like the rest of
``solve(ctx)``: each call returns when the motion is done, checked in the
simulation, and fails the subtask (:class:`~table_1000.experts.context.ExpertFailed`,
or :class:`~table_1000.experts.solver.PlanningFailed` for a move that cannot be
planned) when it is not. Lower-level moves stay available through
``ctx.solver(arm)`` and mix freely with these::

    from table_1000.experts import skills

    g = skills.grasp(ctx, 'cabinet', 'cabinet.drawer1.handle', then=-pull)  # either arm, the nearer first
    skills.move_by_displacement(ctx, g['arm'], *-pull, drag=('cabinet', 'cabinet.drawer1'))
    skills.release(ctx, g['arm'])
    p = skills.pick(ctx, 'pen')                                              # grasp, then lift
    skills.place(ctx, 'pen', (table_frame, (x0, y0), (x1, y1)), near=(x, y, table_height))

What is grasped is an object of the scene (an asset, by name), or a part or body
of it: its shape is the collision box of that part in the frame of its body, or
one given (a :class:`~table_1000.experts.grasps.Box` or
:class:`~table_1000.experts.grasps.Cylinder` placed by ``offset`` in the body
frame, also for a scene primitive that is not an asset). Every asset of the scene
is in each arm's planning world, one planning object per part (``<object>:<part>``);
the table too.

Each skill reaches with the least of the robot it can, in this order: one arm with the torso
lift where it is, the other arm so, the torso lift moving too, the free arms that are out back
at their ready pose (where none is in reach as they are), then the base moving
(:func:`make_room`); a free-space plan (RRTConnect) that may move the torso lift holds it first
where its goal leaves it where it is.

**grasp** -- candidates of the shape at its live pose
(:func:`~table_1000.experts.grasps.grasp_candidates`, cheapest first: close to the
``approach`` direction, default reaching out and down from the arm's shoulder;
centred on ``point``, default the chosen shape's centre, with the additional
gravity moment of a free held body scored about its actual COM) are tried per arm, the arm whose
shoulder is nearer first, in the order above (where none is in reach, the base first moves
to reach the first ones). Every other planning object is an obstacle of the
candidates' hand; the ones within ``SUPPORT_DISTANCE`` of the shape are its supports,
which the hand may touch (overlapping them by at most ``-SUPPORT_CLEARANCE``). A
candidate is kept when the grasp and a pre-grasp ``APPROACH`` behind it have
collision-free IK solutions, and so does the grasp moved by ``then`` (the motion that
follows, if given), straight all along; a ``then`` other than a lift up also needs the hand to
leave straight back ``APPROACH`` from its end, where it lets go (a drawer pulled up to the body
leaves the arm no room to). The hand plans (RRTConnect) to the
pre-grasp with the fingers open to the object's width plus a margin, moves straight
in (the only move in which the fingers may touch the object), and closes; the finger
opposing pads must contact the held body and the gap must agree with its actual
contact surfaces within the geometry error budget, else it backs out, re-observes the
object and chooses again (at most ``MAX_REGRASPS`` times). The grasped part then
moves with the hand in the planning world until :func:`release`.

**pick** -- grasp and lift, checking the body's actual rise. Thin free rigid
bodies are recognized from collision geometry and the hand's contact patch:
sheets are pressed and dragged to a reachable table edge before a horizontal
approach and vertical pinch; filaments use a fingertip pinch first. Very small
sheets whose COM cannot stay supported at a graspable edge landing also use
the fingertip pinch. Thin closures wait for opposing surfaces to touch with the
unchanged drive. See ``docs/specs/thin-grasp.md``.
For thin targets, ``pick`` rejects non-default ``point``, ``approach`` and ``max_angle`` with
``ValueError`` (remove them); ``grasp`` rejects an unprepared flat body on the table (use ``pick``
or ``slide_to_edge`` then ``grasp``), while overhanging or already held thin bodies are allowed.

**slide_to_edge** -- the sheet preparation alone: press its top collision
surface with the closed fingertips, drag while its COM stays supported, then
retract and measure the exposed depth. Only fingers may touch the target and
table; the ordinary collision planners check the rest of the robot and scene.

**move_by_displacement** -- the TCP straight by a world displacement, with the torso
lift too where the arm alone runs into a joint limit, or (``least``) as far along as
plans; straight up beyond the arm's reach, the torso lift rises and the arm does the rest. ``drag``
names a body moved through a joint by the held part (a drawer pulled by its
handle): its planning objects move with the hand and may touch the rest of their
object and whatever rests on them.

**push** -- the closed fingertips against the face of a part that ``direction``
points into, then straight on by ``distance`` (a drawer pushed shut by its front);
the hand comes level with the push or tilted down (``PUSH_TILTS``), and what the
part's body carries moves with it as in a drag; where none is in reach, the base first moves to
reach the first ones.

**release** -- open the fingers to the grasp's opening and back out along the
approach (or up).

**place** -- put the held object down anywhere in a region: a rectangle on a surface
(a frame whose z = 0 plane is the surface), its footprint inside by a margin and out
of keep-out rectangles. Positions every ``PLACE_STEP`` and turns about the surface
normal every ``PLACE_TURN_STEP`` are ranked nearest a given point, then least turned
(``PLACE_TURNS`` per position);
IK screens them above and at the release (``MAX_PLACE_IK``), and the first of those that has a motion
(``MAX_PLACES`` planned) is taken: above it, straight down to ``clearance`` over the
surface, then :func:`release`.

**make_room** -- move the base to where given TCP poses (each a pose, or the poses along a
straight move) are reachable with every joint a margin inside its limits: where it stands
if they are, else the base pose that it gets to soonest among poses sampled around them
(facing them, at a distance), the robot clear of the planning world there and on the way
(the table's legs included). The free arms go back to their ready pose first where they would
sweep something on the way; the base turns in place, drives straight and turns in place. Given no poses, the base goes back to where it
started.

**handover** -- the held object passed to the other hand, held as the next action needs
(:mod:`~table_1000.experts.skills.handover`, after WRS): the giving arm brings it in front of the
body, the receiving one grasps it, the giving one lets go.

**ready** -- the arms that hold nothing back at the robot's declared ready pose, using
their initial values for undeclared joints. The shared torso stays where it is.

**retry** -- a callback run again when it fails, up to the context's ``skill_attempts``
(``ATTEMPTS`` otherwise). Later tries restore its entry state and use new random streams,
including for ``Unreachable``. ``Unmet`` propagates to chain adaptation. Each discarded
failed branch retains its own trajectory and call/solve records.

**chain** -- actions (:class:`Pick`, :class:`Place`) run in order, each asked for what the next
needs of it: a place needs the object held by a hand, and with a hold, from which one of its
placements is in reach (an :class:`InHand`), and the pick takes only such grasps; where none is in
reach (:class:`Unmet`) the object is picked as it can be and handed over to the other hand, else
the pick runs again with the base free to move (see :mod:`~table_1000.experts.skills.chain`)::

    skills.chain(ctx, [skills.Pick('pen', instruction='pick up the pen from the table'),
                       skills.Place('pen', region, near=point, instruction='put the pen into the drawer')])

Each choice of another candidate is noted (``ctx.note``). A skill that finds nothing in reach
(no grasp, push or placement) fails with :class:`~table_1000.experts.context.Unreachable`.

The skills are one :class:`Skills` per context (``ctx.skills()``), put together from
``grasping`` (grasp, pick), ``thin`` (thin-object preparation), ``handover``,
``motion`` (straight moves, push, release), ``placing`` and ``base``
(the base, ready); ``common`` holds what they share.
"""

import numpy as np

from table_1000.experts.context import ExpertFailed, Unreachable
from table_1000.experts.grasps import Hand
from table_1000.experts.skills.base import Base
from table_1000.experts.skills.chain import InHand, Pick, Place, chain
from table_1000.experts.skills.common import Unmet
from table_1000.experts.skills.grasping import LIFT, Grasping
from table_1000.experts.skills.handover import Handing
from table_1000.experts.skills.motion import Motion
from table_1000.experts.skills.placing import PLACE_ABOVE, PLACE_CLEARANCE, PLACE_MARGIN, PLACE_STEP, Placing
from table_1000.experts.skills.thin import ThinPicking

ATTEMPTS = 3            # tries of a step under retry


class Skills(ThinPicking, Grasping, Handing, Motion, Placing, Base):
    """The skills of ``ctx``'s arms (``arms``: names, default all); see the module docstring."""

    def __init__(self, ctx, arms=None):
        self.ctx, self.model = ctx, ctx.model
        self.arms = list(arms or ctx.spec.arms)
        self.solvers = {arm: ctx.solver(arm) for arm in self.arms}
        self.hands = {arm: Hand(self.model, arm, drive=ctx.env.gripper_drive(arm)) for arm in self.arms}
        for solver in self.solvers.values():
            if 'table' not in solver.world.objects:
                solver.add_table()
            for name in ctx.scene.OBJECTS:
                if not any(key.startswith(name + ':') for key in solver.world.objects):
                    # Parts of one asset touch or overlap each other (a cap on a pen, a drawer in its
                    # housing); only their contacts with the robot and other objects are planned.
                    # A part that is about convex (or too thin to get into) is checked as one hull.
                    parts = [obj.name for obj in solver.add_object(name, lambda body, part: (part, body),
                                                                   merged=ctx.geometry[name].solid)]
                    solver.world.allow(parts, parts)
        # The ready arm pose (initial values for undeclared joints) and initial base pose.
        self.home = self.model.vector(dict(zip(ctx.env.joints, ctx.start['qpos'])))
        # A folded start pose need not stay clear when the shared torso lowers.
        # Prefer the robot's declared ready arm pose; leave lift and head alone.
        rest = ctx.spec.keyframes.get('ready', {})
        for arm in self.arms:
            for name in ctx.spec.arms[arm].joints:
                if name in rest:
                    self.home[self.model.index[name]] = rest[name]

        self.start_base = np.asarray(ctx.start['base_pose'], dtype=float) if ctx.spec.base else None
        self.held = {}      # arm -> {'target', 'supports', 'grasp'}
        self.log, self.rejected = [], {}
        self.reaches = {}   # arm -> its reach (see Base.arm_reach)


def _skills(ctx):
    return ctx.skills()


def make_room(ctx, poses, arms=None, path=False):
    """Move the base to where the world TCP ``poses`` are reachable with a margin (see :meth:`Skills.make_room`)."""
    _skills(ctx).make_room(poses, arms, path=path)


def ready(ctx):
    """The arms that hold nothing back at their ready pose (see :meth:`Skills.ready`)."""
    _skills(ctx).ready()


def retry(ctx, step, attempts=ATTEMPTS):
    """Call ``step`` at most ``attempts`` times, restoring its entry before another try.

    The context's ``skill_attempts`` overrides the limit. Later attempts use new
    random streams, including for Unreachable; Unmet propagates to chain adaptation.
    Discarded failed branches have their own trajectories and call/solve records.
    """
    from table_1000.experts.retrying import RetryState
    attempts = getattr(ctx, 'skill_attempts', None) or attempts
    saved = RetryState(ctx) if attempts > 1 else None
    try:
        for attempt in range(1, attempts + 1):
            try:
                return step()
            except Unmet:
                raise
            except ExpertFailed as failure:
                if attempt == attempts:
                    raise
                saved.reset(attempt, failure)
    finally:
        if saved:
            saved.close()


def relaxed(limits, attempt):
    """``attempt(limit)`` for each of ``limits`` (one value, a list or None) in turn while it is
    :class:`Unreachable`; :class:`Unmet` if one of them was."""
    limits = list(np.atleast_1d(limits)) if limits is not None else [None]
    unmet = None
    for limit in limits:
        try:
            return attempt(limit)
        except Unmet as failure:
            unmet = failure
        except Unreachable as failure:
            last = failure
    raise unmet or last


def grasp(ctx, obj, part=None, *, arm=None, shape=None, offset=None, body=None, point=None, approach=None, then=None,
          max_angle=None):
    """Grasp scene object ``obj`` (or its ``part``); see the module docstring. ``max_angle`` (rad): only
    grasps approaching within it of ``approach``; a list of them is tried in order while none is in reach.
    A flat, unprepared thin object on the table raises ``ValueError``: use :func:`pick`, or first
    :func:`slide_to_edge` then :func:`grasp`; overhanging or already held thin objects are allowed.
    """
    skills = _skills(ctx)
    target = skills.target(obj, part, shape, offset, body)
    return relaxed(max_angle, lambda angle: skills.grasp(target, arm, point, approach, then, angle))


def pick(ctx, obj, part=None, *, arm=None, shape=None, offset=None, body=None, point=None, approach=None, height=LIFT,
         max_angle=None, need=None, base=True):
    """Grasp scene object ``obj`` (or its ``part``) and lift it by ``height``; ``max_angle`` as for
    :func:`grasp`; ``need``: held as that :class:`~table_1000.experts.skills.chain.InHand` asks, else
    :class:`Unmet`; ``base``: whether the base may move.
    The thin-object route raises ``ValueError`` for non-default ``point``, ``approach`` or
    ``max_angle`` because preparation chooses its own grasp geometry; remove these arguments.
    """
    skills = _skills(ctx)
    target = skills.target(obj, part, shape, offset, body)
    return relaxed(max_angle, lambda angle: skills.pick(target, arm, point, approach, height, angle, need, base))


def move_by_displacement(ctx, arm, x=0., y=0., z=0., drag=None, least=1.):
    """Move ``arm``'s TCP straight by ``(x, y, z)`` (world), or part way down to the fraction
    ``least``; ``drag``: ``(object, body)`` moved by the held part. Returns the fraction moved."""
    return _skills(ctx).move_by_displacement(arm, (x, y, z), drag, least)


def push(ctx, obj, part=None, *, direction, distance, arm=None, shape=None, offset=None, body=None):
    """See :meth:`Skills.push`; the target as for :func:`grasp`."""
    skills = _skills(ctx)
    return skills.push(skills.target(obj, part, shape, offset, body), direction, distance, arm)


def slide_to_edge(ctx, obj, part=None, *, arm=None, body=None):
    """Pressure-drag a thin object to a graspable table edge; see :meth:`Skills.slide_to_edge`."""
    skills = _skills(ctx)
    return skills.slide_to_edge(skills.target(obj, part, body=body), arm)


def release(ctx, arm):
    """Let go of what ``arm`` holds and back out."""
    _skills(ctx).release(arm)


def place(ctx, obj, region, *, arm=None, near=None, keep_out=(), clearance=PLACE_CLEARANCE, above=PLACE_ABOVE,
          step=PLACE_STEP, margin=PLACE_MARGIN):
    """Put the held ``obj`` down in ``region`` (see :meth:`Skills.place`) and release it."""
    return _skills(ctx).place(obj, region, arm, near, keep_out, clearance, above, step, margin)


def handover(ctx, obj, *, to=None, need=None):
    """Hand the held ``obj`` to the other hand (``to``), held as ``need`` asks (see :meth:`Skills.handover`)."""
    return _skills(ctx).handover(obj, to, need)


def open_gripper(ctx, arm, gap=None):
    solver = ctx.solver(arm)
    solver.open_gripper() if gap is None else solver.open_gripper(gap)


def close_gripper(ctx, arm):
    ctx.solver(arm).close_gripper()


# Public skill calls are the evaluated, replayable boundary.
from table_1000.experts.trace import traced

for _name in ('grasp', 'pick', 'place', 'push', 'move_by_displacement', 'release', 'handover',
              'make_room', 'ready', 'open_gripper', 'close_gripper'):
    globals()[_name] = traced()(globals()[_name])
retry = traced(concrete=False)(retry)
chain = traced(concrete=False)(chain)
