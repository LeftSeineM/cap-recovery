"""Seed-zero x1pro tidy trajectory using the current functional skill layer.

Books remain at their input poses for #106. Pen storage uses immersed material
and interior support, with tilt recorded rather than rejected. Physical search
and recovery are recorded by the common calls-v2 layer (three skill attempts).
Loaded-container transport and tabletop alignment retain seed-zero waypoints;
no common skill defaults are changed by this expert.
"""

from functools import partial
from contextlib import contextmanager
from time import perf_counter, process_time
from types import SimpleNamespace

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from table_1000.experts import skills
from table_1000.experts.skills.hold.frames import Box
from table_1000.experts.context import (DT, ExpertFailed, Step, WallBudgetExceeded,
                                      remaining_wall, run_steps)
from table_1000.experts.skills import semantic
from table_1000.experts.skills.base import BASE_TOLERANCE, drive, wrapped, drive_routes, drive_path
from table_1000.experts.skills.hold.frames import MIN_HOLD_AREA
from table_1000.experts.skills.common import APPROACH
from table_1000.experts.skills.hold.pick import LIFT, MIN_RISE
from table_1000.experts.skills.place.strict import INSERT_MARGIN, PLACE_CLEARANCE, PLACE_MARGIN, PLACE_TURN_STEP
from table_1000.experts.skills.hold.targets import targets as enumerate_targets
from table_1000.experts.skills.push.squaring import yaw_error
from table_1000.experts.skills.place import regions
from table_1000.experts.skills.place.regions import overlaps
from table_1000.experts.skills.drawer import joints as drawer_joints
from table_1000.experts.world import geometry as world_geometry
from .insertion import SampledBox, free_insertion_need
from .squaring import square_on_table

ROBOT = 'x1pro'
SCOPE = {0: [0]}
SOURCE = 'hand'

OPENING = .173
CLEAR = .04
POSITION_TOLERANCE = .04
ANGLE_TOLERANCE = np.radians(20.)
PEN_TILT = np.radians(30.)
# Layout 5 destinations relative to the table centre, never grasp points.
DESTINATIONS = {'jar': (-.48, .04), 'holder': (-.30, .04), 'cup': (-.17, .05),
                'basket': (.06, .01), 'cabinet': (.38, .03)}
# Prefer mutually clear poses inside the unchanged layout acceptance disks.
# The nominal adjacent cup/basket/cabinet poses leave less than PLACE_MARGIN.
CONTAINER_OFFSETS = {'jar': (-.02, -.028), 'holder': (-.025, .030),
                     'cup': (-.015, .015), 'basket': (-.02, .015)}
# Bounded E47 exploration could not grasp the full jar. Declare the revised
# row before execution; retain the already supported jar and rear-left basket.
RETAINED_CONTAINER_GOALS = {'jar': (-.01411243, .0415385),
                            'basket': (-.39198607, .2776098)}
TABLETOP_CONTAINER_GOALS = set()
# Insert long objects before bulky contents obstruct their wrist approach;
# other large footprints precede small ones. The first three top-drawer
# checks retain the already validated strict prefix.
DRAWERS = {3: ('screwdriver', 'lidded_jar', 'charger', 'dish', 'snack'),
           2: ('glue_bottle', 'board_eraser', 'tape', 'stapler', 'remote'),
           1: ('magnet_orange', 'staples', 'eraser', 'glue_stick', 'knife', 'usb_drive')}
# Seed-zero alternate storage is part of the plan; verify the actual container.
STORAGE_OVERRIDES = {'glue_stick': 'basket', 'screwdriver': 'basket',
                     'remote': 'basket', 'tape': 'basket'}
PEN_STORAGE = {}  # Long pens use their declared cup again under storage_result.
DRAWER_OVERRIDES = {}
# Descending authored collision footprint area.
FLAT_BOOKS = ('notebook_slim', 'notebook_a5', 'notebook_a6')
UNRESOLVED_BOOKS = ('notebook_a4', *FLAT_BOOKS)
ITEM_WALL_BUDGET = 600.  # One variant; the orchestration window per step is 40 minutes.
# Cabinet-frame base pose from L2b's successful cycle, also validated for
# drawer3 and drawer2 after the real far-X cleanup in the L4 experiment.
DRAWER_APPROACH = np.array([-.32227720544583194, -.9342566836295899, 1.009551402356736])


def drawer(number):
    return f'cabinet.drawer{number}'


def joint(number):
    return drawer(number) + '.slide'


def workspace(ctx):
    return [world_geometry.body_box(ctx, 'cabinet', 'cabinet.housing', CLEAR),
            *[drawer_joints.travel(ctx, 'cabinet', joint(i), CLEAR) for i in (1, 2, 3)]]


def container_wall_targets(ctx, obj):
    """Select real exterior wall pieces, excluding rims and internal dividers."""
    targets = [target for target in enumerate_targets(ctx.skills(), obj)
               if not target.rim and not target.fallback and 'walls' in target.names[0]]
    for target in targets:
        target.container = True
    return targets


def holder_wall_pick(ctx, original, targets, *args, **options):
    """Retain the ordinary traced Pick while filtering its target groups."""
    group = targets if isinstance(targets, (tuple, list)) else [targets]
    walls = [target for target in group if not target.fallback
             and 'walls' in target.names[0]]
    if not walls:
        ctx.fail('no reachable grasp', skills.Unreachable, kind='planning')
    for target in walls:
        target.container = True
    return original(walls, *args, **options)


def basket_front_wall_pick(ctx, original, targets, *args, **options):
    """Reuse the first-pass front-wall hold, excluding the failed back rim."""
    group = targets if isinstance(targets, (tuple, list)) else [targets]
    front = [target for target in group if not target.fallback
             and any('long_wall1' in name for name in target.names)]
    if not front:
        ctx.fail('no reachable grasp', skills.Unreachable, kind='planning')
    return original(front, *args, **options)


def close_jar_wall(ctx, target, arm, candidate):
    """Centre the open hand on the measured wall before closing both pads."""
    from table_1000.experts.skills.hold.frames import _close, _closing_surface, standoff
    from table_1000.experts.skills.common import APPROACH
    s, solver = ctx.skills(), ctx.solvers[arm]
    hand, tcp = s.hands[arm], ctx.tcp(arm)
    transform = ctx.poses()[target.body] @ target.offset
    surface = _closing_surface(s.contact_surface(target), transform, tcp, hand, candidate.open_gap)
    closed = _close(surface, transform, tcp, hand, candidate.open_gap, check_area=False)
    if closed is not None:
        rotation, origin = transform[:3, :3], transform[:3, 3]
        direction = rotation.T @ tcp[:3, 1]
        reaches = {}
        for sign, points, pads in hand.inner(closed[0]):
            world = points @ tcp[:3, :3].T + tcp[:3, 3]
            low, high = surface.interval((world - origin) @ rotation, direction)
            reaches[sign] = float(np.where(high >= low, high if sign > 0 else -low, -np.inf).max())
        # _close predicts the object sliding by shift. A resting jar need
        # not slide, so move the open hand in the opposite direction instead.
        shift = (reaches[-1.] - reaches[1.]) / 2
        goal = tcp.copy()
        goal[:3, 3] -= shift * tcp[:3, 1]
        ctx.note('jar centre closing', arm=arm, shift_m=shift, predicted_gap_m=closed[0])
        # A sub-mm screw alone lies inside the existing endpoint tolerance.
        # Re-enter from the ordinary approach distance with full collisions.
        solver.move_to_pose_with_screw(standoff(tcp, APPROACH))
        solver.move_to_pose_with_screw(goal)
    solver.close_gripper()


def transfer(ctx, obj, where, pick_options=None, **options):
    pick_options = dict(pick_options or {})
    if obj in ('notebook_a5', 'screwdriver', 'knife'):
        pick_options.setdefault('arm', 'right')
    if obj == 'holder':
        # A vertical exterior pinch keeps the fingers out of the standing
        # pens; slanted divider grips disturbed the initially qualified pen.
        pick_options.setdefault('arm', 'right')
        pick_options.setdefault('approach', (0., 0., -1.))
        pick_options.setdefault('max_angle', np.radians(10.))
    if obj == 'cup':
        pick_options.setdefault('approach', (0., 0., -1.))
        pick_options.setdefault('max_angle', np.radians(45.))
    # Loaded containers keep their current hand; L2a explicitly rejects
    # handing them over because the turn can spill the contents.
    adapt = ('base',) if obj in DESTINATIONS else ('base', 'handover')
    if obj in ('gel_pen', 'pen_black'):
        # A table-safe end hold may put the fingers below the cup rim.
        # Regrasp in free space before spending the item budget on base search.
        adapt = ('handover',)
    started = perf_counter()
    ctx.note('item phase start', item=obj, phase='pick_place')
    try:
        place = skills.Place(obj, where, instruction=f'put away the {obj}', **options)
        if obj in ('eraser', 'knife', 'gel_pen'):
            # These successful R10 holds did not satisfy coupled lookahead.
            # Reuse their stance, then plan pickup and placement independently.
            stance, arm, approach = {
                'eraser': ((-.195261, -.684063, .017445), 'left',
                           (-.2019573, .3719900, -.9060004)),
                'knife': ((-.126585, -.682873, 1.931814), 'right',
                          (-.2332111, -.1122491, -.9659258)),
                'gel_pen': ((.1929769, -.7836518, 1.573489), 'left',
                            (0., 0., -1.)),
            }[obj]
            semantic.require(ctx, checked_approach(
                ctx, np.array(stance), f'{obj} proven pickup stance'),
                f'{obj} pickup stance blocked', kind='check')
            def direct_transfer():
                # Keep the whole built body: the shared capped-pen build
                # merges barrel and cap. Avoid the failed coupled right hold.
                if obj == 'gel_pen':
                    grip = table_pen_grip(ctx, obj) or {}
                    ctx.note('gel whole-body pickup', asset=ctx.scene.OBJECTS[obj]['asset'],
                             bodies=ctx.geometry[obj].bodies, parts=ctx.geometry[obj].parts())
                    skills.pick(ctx, obj, arm=arm, base=False, **grip)
                else:
                    skills.pick(ctx, obj, arm=arm, approach=approach,
                                max_angle=np.radians(30.))
                return place.run(ctx)
            return skills.retry(ctx, direct_transfer)
        if obj in ('jar', 'basket'):
            # Explicitly select the rigid body so a loaded hollow container is
            # pinched on its broad side wall instead of its thin rim sections.
            def loaded_transfer():
                if obj == 'basket':
                    final = where is basket_destination_region
                    s = ctx.skills()
                    ordinary_pick = s.pick
                    s.pick = lambda targets, *a, **kw: basket_front_wall_pick(ctx, ordinary_pick, targets, *a, **kw)
                    approach = ctx.pose(obj)[:3, :3] @ np.array([-np.sin(np.radians(15.)), 0., -np.cos(np.radians(15.))])
                    try:
                        skills.pick(ctx, obj, arm='left', approach=approach,
                                    max_angle=np.radians(10.), **({'height': .10} if final else {}), **pick_options)
                    finally:
                        s.pick = ordinary_pick
                    if final:
                        container_carry_held(ctx, obj)
                        if empty_basket_held(ctx):
                            # Clear the table edge with the whole measured
                            # payload before driving around it.
                            skills.move_by_displacement(ctx, ctx.skills().holding(obj), z=.10)
                            container_carry_held(ctx, obj)
                        # The rear stance reaches the temporary basket, but
                        # its measured hold releases at the front table side.
                        container_table_approach(ctx, rear=False, loaded=obj)
                    return place.run(ctx)
                usable = container_wall_targets(ctx, obj)
                jar_grasp_approach(ctx)
                ctx.note('container body target', obj=obj, targets=len(usable),
                         target='solid_part', carries_contents=True)
                # The final goal needs another side of the table. Keep the
                # physically verified wall pinch, then carry it there rather
                # than asking one stance to reach both ends of the transfer.
                grasp = ctx.skills().grasp(
                    usable, base=False, approach=(0., 0., -1.), max_angle=np.radians(45.),
                    close=lambda selected, arm, candidate: close_jar_wall(ctx, selected, arm, candidate),
                    **pick_options)
                bottom = float(world_geometry.object_points(ctx, obj)[:, 2].min())
                obstacle_top = float(world_geometry.object_points(ctx, 'basket')[:, 2].max())
                skills.move_by_displacement(ctx, grasp['arm'],
                    z=max(.10, obstacle_top + CLEAR + PLACE_CLEARANCE - bottom))
                jar_carry_held(ctx)
                jar_carry_approach(ctx)
                return place.run(ctx)
            skills.retry(ctx, loaded_transfer)
        else:
            s = ctx.skills() if obj == 'holder' else None
            ordinary_pick = s.pick if s is not None else None
            if s is not None:
                s.pick = lambda targets, *a, **kw: holder_wall_pick(ctx, ordinary_pick, targets, *a, **kw)
            try:
                if obj in ('holder', 'cup'):
                    # The coupled search spends its budget making room before
                    # contact. Reuse the first pass's measured working stance
                    # and verify an independent pickup before planning release.
                    def holder_transfer():
                        container_grasp_stance(ctx, obj)
                        skills.pick(ctx, obj, base=False, **pick_options)
                        return place.run(ctx)
                    return skills.retry(ctx, holder_transfer)
                skills.chain(ctx, [skills.Pick(obj, instruction=f'pick up the {obj}', **pick_options), place], adapt=adapt)
            finally:
                if s is not None:
                    s.pick = ordinary_pick
    finally:
        ctx.note('item phase result', item=obj, phase='pick_place',
                 wall_s=perf_counter() - started)


def buffer(ctx, obj, keep_out=None, near=None):
    table = ctx.scene.TABLE
    if near is None:
        near = np.r_[world_geometry.centre(world_geometry.object_points(ctx, obj))[:2], table['height']]
    before = ctx.pose('cabinet').copy()
    if obj == 'dish':
        # The initially open top drawer blocks a collision-free approach to
        # the dish below it. Close it before clearing the front, then reopen
        # only when storing the eraser.
        close(ctx, 1)
    pick_options = ({'approach': -before[:3, 2], 'max_angle': np.radians([10., 20.])}
                    if obj in ('dish', 'charger', 'screwdriver') else None)
    if obj == 'screwdriver':
        pick_options.update(arm='right', part='screwdriver.body.grip')
    if obj == 'dish':
        pick_options['body'] = ctx.geometry[obj].bodies[0]
    transfer(ctx, obj, regions.on_region(obj, 'table', near=near,
                                        keep_out=workspace(ctx) if keep_out is None else keep_out),
             pick_options=pick_options, near=near)
    if obj in ('dish', 'charger'):
        after = ctx.pose('cabinet')
        angle = np.arccos(np.clip((np.trace(before[:3, :3].T @ after[:3, :3]) - 1) / 2, -1., 1.))
        shift = np.linalg.norm(before[:3, 3] - after[:3, 3])
        ctx.note('cabinet motion during front clearing', obj=obj,
                 cabinet_shift=float(shift), cabinet_turn=float(angle))
        semantic.require(ctx, shift <= .020 and angle <= np.radians(5.),
                         'clearing the cabinet front moved the cabinet', kind='check', obj=obj,
                         cabinet_shift=float(shift), cabinet_turn=float(angle))


def buffered(ctx, obj, keep_out=None):
    points = world_geometry.object_points(ctx, obj)
    if obj in DESTINATIONS:
        table_support(ctx, obj)
    else:
        semantic.require(ctx, semantic.on(ctx, obj, 'table'),
                         'temporary object not resting on the table', kind='check', obj=obj)
    regions = workspace(ctx) if keep_out is None else keep_out
    for frame, low, high in regions:
        inverse = np.linalg.inv(frame)
        local = (points @ inverse[:3, :3].T + inverse[:3, 3])[:, :2]
        outline = local[ConvexHull(local).vertices]
        semantic.require(ctx, not overlaps(outline, np.asarray(low)[:2], np.asarray(high)[:2]),
                         'temporary object obstructs the reserved workspace', kind='check', obj=obj)


def _skipped(ctx):
    """Per-episode skipped items; kept out of checkpoints but reported in events."""
    if not hasattr(ctx, '_skipped_items'):
        ctx._skipped_items = {}
    return ctx._skipped_items


def _record_skip(ctx, item, failure):
    reason = getattr(failure, 'reason', str(failure))
    details = getattr(failure, 'details', {})
    _skipped(ctx)[item] = {'reason': reason, 'details': details}
    ctx.note('skip failed item', item=item, reason=reason, details=details)


@contextmanager
def item_phase(ctx, item, phase):
    """One bounded retrieval phase; official checkpoints remain at empty hands."""
    from table_1000.common.batch import Load
    previous = getattr(ctx, 'wall_deadline', None)
    started, cpu, control = perf_counter(), process_time(), ctx.steps
    deadline = started + ITEM_WALL_BUDGET
    ctx.wall_deadline = min(deadline, previous) if previous is not None else deadline
    load, outcome = Load(), 'failed'
    ctx.note('item independent phase start', item=item, phase=phase, wall_budget_s=ITEM_WALL_BUDGET)
    try:
        with load:
            remaining_wall(ctx)
            yield
            remaining_wall(ctx)
            outcome = 'passed'
    finally:
        ctx.wall_deadline = previous
        samples = [sample['load_1m'] / load.cores for sample in load.samples]
        ctx.note('item independent phase result', item=item, phase=phase, outcome=outcome,
                 sim_s=(ctx.steps-control) * DT, cpu_s=process_time()-cpu,
                 raw_wall_s=perf_counter()-started, load=load.record(),
                 load_per_core_mean=sum(samples) / len(samples))


def optional_action(item, action, retries=1, *, phased=False):
    """Bound one trajectory step; skill-level retry owns physical recovery."""
    def run(ctx):
        previous = getattr(ctx, 'wall_deadline', None)
        started_wall, started_cpu, started_control = perf_counter(), process_time(), ctx.steps
        print('X1PRO_V2_ITEM_BEGIN', item, 'control', started_control, flush=True)
        ctx.wall_deadline = min(previous or float('inf'), perf_counter() + ITEM_WALL_BUDGET)
        try:
            action(ctx)
            remaining_wall(ctx)
            _skipped(ctx).pop(item, None)
        finally:
            print('X1PRO_V2_ITEM_END', item, 'cpu_s', process_time() - started_cpu,
                  'wall_s', perf_counter() - started_wall, 'control', ctx.steps, flush=True)
            ctx.wall_deadline = previous
    return run


def optional_check(item, check):
    """A failed item ends this batch; only checked physical states become chain checkpoints."""
    def verify(ctx):
        if item in _skipped(ctx):
            semantic.require(ctx, False, 'item remains incomplete', kind='check', item=item,
                             original_failure=_skipped(ctx)[item]['reason'])
        try:
            check(ctx)
            skill = ctx.skills()
            semantic.require(ctx, not skill.held and not any(s.attachments for s in ctx.solvers.values()),
                             'item checkpoint requires empty hands', kind='check', item=item)
            all_on_desk(ctx)
        except (ExpertFailed, ValueError) as failure:
            _record_skip(ctx, item, failure)
            raise
    return verify


def all_on_desk(ctx):
    """Reject collateral falls while allowing an input object's supported overhang."""
    table = ctx.scene.TABLE
    low = np.asarray(table['center']) - np.asarray(table['size']) / 2
    high = np.asarray(table['center']) + np.asarray(table['size']) / 2
    shapes = world_geometry.scene_shapes(ctx)
    for obj in ctx.scene.OBJECTS:
        points = np.vstack([shape.points for shape in shapes if shape.obj == obj])
        centre = world_geometry.centre(points)
        semantic.require(ctx, points[:, 2].min() >= table['height'] - .005
                         and np.all(centre[:2] >= low) and np.all(centre[:2] <= high),
                         'object fell or flew off the desk', kind='check', obj=obj)


def continuous_no_fall(ctx):
    """Record below-table excursions without rejecting a later physical recovery."""
    if not hasattr(ctx, '_fall_support_points'):
        ctx._fall_support_points = []
        for obj, geometry in ctx.geometry.items():
            for body in geometry.bodies:
                points = geometry.points(body, body)
                if len(points) > 3:
                    points = points[ConvexHull(points).vertices]
                ctx._fall_support_points.append((obj, f'{obj}:{body}', points))
    if not hasattr(ctx, '_observed_falls'):
        ctx._observed_falls = {}
    poses = ctx.poses()
    floor = ctx.scene.TABLE['height'] - .005
    for obj, body, points in ctx._fall_support_points:
        pose = poses[body]
        minimum = float(np.min(points @ pose[2, :3]) + pose[2, 3])
        if minimum < floor:
            if body not in ctx._observed_falls:
                record = dict(obj=obj, body=body, minimum_z_m=minimum,
                              floor_z_m=floor, control_step=ctx.steps)
                ctx._observed_falls[body] = record
                ctx.note('continuous scene fall detected', **record)
            else:
                ctx._observed_falls[body]['minimum_z_m'] = min(
                    minimum, ctx._observed_falls[body]['minimum_z_m'])
        elif body in ctx._observed_falls:
            record = ctx._observed_falls.pop(body)
            ctx.note('continuous scene fall recovered', **record, recovered_step=ctx.steps)


def item_action(obj, action):
    """Annotate a callable with its object for the generic retry cleanup."""
    action.object_name = obj
    return action


def drawer_base_goal(ctx):
    """The validated cabinet-relative stance, using the live cabinet frame."""
    cabinet = ctx.pose('cabinet')
    yaw = np.arctan2(cabinet[1, 0], cabinet[0, 0])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return np.r_[cabinet[:2, 3] + rotation @ DRAWER_APPROACH[:2],
                 wrapped(yaw + DRAWER_APPROACH[2])]


def checked_approach(ctx, goal, purpose, *, position_tolerance=None, angle_tolerance=None):
    """Physically follow a full-robot/payload checked route, correcting once."""
    s = ctx.skills()
    position_tolerance = BASE_TOLERANCE[0] if position_tolerance is None else position_tolerance
    angle_tolerance = BASE_TOLERANCE[1] if angle_tolerance is None else angle_tolerance
    before = s.base_pose().copy()
    if (np.linalg.norm(before[:2] - goal[:2]) < position_tolerance
            and abs(wrapped(before[2] - goal[2])) < angle_tolerance):
        return True
    remaining_wall(ctx)
    s.ready()
    remaining_wall(ctx)
    for attempt in range(2):
        route = s.base_route(s.base_pose(), goal)
        remaining_wall(ctx)
        if route is None:
            ctx.note(purpose + ' blocked', goal=goal.tolist(),
                     collisions=container_route_blockers(ctx, goal))
            break
        ctx.note(purpose + ' base route', goal=goal.tolist(), attempt=attempt + 1,
                 waypoints=[point.tolist() for point in route])
        for point in route:
            remaining_wall(ctx)
            first, distance, last = drive(s.base_pose(), point)
            s.move_base(0., first)
            remaining_wall(ctx)
            s.move_base(distance, last)
        actual = s.base_pose()
        distance = float(np.linalg.norm(actual[:2] - goal[:2]))
        angle = abs(float(wrapped(actual[2] - goal[2])))
        ctx.note(purpose + ' measured', attempt=attempt + 1, pose=actual.tolist(),
                 distance_m=distance, yaw_error_rad=angle)
        if distance < position_tolerance and angle < angle_tolerance:
            break
    remaining_wall(ctx)
    s.note_relocated(before)
    actual = s.base_pose()
    return (np.linalg.norm(actual[:2] - goal[:2]) < position_tolerance
            and abs(wrapped(actual[2] - goal[2])) < angle_tolerance)


def container_route_blockers(ctx, goal):
    """Record the first actual collision on each checked local base route."""
    s = ctx.skills()
    actual, poses = s.model.base.copy(), ctx.poses()
    result = []
    try:
        for route in drive_routes(s.base_pose(), goal):
            previous = s.base_pose()
            for waypoint in route:
                for along in drive_path(previous, waypoint):
                    s.put_base(along, poses)
                    pairs = s.touching(s.model.q)
                    if pairs:
                        result.append(dict(base=list(along), pairs=pairs[:16]))
                        break
                else:
                    previous = waypoint
                    continue
                break
    finally:
        s.model.base = actual
        for solver in s.solvers.values():
            solver.sync()
    return result


def prepare_container_navigation(ctx):
    """Raise the empty torso on a checked path before entering the book edge."""
    s = ctx.skills()
    semantic.require(ctx, not s.held, 'container navigation requires empty hands', kind='planning')
    start = np.asarray(ctx.env.qpos(ctx.spec.lift)[0], dtype=float)
    target = np.full_like(start, .70)
    if np.min(start) < .69:
        for arm in s.arms:
            joints = np.repeat(s.model.q[s.model.arm_indices(arm)][None], 2, axis=0)
            _, pairs = s.model.check_path(arm, joints, .01, np.vstack([start, target]))
            semantic.require(ctx, not pairs, 'empty torso rise blocked before container approach', kind='planning',
                             arm=arm, collisions=pairs)
        count = max(1, int(np.ceil(np.max(abs(target - start)) / ctx.spec.lift_velocity_limit / DT)))
        for k in range(1, count + 1):
            ctx.step({'lift': (start + (target - start) * k / count)[None]})
        ctx.hold(.3)
        for solver in s.solvers.values():
            solver.sync()
        ctx.note('empty torso raised for container route', start=start.tolist(),
                 actual=np.asarray(ctx.env.qpos(ctx.spec.lift)[0]).tolist())
    s.ready()


def container_grasp_stance(ctx, obj):
    """Reach and verify the measured seed-zero stance; never grasp from fallback."""
    prepare_container_navigation(ctx)
    goal = np.array([.01621752, -.66054994, 1.54142070] if obj == 'jar'
                    else [-.03408241, -.653326, 2.1637785])
    reached = checked_approach(ctx, goal, f'{obj} required container stance',
                               position_tolerance=.020, angle_tolerance=np.radians(5.))
    actual = ctx.skills().base_pose().copy()
    distance = float(np.linalg.norm(actual[:2] - goal[:2]))
    angle = abs(float(wrapped(actual[2] - goal[2])))
    ctx.note('container pre-grasp stance verified', obj=obj, target=goal.tolist(),
             actual=actual.tolist(), distance_m=distance, yaw_error_rad=angle)
    semantic.require(ctx, reached and distance <= .020 and angle <= np.radians(5.),
                     'container grasp stance not reached', kind='check', obj=obj, target=goal.tolist(),
                     actual=actual.tolist(), distance_m=distance, yaw_error_rad=angle)


def drawer_approach(ctx):
    """Use existing checked base skills before the normal braced-drawer search."""
    semantic.require(ctx, not ctx.skills().held, 'drawer approach requires empty hands', kind='planning')
    checked_approach(ctx, drawer_base_goal(ctx), 'drawer approach')


def retrieve_buffered(ctx, obj, where, **options):
    """Pick a far-X buffer locally, then return with it to the opened drawer."""
    s = ctx.skills()
    semantic.require(ctx, not s.held, 'buffer retrieval requires empty hands', kind='planning', obj=obj)
    centre = world_geometry.centre(world_geometry.object_points(ctx, obj))
    goal = drawer_base_goal(ctx)
    goal[0] = centre[0]
    goal[1] = min(goal[1], ctx.scene.ROBOTS[ROBOT]['position'][1])
    goal[2] = ctx.scene.ROBOTS[ROBOT]['yaw']
    ctx.note('far buffer retrieval distance', obj=obj,
             distance_m=float(np.linalg.norm(centre[:2] - s.base_pose()[:2])))
    with item_phase(ctx, obj, 'approach'):
        semantic.require(ctx, checked_approach(ctx, goal, 'buffer retrieval approach'),
                         'no checked approach to the table buffer', kind='check', obj=obj)
    pick = dict(approach=-ctx.pose('cabinet')[:3, 2], max_angle=np.radians([30., 45.]))
    if obj == 'screwdriver':
        pick.update(arm='right', part='screwdriver.body.grip')
    if obj == 'dish':
        pick['body'] = ctx.geometry[obj].bodies[0]
    with item_phase(ctx, obj, 'pick'):
        skills.chain(ctx, [skills.Pick(obj, instruction=f'retrieve buffered {obj}', **pick)])
    with item_phase(ctx, obj, 'return'):
        semantic.require(ctx, checked_approach(ctx, drawer_base_goal(ctx), 'carried drawer approach'),
                         'no checked payload route back to the drawer', kind='check', obj=obj)
    with item_phase(ctx, obj, 'place'):
        skills.chain(ctx, [skills.Place(obj, where, instruction=f'put away the retrieved {obj}', **options)])


def open_storage_drawer(ctx, number):
    if not drawer_joints.is_open(ctx, 'cabinet', joint(number), OPENING):
        started = perf_counter()
        ctx.note('item phase start', item=f'drawer{number}', phase='open_drawer', drawer=number)
        with ctx.subtask(f'open drawer {number}'):
            try:
                if number in (2, 3):
                    drawer_approach(ctx)
                semantic.open(ctx, 'cabinet', joint(number), OPENING, brace=True)
            finally:
                ctx.note('item phase result', item=f'drawer{number}', phase='open_drawer', drawer=number,
                         wall_s=perf_counter() - started)


def drawer_opened(ctx, number):
    semantic.require(ctx, drawer_joints.is_open(ctx, 'cabinet', joint(number), OPENING),
                     'drawer not open for storage', kind='check', drawer=number)


def open_loaded_drawer(ctx, number):
    """Pull the loaded drawer with one hand under the original stability guard."""
    if drawer_joints.is_open(ctx, 'cabinet', joint(number), OPENING):
        return
    from table_1000.experts.skills.drawer.bracing import stability
    with ctx.subtask(f'open loaded drawer {number}'):
        entry = drawer_joints.Joint.of(ctx.geometry['cabinet'], joint(number))
        with stability(ctx, 'cabinet', entry):
            drawer_approach(ctx)
            semantic.open(ctx, 'cabinet', joint(number), OPENING, brace=False)


def screw_drawer_half_opened(ctx):
    if contained(ctx, 'screwdriver', 'cabinet', drawer(3)):
        return
    slider = drawer_joints.slider(ctx, 'cabinet', joint(3))
    amount = (slider.limits[1] - slider.limits[0]) / 2
    semantic.require(ctx, drawer_joints.is_open(ctx, 'cabinet', joint(3), amount),
                     'drawer not half open for screwdriver', kind='check', drawer=3)


def open_screw_drawer_half(ctx):
    """The original tool blocks the full pull; half travel exposes its crosswise footprint."""
    if contained(ctx, 'screwdriver', 'cabinet', drawer(3)):
        return
    slider = drawer_joints.slider(ctx, 'cabinet', joint(3))
    amount = (slider.limits[1] - slider.limits[0]) / 2
    if not drawer_joints.is_open(ctx, 'cabinet', joint(3), amount):
        # The R6 physical half-pull succeeded with left support from this
        # seed-zero stance. The ordinary relocated stance blocks the other
        # hand's RRT approach to the bottom handle in the live R7 chain.
        semantic.require(ctx, checked_approach(
            ctx, np.array([.087924, -.937504, 1.051521]),
            'seed zero bottom drawer stance'), 'no checked bottom drawer stance', kind='check')
        skills.braced_drawer(ctx, 'cabinet', joint(3), amount=amount)


def exposed_drawer_region(ctx, obj, number):
    """Restrict the native goal to the floor in front of overhead cabinet material."""
    from table_1000.experts.skills.place.strict import PLACE_MARGIN
    region, blocked = regions.inside_region(
        obj, 'cabinet', drawer(number), top_tolerance=.005)(ctx)
    frame = np.asarray(region[0])
    inverse = np.linalg.inv(frame)
    low, high = np.array(region[1], copy=True), np.array(region[2], copy=True)
    for shape in world_geometry.scene_shapes(ctx):
        if shape.obj != 'cabinet' or world_geometry.selects(shape, 'cabinet', drawer(number)):
            continue
        points = shape.points @ inverse[:3, :3].T + inverse[:3, 3]
        a, b = points.min(0), points.max(0)
        if a[2] > 1e-6 and np.all(b[:2] >= low) and np.all(a[:2] <= high):
            high[1] = min(high[1], a[1] - PLACE_MARGIN)
    semantic.require(ctx, np.all(high > low), 'no exposed drawer floor', kind='planning', drawer=number)
    # Preserve every native height/support bound and blocked object.
    return (frame, low, high, *region[3:]), blocked


def store_screw_first(ctx):
    """After bounded drawer attempts, put the tool through the open basket mouth."""
    if contained(ctx, 'screwdriver', 'basket'):
        return
    close(ctx, 3)
    # Opening drawer1 blocks the grasp. Keep it closed and reuse the right
    # overhead grip that lifted the real tool from the half-drawer entry.
    skills.retry(ctx, lambda: skills.pick(
        ctx, 'screwdriver', part='screwdriver.body.grip', arm='right',
        approach=(0., 0., -1.), max_angle=np.radians(10.), base=False))
    skills.chain(ctx, [skills.Place('screwdriver', screw_basket_drop_region,
                                   instruction='put the screwdriver in the basket')])
    ctx.hold(1.)
    stored(ctx, 'screwdriver', 3)


def screw_basket_drop_region(ctx):
    return basket_drop_region(ctx, obj='screwdriver')


def drawer3_high_release_region(ctx, obj, number=3):
    """Drop through the exposed footprint with both wrist and elbow above the cabinet."""
    region, blocked = exposed_drawer_region(ctx, obj, number)
    frame = region[0].copy()
    release_z = float(world_geometry.object_points(ctx, 'cabinet')[:, 2].max()) + .04
    frame[:3, 3] += frame[:3, 2] * (release_z - frame[2, 3])
    ctx.note('high release above exposed drawer footprint', obj=obj,
             release_plane_z_m=float(frame[2, 3]), clearance_m=PLACE_CLEARANCE)
    return (frame, region[1], region[2]), blocked


def lidded_jar_release_region(ctx):
    return drawer3_high_release_region(ctx, 'lidded_jar')


def charger_release_region(ctx):
    return drawer3_high_release_region(ctx, 'charger')


def dish_release_region(ctx):
    return drawer3_high_release_region(ctx, 'dish')


def snack_release_region(ctx):
    return drawer3_high_release_region(ctx, 'snack')


def glue_bottle_drawer_release_region(ctx):
    native, _ = regions.inside_region('glue_bottle', 'cabinet', drawer(2), top_tolerance=.005)(ctx)
    region, blocked = drawer3_high_release_region(ctx, 'glue_bottle', 2)
    # The release plane is high, while the unchanged drawer height forces
    # the full bottle to lie horizontally before gravity drops it through.
    return (*region, {'maximum_height': native[3]['maximum_height']}), blocked


def board_eraser_drawer_release_region(ctx):
    """Release the flat eraser above the exposed drawer with native height bounds."""
    native, _ = regions.inside_region('board_eraser', 'cabinet', drawer(2), top_tolerance=.005)(ctx)
    region, blocked = drawer3_high_release_region(ctx, 'board_eraser', 2)
    return (*region, {'maximum_height': native[3]['maximum_height']}), blocked


def tape_drawer_release_region(ctx):
    return drawer3_high_release_region(ctx, 'tape', 2)


def stapler_drawer_release_region(ctx):
    return drawer3_high_release_region(ctx, 'stapler', 2)


def remote_basket_release_region(ctx):
    return basket_drop_region(ctx, obj='remote')


def screw_drawer3_high_release_region(ctx):
    return drawer3_high_release_region(ctx, 'screwdriver')


def store_drawer3_high(ctx, obj):
    # Select a real grasp independently from the low drawer placement need.
    # Keep the far-buffer items' existing independent phase deadlines.
    if obj == 'screwdriver':
        semantic.require(ctx, checked_approach(
            ctx, np.array([-.19, -.73, 1.48]), 'basket tool pickup stance'),
            'no checked route to basket tool pickup stance', kind='check')
    if obj == 'lidded_jar':
        semantic.require(ctx, checked_approach(
            ctx, np.array([.15, -.72, np.pi / 2]), 'jar shared pickup-release stance'),
            'no checked route to jar shared stance', kind='check')
    pick = ({'body': ctx.geometry[obj].bodies[0]}
            if obj in ('dish', 'tape', 'lidded_jar', 'board_eraser') else {})
    if obj == 'lidded_jar':
        # Its closed 94 mm shell fits the hand. The floor-only target
        # predicts a thin pinch while the fingers actually meet its walls.
        pick.update(approach=(0., 0., -1.), max_angle=np.radians(10.))
    if obj == 'remote':
        pick = table_pen_grip(ctx, obj) or {}
    if obj == 'charger':
        # The pin pinch missed its measured area, then the next approach
        # swept the loose plug off the table. Pinch the housing from above.
        pick = {'part': 'power_plug.body.shell', 'arm': 'left',
                'approach': (0., 0., -1.), 'max_angle': np.radians(10.)}
    with item_phase(ctx, obj, 'pick'):
        if obj == 'lidded_jar':
            # The first-pass successful hold used a vertical floor-envelope
            # candidate whose real pads closed on both exterior walls. Keep
            # ordinary part enumeration, but exclude the failed tilted pinch.
            skills.retry(ctx, lambda: skills.pick(
                ctx, obj, approach=(0., 0., -1.), max_angle=np.radians(10.), base=False))
        else:
            skills.retry(ctx, lambda: skills.pick(ctx, obj, **pick))
    with item_phase(ctx, obj, 'return'):
        if obj == 'remote':
            arm = ctx.skills().holding(obj)
            bottom = float(world_geometry.object_points(ctx, obj)[:, 2].min())
            height = float(world_geometry.object_points(ctx, 'basket')[:, 2].max()) + CLEAR
            if bottom < height:
                skills.move_by_displacement(ctx, arm, z=height - bottom)
        else:
            drawer3_payload_stance(ctx, obj)
    where = {'lidded_jar': lidded_jar_release_region, 'charger': charger_release_region,
             'dish': dish_release_region, 'snack': snack_release_region,
             'screwdriver': screw_drawer3_high_release_region,
             'glue_bottle': glue_bottle_drawer_release_region,
             'board_eraser': board_eraser_drawer_release_region,
             'tape': tape_drawer_release_region,
             'stapler': stapler_drawer_release_region,
             'remote': remote_basket_release_region}[obj]
    with item_phase(ctx, obj, 'place'):
        region, _ = where(ctx)
        near = region[0][:3, 3] + region[0][:3, :2] @ np.array([0., region[1][1] + .055])
        skills.chain(ctx, [skills.Place(obj, where, near=near,
                                       orientation=('insert' if obj == 'glue_bottle' else
                                                    'flat' if obj == 'board_eraser' else None),
                                       instruction=(f'let {obj} fall through the basket mouth' if obj == 'remote'
                                                    else f'let {obj} fall through the exposed drawer mouth'))])
        ctx.hold(1.)


def drawer3_payload_stance(ctx, obj):
    """Carry above the cabinet, then use the seed-zero front stance for release."""
    arm = ctx.skills().holding(obj)
    bottom = float(world_geometry.object_points(ctx, obj)[:, 2].min())
    height = float(world_geometry.object_points(ctx, 'cabinet')[:, 2].max()) + CLEAR
    if bottom < height:
        skills.move_by_displacement(ctx, arm, z=height - bottom)
    if obj == 'lidded_jar':
        # The old mandatory base goal is blocked with the new full-payload
        # check. First let Place solve from the physically verified hold.
        ctx.note('jar release from pickup stance', base_pose=ctx.skills().base_pose().tolist())
        return
    if not checked_approach(ctx, np.array([.36, -.80, np.pi / 2]),
                            'drawer3 payload release stance'):
        # A fixed historical base goal is optional. Place still checks the
        # complete physical payload and may choose another reachable stance.
        ctx.note('release uses current stance after blocked historical goal',
                 obj=obj, base_pose=ctx.skills().base_pose().tolist())


def bottle_side_done(ctx):
    if contained(ctx, 'glue_bottle', 'cabinet', drawer(2)):
        return
    semantic.require(ctx, semantic.on(ctx, 'glue_bottle', 'table'), 'side bottle not on table', kind='check')
    semantic.require(ctx, abs(principal_axis(ctx, 'glue_bottle')[2]) < np.sin(ANGLE_TOLERANCE),
                     'bottle did not rest on its side', kind='check')


def bottle_side_region(ctx):
    """Release horizontally over the cleared left-front patch, then let it settle."""
    size = np.sort(np.ptp(ctx.geometry['glue_bottle'].points(), axis=0))
    region, blocked = regions.on_region('glue_bottle', 'table',
        keep_out=front_buffer_workspace(ctx))(ctx)
    frame = region[0].copy()
    inverse = np.linalg.inv(frame)
    corners = np.array([[-.61, -.44, ctx.scene.TABLE['height']],
                        [-.39, -.25, ctx.scene.TABLE['height']]])
    local = corners @ inverse[:3, :3].T + inverse[:3, 3]
    low = np.maximum(region[1], local[:, :2].min(0))
    high = np.minimum(region[2], local[:, :2].max(0))
    frame[:3, 3] += frame[:3, 2] * .12
    metadata = region[3] if len(region) > 3 else {}
    return (frame, low, high, {**metadata, 'maximum_height': float(size[1]) + .002}), blocked


def rest_bottle_on_side(ctx):
    """Grasp before asking for the side contact on a clear table patch."""
    if contained(ctx, 'glue_bottle', 'cabinet', drawer(2)):
        return
    semantic.require(ctx, checked_approach(ctx, np.array([-.12, -.80, np.pi / 2]),
                                          'bottle side preparation stance'),
                     'no checked bottle preparation stance', kind='check')
    skills.retry(ctx, lambda: skills.pick(ctx, 'glue_bottle'))
    skills.chain(ctx, [skills.Place('glue_bottle', bottle_side_region, orientation='insert',
                                   instruction='rest glue_bottle on its side on the table')])


def settle_side_bottle(ctx):
    if contained(ctx, 'glue_bottle', 'cabinet', drawer(2)):
        return
    bottle_side_done(ctx)
    before = ctx.pose('glue_bottle').copy()
    ctx.hold(1.)
    after = ctx.pose('glue_bottle')
    travel = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
    turn = float(np.arccos(np.clip((np.trace(after[:3, :3] @ before[:3, :3].T) - 1) / 2, -1., 1.)))
    ctx.note('side bottle stability measured', travel_m=travel, turn_rad=turn)
    semantic.require(ctx, travel <= .01 and turn <= ANGLE_TOLERANCE,
                     'side bottle moved while settling', kind='check', travel_m=travel, turn_rad=turn)
    bottle_side_done(ctx)


def basket_drop_region(ctx, obj='glue_stick'):
    """Keep the live basket floor projection, but release above its open rim."""
    shapes = world_geometry.scene_shapes(ctx, exclude=(obj,))
    basket = [shape for shape in shapes if shape.obj == 'basket']
    region, blocked = world_geometry.support_region(basket, 'basket', floor=True)
    semantic.require(ctx, region is not None, 'basket floor unavailable', kind='planning', obj=obj)
    frame, low, high = region
    inverse = np.linalg.inv(frame)
    points = np.vstack([shape.points for shape in basket]) @ inverse[:3, :3].T + inverse[:3, 3]
    mouth = float(points[:, 2].max())
    frame = frame.copy()
    frame[:3, 3] += frame[:3, 2] * (mouth + .02)
    contents = [shape.points for shape in shapes if shape.obj != 'basket'
                and (shape.points @ inverse[:3, :3].T + inverse[:3, 3])[:, 2].max() > mouth]
    return (frame, low, high), [*blocked, *contents]


def lidded_jar_basket_region(ctx):
    return basket_drop_region(ctx, obj='lidded_jar')


def tape_basket_region(ctx):
    # The basket centre already contains the tool/glue. Select its empty
    # left-rear floor corner so the tape can gain direct internal support.
    region, blocked = basket_drop_region(ctx, obj='tape')
    frame, low, high = region
    low, high = np.array(low, copy=True), np.array(high, copy=True)
    high[0] = min(high[0], low[0] + .105)
    low[1] = max(low[1], high[1] - .105)
    return (frame, low, high), blocked


def store_in_alternate(ctx, obj):
    container = STORAGE_OVERRIDES[obj]
    if contained(ctx, obj, container):
        return
    # The last strategy stays within this step's remaining forty-minute window.
    ctx.wall_deadline = min(ctx.wall_deadline, perf_counter() + 360.)
    if obj == 'tape':
        with item_phase(ctx, obj, 'pick'):
            skills.retry(ctx, lambda: skills.pick(ctx, obj, body=ctx.geometry[obj].bodies[0]))
        with item_phase(ctx, obj, 'mouth_clearance'):
            bottom = float(world_geometry.object_points(ctx, obj)[:, 2].min())
            mouth = float(world_geometry.object_points(ctx, 'basket')[:, 2].max()) + CLEAR
            if bottom < mouth:
                skills.move_by_displacement(ctx, ctx.skills().holding(obj), z=mouth - bottom)
        with item_phase(ctx, obj, 'place'):
            skills.chain(ctx, [skills.Place(obj, tape_basket_region,
                                           instruction='put tape through the basket mouth')])
        ctx.hold(1.)
        return
    if obj == 'lidded_jar':
        skills.retry(ctx, lambda: skills.pick(ctx, obj))
        skills.chain(ctx, [skills.Place(obj, lidded_jar_basket_region,
                                       instruction='put the small lidded jar in the basket')])
        ctx.hold(1.)
        return
    checked_approach(ctx, np.array([.36, -.80, np.pi / 2]), 'seed zero independent pickup')
    # Separate actions deliberately omit the drawer's lookahead constraint.
    skills.chain(ctx, [skills.Pick(obj, body=ctx.geometry[obj].bodies[0],
                                  instruction=f'pick up the {obj}')])
    skills.chain(ctx, [skills.Place(obj, basket_drop_region,
                                   instruction=f'put the {obj} in the basket')])
    ctx.hold(1.)


def store(ctx, obj, number, *, open_if_needed=True):
    if obj in STORAGE_OVERRIDES:
        return store_in_alternate(ctx, obj)
    if contained(ctx, obj, 'cabinet', drawer(number)):
        return
    if open_if_needed:
        open_storage_drawer(ctx, number)
    else:
        drawer_opened(ctx, number)
    if (number == 3 and obj in ('lidded_jar', 'charger', 'dish', 'snack')
            or number == 2 and obj in ('board_eraser', 'tape', 'stapler')):
        return store_drawer3_high(ctx, obj)
    size = np.sort(np.ptp(ctx.geometry[obj].points(), axis=0))
    body = ctx.geometry['cabinet'].parts(drawer(number))[0][0]
    low, high = ctx.geometry['cabinet'].bounds(drawer(number), frame=body)
    height = float(high[2] - low[2])
    orientation = ('flat' if size[1] >= 2 * size[0] else
                   'insert' if size[2] >= 2 * size[1] and size[2] > height else None)
    if obj == 'glue_bottle':
        bottle_side_done(ctx)
        return store_drawer3_high(ctx, obj)
    far_buffer = (obj in ('screwdriver', 'charger', 'dish')
                  and world_geometry.centre(world_geometry.object_points(ctx, obj))[0]
                  < ctx.scene.TABLE['center'][0] - ctx.scene.TABLE['size'][0] / 4)
    if far_buffer and obj == 'screwdriver':
        # Retrieval keeps the table-resting tilt. A shallow drawer does not
        # require the long vertical descent used for a narrow-mouth insert.
        # The measured full-object resting-height bound still applies.
        orientation = None
    where = regions.inside_region(obj, 'cabinet', drawer(number),
                                   orientation=orientation, top_tolerance=.005)
    if far_buffer:
        retrieve_buffered(ctx, obj, where, **({'orientation': orientation} if orientation else {}))
    elif obj in ('screwdriver', 'charger', 'dish'):
        with item_phase(ctx, obj, 'pick_place'):
            transfer(ctx, obj, where, **({'orientation': orientation} if orientation else {}))
    else:
        transfer(ctx, obj, where, **({'orientation': orientation} if orientation else {}))


def front_buffer_workspace(ctx):
    """Keep buffers at the desk end opposite the cabinet, clear of both drawer approaches."""
    centre, size = np.asarray(ctx.scene.TABLE['center']), np.asarray(ctx.scene.TABLE['size'])
    low, high = centre - size / 2, centre + size / 2
    if ctx.pose('cabinet')[0, 3] >= centre[0]:
        low[0] = centre[0] - size[0] / 4
    else:
        high[0] = centre[0] + size[0] / 4
    goals = []
    for obj in DESTINATIONS:
        points = ctx.geometry[obj].points()[:, :2] + destination(ctx, obj)[:2]
        goals.append((np.eye(4), points.min(0) - CLEAR, points.max(0) + CLEAR))
    return [*workspace(ctx), *goals, (np.eye(4), low, high)]


def clear_drawer_front(ctx, obj):
    """Buffer a proven drawer3 deadlock before opening that drawer."""
    if not contained(ctx, obj, 'cabinet', drawer(3)):
        table = ctx.scene.TABLE
        centre, size = np.asarray(table['center']), np.asarray(table['size'])
        side = -1. if ctx.pose('cabinet')[0, 3] >= centre[0] else 1.
        near = np.r_[centre + size * [side * 3 / 8, -1 / 3], table['height']]
        buffer(ctx, obj, keep_out=front_buffer_workspace(ctx), near=near)


def drawer_front_clear(ctx, obj):
    if not contained(ctx, obj, 'cabinet', drawer(3)):
        buffered(ctx, obj, keep_out=front_buffer_workspace(ctx))


def remote_to_drawer(ctx):
    """Use the close seed-zero stance for the low remote resting on A5."""
    drawer_opened(ctx, 3)
    if not checked_approach(ctx, np.array([.055, -.82, np.pi / 2]),
                            'close remote pickup stance'):
        ctx.note('remote uses current stance after blocked historical goal',
                 base_pose=ctx.skills().base_pose().tolist())
    # The old panel push has no collision-free candidate from this chain.
    # A5 remains a support; let the ordinary pick plan its approach directly.
    ctx.note('remote direct pickup without unreachable preliminary push')
    store_drawer3_high(ctx, 'remote')


def contained(ctx, obj, container, part=None):
    """Use actual immersion and inner-wall/floor support for all storage."""
    return semantic.storage_result(ctx, obj, container, part)


def stored(ctx, obj, number):
    # Accept the original drawer target or the explicitly documented basket plan.
    if obj == 'screwdriver' and contained(ctx, obj, 'cabinet', drawer(3)):
        return
    if obj in STORAGE_OVERRIDES:
        container = STORAGE_OVERRIDES[obj]
        semantic.require(ctx, contained(ctx, obj, container),
                         'object not completely in its alternate container', kind='check', obj=obj, container=container)
        return
    number = DRAWER_OVERRIDES.get(obj, number)
    semantic.require(ctx, contained(ctx, obj, 'cabinet', drawer(number)),
                     'object not completely in its drawer', kind='check', obj=obj, drawer=number)


def close(ctx, number):
    if not drawer_joints.is_closed(ctx, 'cabinet', joint(number)):
        with ctx.subtask(f'close drawer {number}'):
            if number == 2:
                # Reuse the measured step30 opening stance instead of
                # searching from the high-release stance beside the cabinet.
                # Observe from the action-entry pose, including the checked
                # base approach, then push with one hand under the same guard.
                from table_1000.experts.skills.drawer.bracing import stability
                entry = drawer_joints.Joint.of(ctx.geometry['cabinet'], joint(number))
                with stability(ctx, 'cabinet', entry):
                    semantic.require(ctx, checked_approach(
                        ctx, np.array([0., -.90, .96]), 'middle drawer close stance'),
                        'no checked route to middle drawer close stance', kind='check')
                    semantic.close(ctx, 'cabinet', joint(number), brace=False)
            else:
                semantic.close(ctx, 'cabinet', joint(number), brace=True)


def drawer_done(ctx, number):
    semantic.require(ctx, drawer_joints.is_closed(ctx, 'cabinet', joint(number)),
                     'drawer not closed', kind='check', drawer=number)
    for obj in DRAWERS[number]:
        stored(ctx, obj, number)


def drawer_closed(ctx, number):
    semantic.require(ctx, drawer_joints.is_closed(ctx, 'cabinet', joint(number)),
                     'drawer not closed', kind='check', drawer=number)


def principal_axis(ctx, obj, longest=True):
    points = ctx.geometry[obj].points()
    _, _, axes = np.linalg.svd(points - points.mean(0), full_matrices=False)
    return ctx.pose(obj)[:3, :3] @ axes[0 if longest else -1]


def standing(ctx, obj, container):
    """Use the same immersed-segment predicate before and after insertion."""
    return semantic.storage_result(ctx, obj, container)


def withdraw_pen(ctx, obj, container):
    """Leave the surrounding basket before lifting; choose the insertion stance afterwards."""
    if obj == 'pen_white':
        semantic.require(ctx, checked_approach(
            ctx, np.array([-.217345, -.791759, 1.01820]), 'seed zero white pen approach'),
            'no checked white pen stance', kind='check')
    elif obj == 'highlighter_yellow':
        semantic.require(ctx, checked_approach(
            ctx, np.array([-.192248, -.738335, .805276]), 'seed zero yellow pen approach'),
            'no checked yellow pen stance', kind='check')
    skill = ctx.skills()
    # The end-contact stream leaves the fingers above the mouth on insertion.
    # Do not require one stance to reach the basket, lift and holder together.
    need = free_insertion_need(obj, skill.arms)
    options = ({**(table_pen_grip(ctx, obj) or {}), 'arm': 'left',
                'approach': (0., 0., -1.), 'max_angle': np.radians([30., 45.]),
                'base': False}
               if obj == 'pen_white' else {'arm': 'left', 'base': False}
               if obj == 'highlighter_yellow' else {})
    shape_options = {key: options.pop(key) for key in ('shape', 'offset', 'body') if key in options}
    target = skill.target(obj, **shape_options)
    angles = options.pop('max_angle', None)
    result = skills.relaxed(angles, lambda angle: skill.grasp(target, need=need, max_angle=angle, **options))
    arm = result['arm']
    solver, target = ctx.solver(arm), skill.held[arm]['target']
    before = ctx.poses()[target.body].copy()
    backward = -solver.tcp()[:3, 0] * APPROACH
    skills.move_by_displacement(ctx, arm, x=float(backward[0]), y=float(backward[1]), z=float(backward[2]))
    skills.move_by_displacement(ctx, arm, z=LIFT)
    ctx.hold(.2)
    poses = ctx.poses()
    after = poses[target.body]
    rise = float(after[2, 3] - before[2, 3])
    if rise < MIN_RISE:
        skills.release(ctx, arm)
        ctx.fail('object not lifted', kind='check', rise=rise, finger_gap=solver.finger_gap(), arm=arm,
                 grasp=result['grasp'], object_before=before.tolist())
    area = skill.measured_held_area(target, arm, after @ target.offset, solver.tcp(), solver.finger_gap())
    ctx.note('withdrawal area diagnostic', obj=obj, held_area_m2=area)
    for name in list(solver.attachments):
        solver.attach(name, poses[solver.world.objects[name].body])
    ctx.note('pen withdrawn before insertion', obj=obj, arm=arm, rise=rise, held_area_m2=area)


def insert_pen(ctx, obj, container):
    ctx._standing_at_pen_entry = {
        name: destination for destination, names in (
            ('holder', getattr(ctx.scene, 'SHORT_PENS', ())),
            ('cup', getattr(ctx.scene, 'LONG_PENS', ())))
        for name in names if name != obj and standing(ctx, name, destination)}
    qualified = standing(ctx, obj, container)
    if obj in ('highlighter_pink', 'marker_black'):
        # These pens already qualify in layout 0 seed 0. A later disturbance
        # must fail instead of silently repairing it with an individual pick.
        semantic.require(ctx, qualified, 'initially qualified pen was disturbed', kind='check',
                         obj=obj, container=container)
    if qualified:
        ctx.note('leave qualified standing pen in place', obj=obj, container=container)
        return
    if obj in ('gel_pen', 'pen_black'):
        source = world_geometry.centre(world_geometry.object_points(ctx, obj))
        destination = world_geometry.centre(world_geometry.object_points(ctx, container))
        table = ctx.scene.TABLE
        goal = np.array([(source[0] + destination[0]) / 2,
                         table['center'][1] - table['size'][1] / 2 - .32, np.pi / 2])
        semantic.require(ctx, checked_approach(ctx, goal, 'pen and cup approach'),
                         'no checked approach for pen and cup', kind='check', obj=obj, container=container)
    where = (pen_release_region(obj, container) if obj in ('gel_pen', 'pen_black', 'pen_white') else
             regions.inside_region(obj, container, orientation='upright'))
    for attempt in range(2):
        try:
            if not attempt and obj in ('highlighter_yellow', 'pen_white'):
                def withdraw_and_place():
                    withdraw_pen(ctx, obj, container)
                    return skills.chain(ctx, [skills.Place(
                        obj, where, orientation='upright', turns=int(360 / PLACE_TURN_STEP))])
                skills.retry(ctx, withdraw_and_place)
            else:
                pick = table_pen_grip(ctx, obj) if obj in ('gel_pen', 'pen_black') else None
                options = {'turns': int(360 / PLACE_TURN_STEP)} if obj in ('gel_pen', 'pen_black') else {}
                transfer(ctx, obj, where, **({'pick_options': pick} if pick is not None else {}),
                         orientation='upright', **options)
            if obj in ('gel_pen', 'pen_black', 'pen_white'):
                semantic.require(ctx, standing(ctx, obj, container),
                                 'released pen lacks immersion or inner support; pick it again', kind='check',
                                 obj=obj, container=container, recoverable=True)
            return
        except ExpertFailed as failure:
            if attempt or not failure.details.get('recoverable'):
                raise
            # Continue from the measured failure state, without restoring a
            # snapshot or disturbing qualified pens. A held pen is put back
            # on a geometry-selected table patch before this one regrasp.
            ctx.note('regrasp pen after insertion failure', obj=obj, reason=failure.reason)
            if ctx.skills().holding(obj) is not None:
                skills.chain(ctx, [skills.Place(obj, regions.on_region(obj, 'table', keep_out=workspace(ctx)),
                                               instruction=f'put the {obj} back on the table')])


def table_pen_grip(ctx, obj, part=None):
    """Search a table-resting pen along its whole measured horizontal axis."""
    target = ctx.skills().target(obj, part=part)
    pose = ctx.poses()[target.body]
    axis = principal_axis(ctx, obj)
    if np.linalg.norm(axis[:2]) <= abs(axis[2]):
        return None
    axis = axis.copy()
    axis[2] = 0.
    axis /= np.linalg.norm(axis)
    rotation = np.column_stack([axis, np.cross([0., 0., 1.], axis), [0., 0., 1.]])
    world = (target.points @ pose[:3, :3].T + pose[:3, 3]
             if part is not None else world_geometry.object_points(ctx, obj))
    points = world @ rotation
    frame = np.eye(4)
    frame[:3, :3] = rotation
    frame[:3, 3] = rotation @ ((points.min(0) + points.max(0)) / 2)
    return dict(shape=SampledBox(tuple(np.ptp(points, axis=0)), contact_step=.005),
                offset=np.linalg.inv(pose) @ frame, body=target.body.split(':', 1)[1],
                approach=[0., 0., -1.], max_angle=np.radians([30., 45.]))


def pen_release_region(obj, container):
    """Release a vertical loose pen at the measured mouth and let it settle."""
    inside = regions.inside_region(obj, container, orientation='upright')
    def where(ctx):
        region, blocked = inside(ctx)
        frame = region[0].copy()
        points = world_geometry.object_points(ctx, container)
        mouth = float(np.max((points - frame[:3, 3]) @ frame[:3, 2]))
        # Immersion and support decide storage. The former tilt limit drove
        # deep release and tip-only holds; keep the opening above the mouth.
        height = .02
        frame[:3, 3] += frame[:3, 2] * (mouth + height)
        ctx.note('pen release at container mouth', obj=obj, container=container,
                 mouth_m=mouth, release_bottom_above_m=height + min(PLACE_CLEARANCE, INSERT_MARGIN))
        # This release plane preserves the valid mouth projection, rounded
        # corners and real keepouts; the final standing check is unchanged.
        return (frame, region[1], region[2]), blocked
    return where


def pen_done(ctx, obj, container):
    semantic.require(ctx, standing(ctx, obj, container), 'pen not standing in its holder', kind='check',
                     obj=obj, container=container)
    for name, destination in getattr(ctx, '_standing_at_pen_entry', {}).items():
        semantic.require(ctx, standing(ctx, name, destination),
                         'previously standing pen disturbed', kind='check', obj=name, container=destination)


def store_clip(ctx, obj):
    if not contained(ctx, obj, 'jar'):
        # On the seed-0 front-left pink clip the left-hand pinch only catches
        # the articulated wire tips. Approach from its other side instead.
        pick_options = {'arm': 'right'} if obj == 'clip_pink' else None
        transfer(ctx, obj, clip_release_region(obj), pick_options=pick_options)


def clip_release_region(obj):
    """Release loose clips above the live mouth, then let them settle inside."""
    def where(ctx):
        shapes = world_geometry.scene_shapes(ctx, exclude=(obj,))
        jar = [shape for shape in shapes if world_geometry.selects(shape, 'jar', None)]
        region, blocked = world_geometry.support_region(jar, 'jar', floor=True)
        semantic.require(ctx, region is not None, 'no jar floor for drop', kind='planning', obj=obj)
        frame, low, high = region
        local = np.linalg.inv(frame)
        points = np.vstack([shape.points for shape in jar]) @ local[:3, :3].T + local[:3, 3]
        mouth = float(points[:, 2].max())
        frame = frame.copy()
        # The ordinary place clearance adds another centimetre: the clip's
        # lowest point is released 3 cm above the measured rim.
        frame[:3, 3] += frame[:3, 2] * (mouth + .02)
        ctx.note('clip release above jar mouth', obj=obj, mouth_m=mouth,
                 release_bottom_above_m=.02 + PLACE_CLEARANCE)
        # Retain the floor's unsupported cells, including the rounded corners,
        # and check protruding contents as real 3D collision geometry.
        contents = [shape.points for shape in shapes if shape.obj != 'jar'
                    and (shape.points @ local[:3, :3].T + local[:3, 3])[:, 2].max() > mouth]
        return (frame, low, high), [*blocked, *contents]
    return where


def clip_done(ctx, obj):
    semantic.require(ctx, contained(ctx, obj, 'jar'), 'clip not in the jar', kind='check', obj=obj)


def leave_book(ctx, obj):
    """Keep an unresolved book at its input pose for the skip-books chain."""
    ctx.note('leave unresolved book in its initial position', obj=obj)


def book_left(ctx, obj):
    points = world_geometry.object_points(ctx, obj)
    table = ctx.scene.TABLE
    centre = world_geometry.centre(points)
    low = np.asarray(table['center']) - np.asarray(table['size']) / 2
    high = np.asarray(table['center']) + np.asarray(table['size']) / 2
    semantic.require(ctx, points[:, 2].min() >= table['height'] - .005
                     and np.all(centre[:2] >= low) and np.all(centre[:2] <= high),
                     'deferred book fell off the desk', kind='check', obj=obj)
    ctx.note('unresolved book remains at its initial position', obj=obj)


def destination(ctx, obj):
    xy = NEAT_GOALS[obj][0] if obj in ('holder', 'cup') else np.asarray(ctx.scene.TABLE['center']) + DESTINATIONS[obj]
    return np.r_[xy, ctx.scene.TABLE['height']]


def container_placement_goal(ctx, obj):
    near = destination(ctx, obj).copy()
    if obj not in ('holder', 'cup'):
        near[:2] += CONTAINER_OFFSETS.get(obj, (0., 0.))
    return near


def container_goal_workspace(ctx):
    """Reserve the final footprints while the basket is held at a staging pose."""
    regions = list(workspace(ctx))
    table_height = ctx.scene.TABLE['height']
    for obj in ('cabinet', 'jar', 'holder', 'cup'):
        body = ctx.geometry[obj].bodies[0]
        _, low, high = world_geometry.body_box(ctx, obj, body, CLEAR)
        goal = np.eye(4)
        goal[:3, 3] = destination(ctx, obj)
        # Layout 5 destinations use zero heading; keep the local body bounds
        # conservative while projecting them into the final world frame.
        goal[2, 3] = table_height
        regions.append((goal, low, high))
    return regions


def at_destination(ctx, obj):
    return (np.linalg.norm(ctx.pose(obj)[:2, 3] - destination(ctx, obj)[:2]) <= POSITION_TOLERANCE
            and abs(np.arctan2(ctx.pose(obj)[1, 0], ctx.pose(obj)[0, 0])) <= np.radians(10.))


def align_container(ctx, obj):
    if obj in TABLETOP_CONTAINER_GOALS:
        return square_container(ctx, obj)
    if obj in RETAINED_CONTAINER_GOALS:
        ctx.note('revised container plan retains supported input', obj=obj,
                 goal=RETAINED_CONTAINER_GOALS[obj])
        neat_container_pose(ctx, obj)
        ctx.hold(1.)
        return
    near = container_placement_goal(ctx, obj)
    if at_destination(ctx, obj) and (obj not in CONTAINER_OFFSETS
            or np.linalg.norm(ctx.pose(obj)[:2, 3] - near[:2]) <= .01):
        return
    if obj == 'basket':
        return align_basket_on_table(ctx)
    if obj in ('jar', 'holder', 'cup'):
        prepare_container_navigation(ctx)
        container_table_approach(ctx, rear=False)
    where = ({'jar': jar_destination_region, 'basket': basket_destination_region}[obj]
             if obj in ('jar', 'basket') else
             (lambda ctx: destination_region(ctx, obj)) if obj in ('cup', 'holder') else
             regions.on_region(obj, 'table', near=near))
    if obj not in ('jar', 'basket', 'holder', 'cup'):
        return transfer(ctx, obj, where, near=near)
    # A layout asks for an absolute heading, while ordinary table placement
    # deliberately preserves the measured heading. Keep the common planner
    # and all its checks; seed its candidates at the layout heading only for
    # this final container move, including Pick's look-ahead requirement.
    s = ctx.skills()
    ordinary = s.placements
    def layout_placements(target, current, *args, **kwargs):
        if target.obj != obj:
            return ordinary(target, current, *args, **kwargs)
        aligned = current.copy()
        aligned[:3, :3] = np.eye(3)
        return [(pose, spot) for pose, spot in ordinary(target, aligned, *args, **kwargs)
                if np.linalg.norm(pose[:2, 3] - destination(ctx, obj)[:2]) <= POSITION_TOLERANCE
                and abs(np.arctan2(pose[1, 0], pose[0, 0])) <= np.radians(10.)]
    s.placements = layout_placements
    try:
        return transfer(ctx, obj, where, near=near, **({'step': .01} if obj in CONTAINER_OFFSETS else {}))
    finally:
        s.placements = ordinary


def align_basket_on_table(ctx):
    """Return the rear-staged full basket by two measured tabletop push legs."""
    goal = container_placement_goal(ctx, 'basket')
    container_table_approach(ctx, rear=True)
    for axis, part, stance in (
            (0, 'file_basket.shell.end_wall1', (-.50, .84, -np.pi / 2)),
            (1, 'file_basket.shell.long_wall2', (.04, .81, -np.pi / 2))):
        semantic.require(ctx, checked_approach(ctx, np.array(stance),
                         f'basket final push axis {axis} stance'),
                         'no checked stance for basket final push', kind='check', axis=axis)
        for attempt in range(4):
            before = ctx.pose('basket')[:3, 3].copy()
            error = float(goal[axis] - before[axis])
            if abs(error) <= .012:
                break
            direction = np.zeros(3)
            direction[axis] = np.sign(error)
            distance = min(.20, abs(error))
            ctx.note('basket measured final tabletop leg', axis=axis,
                     attempt=attempt + 1, start=before.tolist(), goal=goal.tolist(),
                     direction=direction.tolist(), distance_m=distance)
            selected_part = ('file_basket.shell.end_wall' if axis == 0 else
                             'file_basket.shell.long_wall') + ('1' if error > 0 else '2')
            skills.retry(ctx, lambda: skills.push(
                ctx, 'basket', part=selected_part, direction=direction, distance=distance))
            ctx.hold(.5)
        semantic.require(ctx, abs(float(goal[axis] - ctx.pose('basket')[axis, 3])) <= .020,
                         'basket tabletop leg did not converge', kind='check', axis=axis)


def jar_destination_region(ctx):
    return container_destination_region(ctx, 'jar')


def basket_destination_region(ctx):
    return container_destination_region(ctx, 'basket')


def container_destination_region(ctx, obj):
    """Bound final placement by the layout footprint and its acceptance tolerance."""
    near = destination(ctx, obj)
    region, blocked = regions.on_region(obj, 'table', near=near)(ctx)
    frame, low, high = region[:3]
    inverse = np.linalg.inv(frame)
    points = ctx.geometry[obj].points() + near
    local = points @ inverse[:3, :3].T + inverse[:3, 3]
    allowance = POSITION_TOLERANCE / np.sqrt(2.) + PLACE_MARGIN
    return (frame, np.maximum(low, local[:, :2].min(0) - allowance),
            np.minimum(high, local[:, :2].max(0) + allowance), *region[3:]), world_keepouts(blocked)


def world_keepouts(blocked):
    """Preserve local solid bounds in the frame used by placement collision checks."""
    # A tilted pen's local XY rectangle is not a vertical table keep-out:
    # projecting the table into it can exclude remote positions. Preserve
    # the same measured 3D bounds as a world convex solid instead.
    solids = []
    for obstacle in blocked:
        if isinstance(obstacle, np.ndarray):
            solids.append(obstacle)
            continue
        obstacle_frame, obstacle_low, obstacle_high = obstacle
        if len(obstacle_low) == 3 and np.all(np.asarray(obstacle_high) > obstacle_low):
            corners = np.array([[x, y, z]
                                for x in (obstacle_low[0], obstacle_high[0])
                                for y in (obstacle_low[1], obstacle_high[1])
                                for z in (obstacle_low[2], obstacle_high[2])])
            solids.append(corners @ obstacle_frame[:3, :3].T + obstacle_frame[:3, 3])
        else:
            solids.append(obstacle)
    return solids


def container_table_approach(ctx, *, rear, loaded=None):
    """Reach a table side through collision-checked exterior waypoints."""
    s, table = ctx.skills(), ctx.scene.TABLE
    semantic.require(ctx, (s.holding(loaded) is not None and len(s.held) == 1) if loaded else not s.held,
                     'container approach requires its declared payload' if loaded else
                     'container approach requires empty hands', kind='planning')
    s.ready()
    centre, half = np.asarray(table['center']), np.asarray(table['size']) / 2
    before = s.base_pose().copy()
    # Retain at least the authored initial stand-off when travelling around
    # the table; a later grasp stance can be closer than the base may turn.
    initial = np.asarray(ctx.scene.ROBOTS[ROBOT]['position'])
    clearance = max(.64, max(abs(before[1] - centre[1]), abs(initial[1] - centre[1])) - half[1] + CLEAR)
    sign_y = 1. if rear else -1.
    destination_y = centre[1] + sign_y * (half[1] + clearance)
    heading = -sign_y * np.pi / 2
    if sign_y * (before[1] - centre[1]) >= half[1]:
        return
    retreat = before.copy()
    retreat[:2] -= clearance * np.array([np.cos(before[2]), np.sin(before[2])])
    poses, actual = ctx.poses(), s.model.base.copy()
    final = [np.array([centre[0], destination_y, heading])]
    entry = np.array([.01621752, -.86, 1.54142070])
    if not rear and entry is not None:
        # Turn beyond the overhanging books, then enter the free left-front
        # work patch by a straight, fully checked approach.
        final = [np.array([entry[0], destination_y, heading]), entry.copy()]
    selected = None
    for sign in (-1., 1.):
        side = centre[0] + sign * (half[0] + clearance)
        route = [retreat, np.array([side, retreat[1], before[2]]),
                 np.array([side, destination_y, heading]), *final]
        try:
            previous, clear = before, True
            for index, goal in enumerate(route):
                if s.base_route(previous, goal) is None:
                    ctx.note('container exterior route blocked', sign=sign, segment=index,
                             start=previous.tolist(), goal=goal.tolist())
                    clear = False
                    break
                s.put_base(goal, poses)
                previous = goal
            if clear:
                selected = route
                break
        finally:
            s.model.base = actual
            for solver in s.solvers.values():
                solver.sync()
    semantic.require(ctx, selected is not None, 'no clear table exterior route for container approach', kind='planning')
    ctx.note('container table approach', side='rear' if rear else 'front',
             waypoints=[point.tolist() for point in selected])
    for goal in selected:
        route = s.base_route(s.base_pose(), goal)
        semantic.require(ctx, route is not None, 'container exterior route changed', kind='planning')
        for point in route:
            first, distance, last = drive(s.base_pose(), point)
            s.move_base(0., first)
            s.move_base(distance, last)
        if loaded:
            container_carry_held(ctx, loaded)
    s.note_relocated(before)


def basket_front_stage_region(ctx):
    near = np.r_[np.asarray(ctx.scene.TABLE['center']) + np.asarray((-.44, -.29)),
                 ctx.scene.TABLE['height']]
    region, blocked = regions.on_region(
        'basket', 'table', near=near, keep_out=container_goal_workspace(ctx))(ctx)
    return region, world_keepouts(blocked)


def buffer_basket_for_containers(ctx):
    """Push the loaded basket into the rear-left free strip without lifting."""
    if not hasattr(ctx, '_container_front_entry'):
        ctx._container_front_entry = ctx.skills().base_pose().copy()
    before_base = ctx.skills().base_pose().copy()
    retreat = before_base.copy()
    retreat[:2] -= .35 * np.array([np.cos(before_base[2]), np.sin(before_base[2])])
    side = ctx.scene.TABLE['center'][0] - ctx.scene.TABLE['size'][0] / 2 - .40
    route = [retreat, np.array([side, retreat[1], before_base[2]]),
             np.array([side, -.10, 0.])]
    for index, goal in enumerate(route):
        semantic.require(ctx, checked_approach(
            ctx, goal, f'basket tabletop push exterior segment {index}'),
            'no checked exterior route for loaded basket push', kind='check', segment=index)
    before = ctx.pose('basket')[:3, 3].copy()
    distance = max(0., .31 - before[1])
    ctx.note('basket rear-left tabletop staging', start=before.tolist(),
             goal_y_m=.31, distance_m=distance)
    if distance > .01:
        skills.retry(ctx, lambda: skills.push(
            ctx, 'basket', part='file_basket.shell.long_wall1',
            direction=(0., 1., 0.), distance=distance))
    ctx.hold(1.)


def jar_carry_approach(ctx):
    """Carry the upright jar outside the table to its reachable release side."""
    s, table = ctx.skills(), ctx.scene.TABLE
    s.ready()
    before = s.base_pose().copy()
    centre, half = np.asarray(table['center']), np.asarray(table['size']) / 2
    clearance = max(.40, abs(np.asarray(ctx.scene.ROBOTS[ROBOT]['position'])[1] - centre[1]) - half[1] + CLEAR)
    retreat = before.copy()
    retreat[:2] -= clearance * np.array([np.cos(before[2]), np.sin(before[2])])
    side = centre[0] - half[0] - clearance
    route = [retreat, np.array([side, retreat[1], before[2]]),
             np.array([side, destination(ctx, 'jar')[1], 0.])]
    poses, actual = ctx.poses(), s.model.base.copy()
    try:
        previous = before
        for goal in route:
            semantic.require(ctx, s.base_route(previous, goal) is not None,
                             'no clear loaded jar route to release side', kind='planning')
            s.put_base(goal, poses)
            previous = goal
    finally:
        s.model.base = actual
        for solver in s.solvers.values():
            solver.sync()
    ctx.note('jar loaded exterior route', waypoints=[point.tolist() for point in route])
    for goal in route:
        checked = s.base_route(s.base_pose(), goal)
        semantic.require(ctx, checked is not None, 'loaded jar route changed', kind='planning')
        for point in checked:
            first, distance, last = drive(s.base_pose(), point)
            s.move_base(0., first)
            s.move_base(distance, last)
        jar_carry_held(ctx)
    s.note_relocated(before)


def jar_grasp_approach(ctx):
    """Reach deep wall contacts from the collision-checked front table edge."""
    container_grasp_stance(ctx, 'jar')


def jar_carry_held(ctx):
    """Reobserve the pinch after lifting and each exterior drive segment."""
    container_carry_held(ctx, 'jar')


def empty_basket_held(ctx):
    """Only a physically empty basket can travel in a tilted pinch."""
    s = ctx.skills()
    arm = s.holding('basket')
    return (arm is not None and not s.held[arm]['riders']
            and not any(semantic.inside(ctx, name, 'basket')
                        for name in ctx.scene.OBJECTS if name != 'basket'))


def container_carry_held(ctx, obj):
    """Require measured pad support and uprightness during a loaded drive."""
    from table_1000.experts.skills.hold.frames import MIN_HOLD_AREA
    s = ctx.skills()
    arm = s.holding(obj)
    semantic.require(ctx, arm is not None, f'{obj} hold lost during transport', kind='check')
    held = s.held[arm]
    target = held['target']
    body = ctx.poses()[target.body]
    tcp, gap = ctx.tcp(arm), s.solvers[arm].finger_gap()
    checked = s.pad_closure(target, arm, held['grasp']['open_gap'])
    area = s.measured_held_area(target, arm, body @ target.offset, tcp, gap)
    empty = obj == 'basket' and empty_basket_held(ctx)
    ctx.note(f'{obj} transport hold checked', arm=arm, actual_held_area_m2=area,
             min_hold_area_m2=MIN_HOLD_AREA, held=checked['held'],
             upright_z=float(ctx.pose(obj)[2, 2]), empty_basket=empty)
    semantic.require(ctx, checked['held'],
                     f'{obj} hold unstable during transport', kind='check', actual_held_area_m2=area)
    if empty:
        # Plan each exterior segment using its actual settled pose, including
        # every attached piece, rather than the pre-lift TCP transform.
        solver, poses = s.solvers[arm], ctx.poses()
        for name in list(solver.attachments):
            solver.attach(name, poses[solver.world.objects[name].body])


def basket_staged(ctx):
    buffered(ctx, 'basket', keep_out=container_goal_workspace(ctx))
    for item, container in STORAGE_OVERRIDES.items():
        if container == 'basket':
            stored(ctx, item, 0)


def loaded_table_region(ctx, obj, near):
    """Keep actual attached contents with their container, outside external keep-outs."""
    contents = set()
    for held in ctx.skills().held.values():
        if held['target'].obj == obj:
            contents.update(name.split(':', 1)[0] for name in held.get('riders', ()))
    shapes = world_geometry.scene_shapes(ctx, exclude=(obj, *sorted(contents)))
    region, blocked = world_geometry.support_region(
        shapes, 'table', headroom=regions.headroom(ctx, obj))
    semantic.require(ctx, region is not None, 'no support', kind='planning', surface='table')
    ctx.note('container release excludes its attached contents', obj=obj,
             contents=sorted(contents))
    return regions.around(regions.inset(region, regions.EDGE), near), blocked


def destination_region(ctx, obj):
    """Constrain placement search to the footprint at the layout-5 destination."""
    near = destination(ctx, obj)
    region, blocked = loaded_table_region(ctx, obj, near)
    frame, low, high = region[:3]
    inverse = np.linalg.inv(frame)
    # Layout 5 uses zero heading. Keep enough room for the entire collision
    # footprint, plus the existing placement margin and position tolerance.
    points = ctx.geometry[obj].points() + near
    local = points @ inverse[:3, :3].T + inverse[:3, 3]
    allowance = POSITION_TOLERANCE + PLACE_MARGIN
    return (frame, np.maximum(low, local[:, :2].min(0) - allowance),
            np.minimum(high, local[:, :2].max(0) + allowance), *region[3:]), world_keepouts(blocked)


def table_support(ctx, obj):
    bottom = float(world_geometry.object_points(ctx, obj)[:, 2].min())
    semantic.require(ctx, abs(bottom - ctx.scene.TABLE['height']) <= .01,
                     'container not resting on the table', kind='check', obj=obj)


def aligned(ctx, obj):
    if obj in RETAINED_CONTAINER_GOALS or obj in ('holder', 'cup'):
        return neat_container_pose(ctx, obj)
    table_support(ctx, obj)
    semantic.require(ctx, at_destination(ctx, obj), 'container away from layout 5 destination', kind='check', obj=obj)


def container_done_without_books(ctx, obj, *, neat=False):
    if neat:
        neat_container_pose(ctx, obj)
    elif obj == 'basket':
        aligned(ctx, obj)
    else:
        aligned(ctx, obj)
    if obj in ('holder', 'cup'):
        for pen in (ctx.scene.SHORT_PENS if obj == 'holder' else ctx.scene.LONG_PENS):
            if pen in PEN_STORAGE:
                semantic.require(ctx, contained(ctx, pen, *PEN_STORAGE[pen]),
                                 'alternate pen not contained', kind='check', obj=pen)
            else:
                pen_done(ctx, pen, obj)
    elif obj == 'jar':
        for clip in ctx.scene.CLIPS:
            clip_done(ctx, clip)
        semantic.require(ctx, contained(ctx, 'magnet_red', 'jar'), 'red magnet left the jar', kind='check')
    elif obj == 'cabinet':
        for number in (1, 2, 3):
            drawer_done(ctx, number)
    elif obj == 'basket':
        for item, container in STORAGE_OVERRIDES.items():
            if container == 'basket':
                stored(ctx, item, 0)


def verify_storage(ctx, *, neat=False):
    """Accept the non-book #95 chain; book placement belongs to issue #106."""
    if _skipped(ctx):
        pending = {item: value['reason'] for item, value in _skipped(ctx).items()}
        ctx.note('unfinished items', items=pending)
        semantic.require(ctx, False, 'skipped items remain incomplete', kind='check', items=pending)
    semantic.require(ctx, not ctx.skills().held and not any(s.attachments for s in ctx.solvers.values()),
                     'final checkpoint requires empty hands', kind='check')
    for number in (1, 2, 3):
        drawer_done(ctx, number)
    for obj in DESTINATIONS:
        container_done_without_books(ctx, obj, neat=neat)
    for obj in UNRESOLVED_BOOKS:
        book_left(ctx, obj)
    table = ctx.scene.TABLE
    low = np.asarray(table['center']) - np.asarray(table['size']) / 2
    high = np.asarray(table['center']) + np.asarray(table['size']) / 2
    for obj in ctx.scene.OBJECTS:
        if obj in UNRESOLVED_BOOKS:
            # #106 owns these already overhanging input books; book_left
            # above still checks desk height and centre. Leave their poses.
            continue
        points = world_geometry.object_points(ctx, obj)
        semantic.require(ctx, points[:, 2].min() >= table['height'] - .005
                         and np.all(points[:, :2].min(0) >= low)
                         and np.all(points[:, :2].max(0) <= high), 'object fell or flew off the desk', kind='check', obj=obj)
        if obj not in DESTINATIONS and obj not in UNRESOLVED_BOOKS:
            semantic.require(ctx, not semantic.on(ctx, obj, 'table'),
                             'loose object left in the front work area', kind='check', obj=obj)
    if neat:
        neat_group_check(ctx)


def finish(ctx):
    """Final storage acceptance with mutually aligned container groups."""
    verify_storage(ctx, neat=True)


# Revised small-container order is holder/jar/cup, with a common rear edge
# and the same 40 +/- 20 mm gaps. Cabinet remains the stationary anchor.
NEAT_GOALS = {'jar': (RETAINED_CONTAINER_GOALS['jar'], None),
 'holder': ((-.180, .075), 0.),
 'cup': ((.140, .065), 0.),
 'basket': (RETAINED_CONTAINER_GOALS['basket'], 0.),
 'cabinet': ((0.35728380084, 0.02270017006), 0.)}
NEAT_ROW = ('holder', 'jar', 'cup')
NEAT_KEEP = {}
NEAT_YAW_ONLY = {}


def neat_container_pose(ctx, obj):
    table_support(ctx, obj)
    goal, heading = NEAT_GOALS[obj]
    pose = ctx.pose(obj)
    distance = float(np.linalg.norm(pose[:2, 3] - goal))
    angle = 0. if heading is None else abs(yaw_error(heading, np.arctan2(pose[1, 0], pose[0, 0])))
    ctx.note('neat container terminal error', obj=obj, position_m=distance, yaw_rad=angle)
    semantic.require(ctx, distance <= .020 and angle <= np.radians(5.),
                     'container not at its neat group pose', kind='check', obj=obj, position_m=distance, yaw_rad=angle)
    ctx.note('neat container tilt diagnostic', obj=obj, up_z=float(pose[2, 2]))


def neat_group_check(ctx):
    bounds = {obj: (world_geometry.object_points(ctx, obj).min(0), world_geometry.object_points(ctx, obj).max(0))
              for obj in NEAT_GOALS}
    rear = [bounds[obj][1][1] for obj in NEAT_ROW]
    gaps = [bounds[right][0][0] - bounds[left][1][0]
            for left, right in zip(NEAT_ROW, NEAT_ROW[1:])]
    left_error = abs(bounds['basket'][0][0] - bounds['jar'][0][0])
    ctx.note('neat group measured', rear_edges=rear, adjacent_gaps=gaps, left_edge_error=left_error,
             retained_goals=RETAINED_CONTAINER_GOALS, yaw_only_anchors=NEAT_YAW_ONLY)
    semantic.require(ctx, max(rear) - min(rear) <= .020
                     and all(abs(gap - .04) <= .020 for gap in gaps)
,
                     'container row or consistent gaps not aligned', kind='check')


def neat_payload_guard(ctx):
    """Watch spills during pushes; pen tilt is diagnostic, not a rejection."""
    # Collision pieces are rigid. Transform their precomputed hull planes
    # rather than rerunning Qhull for the whole scene on every control frame.
    # The enclosing rays, extents and immersed-segment tests below are the
    # original contained/upright_inside algorithms and original thresholds.
    if not hasattr(ctx, '_neat_local_shapes'):
        names = set(('holder', 'cup', 'jar', 'basket', *ctx.scene.SHORT_PENS,
                     *ctx.scene.LONG_PENS, *ctx.scene.CLIPS, 'magnet_red', *STORAGE_OVERRIDES))
        ctx._neat_local_shapes = {
            name: world_geometry.asset_shapes(name, ctx.geometry[name],
                                        {body: np.eye(4) for body in ctx.geometry[name].bodies})
            for name in names}
        ctx._neat_pen_targets = {pen: ctx.skills().target(pen)
                                 for pen in (*ctx.scene.SHORT_PENS, *ctx.scene.LONG_PENS)
                                 if pen not in PEN_STORAGE}
    poses, shapes, points = ctx.poses(), {}, {}
    for name, pieces in ctx._neat_local_shapes.items():
        shapes[name] = []
        for piece in pieces:
            frame = poses[name + ':' + piece.body]
            rotation, shift = frame[:3, :3], frame[:3, 3]
            normals = piece.planes[:, :3] @ rotation.T
            shapes[name].append(SimpleNamespace(
                frame=frame, points=piece.points @ rotation.T + shift,
                planes=np.c_[normals, piece.planes[:, 3] - normals @ shift]))
        points[name] = np.vstack([piece.points for piece in shapes[name]])

    def enclosed(obj, container):
        pieces, payload = shapes[container], points[obj]
        if not semantic._inside_point(pieces, world_geometry.centre(payload)):
            return False
        inverse = np.linalg.inv(pieces[0].frame)
        bounds = points[container] @ inverse[:3, :3].T + inverse[:3, 3]
        payload = payload @ inverse[:3, :3].T + inverse[:3, 3]
        low, high = bounds.min(0), bounds.max(0)
        return bool(np.all(payload[:, :2].min(0) >= low[:2] - .005)
                    and np.all(payload[:, :2].max(0) <= high[:2] + .005)
                    and payload[:, 2].min() >= low[2] - .005)

    def upright(pen, container):
        target = ctx._neat_pen_targets[pen]
        size = np.ptp(target.points, axis=0)
        pose, pieces = poses[target.body], shapes[container]
        normal = pieces[0].frame[:3, 2]
        cosine = abs(float(pose[:3, int(np.argmax(size))] @ normal))
        if cosine < np.cos(PEN_TILT) and pen not in getattr(ctx, '_tilt_noted', set()):
            ctx._tilt_noted = getattr(ctx, '_tilt_noted', set()) | {pen}
            ctx.note('payload tilt diagnostic', obj=pen,
                     tilt_deg=float(np.degrees(np.arccos(np.clip(cosine, 0., 1.)))))
        # Storage is tested with semantic.storage_result at the step boundary.
        # A transient tilt alone cannot abandon an otherwise contained payload.
        return True

    for container, pens in (('holder', ctx.scene.SHORT_PENS), ('cup', ctx.scene.LONG_PENS)):
        for pen in pens:
            if pen in PEN_STORAGE:
                destination, part = PEN_STORAGE[pen]
                semantic.require(ctx, part is None and enclosed(pen, destination),
                                 'pen left its storage during squaring', kind='check', obj=pen)
            else:
                semantic.require(ctx, upright(pen, container),
                                 'standing pen left its upright storage during squaring', kind='check', obj=pen)
    for clip in (*ctx.scene.CLIPS, 'magnet_red'):
        semantic.require(ctx, enclosed(clip, 'jar'),
                         'jar payload escaped during squaring', kind='check', obj=clip)
    for obj, container in STORAGE_OVERRIDES.items():
        semantic.require(ctx, enclosed(obj, container),
                         'basket payload escaped during squaring', kind='check', obj=obj)


def square_container(ctx, obj):
    goal, heading = NEAT_GOALS[obj]
    if obj in NEAT_KEEP:
        ctx.note('retain container after bounded squaring case', obj=obj, reason=NEAT_KEEP[obj])
        ctx.observers.append(neat_payload_guard)
        try:
            neat_payload_guard(ctx)
            ctx.hold(1.)
        finally:
            ctx.observers.remove(neat_payload_guard)
        all_on_desk(ctx)
        semantic.require(ctx, not ctx.skills().held and not any(s.attachments for s in ctx.solvers.values()),
                         'retained container checkpoint requires empty hands', kind='check')
        return
    previous_deadline = getattr(ctx, 'wall_deadline', None)
    ctx.wall_deadline = min(previous_deadline or float('inf'), perf_counter() + (300. if obj == 'basket' else 650. if obj == 'holder' else 900.))
    ctx.observers.append(neat_payload_guard)
    try:
        neat_payload_guard(ctx)
        pose = ctx.pose(obj)
        already_neat = (np.linalg.norm(pose[:2, 3] - goal) <= .020
                        and (heading is None or abs(yaw_error(heading,
                             np.arctan2(pose[1, 0], pose[0, 0]))) <= np.radians(5.)))
        if already_neat:
            ctx.note('container already within declared neat goal', obj=obj)
        elif obj == 'basket':
            container_table_approach(ctx, rear=False)
            base = ctx.skills().base_pose().copy()
            retreat = base.copy()
            retreat[:2] -= .4 * np.array([np.cos(base[2]), np.sin(base[2])])
            for goal_base in (retreat, np.array([-1.08, -1.02, np.pi / 2]),
                              np.array([-1.08, -.30, 0.])):
                semantic.require(ctx, checked_approach(ctx, goal_base, 'basket left exterior approach'),
                                 'basket left exterior approach blocked', kind='check')
        elif obj == 'jar':
            container_table_approach(ctx, rear=True)
        elif obj == 'cup':
            container_table_approach(ctx, rear=True)
            checked_approach(ctx, np.array([-.00522568, .80853415, -1.58384442]),
                             'cup proven rear squaring stance')
        elif obj == 'holder':
            container_table_approach(ctx, rear=False)
            checked_approach(ctx, np.array([-.40, -.70, 1.30]), 'holder squaring front stance')
        result = square_on_table(ctx, obj, goal, heading, max_rounds=2 if obj == 'basket' else 12,
                                 position_tolerance=.020, angle_tolerance=np.radians(5.),
                                 max_push=.004 if obj == 'cup' else .01 if obj == 'holder' else .02, contact_height=.065 if obj == 'basket' else .035,
                                 yaw_direction=(1., 0.) if obj == 'basket' else (0., -1.) if obj in ('jar', 'cup') else (0., 1.),
                                 translation_lever=.035 if obj == 'holder' else 0.,
                                 check=neat_payload_guard)
        ctx.note('container squared by pushing', obj=obj, result=result)
        ctx.skills().ready()
        before = ctx.pose(obj).copy()
        ctx.hold(1.)
        semantic.require(ctx, np.linalg.norm(ctx.pose(obj)[:3, 3] - before[:3, 3]) <= .020,
                         'squared container not stable after retreat', kind='check', obj=obj)
    finally:
        ctx.observers.remove(neat_payload_guard)
        ctx.wall_deadline = previous_deadline
    semantic.require(ctx, not ctx.skills().held and not any(s.attachments for s in ctx.solvers.values()),
                     'container squaring checkpoint requires empty hands', kind='check')
    all_on_desk(ctx)


def settle_final_basket(ctx):
    """Observe the staged basket at rest before checking the final inventory."""
    for obj in ('pen_white', 'gel_pen'):
        axis = principal_axis(ctx, obj)
        ctx.note('authorized cup pen containment target', obj=obj,
                 tilt_degrees=float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0., 1.)))),
                 contained=bool(contained(ctx, obj, 'cup')))
    before = ctx.pose('basket').copy()
    ctx.hold(1.)
    displacement = float(np.linalg.norm(ctx.pose('basket')[:3, 3] - before[:3, 3]))
    ctx.note('final staged basket stability', displacement_m=displacement, observation_s=1.)
    semantic.require(ctx, displacement <= POSITION_TOLERANCE,
                     'staged basket moved during final settling', kind='check', displacement_m=displacement)


# Main chain: easy objects go directly to their destinations. Three proven
# drawer-front deadlocks need a checked table buffer before their final store.
def optional_step(item, name, action, check, obj=None, retries=1, *, phased=False):
    action = item_action(obj, action) if obj is not None else action
    return Step(name, optional_action(item, action, retries=retries, phased=phased),
                optional_check(item, check))


# A: the top drawer starts open, so its contents can go straight to drawer 1.
STEPS = [optional_step(obj, (f'A: put {obj} in the {STORAGE_OVERRIDES[obj]}'
                            if obj in STORAGE_OVERRIDES else
                            f'A: put {obj} directly in drawer 1'),
                       partial(store, obj=obj, number=1),
                       partial(stored, obj=obj, number=1), obj=obj)
         for obj in DRAWERS[1]]
STEPS += [optional_step('drawer1', 'A: close and check the top drawer',
                       partial(close, number=1), partial(drawer_done, number=1),
                       retries=0)]

# C: these destinations are already accessible, so clear the loose pens and
# clips before approaching the obstructed lower drawers.
# The yellow pen's first arm motion crosses the white pen at the front edge.
# Clear that real obstacle before withdrawing the yellow pen.
for container, names in (('cup', ('pen_white',)), ('holder', ('pencil',)),
                         ('holder', ('highlighter_yellow', 'highlighter_pink', 'marker_black')),
                         ('cup', ('pen_black', 'marker_blue', 'gel_pen'))):
    STEPS += [optional_step(obj, f'C: stand {obj} in the {container}',
                           partial(insert_pen, obj=obj, container=container),
                           partial(pen_done, obj=obj, container=container), obj=obj,
                           retries=0) for obj in names]
STEPS += [optional_step(obj, f'C: put {obj} in the jar', partial(store_clip, obj=obj),
                       partial(clip_done, obj=obj), obj=obj)
          for obj in ('clip_large', 'clip_pink', 'clip_small', 'clip_blue')]

# B: charger/dish block the approach to their own target drawer.
# Their real table buffering was validated before this sequence was added.
STEPS += [optional_step(obj, f'B: clear {obj} from the drawer approach',
                       partial(clear_drawer_front, obj=obj),
                       partial(drawer_front_clear, obj=obj), obj=obj)
          for obj in ('charger', 'dish')]
# Remote remains deferred until the loose objects around its support are stored.
for number in (3, 2):
    objects = tuple(obj for obj in DRAWERS[number] if obj not in ('remote', 'screwdriver'))
    if number == 3:
        STEPS += [optional_step('drawer3-half', 'B: half open drawer 3 for screwdriver',
                               open_screw_drawer_half, screw_drawer_half_opened, retries=0),
                  optional_step('screwdriver', 'B: put screwdriver in the basket',
                                store_screw_first, partial(stored, obj='screwdriver', number=3),
                                obj='screwdriver', retries=0)]
    STEPS += [optional_step(f'drawer{number}-open', f'B: open drawer {number} for storage',
                           partial(open_storage_drawer, number=number),
                           partial(drawer_opened, number=number))]
    if number == 2:
        STEPS += [optional_step('glue_bottle-side', 'B: rest glue_bottle on its side on the table',
                               rest_bottle_on_side, bottle_side_done, obj='glue_bottle', retries=0),
                  optional_step('glue_bottle-stable', 'B: check side bottle stability',
                               settle_side_bottle, bottle_side_done, retries=0)]
    STEPS += [optional_step(obj, (f'B: put {obj} in the {STORAGE_OVERRIDES[obj]}'
                                if obj in STORAGE_OVERRIDES else
                                f'B: put {obj} directly in drawer {number}'),
                           partial(store, obj=obj, number=number, open_if_needed=False),
                           partial(stored, obj=obj, number=number), obj=obj,
                           phased=obj in ('screwdriver', 'charger', 'dish'),
                           retries=0 if obj in ('screwdriver', 'charger', 'dish') else 1)
              for obj in objects]
    check = partial(drawer_closed, number=number)
    STEPS += [optional_step(f'drawer{number}', f'B: close drawer {number}',
                           partial(close, number=number), check, retries=0)]

# D: unresolved books stay at their input poses for issue #106.
for obj in FLAT_BOOKS + ('notebook_a4',):
    STEPS += [Step(f'D: leave {obj} in its initial position', partial(leave_book, obj=obj),
                   partial(book_left, obj=obj))]

# The supported remote push needs drawer 3 open for the close front stance.
STEPS += [optional_step('drawer3-tail-open', 'B: open drawer 3 for remote',
                       partial(open_loaded_drawer, number=3), partial(drawer_opened, number=3)),
          optional_step('remote', 'A: pick remote directly into the basket',
                       remote_to_drawer, partial(stored, obj='remote', number=2), obj='remote'),
          optional_step('drawer3-tail-final', 'B: close drawer 3 after remote',
                       partial(close, number=3), partial(drawer_done, number=3))]
STEPS += [optional_step('basket-staging', 'E: temporarily stage basket clear of final container goals',
                       buffer_basket_for_containers, basket_staged, obj='basket')]
STEPS += [optional_step(obj, f'E: align the {obj}', partial(align_container, obj=obj),
                       partial(container_done_without_books, obj=obj), obj=obj)
          for obj in ('cabinet', 'jar', 'basket')]
# Holder exhausted its bounded exploration window. Preserve its final goal,
# but finish the independent container checks before reporting that blocker.
STEPS += [Step(f'G: square loaded {obj} on the table', partial(square_container, obj=obj),
               partial(container_done_without_books, obj=obj, neat=True))
          for obj in ('jar', 'basket', 'cabinet')]
STEPS += [optional_step('cup', 'E: align the cup', partial(align_container, obj='cup'),
                       partial(container_done_without_books, obj='cup'), obj='cup'),
          Step('G: square loaded cup on the table', partial(square_container, obj='cup'),
               partial(container_done_without_books, obj='cup', neat=True))]
STEPS += [optional_step('holder', 'E: align the holder', partial(align_container, obj='holder'),
                       partial(container_done_without_books, obj='holder'), obj='holder'),
          Step('G: square loaded holder on the table', partial(square_container, obj='holder'),
               partial(container_done_without_books, obj='holder', neat=True))]
STEPS += [Step('F: verify storage and mutually aligned container groups', settle_final_basket, finish)]


def solve(ctx):
    print('X1PRO_V2_SOLVE_BEGIN', ctx.only, 'control', ctx.steps, flush=True)
    continuous_no_fall(ctx)
    ctx.observers.append(continuous_no_fall)
    try:
        run_steps(ctx, STEPS)
    finally:
        ctx.observers.remove(continuous_no_fall)
