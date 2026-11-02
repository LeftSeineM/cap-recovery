"""RM001 seed-zero tidy trajectory using physical storage and measured grasp outcomes."""

from functools import partial
from contextlib import contextmanager, nullcontext
from time import perf_counter, process_time

import numpy as np
from mplib.collision_detection import fcl
from scipy.spatial import ConvexHull

from table_1000.experts import skills
from table_1000.experts.skills.hold.frames import Box
from table_1000.experts.context import (DT, ExpertFailed, Step, WallBudgetExceeded,
                                      remaining_wall, run_steps)
from table_1000.experts.skills import semantic
from table_1000.experts.skills.base import BASE_TOLERANCE, drive, drive_path, drive_routes, wrapped
from table_1000.experts.skills.drawer.bracing import stability, _brace
from table_1000.experts.skills.common import APPROACH, SUPPORT_DISTANCE
from table_1000.experts.skills.hold.pick import LIFT, MIN_RISE
from table_1000.experts.skills.place.strict import INSERT_MARGIN, PLACE_CLEARANCE, PLACE_MARGIN, PLACE_TURN_STEP
from table_1000.experts.skills.place.regions import overlaps
from table_1000.experts.skills.hold.targets import targets as enumerate_targets

from table_1000.experts.world import geometry as _world
from table_1000.experts.skills.place import regions as _regions
from table_1000.experts.skills.drawer import joints as _joints

ROBOT = 'x1pro_rm001'
SCOPE = {0: [0]}
SOURCE = 'hand'

OPENING = .173
CLEAR = .04
POSITION_TOLERANCE = .04
ANGLE_TOLERANCE = np.radians(20.)
# Layout 5 destinations relative to the table centre, never grasp points.
DESTINATIONS = {'jar': (-.48, .04), 'holder': (-.30, .04), 'cup': (-.17, .05),
                'basket': (.06, .01), 'cabinet': (.38, .03)}
# Prefer mutually clear poses inside the unchanged layout acceptance disks.
# The nominal adjacent cup/basket/cabinet poses leave less than PLACE_MARGIN.
CONTAINER_OFFSETS = {'holder': (-.025, 0.), 'cup': (-.025, 0.),
                     'basket': (-.02, .015)}
# Authorized by msg_7e4512db2e6b after eight strict basket staging trials.
# Preserve the real fourth-push pose; move the three small-container goals
# together by the minimum common-Y shift on a 1 mm planning grid. Original
# position/upright/yaw tolerances and every physical containment check remain.
ORIGINAL_LAYOUT5_DESTINATIONS = dict(DESTINATIONS)
REVISED_TARGET_SHIFT = (0., -.019)
for _container in ('jar', 'holder', 'cup'):
    _x, _y = DESTINATIONS[_container]
    DESTINATIONS[_container] = (_x, _y + REVISED_TARGET_SHIFT[1])
DESTINATIONS['basket'] = (-.3187319338321686, .30077266693115234)
TARGET_YAWS = {'basket': -.7398276783589862}
CONTAINER_OFFSETS.pop('basket')
# After four bounded jar variants, retain the canonical s54 jar instead of
# continuing the blocked tabletop traversal. Declare this whole plan before
# s55 executes; never redefine a goal from each step's measured outcome.
# Holder/jar/cup have one rear edge and 40 mm footprint gaps at heading zero.
# Authorized by msg_0a363761c4b7 / msg_909cb025e037, following x1pro's finish.
RETAINED_CONTAINER_GOALS = {
    'jar': (-.014111942611634731, .041538216173648834),
    'basket': (-.40014737844467163, .26624298095703125),
    'cabinet': (.36391842170596966, .0344979227659956),
}
DESTINATIONS.update(RETAINED_CONTAINER_GOALS)
DESTINATIONS.update(holder=(-.1889888191823265, .07641506864496682),
                    cup=(.15076490044292437, .06641506700583909))
TARGET_YAWS['basket'] = 0.
CONTAINER_OFFSETS.clear()
NEAT_ROW = ('holder', 'jar', 'cup')
# Insert long objects before bulky contents obstruct their wrist approach;
# other large footprints precede small ones. The first three top-drawer
# checks retain the already validated strict prefix.
DRAWERS = {3: ('screwdriver', 'lidded_jar', 'charger', 'dish', 'snack'),
           2: ('glue_bottle', 'board_eraser', 'tape', 'stapler', 'remote'),
           1: ('magnet_orange', 'staples', 'eraser', 'glue_stick', 'knife', 'usb_drive')}
# Seed-zero alternate storage is part of the plan; verify the actual container.
STORAGE_OVERRIDES = {'glue_stick': 'basket', 'screwdriver': 'basket',
                     'glue_bottle': 'basket'}
FLAT_PEN_DRAWERS = {'pen_black': 1}
DRAWER_OVERRIDES = {}
# Descending authored collision footprint area.
FLAT_BOOKS = ('notebook_slim', 'notebook_a5', 'notebook_a6')
UNRESOLVED_BOOKS = ('notebook_a4', *FLAT_BOOKS)
ITEM_WALL_BUDGET = 2400.


def drawer(number):
    return f'cabinet.drawer{number}'


def joint(number):
    return drawer(number) + '.slide'


def workspace(ctx):
    return [_world.body_box(ctx, 'cabinet', 'cabinet.housing', CLEAR),
            *[_joints.travel(ctx, 'cabinet', joint(i), CLEAR) for i in (1, 2, 3)]]


def transfer(ctx, obj, where, pick_options=None, **options):
    """Search ordinary RM001 grasps and base poses without an inherited hand."""
    pick_options = dict(pick_options or {})
    if obj == 'eraser':
        pick_options.update(approach=(0., 0., -1.), max_angle=np.radians(10.))
        ctx.note('RM001 eraser overhead grip', direction=[0., 0., -1.], cone_deg=10.)
    adapt = ('base',) if obj in DESTINATIONS else ('base', 'handover')
    if obj in ('knife', 'usb_drive'):
        # Seed-zero entry trials checked 37/60 common stances without a solution.
        # Their successful fallback picked first, then adapted the hold/place.
        adapt = ('handover',)
        ctx.note('RM001 skip exhausted common base search', obj=obj,
                 evidence_checked=37 if obj == 'knife' else 60)
    skills.chain(ctx, [skills.Pick(obj, instruction=f'pick up the {obj}',
                                   **pick_options),
                       skills.Place(obj, where, instruction=f'put away the {obj}', **options)],
                 adapt=adapt)


def buffer(ctx, obj, keep_out=None, near=None):
    table = ctx.scene.TABLE
    if near is None:
        near = np.r_[_world.centre(_world.object_points(ctx, obj))[:2], table['height']]
    before = ctx.pose('cabinet').copy()
    if obj == 'dish':
        # The initially open top drawer blocks a collision-free approach to
        # the dish below it. Close it before clearing the front, then reopen
        # only when storing the eraser.
        close(ctx, 1)
    pick_options = {}
    if obj == 'dish':
        target = ctx.skills().target(obj)
        pick_options = dict(shape=target.shape, offset=target.offset,
                            approach=(0., 0., -1.), max_angle=np.radians(15.))
        deadline = perf_counter() + 600.
        previous = getattr(ctx, 'wall_deadline', None)
        ctx.wall_deadline = min(previous, deadline) if previous is not None else deadline
        ctx.note('RM001 dish whole overhead grip', cone_deg=15., local_budget_s=600.)
    if obj == 'charger':
        # The right-hand trial reached eight planned above motions but all
        # failed tracking at the left-side patch after base adaptation.
        target = ctx.skills().target(obj)
        pick_options = dict(arm='left', shape=target.shape, offset=target.offset,
                            approach=(0., 0., -1.), max_angle=np.radians(15.))
        deadline = perf_counter() + 600.
        previous = getattr(ctx, 'wall_deadline', None)
        ctx.wall_deadline = min(previous, deadline) if previous is not None else deadline
        ctx.note('RM001 charger left-hand buffer', arm='left',
                 reason='right-hand above-patch motions failed tracking')
        ctx.note('RM001 charger overhead drop', cone_deg=15., release_above_table_m=.03,
                 approach_above_m=.04, local_budget_s=600.)
    where = (charger_buffer_drop_region if obj == 'charger' else
             dish_buffer_region if obj == 'dish' else
             _regions.on_region(obj, 'table', near=near,
                                keep_out=workspace(ctx) if keep_out is None else keep_out))
    try:
        transfer(ctx, obj, where, near=near, pick_options=pick_options,
                 **({'above': .04} if obj == 'charger' else {}))
    except ExpertFailed as error:
        if not (obj in ('charger', 'dish') and error.reason == 'no placement'
                and error.details.get('last') == 'tracking'
                and release_measured_buffer(ctx, obj)):
            raise
    if obj == 'charger':
        ctx.hold(1.)
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
    points = _world.object_points(ctx, obj)
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
    """Each physical retry gets a new official batch, never a silent skip."""
    return action


def optional_check(item, check):
    return check


def all_on_desk(ctx):
    """Reject collateral falls while allowing an input object's supported overhang."""
    table = ctx.scene.TABLE
    low = np.asarray(table['center']) - np.asarray(table['size']) / 2
    high = np.asarray(table['center']) + np.asarray(table['size']) / 2
    shapes = _world.scene_shapes(ctx)
    for obj in ctx.scene.OBJECTS:
        points = np.vstack([shape.points for shape in shapes if shape.obj == obj])
        centre = _world.centre(points)
        semantic.require(ctx, points[:, 2].min() >= table['height'] - .005
                         and np.all(centre[:2] >= low) and np.all(centre[:2] <= high),
                         'object fell or flew off the desk', kind='check', obj=obj)


def continuous_no_fall(ctx):
    """Record intermediate falls; user accepts real recovery before the checkpoint."""
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
            event = dict(obj=obj, body=body, minimum_z_m=minimum,
                         floor_z_m=floor, control_step=ctx.steps)
            if obj not in ctx._observed_falls:
                ctx.note('continuous scene fall detected; recovery permitted', **event)
            if (obj not in ctx._observed_falls
                    or minimum < ctx._observed_falls[obj]['minimum_z_m']):
                ctx._observed_falls[obj] = event


def item_action(obj, action):
    """Annotate a callable with its object for the generic retry cleanup."""
    action.object_name = obj
    return action


def checked_approach(ctx, goal, purpose, *, attempts=2, tolerance=BASE_TOLERANCE):
    """Follow checked routes and correct the measured residual to the same tolerance."""
    s = ctx.skills()
    before = s.base_pose().copy()
    if (np.linalg.norm(before[:2] - goal[:2]) < tolerance[0]
            and abs(wrapped(before[2] - goal[2])) < tolerance[1]):
        return True
    remaining_wall(ctx)
    s.ready()
    remaining_wall(ctx)
    for attempt in range(attempts):
        route = s.base_route(s.base_pose(), goal)
        remaining_wall(ctx)
        if route is None:
            actual = s.model.base.copy()
            poses, q = ctx.poses(), s.model.q.copy()
            rejected = []
            try:
                for candidate in drive_routes(s.base_pose(), goal):
                    previous = s.base_pose()
                    found = None
                    for waypoint in candidate:
                        for along in drive_path(previous, waypoint):
                            s.put_base(along, poses)
                            pairs = s.touching(q)
                            if pairs:
                                found = dict(pose=np.asarray(along).tolist(), pairs=pairs)
                                break
                        if found:
                            break
                        previous = waypoint
                    rejected.append(found)
            finally:
                s.model.base = actual
                for solver in s.solvers.values():
                    solver.sync()
            ctx.note(purpose + ' blocked', goal=goal.tolist(),
                     measured_lift=s.model.q[s.model.lift_indices()].tolist(),
                     first_collisions=rejected)
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
        if distance < tolerance[0] and angle < tolerance[1]:
            break
    remaining_wall(ctx)
    s.note_relocated(before)
    actual = s.base_pose()
    return (np.linalg.norm(actual[:2] - goal[:2]) < tolerance[0]
            and abs(wrapped(actual[2] - goal[2])) < tolerance[1])


def travel_ready(ctx):
    """Raise the empty folded RM001 hands before driving beside loose pens."""
    s = ctx.skills()
    semantic.require(ctx, not s.held, 'travel pose requires empty hands', kind='planning')
    s.ready()
    goal = s.model.q.copy()
    goal[s.model.lift_indices()] = .70
    if np.max(np.abs(goal - s.model.q)) > .01:
        s.solvers[s.arms[0]].move_to_pose_with_RRTConnect(None, lift=True, goals=[goal])
    ctx.note('RM001 travel torso', lift_m=.70,
             tcp_heights={arm: float(ctx.tcp(arm)[2, 3]) for arm in s.arms})


def drawer_approach(ctx):
    """Face the cabinet from a relative, fully checked RM001 working pose."""
    semantic.require(ctx, not ctx.skills().held, 'drawer approach requires empty hands', kind='planning')
    s = ctx.skills()
    travel_ready(ctx)
    cabinet = ctx.pose('cabinet')
    yaw = np.arctan2(cabinet[1, 0], cabinet[0, 0])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                         [np.sin(yaw), np.cos(yaw)]])
    goal = np.r_[cabinet[:2, 3] + rotation @ [0., -.90], wrapped(yaw + np.pi / 2)]
    start = s.base_pose().copy()
    retreat = start.copy()
    retreat[:2] -= .20 * np.array([np.cos(start[2]), np.sin(start[2])])
    outside = goal.copy()
    outside[:2] -= .20 * np.array([np.cos(goal[2]), np.sin(goal[2])])
    selected = next((route for route in ([goal], [retreat, outside, goal])
                     if all(s.base_route(previous, target) is not None
                            for previous, target in zip([start, *route], route))), None)
    semantic.require(ctx, selected is not None, 'no checked cabinet-relative approach', kind='planning')
    ctx.note('RM001 cabinet approach', cabinet_front_distance_m=.90,
             waypoints=[point.tolist() for point in selected])
    for target in selected:
        semantic.require(ctx, checked_approach(ctx, target, 'RM001 cabinet approach'),
                         'cabinet-relative route changed', kind='check')


def retrieve_buffered(ctx, obj, where, **options):
    transfer(ctx, obj, where, **options)


def bottom_drawer_stance(ctx):
    """Reuse the supported half-pull stance physically verified in this chain."""
    travel_ready(ctx)
    cabinet = ctx.pose('cabinet')
    local = np.array([-.488269983, -.726787482, 0.])
    goal = np.r_[(cabinet[:3, :3] @ local + cabinet[:3, 3])[:2],
                 wrapped(np.arctan2(cabinet[1, 0], cabinet[0, 0]) + .514801296)]
    ctx.note('RM001 bottom drawer common stance', local_xy_m=local[:2].tolist(),
             yaw_offset_rad=.514801296,
             source='pass-two step22 a01 actual base at the successful supported handle grasp')
    semantic.require(ctx, checked_approach(ctx, goal, 'RM001 bottom drawer approach'),
                     'no checked route to verified bottom drawer stance', kind='check')


def open_storage_drawer(ctx, number, *, amount=OPENING):
    if not _joints.is_open(ctx, 'cabinet', joint(number), amount):
        started = perf_counter()
        ctx.note('item phase start', item=f'drawer{number}', phase='open_drawer', drawer=number)
        with ctx.subtask(f'open drawer {number}'):
            try:
                if number == 3:
                    bottom_drawer_stance(ctx)
                else:
                    drawer_approach(ctx)
                if number in (2, 3):
                    seat_top_drawer(ctx)
                try:
                    with sloped_bottom_handle(ctx) if number == 3 else nullcontext():
                        semantic.open(ctx, 'cabinet', joint(number), amount, brace=True)
                except ExpertFailed as error:
                    # The real slider can reach its requested opening before
                    # a loaded arm settles at its final joint waypoint. The
                    # original braced call has already checked cabinet
                    # stability and withdrawn both hands on this exit path.
                    if error.reason != 'tracking' or not _joints.is_open(
                            ctx, 'cabinet', joint(number), amount):
                        raise
                    semantic.require(ctx, not ctx.skills().held
                                     and not any(s.attachments for s in ctx.solvers.values()),
                                     'opened drawer tracking exit left a hand occupied', kind='check')
                    all_on_desk(ctx)
                    ctx.note('RM001 drawer opening physically complete', drawer=number,
                             opening_m=_joints.opening(ctx, 'cabinet', joint(number)),
                             arm_failure=error.reason, arm_failure_details=error.details)
            finally:
                ctx.note('item phase result', item=f'drawer{number}', phase='open_drawer', drawer=number,
                         wall_s=perf_counter() - started)


def drawer_opened(ctx, number):
    semantic.require(ctx, _joints.is_open(ctx, 'cabinet', joint(number), OPENING),
                     'drawer not open for storage', kind='check', drawer=number)


def screw_drawer_half_opened(ctx):
    if contained(ctx, 'screwdriver', 'cabinet', drawer(3)):
        return
    slider = _joints.slider(ctx, 'cabinet', joint(3))
    amount = (slider.limits[1] - slider.limits[0]) / 2
    semantic.require(ctx, _joints.is_open(ctx, 'cabinet', joint(3), amount),
                     'drawer not half open for screwdriver', kind='check', drawer=3)


def open_screw_drawer_half(ctx):
    """Retain the inherited half-pull plan, searching an RM001 stance."""
    if contained(ctx, 'screwdriver', 'cabinet', drawer(3)):
        return
    slider = _joints.slider(ctx, 'cabinet', joint(3))
    amount = (slider.limits[1] - slider.limits[0]) / 2
    if not _joints.is_open(ctx, 'cabinet', joint(3), amount):
        deadline = perf_counter() + 600.
        previous = getattr(ctx, 'wall_deadline', None)
        ctx.wall_deadline = min(previous, deadline) if previous is not None else deadline
        before = ctx.pose('cabinet').copy()
        try:
            with sloped_bottom_handle(ctx):
                skills.braced_drawer(ctx, 'cabinet', joint(3), amount=amount)
        except ExpertFailed as error:
            if error.reason != 'tracking' or not _joints.is_open(ctx, 'cabinet', joint(3), amount):
                raise
            s = ctx.skills()
            semantic.require(ctx, not s.held and not any(solver.attachments for solver in ctx.solvers.values()),
                             'half-open drawer tracking exit left a hand occupied', kind='check')
            after = ctx.pose('cabinet')
            shift = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
            turn = float(np.arccos(np.clip((np.trace(before[:3, :3].T @ after[:3, :3]) - 1) / 2, -1., 1.)))
            semantic.require(ctx, shift <= .020 and turn <= np.radians(5.),
                             'half-open drawer moved the cabinet', kind='check', shift=shift, turn=turn)
            all_on_desk(ctx)
            ctx.note('RM001 bottom drawer half physically complete',
                     opening_m=_joints.opening(ctx, 'cabinet', joint(3)),
                     cabinet_shift=shift, cabinet_turn=turn,
                     arm_failure=error.reason, arm_failure_details=error.details)


@contextmanager
def sloped_bottom_handle(ctx):
    """Use an explicit RM001 approach preference for this one low handle."""
    s = ctx.skills()
    slider = _joints.slider(ctx, 'cabinet', joint(3))
    direction = slider.direction(ctx.pose('cabinet', slider.child))
    approach = -direction + np.array([0., 0., -1.])
    approach /= np.linalg.norm(approach)
    ordinary_paths, ordinary_grasp = s.grasp_paths, s.grasp
    def applies(target):
        return target.obj == 'cabinet' and 'cabinet:cabinet.drawer3.handle' in target.names
    def paths(target, *args, **options):
        if applies(target):
            options.update(approach=approach, max_angle=np.radians(20.))
        return ordinary_paths(target, *args, **options)
    def grasp(target, *args, **options):
        if applies(target):
            options.update(approach=approach, max_angle=np.radians(20.))
        return ordinary_grasp(target, *args, **options)
    ctx.note('RM001 low handle approach', approach=approach.tolist(),
             cone_deg=20., local_budget_s=600.,
             scope='drawer3 handle; expert entry restores this local preference')
    s.grasp_paths, s.grasp = paths, grasp
    try:
        yield
    finally:
        s.grasp_paths, s.grasp = ordinary_paths, ordinary_grasp


def store_screw_first(ctx):
    """Keep the inherited basket destination without the x1pro grip."""
    if contained(ctx, 'screwdriver', 'basket'):
        return
    close(ctx, 3)
    transfer(ctx, 'screwdriver', screw_basket_drop_region)
    ctx.hold(1.)
    stored(ctx, 'screwdriver', 3)


def screw_basket_drop_region(ctx):
    return basket_drop_region(ctx, obj='screwdriver')




def bottle_side_done(ctx):
    if 'glue_bottle' in STORAGE_OVERRIDES:
        return stored(ctx, 'glue_bottle', 2)
    if contained(ctx, 'glue_bottle', 'cabinet', drawer(2)):
        return
    semantic.require(ctx, semantic.on(ctx, 'glue_bottle', 'table'), 'side bottle not on table', kind='check')
    semantic.require(ctx, abs(principal_axis(ctx, 'glue_bottle')[2]) < np.sin(ANGLE_TOLERANCE),
                     'bottle did not rest on its side', kind='check')


def rest_bottle_on_side(ctx):
    """Change the table contact before choosing a hold for the shallow drawer."""
    if 'glue_bottle' in STORAGE_OVERRIDES:
        return store_in_alternate(ctx, 'glue_bottle')
    if contained(ctx, 'glue_bottle', 'cabinet', drawer(2)):
        return
    size = np.sort(np.ptp(ctx.geometry['glue_bottle'].points(), axis=0))
    where = _regions.on_region('glue_bottle', 'table',
        near=_world.centre(_world.object_points(ctx, 'glue_bottle')),
        keep_out=front_buffer_workspace(ctx))
    def side_region(ctx):
        region, blocked = where(ctx)
        metadata = region[3] if len(region) > 3 else {}
        return (*region[:3], {**metadata, 'maximum_height': float(size[1]) + .002}), blocked
    transfer(ctx, 'glue_bottle', side_region, orientation='insert')


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
    shapes = _world.scene_shapes(ctx, exclude=(obj,))
    basket = [shape for shape in shapes if shape.obj == 'basket']
    region, blocked = _world.support_region(basket, 'basket', floor=True)
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


def store_in_alternate(ctx, obj):
    if contained(ctx, obj, STORAGE_OVERRIDES[obj]):
        return
    if obj in ('dish', 'glue_bottle'):
        # The drawer strategies failed in the real chain. Keep the verified
        # dish grip or ordinary bottle grip, then use the basket.
        target = ctx.skills().target(obj)
        previous = getattr(ctx, 'wall_deadline', None)
        deadline = perf_counter() + 600.
        ctx.wall_deadline = min(previous, deadline) if previous is not None else deadline
        ctx.note('RM001 dish basket plan' if obj == 'dish' else 'RM001 bottle basket plan',
                 local_budget_s=600., cone_deg=15. if obj == 'dish' else None,
                 source='drawer strategy failed; preserve the live resting orientation in the basket')
        try:
            with ctx.subtask(f'pick up the {obj} independently for the basket'):
                options = (dict(shape=target.shape, offset=target.offset,
                                approach=(0., 0., -1.), max_angle=np.radians(15.))
                           if obj == 'dish' else {})
                skills.Pick(obj, **options).run(ctx, base=obj == 'dish')
            try:
                with ctx.subtask(f'put the {obj} in the basket'):
                    where = dish_basket_drop_region if obj == 'dish' else bottle_basket_drop_region
                    skills.Place(obj, where, above=.04).run(ctx)
            except ExpertFailed as error:
                floor = _regions.inside_region(obj, 'basket',
                    **({'top_tolerance': .005} if obj == 'dish' else {}))
                if not (error.reason == 'no placement' and error.details.get('last') == 'tracking'
                        and release_measured_container(ctx, obj, 'basket', floor)):
                    raise
            ctx.hold(1.)
            stored(ctx, obj, 3 if obj == 'dish' else 2)
        finally:
            ctx.wall_deadline = previous
        return
    transfer(ctx, obj, screw_basket_drop_region if obj == 'screwdriver' else basket_drop_region)
    ctx.hold(1.)


def dish_basket_drop_region(ctx):
    return basket_drop_region(ctx, obj='dish')


def bottle_basket_drop_region(ctx):
    return basket_drop_region(ctx, obj='glue_bottle')


def top_drawer_drop_region(ctx, obj, number=1):
    """Release above a live exposed drawer floor; existing callers use the top one."""
    shapes = _world.scene_shapes(ctx, exclude=(obj,))
    region, blocked = _world.support_region(shapes, 'cabinet', drawer(number), floor=True)
    semantic.require(ctx, region is not None, 'no exposed drawer floor', kind='planning', drawer=number)
    frame, low, high = region
    inverse = np.linalg.inv(frame)
    own = [s for s in shapes if _world.selects(s, 'cabinet', drawer(number))]
    points = np.vstack([s.points for s in own]) @ inverse[:3, :3].T + inverse[:3, 3]
    rim = float(points[:, 2].max())
    frame = frame.copy()
    frame[:3, 3] += frame[:3, 2] * (rim + .02)
    ctx.note('RM001 top drawer release' if number == 1 else 'RM001 drawer mouth release',
             obj=obj, drawer=number, above_rim_m=.02, approach_above_m=.04)
    return (frame, low, high), blocked


def knife_drawer_drop_region(ctx):
    return top_drawer_drop_region(ctx, 'knife')


def black_pen_drawer_drop_region(ctx):
    return top_drawer_drop_region(ctx, 'pen_black')




def eraser_drawer_drop_region(ctx):
    region, blocked = _regions.inside_region(
        'eraser', 'cabinet', drawer(1), top_tolerance=.005)(ctx)
    mouth, _ = top_drawer_drop_region(ctx, 'eraser')
    frame = region[0].copy()
    frame[:3, 3] += frame[:3, 2] * (mouth[0][2, 3] - frame[2, 3])
    columns = []
    for obstacle in world_keepouts(blocked):
        if isinstance(obstacle, np.ndarray):
            columns.append((np.eye(4), obstacle[:, :2].min(0) - .003,
                            obstacle[:, :2].max(0) + .003))
        else:
            f, low, high = obstacle
            columns.append((f, np.asarray(low)[:2] - .003,
                            np.asarray(high)[:2] + .003))
    return (frame, *region[1:]), columns


def eraser_to_drawer(ctx):
    """Keep the proven overhead grip and release through the exposed mouth."""
    previous = getattr(ctx, 'wall_deadline', None)
    deadline = perf_counter() + 600.
    ctx.wall_deadline = min(previous, deadline) if previous is not None else deadline
    floor = _regions.inside_region('eraser', 'cabinet', drawer(1), top_tolerance=.005)
    ctx.note('RM001 eraser drawer mouth plan', local_budget_s=600.,
             above_mouth_m=.02, above_m=.04, original_floor_release_guard=True,
             world_vertical_columns=True, extra_planning_margin_m=.003,
             source='step53 a01 real overhead grip; low placement stopped tracking')
    s = ctx.skills()
    ordinary_lower = s.lower
    def inspect_eraser(arm, placed, spot, tcp_object, clearance, above, lift, result, orientation=None):
        done = ordinary_lower(arm, placed, spot, tcp_object, clearance, above, lift, result, orientation)
        if not done and result['last'] == 'tracking' and s.held[arm]['target'].obj == 'eraser':
            ctx.fail('no placement', kind='check', last='tracking', measured_eraser_stop=True)
        return done
    s.lower = inspect_eraser
    try:
        try:
            transfer(ctx, 'eraser', eraser_drawer_drop_region, above=.04, step=.005)
        except ExpertFailed as error:
            if not (error.reason == 'no placement' and error.details.get('last') == 'tracking'
                    and release_measured_drawer(ctx, 'eraser', 1, floor)):
                raise
        ctx.hold(1.)
        stored(ctx, 'eraser', 1)
    finally:
        s.lower = ordinary_lower
        ctx.wall_deadline = previous






def store(ctx, obj, number, *, open_if_needed=True):
    if obj in STORAGE_OVERRIDES:
        return store_in_alternate(ctx, obj)
    if contained(ctx, obj, 'cabinet', drawer(number)):
        return
    if open_if_needed:
        open_storage_drawer(ctx, number)
    else:
        drawer_opened(ctx, number)
    if obj == 'eraser' and number == 1:
        return eraser_to_drawer(ctx)
    if obj == 'knife':
        transfer(ctx, obj, knife_drawer_drop_region,
                 pick_options=dict(approach=(0., 0., -1.), max_angle=np.radians(15.)),
                 above=.04)
        ctx.hold(1.)
        return
    if obj == 'charger' and number == 3:
        return charger_to_drawer(ctx)
    if obj == 'snack' and number == 3:
        return snack_to_drawer(ctx)
    if obj == 'board_eraser' and number == 2:
        return board_eraser_to_drawer(ctx)
    if obj == 'tape' and number == 2:
        return tape_to_drawer(ctx)
    if obj == 'stapler' and number == 2:
        return stapler_to_drawer(ctx)
    if obj == 'dish' and number == 3:
        return dish_to_drawer(ctx)
    size = np.sort(np.ptp(ctx.geometry[obj].points(), axis=0))
    body = ctx.geometry['cabinet'].parts(drawer(number))[0][0]
    low, high = ctx.geometry['cabinet'].bounds(drawer(number), frame=body)
    height = float(high[2] - low[2])
    orientation = ('flat' if size[1] >= 2 * size[0] else
                   'insert' if size[2] >= 2 * size[1] and size[2] > height else None)
    if obj == 'glue_bottle':
        bottle_side_done(ctx)
        orientation = None
    far_buffer = (obj in ('screwdriver', 'charger', 'dish')
                  and _world.centre(_world.object_points(ctx, obj))[0]
                  < ctx.scene.TABLE['center'][0] - ctx.scene.TABLE['size'][0] / 4)
    if far_buffer and obj == 'screwdriver':
        # Retrieval keeps the table-resting tilt. A shallow drawer does not
        # require the long vertical descent used for a narrow-mouth insert.
        # The measured full-object resting-height bound still applies.
        orientation = None
    where = _regions.inside_region(obj, 'cabinet', drawer(number),
                                   orientation=orientation, top_tolerance=.005)
    if far_buffer:
        retrieve_buffered(ctx, obj, where, **({'orientation': orientation} if orientation else {}))
    elif obj in ('screwdriver', 'charger', 'dish'):
        with item_phase(ctx, obj, 'pick_place'):
            transfer(ctx, obj, where, **({'orientation': orientation} if orientation else {}))
    elif obj == 'lidded_jar' and number == 3:
        before = ctx.pose('cabinet').copy()
        deadline = perf_counter() + 600.
        previous = getattr(ctx, 'wall_deadline', None)
        ctx.wall_deadline = min(previous, deadline) if previous is not None else deadline
        ctx.note('RM001 drawer hover fallback', obj=obj, drawer=number,
                 local_budget_s=600., source='step26 a01 stopped hover; recompute the live floor')
        try:
            transfer(ctx, obj, where, **({'orientation': orientation} if orientation else {}))
        except ExpertFailed as error:
            if not (error.reason == 'no placement' and error.details.get('last') == 'tracking'
                    and release_measured_drawer(ctx, obj, number, where)):
                raise
        after = ctx.pose('cabinet')
        shift = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
        turn = float(np.arccos(np.clip((np.trace(before[:3, :3].T @ after[:3, :3]) - 1) / 2, -1., 1.)))
        semantic.require(ctx, shift <= .020 and turn <= np.radians(5.),
                         'jar placement moved the cabinet', kind='check', shift=shift, turn=turn)
    else:
        transfer(ctx, obj, where, **({'orientation': orientation} if orientation else {}))


def release_measured_drawer(ctx, obj, number, where):
    """Drop from a measured clear hover over the original exposed drawer floor."""
    return release_measured_container(ctx, obj, 'cabinet', where, drawer(number), number)


def release_measured_container(ctx, obj, container, where, part=None, number=None):
    """Execute a release only over a measured clear original container floor."""
    def rejected(reason, **details):
        ctx.note('RM001 measured release rejected', obj=obj, container=container,
                 reason=reason, **details)
        return False
    s = ctx.skills()
    held = [(arm, value) for arm, value in s.held.items() if value['target'].obj == obj]
    if len(held) != 1:
        return rejected('requires exactly one held object', matches=len(held))
    arm, value = held[0]
    region, blocked = where(ctx)
    frame, low, high = region[:3]
    inverse = np.linalg.inv(frame)
    points = _world.object_points(ctx, obj)
    local = points @ inverse[:3, :3].T + inverse[:3, 3]
    height = float(local[:, 2].min())
    if not .02 <= height <= .15:
        return rejected('bottom outside original hover interval', bottom_above_floor_m=height)
    if (np.any(local[:, :2].min(0) < low + PLACE_MARGIN)
            or np.any(local[:, :2].max(0) > high - PLACE_MARGIN)):
        return rejected('footprint outside original region margin',
                        object_low=local[:, :2].min(0).tolist(), object_high=local[:, :2].max(0).tolist(),
                        region_low=low.tolist(), region_high=high.tolist())
    metadata = region[3] if len(region) > 3 else {}
    resting_top = float(np.ptp(local[:, 2]))
    if resting_top > min(metadata.get('maximum_height', np.inf),
                         metadata.get('maximum_release_top', np.inf)):
        return rejected('resting height exceeds original headroom', resting_top_m=resting_top)
    outline = local[:, :2][ConvexHull(local[:, :2]).vertices]
    for obstacle in world_keepouts(blocked):
        if isinstance(obstacle, np.ndarray):
            other = obstacle @ inverse[:3, :3].T + inverse[:3, 3]
            if other[:, 2].max() <= 0. or other[:, 2].min() > local[:, 2].max():
                continue
            if overlaps(outline, other[:, :2].min(0) - PLACE_MARGIN,
                        other[:, :2].max(0) + PLACE_MARGIN):
                return rejected('solid obstructs vertical release')
        else:
            other_frame, other_low, other_high = obstacle
            other_inverse = np.linalg.inv(other_frame)
            projected = (points @ other_inverse[:3, :3].T + other_inverse[:3, 3])[:, :2]
            polygon = projected[ConvexHull(projected).vertices]
            if overlaps(polygon, np.asarray(other_low)[:2] - PLACE_MARGIN,
                        np.asarray(other_high)[:2] + PLACE_MARGIN):
                return rejected('projected keepout obstructs vertical release')
    closure = s.pad_closure(value['target'], arm, value['grasp']['open_gap'])
    if not closure['held']:
        return rejected('actual pad closure lost the object', closure=closure)
    ctx.note('RM001 measured drawer release' if container == 'cabinet'
             else 'RM001 measured basket release', obj=obj, drawer=number, arm=arm,
             bottom_above_floor_m=height, resting_top_m=resting_top,
             low=points.min(0).tolist(), high=points.max(0).tolist(),
             source='actual stopped pose after placement tracking; original floor and keep-outs')
    with ctx.subtask(f'release {obj} over the validated {container} floor'):
        try:
            skills.release(ctx, arm)
        except ExpertFailed as error:
            if error.reason != 'tracking':
                raise
            ctx.note('RM001 container object released before retreat tracking stopped',
                     obj=obj, reason=error.reason)
    ctx.hold(1.)
    semantic.require(ctx, contained(ctx, obj, container, part),
                     'measured release did not contain the whole object', kind='check', obj=obj, container=container)
    closure = s.pad_closure(value['target'], arm, value['grasp']['open_gap'])
    semantic.require(ctx, not closure['held'], 'container object still held after release', kind='check', obj=obj)
    return True


def charger_drawer_drop_region(ctx):
    return top_drawer_drop_region(ctx, 'charger', number=3)


def loose_front_stance(ctx, obj):
    """Reach a measured front work position before an independent loose-object pick."""
    travel_ready(ctx)
    goal = np.r_[_world.centre(_world.object_points(ctx, obj))[:2] + [0., -.55], np.pi / 2]
    ctx.note('RM001 loose object front stance', obj=obj, relative_xy_m=[0., -.55],
             yaw_rad=float(np.pi / 2),
             source='pass-two charger rescue: previous placement stances had no executable grasp')
    semantic.require(ctx, checked_approach(ctx, goal, 'RM001 loose object approach', attempts=5),
                     'no checked route to loose object front stance', kind='check', obj=obj)


def charger_to_drawer(ctx):
    """Keep the real buffer-entry grip, then drop through the exposed drawer mouth."""
    deadline = perf_counter() + 600.
    previous = getattr(ctx, 'wall_deadline', None)
    ctx.wall_deadline = min(previous, deadline) if previous is not None else deadline
    before = ctx.pose('cabinet').copy()
    ctx.note('RM001 charger drawer mouth plan', local_budget_s=600.,
             independent_pick=True, pick_base=False, above_mouth_m=.02, above_m=.04,
             source='step27 a01 lifted with left hand but no deep placement IK')
    loose_front_stance(ctx, 'charger')
    with ctx.subtask('pick up the buffered charger independently'):
        skills.Pick('charger').run(ctx, base=False)
    floor = _regions.inside_region('charger', 'cabinet', drawer(3), top_tolerance=.005)
    try:
        with ctx.subtask('drop the charger through the exposed bottom drawer mouth'):
            skills.Place('charger', charger_drawer_drop_region, above=.04).run(ctx)
    except ExpertFailed as error:
        if not (error.reason == 'no placement' and error.details.get('last') == 'tracking'
                and release_measured_drawer(ctx, 'charger', 3, floor)):
            raise
    ctx.hold(1.)
    after = ctx.pose('cabinet')
    shift = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
    turn = float(np.arccos(np.clip((np.trace(before[:3, :3].T @ after[:3, :3]) - 1) / 2, -1., 1.)))
    semantic.require(ctx, shift <= .020 and turn <= np.radians(5.),
                     'charger placement moved the cabinet', kind='check', shift=shift, turn=turn)
    stored(ctx, 'charger', 3)


def dish_drawer_drop_region(ctx):
    return top_drawer_drop_region(ctx, 'dish', number=3)


def dish_high_drawer_drop_region(ctx):
    """Keep the exposed floor footprint, with the release plane above the cabinet."""
    region, blocked = dish_drawer_drop_region(ctx)
    frame = region[0].copy()
    height = float(_world.object_points(ctx, 'cabinet')[:, 2].max()) + .04
    frame[:3, 3] += frame[:3, 2] * (height - frame[2, 3])
    ctx.note('RM001 dish high drawer release', release_plane_z_m=float(frame[2, 3]),
             above_cabinet_m=.04, margin_m=PLACE_MARGIN,
             source='coordinator requested drawer priority; preserve all floor keep-outs')
    return (frame, *region[1:]), blocked


def release_high_dish(ctx):
    """Validate the entire drawer fall column before a real high release."""
    points = _world.object_points(ctx, 'dish')
    cabinet_top = float(_world.object_points(ctx, 'cabinet')[:, 2].max())
    clearance = float(points[:, 2].min() - cabinet_top)
    if not .06 <= clearance <= .12:
        ctx.note('RM001 high dish release rejected', reason='outside high approach band',
                 bottom_above_cabinet_m=clearance)
        return False
    def high_column(ctx):
        region, blocked = dish_high_drawer_drop_region(ctx)
        # Keep every original obstacle throughout the fall, including those
        # below this high plane. Two-dimensional boxes cannot be z-culled.
        columns = []
        for obstacle in blocked:
            if isinstance(obstacle, np.ndarray):
                columns.append((np.eye(4), obstacle[:, :2].min(0), obstacle[:, :2].max(0)))
            else:
                frame, low, high = obstacle
                columns.append((frame, np.asarray(low)[:2], np.asarray(high)[:2]))
        return region, columns
    ctx.note('RM001 measured high dish release plan',
             cabinet_top_z_m=cabinet_top, bottom_above_cabinet_m=clearance,
             release_band_m=[.06, .12], full_original_fall_column=True,
             source='actual above-target stop; original closure and 10mm margin retained')
    return release_measured_container(ctx, 'dish', 'cabinet', high_column, drawer(3), 3)


def snack_drawer_drop_region(ctx):
    return top_drawer_drop_region(ctx, 'snack', number=3)


def board_eraser_drawer_drop_region(ctx):
    return top_drawer_drop_region(ctx, 'board_eraser', number=2)


def board_eraser_repack_region(ctx):
    region, blocked = board_eraser_drawer_drop_region(ctx)
    frame, low, high = region[:3]
    inverse = np.linalg.inv(frame)
    points = _world.object_points(ctx, 'board_eraser')
    local = points @ inverse[:3, :3].T + inverse[:3, 3]
    centre = np.array([-.04, .08])
    allowance = np.ptp(local[:, :2], axis=0) / 2 + PLACE_MARGIN + .004
    return (frame, np.maximum(low, centre - allowance),
            np.minimum(high, centre + allowance)), world_keepouts(blocked)


def board_eraser_to_drawer(ctx, *, repack=False):
    """Use a whole-object overhead pinch, then release over the exposed floor."""
    previous = getattr(ctx, 'wall_deadline', None)
    deadline = perf_counter() + 600.
    ctx.wall_deadline = min(previous, deadline) if previous is not None else deadline
    before = ctx.pose('cabinet').copy()
    target = ctx.skills().target('board_eraser')
    ctx.note('RM001 board eraser drawer plan', local_budget_s=600., cone_deg=15.,
             independent_pick=True, above_mouth_m=.02, above_m=.04,
             repack_floor_xy_m=[-.04, .08] if repack else None,
             source='ordinary felt-pad hold could not complete flat release in step35 a01')
    try:
        with ctx.subtask('pick up the whole board eraser from above'):
            skills.Pick('board_eraser', shape=target.shape, offset=target.offset,
                        approach=(0., 0., -1.), max_angle=np.radians(15.)).run(ctx)
        floor = _regions.inside_region('board_eraser', 'cabinet', drawer(2), top_tolerance=.005)
        try:
            with ctx.subtask('drop the board eraser through the exposed middle drawer mouth'):
                skills.Place('board_eraser',
                             board_eraser_repack_region if repack else board_eraser_drawer_drop_region,
                             above=.04, **({'step': .002} if repack else {})).run(ctx)
        except ExpertFailed as error:
            if not (error.reason == 'no placement' and error.details.get('last') == 'tracking'
                    and release_measured_drawer(ctx, 'board_eraser', 2, floor)):
                raise
        ctx.hold(1.)
        after = ctx.pose('cabinet')
        shift = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
        turn = float(np.arccos(np.clip((np.trace(before[:3, :3].T @ after[:3, :3]) - 1) / 2, -1., 1.)))
        semantic.require(ctx, shift <= .020 and turn <= np.radians(5.),
                         'board eraser placement moved the cabinet', kind='check', shift=shift, turn=turn)
        stored(ctx, 'board_eraser', 2)
    finally:
        ctx.wall_deadline = previous


def stapler_drawer_region(ctx):
    """Use the exposed mouth; existing contents remain real collision solids."""
    region, blocked = top_drawer_drop_region(ctx, 'stapler', number=2)
    ctx.note('RM001 stapler mouth above contents',
             source='v2 s36 a01 floor strip occupied; allow real supported stacking')
    return region, world_keepouts(blocked)


def stapler_to_drawer(ctx):
    """Retain the working grip and verify the actual hover before any release."""
    previous = getattr(ctx, 'wall_deadline', None)
    deadline = perf_counter() + 600.
    ctx.wall_deadline = min(previous, deadline) if previous is not None else deadline
    before = ctx.pose('cabinet').copy()
    where = _regions.inside_region('stapler', 'cabinet', drawer(2), top_tolerance=.005)
    ctx.note('RM001 stapler drawer hover plan', local_budget_s=600., target_inset_m=.005,
             independent_pick=True, pick_base=False,
             above_mouth_m=.02, above_m=.04,
             source='deep drawer RRT failed; keep the working narrow grip over the live mouth')
    try:
        try:
            with ctx.subtask('pick up the stapler independently of its drawer placement'):
                skills.Pick('stapler').run(ctx, base=False)
            with ctx.subtask('place the stapler inside the inset middle drawer target'):
                skills.Place('stapler', stapler_drawer_region, above=.04, step=.005).run(ctx)
        except ExpertFailed as error:
            if not (error.reason == 'no placement' and error.details.get('last') == 'tracking'
                    and release_measured_drawer(ctx, 'stapler', 2, where)):
                raise
        ctx.hold(1.)
        after = ctx.pose('cabinet')
        shift = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
        turn = float(np.arccos(np.clip((np.trace(before[:3, :3].T @ after[:3, :3]) - 1) / 2, -1., 1.)))
        semantic.require(ctx, shift <= .020 and turn <= np.radians(5.),
                         'stapler placement moved the cabinet', kind='check', shift=shift, turn=turn)
        stored(ctx, 'board_eraser', 2)
        stored(ctx, 'tape', 2)
        stored(ctx, 'stapler', 2)
    finally:
        ctx.wall_deadline = previous


def tape_drawer_drop_region(ctx):
    region, blocked = top_drawer_drop_region(ctx, 'tape', number=2)
    board = _world.object_points(ctx, 'board_eraser')
    # Storage currently requires direct drawer support. Reserve the entire
    # board footprint through the fall instead of settling on top of it.
    return region, [*world_keepouts(blocked),
                    (np.eye(4), board[:, :2].min(0), board[:, :2].max(0))]


def tape_to_drawer(ctx):
    """Release flat over the exposed mouth, then check the settled drawer result."""
    previous = getattr(ctx, 'wall_deadline', None)
    deadline = perf_counter() + 600.
    ctx.wall_deadline = min(previous, deadline) if previous is not None else deadline
    before = ctx.pose('cabinet').copy()
    target = ctx.skills().target('tape')
    ctx.note('RM001 tape mouth release plan', local_budget_s=600., cone_deg=15.,
             orientation='flat', above_mouth_m=.02, above_m=.04,
             source='v2 stack settled on the board but storage requires direct drawer support')
    try:
        stored(ctx, 'board_eraser', 2)
        slider = _joints.slider(ctx, 'cabinet', joint(2))
        open_storage_drawer(ctx, 2, amount=float(slider.limits[1] - slider.limits[0]))
        board_eraser_to_drawer(ctx, repack=True)
        loose_front_stance(ctx, 'tape')
        with ctx.subtask('pick up the whole tape roll from above'):
            skills.Pick('tape', shape=target.shape, offset=target.offset,
                        arm='right', approach=(0., 0., -1.),
                        max_angle=np.radians(15.)).run(ctx, base=False)
        where = _regions.inside_region('tape', 'cabinet', drawer(2), top_tolerance=.005)
        try:
            with ctx.subtask('release the tape flat over the exposed middle drawer mouth'):
                skills.Place('tape', tape_drawer_drop_region,
                             orientation='flat', above=.04).run(ctx)
        except ExpertFailed as error:
            if not (error.details.get('last') == 'tracking'
                    and release_measured_drawer(ctx, 'tape', 2, where)):
                raise
        ctx.hold(1.)
        after = ctx.pose('cabinet')
        shift = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
        turn = float(np.arccos(np.clip((np.trace(before[:3, :3].T @ after[:3, :3]) - 1) / 2, -1., 1.)))
        semantic.require(ctx, shift <= .020 and turn <= np.radians(5.),
                         'tape stacking moved the cabinet', kind='check', shift=shift, turn=turn)
        stored(ctx, 'board_eraser', 2)
        stored(ctx, 'tape', 2)
    finally:
        ctx.wall_deadline = previous


def snack_to_drawer(ctx):
    """Keep the ordinary working grip, without requiring a deep drawer pose."""
    previous = getattr(ctx, 'wall_deadline', None)
    deadline = perf_counter() + 600.
    ctx.wall_deadline = min(previous, deadline) if previous is not None else deadline
    before = ctx.pose('cabinet').copy()
    ctx.note('RM001 snack drawer mouth plan', local_budget_s=600.,
             independent_pick=True, pick_base=False, above_mouth_m=.02, above_m=.04,
             source='step29 a01 held the snack but 52 placement IK checks produced no plan')
    try:
        loose_front_stance(ctx, 'snack')
        with ctx.subtask('pick up the snack independently'):
            skills.Pick('snack', arm='left', approach=(0., 0., -1.),
                        max_angle=np.radians(15.)).run(ctx, base=False)
        floor = _regions.inside_region('snack', 'cabinet', drawer(3), top_tolerance=.005)
        try:
            with ctx.subtask('drop the snack through the exposed bottom drawer mouth'):
                skills.Place('snack', snack_drawer_drop_region, above=.04).run(ctx)
        except ExpertFailed as error:
            if not (error.reason == 'no placement' and error.details.get('last') == 'tracking'
                    and release_measured_drawer(ctx, 'snack', 3, floor)):
                raise
        ctx.hold(1.)
        after = ctx.pose('cabinet')
        shift = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
        turn = float(np.arccos(np.clip((np.trace(before[:3, :3].T @ after[:3, :3]) - 1) / 2, -1., 1.)))
        semantic.require(ctx, shift <= .020 and turn <= np.radians(5.),
                         'snack placement moved the cabinet', kind='check', shift=shift, turn=turn)
        stored(ctx, 'snack', 3)
    finally:
        ctx.wall_deadline = previous


def charger_repack_region(ctx):
    region, blocked = top_drawer_drop_region(ctx, 'charger', number=3)
    frame, low, high = region[:3]
    centre = np.array([-.08, .12])
    allowance = .03 + PLACE_MARGIN + .003
    return (frame, np.maximum(low, centre - allowance),
            np.minimum(high, centre + allowance)), blocked


def repack_charger_for_dish(ctx):
    """Use the rear-left floor patch, leaving the front-left patch for the dish."""
    previous = getattr(ctx, 'wall_deadline', None)
    deadline = perf_counter() + 240.
    ctx.wall_deadline = min(previous, deadline) if previous is not None else deadline
    try:
        s = ctx.skills()
        target = s.target('charger')
        ctx.note('RM001 repack charger for dish', floor_xy_m=[-.08, .12],
                 local_budget_s=240., cone_deg=15.,
                 source='original margin leaves zero flat-dish locations beside the first packing')
        with ctx.subtask('lift the charger to reserve a flat dish location'):
            skills.Pick('charger', arm='left', shape=target.shape, offset=target.offset,
                        approach=(0., 0., -1.), max_angle=np.radians(15.)).run(ctx, base=False)
        floor = _regions.inside_region('charger', 'cabinet', drawer(3), top_tolerance=.005)
        try:
            with ctx.subtask('repack the charger at the rear-left of the exposed drawer'):
                skills.Place('charger', charger_repack_region, above=.04, step=.002).run(ctx)
        except ExpertFailed as error:
            if not (error.reason == 'no placement' and error.details.get('last') == 'tracking'
                    and release_measured_drawer(ctx, 'charger', 3, floor)):
                raise
        ctx.hold(1.)
        stored(ctx, 'charger', 3)
        semantic.require(ctx, not s.held and not any(v.attachments for v in ctx.solvers.values()),
                         'charger repack left a hand occupied', kind='check')
        all_on_desk(ctx)
        region, blocked = dish_drawer_drop_region(ctx)
        target = s.target('dish')
        frame, low, high = region[:3]
        near = frame @ np.r_[(low + high) / 2, 0., 1.]
        options = s.placements(target, ctx.poses()[target.body],
                               region, near[:3], blocked, .02, PLACE_MARGIN)
        ctx.note('RM001 dish room after charger repack', candidates=len(options))
        semantic.require(ctx, bool(options), 'charger repack did not create a flat dish location', kind='planning')
    finally:
        ctx.wall_deadline = previous


def dish_left_stance(ctx):
    """Reach the cabinet-relative RM001 stance physically verified in step28."""
    cabinet = ctx.pose('cabinet')
    local = np.array([-.5533640124694741, -.7233894711273029, 0.])
    goal = np.r_[(cabinet[:3, :3] @ local + cabinet[:3, 3])[:2],
                 np.arctan2(cabinet[1, 0], cabinet[0, 0]) + .5449476619410601]
    travel_ready(ctx)
    semantic.require(ctx, checked_approach(ctx, goal, 'RM001 dish proven left stance'),
                     'no checked route to proven dish stance', kind='check')
    return local


def dish_to_drawer(ctx):
    """Reuse the verified whole-dish pinch instead of the thin rim."""
    deadline = perf_counter() + 600.
    previous = getattr(ctx, 'wall_deadline', None)
    ctx.wall_deadline = min(previous, deadline) if previous is not None else deadline
    before = ctx.pose('cabinet').copy()
    target = ctx.skills().target('dish')
    ctx.note('RM001 dish drawer mouth plan', local_budget_s=600.,
             grip='whole fitted object', cone_deg=15., independent_pick=True,
             above_mouth_m=.02, above_m=.04, margin_m=PLACE_MARGIN, grid_m=.005,
             source='test original floor at 5mm grid before repacking; release above the cabinet')
    region, blocked = dish_drawer_drop_region(ctx)
    frame, low, high = region[:3]
    near = frame @ np.r_[(low + high) / 2, 0., 1.]
    options = ctx.skills().placements(target, ctx.poses()[target.body],
                                      region, near[:3], blocked, .005, PLACE_MARGIN)
    ctx.note('RM001 dish room before charger repack', candidates=len(options),
             grid_m=.005, margin_m=PLACE_MARGIN)
    local = dish_left_stance(ctx)
    if not options:
        ctx.note('RM001 dish reserve reachable left patch',
                 source='no existing flat-dish location; repack the blocking charger')
        repack_charger_for_dish(ctx)
    else:
        ctx.note('RM001 dish use existing drawer room', candidates=len(options),
                 source='v2 entry has room; avoid the inherited unconditional charger repack')
    # This left-side RM001 stance and the dish-local Y pinch actually held
    # this buffered dish in step28 a04. The perpendicular pinch in step44
    # a02 had only 4.5 mm2 of measured contact and failed the original check.
    local = dish_left_stance(ctx)
    s = ctx.skills()
    ordinary_options = s.options
    def dish_options(target, *args, **kwargs):
        for result in ordinary_options(target, *args, **kwargs):
            arm, grasp = result[:2]
            if grasp.target.obj == 'dish':
                closing = ctx.pose('dish')[:3, 1]
                if abs(float(grasp.pose[:3, 1] @ closing)) < np.cos(np.radians(5.)):
                    continue
            yield result
    ctx.note('RM001 dish proven grip plan', stance_kind='cabinet-relative',
             local_xy_m=local[:2].tolist(), yaw_offset_rad=.5449476619410601,
             arm='left', closing_axis='dish local Y', closing_cone_deg=5.,
             source='step28 a04 real held contact; original options and closure checks retained')
    s.options = dish_options
    try:
        with ctx.subtask('pick up the buffered whole dish'):
            skills.Pick('dish', arm='left', shape=target.shape, offset=target.offset,
                        approach=(0., 0., -1.), max_angle=np.radians(15.)).run(ctx, base=False)
    finally:
        s.options = ordinary_options
    ordinary_placements = s.placements
    ordinary_lower = s.lower
    def level_dish_placements(target, current, *args, **kwargs):
        if target.obj == 'dish':
            current = current.copy()
            heading = float(np.arctan2(current[1, 0], current[0, 0]))
            c, t = np.cos(heading), np.sin(heading)
            current[:3, :3] = [[c, -t, 0.], [t, c, 0.], [0., 0., 1.]]
        return ordinary_placements(target, current, *args, **kwargs)
    def high_dish_lower(arm, placed, spot, tcp_object, clearance, above, lift, result, orientation=None):
        done = ordinary_lower(arm, placed, spot, tcp_object, clearance, above, lift, result, orientation)
        if not done and result['last'] == 'tracking' and s.held[arm]['target'].obj == 'dish':
            ctx.note('RM001 inspect first high dish tracking stop',
                     spot=spot.tolist(), planned=result['planned'])
            ctx.fail('no placement', kind='check', last='tracking', high_stop=True)
        return done
    ctx.note('RM001 level dish release plan', normal=[0., 0., 1.],
             source='a03 lifted dish tilted 7.67 degrees; preserve original floor margin and keep-outs')
    s.placements = level_dish_placements
    s.lower = high_dish_lower
    try:
        try:
            with ctx.subtask('drop the dish from above the cabinet through the exposed drawer footprint'):
                skills.Place('dish', dish_high_drawer_drop_region, above=.04, step=.005).run(ctx)
        except ExpertFailed as error:
            if not (error.reason == 'no placement' and error.details.get('last') == 'tracking'
                    and release_high_dish(ctx)):
                raise
    finally:
        s.placements = ordinary_placements
        s.lower = ordinary_lower
    ctx.hold(1.)
    after = ctx.pose('cabinet')
    shift = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
    turn = float(np.arccos(np.clip((np.trace(before[:3, :3].T @ after[:3, :3]) - 1) / 2, -1., 1.)))
    semantic.require(ctx, shift <= .020 and turn <= np.radians(5.),
                     'dish placement moved the cabinet', kind='check', shift=shift, turn=turn)
    stored(ctx, 'dish', 3)


def buffer_then_store(ctx, obj, number):
    """Rescue a known front-clearing deadlock, then make the one final placement."""
    if obj in STORAGE_OVERRIDES:
        return store_in_alternate(ctx, obj)
    if contained(ctx, obj, 'cabinet', drawer(number)):
        return
    if obj == 'dish':
        buffered(ctx, obj, keep_out=front_buffer_workspace(ctx))
        ctx.note('RM001 reuse existing dish buffer',
                 source='step21 already moved this real object clear; original buffer predicate rechecked')
        return store(ctx, obj, number, open_if_needed=False)
    with item_phase(ctx, obj, 'buffer'):
        buffer(ctx, obj)
    store(ctx, obj, number, open_if_needed=False)


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


def charger_buffer_drop_region(ctx):
    """Release above the original table-relative front buffer patch."""
    table = ctx.scene.TABLE
    centre, size = np.asarray(table['center']), np.asarray(table['size'])
    side = -1. if ctx.pose('cabinet')[0, 3] >= centre[0] else 1.
    near = np.r_[centre + size * [side * 3 / 8, -1 / 3], table['height']]
    region, blocked = _regions.on_region(
        'charger', 'table', near=near, keep_out=front_buffer_workspace(ctx))(ctx)
    frame = region[0].copy()
    frame[:3, 3] += frame[:3, 2] * .03
    return (frame, *region[1:]), blocked


def release_measured_charger(ctx):
    return release_measured_buffer(ctx, 'charger')


def dish_buffer_region(ctx):
    """Reserve extra room from the charger after measured tracking drift."""
    table = ctx.scene.TABLE
    centre, size = np.asarray(table['center']), np.asarray(table['size'])
    side = -1. if ctx.pose('cabinet')[0, 3] >= centre[0] else 1.
    near = np.r_[centre + size * [side * 3 / 8, -1 / 3], table['height']]
    charger = _world.object_points(ctx, 'charger')[:, :2]
    reserve = (np.eye(4), charger.min(0) - .02, charger.max(0) + .02)
    ctx.note('RM001 dish charger reserve', extra_clearance_m=.02)
    return _regions.on_region('dish', 'table', near=near,
                              keep_out=[*front_buffer_workspace(ctx), reserve])(ctx)


def release_measured_buffer(ctx, obj):
    """Use a verified actual hover as a new release action after tracking stops."""
    s = ctx.skills()
    held = [(arm, value) for arm, value in s.held.items()
            if value['target'].obj == obj]
    if len(held) != 1:
        return False
    arm, value = held[0]
    points = _world.object_points(ctx, obj)
    height = float(points[:, 2].min() - ctx.scene.TABLE['height'])
    if not .02 <= height <= .15:
        return False
    if obj == 'charger':
        region, blocked = charger_buffer_drop_region(ctx)
        release_height = .03
    else:
        region, blocked = dish_buffer_region(ctx)
        release_height = 0.
    frame, low, high = region[:3]
    frame = frame.copy()
    frame[:3, 3] -= frame[:3, 2] * release_height
    inverse = np.linalg.inv(frame)
    local = points @ inverse[:3, :3].T + inverse[:3, 3]
    if (np.any(local[:, :2].min(0) < low + PLACE_MARGIN)
            or np.any(local[:, :2].max(0) > high - PLACE_MARGIN)):
        return False
    outline = local[:, :2][ConvexHull(local[:, :2]).vertices]
    for obstacle in world_keepouts(blocked):
        if isinstance(obstacle, np.ndarray):
            other = obstacle @ inverse[:3, :3].T + inverse[:3, 3]
            if other[:, 2].max() <= 0. or other[:, 2].min() > local[:, 2].max():
                continue
            if overlaps(outline, other[:, :2].min(0) - PLACE_MARGIN,
                        other[:, :2].max(0) + PLACE_MARGIN):
                return False
        else:
            other_frame, other_low, other_high = obstacle
            other_inverse = np.linalg.inv(other_frame)
            projected = (points @ other_inverse[:3, :3].T + other_inverse[:3, 3])[:, :2]
            polygon = projected[ConvexHull(projected).vertices]
            if overlaps(polygon, np.asarray(other_low)[:2] - PLACE_MARGIN,
                        np.asarray(other_high)[:2] + PLACE_MARGIN):
                return False
    closure = s.pad_closure(value['target'], arm, value['grasp']['open_gap'])
    if not closure['held']:
        return False
    ctx.note(f'RM001 measured {obj} release', arm=arm, bottom_above_table_m=height,
             low=points.min(0).tolist(), high=points.max(0).tolist(),
             source='actual stopped pose after failed placement tracking')
    with ctx.subtask(f'release {obj} over the validated table patch'):
        try:
            skills.release(ctx, arm)
        except ExpertFailed as error:
            if error.reason != 'tracking':
                raise
            ctx.note(f'RM001 {obj} released before retreat tracking stopped', reason=error.reason)
    ctx.hold(1.)
    buffered(ctx, obj, keep_out=front_buffer_workspace(ctx))
    closure = s.pad_closure(value['target'], arm, value['grasp']['open_gap'])
    semantic.require(ctx, not closure['held'], f'{obj} still held after measured release', kind='check')
    return True


def drawer_front_clear(ctx, obj):
    if not contained(ctx, obj, 'cabinet', drawer(3)):
        buffered(ctx, obj, keep_out=front_buffer_workspace(ctx))


def remote_drawer_drop_region(ctx, *, search=False):
    """Keep the native stack COM and solids throughout an above-mouth drop."""
    region, blocked = _regions.inside_region(
        'remote', 'cabinet', drawer(2), support='board_eraser',
        orientation='flat', top_tolerance=.005)(ctx)
    mouth, _ = top_drawer_drop_region(ctx, 'remote', number=2)
    frame = region[0].copy()
    frame[:3, 3] += frame[:3, 2] * (mouth[0][2, 3] - frame[2, 3])
    columns = []
    for obstacle in blocked:
        if isinstance(obstacle, np.ndarray):
            columns.append((np.eye(4), obstacle[:, :2].min(0), obstacle[:, :2].max(0)))
        else:
            f, low, high = obstacle
            columns.append((f, np.asarray(low)[:2], np.asarray(high)[:2]))
    if search:
        # a03 missed the top-drawer handle's original 10mm projection by
        # 0.8mm at its real tracking stop. Reserve an extra 3mm in planning;
        # the release guard below still checks the unchanged original region.
        columns = [(f, np.asarray(low) - .003, np.asarray(high) + .003)
                   for f, low, high in columns]
        ctx.note('RM001 remote extra planning reserve', obstacle_reserve_m=.003,
                 source='a03 actual stop overlapped the original top-handle margin by 0.8mm')
    return (frame, *region[1:]), columns


def release_measured_remote(ctx):
    """Require the actual mass COM inside the unchanged live support margin."""
    region, _ = remote_drawer_drop_region(ctx)
    surface, low, high = region[3]['surface']
    target = ctx.skills().target('remote')
    com = (np.linalg.inv(surface) @ np.r_[ctx.env.center_of_mass(target.body)[0], 1.])[:2]
    supported = bool(np.all(com >= np.asarray(low)[:2] + PLACE_MARGIN)
                     and np.all(com <= np.asarray(high)[:2] - PLACE_MARGIN))
    ctx.note('RM001 remote measured stack support', actual_mass_com_local_m=com.tolist(),
             support_low_m=np.asarray(low).tolist(), support_high_m=np.asarray(high).tolist(),
             margin_m=PLACE_MARGIN, supported=supported)
    if not supported:
        return False
    return release_measured_drawer(ctx, 'remote', 2, remote_drawer_drop_region)


def remote_drawer_search_region(ctx):
    return remote_drawer_drop_region(ctx, search=True)


def remote_to_drawer(ctx):
    """Use the clear right side of the board eraser's live top as a stack support."""
    drawer_opened(ctx, 2)
    previous = getattr(ctx, 'wall_deadline', None)
    deadline = perf_counter() + 600.
    ctx.wall_deadline = min(previous, deadline) if previous is not None else deadline
    ctx.note('RM001 remote stack plan', support='board_eraser', orientation='flat at rest; above-mouth release',
             independent_pick=True, local_budget_s=600.,
             source='a01 physically lifted the remote; original floor had no empty footprint')
    try:
        with ctx.subtask('pick up the remote independently for a flat stack'):
            skills.Pick('remote').run(ctx)
        s = ctx.skills()
        ordinary_placements, ordinary_lower = s.placements, s.lower
        def level_remote(target, current, *args, **kwargs):
            if target.obj == 'remote':
                current = current.copy()
                heading = float(np.arctan2(current[1, 0], current[0, 0]))
                c, t = np.cos(heading), np.sin(heading)
                current[:3, :3] = [[c, -t, 0.], [t, c, 0.], [0., 0., 1.]]
            return ordinary_placements(target, current, *args, **kwargs)
        def inspect_remote(arm, placed, spot, tcp_object, clearance, above, lift, result, orientation=None):
            done = ordinary_lower(arm, placed, spot, tcp_object, clearance, above, lift, result, orientation)
            if not done and result['last'] == 'tracking' and s.held[arm]['target'].obj == 'remote':
                ctx.fail('no placement', kind='check', last='tracking', measured_remote_stop=True)
            return done
        s.placements, s.lower = level_remote, inspect_remote
        try:
            try:
                with ctx.subtask('drop the remote over the clear board eraser top inside drawer 2'):
                    skills.Place('remote', remote_drawer_search_region,
                                 above=.04, step=.005).run(ctx)
            except ExpertFailed as error:
                if not (error.reason == 'no placement' and error.details.get('last') == 'tracking'
                        and release_measured_remote(ctx)):
                    raise
        finally:
            s.placements, s.lower = ordinary_placements, ordinary_lower
        ctx.hold(1.)
        semantic.require(ctx, semantic.on(ctx, 'remote', 'board_eraser'),
                         'remote did not settle on its stack support', kind='check')
        for obj in ('board_eraser', 'tape', 'stapler', 'remote'):
            stored(ctx, obj, 2)
    finally:
        ctx.wall_deadline = previous


def contained(ctx, obj, container, part=None):
    """Judge actual immersion and internal support without an angle gate."""
    return semantic.storage_result(ctx, obj, container, part)


def stored(ctx, obj, number):
    if obj in STORAGE_OVERRIDES:
        container = STORAGE_OVERRIDES[obj]
        semantic.require(ctx, contained(ctx, obj, container),
                         'object not completely in its alternate container', kind='check', obj=obj, container=container)
        return
    number = DRAWER_OVERRIDES.get(obj, number)
    semantic.require(ctx, contained(ctx, obj, 'cabinet', drawer(number)),
                     'object not completely in its drawer', kind='check', obj=obj, drawer=number)


def drawer_front_target(ctx, entry):
    """Define a patch on actual low front-panel material beside the tape stack."""
    part = entry.child + '.front'
    low, high = ctx.geometry['cabinet'].bounds(part, frame=entry.child)
    offset = np.eye(4)
    offset[:3, 3] = [high[0] - .03, (low[1] + high[1]) / 2, low[2] + .02]
    size = (.02, float(high[1] - low[1]), .012)
    return part, Box(size), offset


def seat_top_drawer(ctx):
    """Physically reset accumulated drift before another drawer's stability check."""
    opening = _joints.opening(ctx, 'cabinet', joint(1))
    if opening <= _joints.CLOSED:
        return
    entry = _joints.slider(ctx, 'cabinet', joint(1))
    part, shape, offset = drawer_front_target(ctx, entry)
    ctx.note('RM001 top drawer reseat start', opening_m=opening, goal_m=_joints.CLOSED,
             arm='right', front_local_point=offset[:3, 3].tolist(), patch_size_m=list(shape.size),
             source='only reseat a drawer beyond the visible closed threshold')
    with ctx.subtask('physically seat the top drawer before opening another drawer'):
        with stability(ctx, 'cabinet', entry):
            skills.push(ctx, 'cabinet', part, arm='right', shape=shape, offset=offset,
                        body=entry.child, direction=-entry.direction(ctx.pose('cabinet', entry.child)),
                        distance=opening + semantic.PUSH_PAST)
        actual = _joints.opening(ctx, 'cabinet', joint(1))
        semantic.require(ctx, actual <= _joints.CLOSED, 'top drawer not seated before lower drawer operation', kind='check',
                         opening_m=actual)
        ctx.note('RM001 top drawer reseat complete', opening_m=actual)


def push_middle_drawer_front(ctx, entry):
    """Push actual low front-panel material beside the central tape stack."""
    part, shape, offset = drawer_front_target(ctx, entry)
    direction = -entry.direction(ctx.pose('cabinet', entry.child))
    distance = _joints.opening(ctx, 'cabinet', joint(2)) + semantic.PUSH_PAST
    low, high = ctx.geometry['cabinet'].bounds(part, frame=entry.child)
    ctx.note('RM001 middle drawer low front push', arm='right',
             part=part, front_local_point=offset[:3, 3].tolist(), patch_size_m=list(shape.size),
             right_edge_margin_m=float(high[0] - offset[0, 3]), height_above_front_bottom_m=.02)
    skills.push(ctx, 'cabinet', part, arm='right', shape=shape, offset=offset,
                body=entry.child, direction=direction, distance=distance)
    semantic.require(ctx, _joints.is_closed(ctx, 'cabinet', joint(2)),
                     'middle drawer not closed after low front push', kind='check')


def close(ctx, number):
    if not _joints.is_closed(ctx, 'cabinet', joint(number)):
        previous = getattr(ctx, 'wall_deadline', None)
        loaded_top = number == 1 and contained(ctx, 'pen_black', 'cabinet', drawer(1))
        if number == 2 or loaded_top:
            deadline = perf_counter() + 600.
            ctx.wall_deadline = min(previous, deadline) if previous is not None else deadline
        try:
            with ctx.subtask(f'close drawer {number}'):
                if number == 2:
                    ctx.note('RM001 middle drawer close approach', local_budget_s=600.,
                             brace=False, original_stability_monitor=True, active_push='low right front',
                             source='first-pass unbraced push moved housing only 1.2mm; remove old 5mm drift workaround')
                    drawer_approach(ctx)
                    entry = _joints.slider(ctx, 'cabinet', joint(number))
                    with stability(ctx, 'cabinet', entry):
                        push_middle_drawer_front(ctx, entry)
                elif loaded_top:
                    final_top = contained(ctx, 'eraser', 'cabinet', drawer(1))
                    ctx.note('RM001 loaded top drawer close', local_budget_s=600.,
                             brace=not final_top, original_stability_monitor=True,
                             keep_current_stance=final_top,
                             source='v2 s52 approach preparation let middle slide open; push directly after eraser release')
                    if not final_top:
                        drawer_approach(ctx)
                    entry = _joints.slider(ctx, 'cabinet', joint(number))
                    part, shape, offset = drawer_front_target(ctx, entry)
                    if final_top:
                        low, high = ctx.geometry['cabinet'].bounds(part, frame=entry.child)
                        offset[:3, 3] = [(low[0] + high[0]) / 2,
                                         (low[1] + high[1]) / 2,
                                         (low[2] + high[2]) / 2]
                    s = ctx.skills()
                    ordinary_target = s.target
                    def front_target(obj, part=None, shape=None, offset=None, body=None):
                        if obj == 'cabinet' and part == entry.handle:
                            patch_part, patch_shape, patch_offset = drawer_front_target(ctx, entry)
                            return ordinary_target(obj, patch_part, shape=patch_shape,
                                                   offset=patch_offset, body=entry.child)
                        return ordinary_target(obj, part, shape, offset, body)
                    ctx.note('RM001 loaded top front plan', active='either' if final_top else 'right',
                             support=None if final_top else 'left',
                             stance='current release pose' if final_top else 'cabinet relative front',
                             front_local_point=offset[:3, 3].tolist(),
                             patch_size_m=list(shape.size), original_stability_and_pressure=True,
                             source='two original handle searches exhausted preflight; use tested solid front push')
                    s.target = front_target
                    try:
                        with stability(ctx, 'cabinet', entry):
                            with (nullcontext() if final_top else _brace(ctx, 'cabinet', entry, 'right', relocate=False)):
                                skills.push(ctx, 'cabinet', part, arm=None if final_top else 'right',
                                            shape=shape, offset=offset,
                                            body=entry.child,
                                            direction=-entry.direction(ctx.pose('cabinet', entry.child)),
                                            distance=_joints.opening(ctx, 'cabinet', joint(number))
                                                     + semantic.PUSH_PAST)
                    finally:
                        s.target = ordinary_target
                    semantic.require(ctx, _joints.is_closed(ctx, 'cabinet', joint(number)),
                                     'loaded top drawer not closed after front push', kind='check')
                else:
                    semantic.close(ctx, 'cabinet', joint(number), brace=True)
        except ExpertFailed as error:
            details = error.details
            recover_middle = (
                loaded_top
                and error.reason == 'drawer housing stability exceeded (20 mm / 5 deg / other drawers 20 mm)'
                and details.get('translation_m', np.inf) <= .020
                and details.get('rotation_deg', np.inf) <= 5.
                and _joints.is_closed(ctx, 'cabinet', joint(1))
                and not _joints.is_closed(ctx, 'cabinet', joint(2))
                and _joints.is_closed(ctx, 'cabinet', joint(3)))
            if not recover_middle:
                raise
            ctx.note('RM001 recover middle drawer after completed top push',
                     original_failure=error.reason, original_details=details,
                     top_opening_m=_joints.opening(ctx, 'cabinet', joint(1)),
                     middle_opening_m=_joints.opening(ctx, 'cabinet', joint(2)),
                     recovery='original middle front push with its own full stability monitor')
            close(ctx, 2)
            for closed_number in (1, 2, 3):
                drawer_closed(ctx, closed_number)
        finally:
            ctx.wall_deadline = previous


def drawer_done(ctx, number):
    semantic.require(ctx, _joints.is_closed(ctx, 'cabinet', joint(number)),
                     'drawer not closed', kind='check', drawer=number)
    for obj in DRAWERS[number]:
        stored(ctx, obj, number)


def drawer_closed(ctx, number):
    semantic.require(ctx, _joints.is_closed(ctx, 'cabinet', joint(number)),
                     'drawer not closed', kind='check', drawer=number)


def principal_axis(ctx, obj, longest=True):
    points = ctx.geometry[obj].points()
    _, _, axes = np.linalg.svd(points - points.mean(0), full_matrices=False)
    return ctx.pose(obj)[:3, :3] @ axes[0 if longest else -1]


def standing(ctx, obj, container):
    """Require immersion and internal support; record tilt without rejecting it."""
    return semantic.storage_result(ctx, obj, container)


def pen_mouth_release(obj, container):
    """Release a pen above its original container, then accept its real support."""
    def where(ctx):
        shapes = _world.scene_shapes(ctx, exclude=(obj,))
        own = [shape for shape in shapes if shape.obj == container]
        region, blocked = _world.support_region(own, container, floor=True)
        semantic.require(ctx, region is not None, 'pen container floor unavailable', kind='planning',
                         obj=obj, container=container)
        frame, low, high = region
        inverse = np.linalg.inv(frame)
        points = np.vstack([shape.points for shape in own]) @ inverse[:3, :3].T + inverse[:3, 3]
        mouth = float(points[:, 2].max())
        frame = frame.copy()
        frame[:3, 3] += frame[:3, 2] * (mouth + .02)
        contents = [shape.points for shape in shapes if shape.obj != container
                    and (shape.points @ inverse[:3, :3].T + inverse[:3, 3])[:, 2].max() > mouth]
        # The release pose can be clear while a pen lands on another pen.
        # Reserve a free vertical column down to the actual floor, including
        # dividers, instead of treating an overhead release as insertion.
        columns = []
        for shape in shapes:
            if shape.obj == container and 'floor' in shape.part:
                continue
            local_points = shape.points @ inverse[:3, :3].T + inverse[:3, 3]
            if local_points[:, 2].max() <= 0.:
                continue
            columns.append((frame, local_points[:, :2].min(0),
                            local_points[:, :2].max(0)))
        ctx.note('RM001 pen release above original mouth', obj=obj, container=container,
                 above_mouth_m=.02 + PLACE_CLEARANCE,
                 free_drop_columns=len(columns), acceptance='storage_result after settling')
        return (frame, low, high), [*blocked, *contents, *columns]
    return where


def gel_cup_drop_region(ctx):
    """Release vertically above the cup, then check the settled original goal."""
    shapes = _world.scene_shapes(ctx, exclude=('gel_pen',))
    cup = [shape for shape in shapes if shape.obj == 'cup']
    region, blocked = _world.support_region(cup, 'cup', floor=True)
    semantic.require(ctx, region is not None, 'cup floor unavailable', kind='planning')
    frame, low, high = region
    inverse = np.linalg.inv(frame)
    points = np.vstack([shape.points for shape in cup]) @ inverse[:3, :3].T + inverse[:3, 3]
    mouth = float(points[:, 2].max())
    centre, half = (low + high) / 2, (high - low) / 4
    frame = frame.copy()
    frame[:3, 3] += frame[:3, 2] * (mouth + .08)
    contents = [shape.points for shape in shapes if shape.obj != 'cup'
                and (shape.points @ inverse[:3, :3].T + inverse[:3, 3])[:, 2].max() > mouth]
    ctx.note('RM001 gel upright release above cup', above_mouth_m=.08,
             floor_central_fraction=.5)
    return (frame, centre - half, centre + half), [*blocked, *contents]


def gel_end_grip(ctx):
    """Retry the measured cap hold after clearing the lower-drawer clutter."""
    target = ctx.skills().target('gel_pen', part='cap.shell')
    toward = ctx.pose('cabinet')[:3, 3] - ctx.skills().centre(target)
    axis = principal_axis(ctx, 'gel_pen')
    if axis @ toward < 0.:
        axis = -axis
    axis[2] = 0.
    axis /= np.linalg.norm(axis)
    approach = (axis + np.array([0., 0., -1.])) / np.sqrt(2.)
    ctx.note('RM001 gel fitted cap grip', approach=approach.tolist(),
             cone_deg=20., axis_toward='cabinet', geometry='cap.shell',
             declared='whole object')
    return dict(shape=target.shape, offset=target.offset,
                approach=approach, max_angle=np.radians(20.))




def insert_pen(ctx, obj, container):
    ctx._standing_at_pen_entry = {
        name: destination for destination, names in (
            ('holder', getattr(ctx.scene, 'SHORT_PENS', ())),
            ('cup', getattr(ctx.scene, 'LONG_PENS', ())))
        for name in names if name != obj and standing(ctx, name, destination)}
    if obj in FLAT_PEN_DRAWERS:
        number = FLAT_PEN_DRAWERS[obj]
        if contained(ctx, obj, 'cabinet', drawer(number)):
            return
        # The v2 prefix really closed the top drawer at s7; expose its
        # floor before searching a release instead of aiming at the slit.
        ctx.note('RM001 reopen flat-pen drawer', obj=obj, drawer=number,
                 opening_m=_joints.opening(ctx, 'cabinet', joint(number)))
        open_storage_drawer(ctx, number)
        target = ctx.skills().target(obj)
        skills.chain(ctx, [skills.Pick(obj, shape=target.shape, offset=target.offset,
                                      approach=(0., 0., -1.), max_angle=np.radians(30.),
                                      instruction=f'pick up the {obj}'),
                           skills.Place(obj, black_pen_drawer_drop_region,
                                        above=.04,
                                        instruction=f'put the {obj} flat in drawer {number}')],
                     adapt=('handover',))
        ctx.hold(1.)
        semantic.require(ctx, contained(ctx, obj, 'cabinet', drawer(number)),
                         'flat pen not completely in its drawer', kind='check', obj=obj, drawer=number)
        close(ctx, number)
        drawer_closed(ctx, number)
        return
    qualified = standing(ctx, obj, container)
    if obj in ('highlighter_pink', 'marker_black'):
        semantic.require(ctx, qualified, 'initially qualified pen was disturbed', kind='check',
                         obj=obj, container=container)
    if qualified:
        ctx.note('leave qualified standing pen in place', obj=obj, container=container)
        return
    pick_options = {}
    if obj == 'pen_white':
        travel_ready(ctx)
        centre = _world.centre(_world.object_points(ctx, obj))
        front = ctx.scene.TABLE['center'][1] - ctx.scene.TABLE['size'][1] / 2
        goal = np.array([centre[0] - .22, front - .31, np.pi / 2])
        semantic.require(ctx, checked_approach(ctx, goal, 'RM001 white pen approach'),
                         'no checked white pen relative stance', kind='check')
        target = ctx.skills().target(obj)
        pick_options = dict(arm='right', shape=target.shape, offset=target.offset,
                            approach=(0., 0., -1.), max_angle=np.radians(30.))
        ctx.note('RM001 white pen cup plan', cone_deg=30., object_left_m=.22,
                 table_front_back_m=.31, container=container)
        where = _regions.inside_region(obj, container, orientation='upright')
        place = skills.Place(obj, where, orientation='upright', turns=24)
        need = place.requirements(ctx)
        with ctx.subtask(f'pick up the {obj}'):
            skills.Pick(obj, **pick_options).run(ctx, need)
        with ctx.subtask(f'put away the {obj}'):
            place.run(ctx)
        return
    if obj == 'pencil':
        # The first-pass s9a03 succeeded near this object-relative stance;
        # the v2 entry exhausted three coupled stance searches with no IK.
        travel_ready(ctx)
        centre = _world.centre(_world.object_points(ctx, obj))
        front = ctx.scene.TABLE['center'][1] - ctx.scene.TABLE['size'][1] / 2
        s = ctx.skills()
        goals = [np.array([centre[0] - left, front - back, np.pi / 2])
                 for back in (.14, .22, .30) for left in (.22, .10, .34)]
        goal = next((pose for pose in goals
                     if s.base_route(s.base_pose(), pose) is not None), None)
        ctx.note('RM001 pencil relative stance',
                 candidates=[pose.tolist() for pose in goals],
                 chosen=None if goal is None else goal.tolist(),
                 source='front-edge route checked before grasp search')
        semantic.require(ctx, goal is not None and checked_approach(
            ctx, goal, 'RM001 pencil approach'), 'no checked pencil relative stance', kind='check')
        # The whole-object fit includes the cap clip and is wider than the
        # exposed shell. Fit the shell, while declaring the entire pencil.
        target = ctx.skills().target(obj, part='cap.shell')
        pick_options = dict(shape=target.shape, offset=target.offset,
                            approach=(0., 0., -1.), max_angle=np.radians(30.))
        ctx.note('RM001 pencil fitted cap grip', geometry='cap.shell',
                 declared='whole object', approach=[0., 0., -1.], cone_deg=30.)
    elif obj == 'highlighter_yellow':
        # First-pass s10a01 physically succeeded with the left hand, down
        # 45 degrees along the exposed pen axis toward its negative-X end.
        axis = principal_axis(ctx, obj)
        if axis[0] > 0.:
            axis = -axis
        approach = axis - np.array([0., 0., 1.])
        approach /= np.linalg.norm(approach)
        target = ctx.skills().target(obj)
        pick_options = dict(arm='left', shape=target.shape, offset=target.offset,
                            approach=approach, max_angle=np.radians(20.))
        ctx.note('RM001 yellow pen learned diagonal grip', arm='left',
                 approach=approach.tolist(), cone_deg=20.,
                 source='first-pass s10a01 physical success')
        with ctx.subtask(f'pick up the {obj}'):
            skills.Pick(obj, **pick_options).run(ctx)
        with ctx.subtask(f'put away the {obj}'):
            skills.Place(obj, pen_mouth_release(obj, container),
                         orientation='upright', turns=24).run(ctx)
        ctx.hold(1.)
        return
    elif obj in ('pen_black', 'marker_blue'):
        # Keep the full object; avoid enumerating each small authored part.
        # Its box axes leave no approach within five degrees of vertical.
        target = ctx.skills().target(obj)
        pick_options = dict(shape=target.shape, offset=target.offset,
                            approach=(0., 0., -1.), max_angle=np.radians(30.))
        ctx.note('RM001 whole pen overhead grip', obj=obj, cone_deg=30.)
    elif obj == 'gel_pen':
        # The overhead cone is blocked by books. The exposed end admits a
        # diagonal approach along the pen toward the cabinet side.
        pick_options = gel_end_grip(ctx)
        deadline = perf_counter() + 600.
        previous = getattr(ctx, 'wall_deadline', None)
        ctx.wall_deadline = min(previous, deadline) if previous is not None else deadline
        with ctx.subtask(f'pick up the {obj}'):
            skills.Pick(obj, **pick_options).run(ctx)
        with ctx.subtask(f'put away the {obj}'):
            skills.Place(obj, gel_cup_drop_region, orientation='upright').run(ctx)
        ctx.hold(1.)
        return
    transfer(ctx, obj, _regions.inside_region(obj, container, orientation='upright'),
             pick_options=pick_options, orientation='upright')


def pen_done(ctx, obj, container):
    if obj in FLAT_PEN_DRAWERS:
        number = FLAT_PEN_DRAWERS[obj]
        semantic.require(ctx, contained(ctx, obj, 'cabinet', drawer(number)),
                         'flat pen not completely in its drawer', kind='check', obj=obj, drawer=number)
    else:
        semantic.require(ctx, standing(ctx, obj, container), 'pen not standing in its holder', kind='check',
                         obj=obj, container=container)
    for name, destination in getattr(ctx, '_standing_at_pen_entry', {}).items():
        semantic.require(ctx, standing(ctx, name, destination),
                         'previously standing pen disturbed', kind='check', obj=name, container=destination)


def store_clip(ctx, obj):
    if contained(ctx, obj, 'jar'):
        return
    if obj == 'clip_large':
        # The first-pass s16a01 physically picked at this relative stance
        # with the right hand almost vertically down, then dropped into jar.
        travel_ready(ctx)
        centre = _world.centre(_world.object_points(ctx, obj))
        goal = np.r_[centre[:2] + [-.25, -.55], 2.]
        ctx.note('RM001 large clip learned stance', relative_xy_m=[-.25, -.55],
                 yaw_rad=2., source='first-pass s16a01 physical success')
        semantic.require(ctx, checked_approach(ctx, goal, 'RM001 large clip approach'),
                         'no checked large clip stance', kind='check')
        target = ctx.skills().target(obj)
        with ctx.subtask(f'pick up the {obj}'):
            skills.Pick(obj, arm='right', shape=target.shape, offset=target.offset,
                        approach=(0., 0., -1.), max_angle=np.radians(15.)).run(ctx)
        with ctx.subtask(f'put away the {obj}'):
            skills.Place(obj, clip_release_region(obj)).run(ctx)
        ctx.hold(1.)
        return
    transfer(ctx, obj, clip_release_region(obj))


def clip_release_region(obj):
    """Release loose clips above the live mouth, then let them settle inside."""
    def where(ctx):
        shapes = _world.scene_shapes(ctx, exclude=(obj,))
        jar = [shape for shape in shapes if _world.selects(shape, 'jar', None)]
        region, blocked = _world.support_region(jar, 'jar', floor=True)
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
    points = _world.object_points(ctx, obj)
    table = ctx.scene.TABLE
    centre = _world.centre(points)
    low = np.asarray(table['center']) - np.asarray(table['size']) / 2
    high = np.asarray(table['center']) + np.asarray(table['size']) / 2
    semantic.require(ctx, points[:, 2].min() >= table['height'] - .005
                     and np.all(centre[:2] >= low) and np.all(centre[:2] <= high),
                     'deferred book fell off the desk', kind='check', obj=obj)
    ctx.note('unresolved book remains at its initial position', obj=obj)


def destination(ctx, obj):
    return np.r_[np.asarray(ctx.scene.TABLE['center']) + DESTINATIONS[obj], ctx.scene.TABLE['height']]


def container_placement_goal(ctx, obj):
    near = destination(ctx, obj).copy()
    near[:2] += CONTAINER_OFFSETS.get(obj, (0., 0.))
    return near


def container_goal_workspace(ctx):
    """Reserve the final footprints while the basket is held at a staging pose."""
    regions = list(workspace(ctx))
    table_height = ctx.scene.TABLE['height']
    for obj in ('cabinet', 'jar', 'holder', 'cup'):
        body = ctx.geometry[obj].bodies[0]
        _, low, high = _world.body_box(ctx, obj, body, CLEAR)
        goal = np.eye(4)
        goal[:3, 3] = destination(ctx, obj)
        # Layout 5 destinations use zero heading; keep the local body bounds
        # conservative while projecting them into the final world frame.
        goal[2, 3] = table_height
        regions.append((goal, low, high))
    return regions


def at_destination(ctx, obj):
    return (np.linalg.norm(ctx.pose(obj)[:2, 3] - destination(ctx, obj)[:2]) <= .020
            and (obj == 'jar' or abs(wrapped(np.arctan2(ctx.pose(obj)[1, 0], ctx.pose(obj)[0, 0])
                            - TARGET_YAWS.get(obj, 0.))) <= np.radians(5.)))


def table_push_approach(ctx, obj, goal):
    """Go round the table to face the supported object's pushing side."""
    s = ctx.skills()
    pose = ctx.pose(obj)
    delta = np.asarray(goal) - pose[:2, 3]
    # Sideways pushes are reached from the clear rear row. A northward
    # correction needs the opposite side; southward pushes need the rear.
    north = not (abs(delta[1]) > abs(delta[0]) and delta[1] > 0.)
    if obj in ('holder', 'cup') and abs(wrapped(np.arctan2(pose[1, 0], pose[0, 0]))) > np.radians(5.):
        # square_on_table corrects heading with its +/-Y off-centre push
        # before translation. Stand on that push's entering side.
        north = delta[1] < -.020
    sign = 1. if north else -1.
    sideways_jar = obj == 'jar' and abs(delta[0]) > abs(delta[1])
    stand_off = .65 if sideways_jar else .80
    target = np.r_[pose[:2, 3] + [0., sign * stand_off], -sign * np.pi/2]
    start = s.base_pose()
    if (np.linalg.norm(start[:2] - target[:2]) <= (.02 if sideways_jar else .12)
            and abs(wrapped(start[2]-target[2])) <= .1):
        return
    travel_ready(ctx)
    if start[1] * sign < .55:
        edge = sign * .95
        retreat = start.copy()
        retreat[:2] -= .25 * np.array([np.cos(start[2]), np.sin(start[2])])
        route = [retreat, np.array([-1.05, retreat[1], retreat[2]]),
                 np.array([-1.05, edge, 0.]), target]
    else:
        route = [target]
        if s.base_route(start, target) is None:
            # A short sideways correction otherwise turns the long base
            # into the table edge. Back out before turning, then approach
            # the same reachable contact stance; check every segment.
            route = None
            for distance in (.08, .12, .18, .25):
                retreat = start.copy()
                retreat[:2] -= distance * np.array([np.cos(start[2]), np.sin(start[2])])
                outside = target.copy()
                outside[:2] -= distance * np.array([np.cos(target[2]), np.sin(target[2])])
                candidate = [retreat, outside, target]
                if all(s.base_route(a, b) is not None
                       for a, b in zip([start, *candidate], candidate)):
                    route = candidate
                    ctx.note('RM001 table push retreat before lateral correction', obj=obj,
                             retreat_m=distance, same_contact_stance=True)
                    break
            semantic.require(ctx, route is not None, 'no checked retreat route for table pushing side', kind='planning', obj=obj)
    ctx.note('RM001 table push approach', obj=obj, side='north' if north else 'south',
             waypoints=[point.tolist() for point in route], relative_distance_m=stand_off)
    for point in route:
        reached = checked_approach(ctx, point, 'RM001 container push approach',
                                   tolerance=(.020, np.radians(5.)))
        if not reached:
            reached = checked_approach(ctx, point, 'RM001 container push final correction',
                                       tolerance=(.020, np.radians(5.)))
        semantic.require(ctx, reached,
                         'no checked route to table pushing side', kind='check', obj=obj)




def steep_jar_sideways(ctx, goal, check):
    """Push the round jar with downward RM001 fingers, clear of the cabinet."""
    from table_1000.experts.skills.push.squaring import push_contact, _payload_contacts
    from table_1000.experts.skills.common import CANDIDATE_PLANS, IK_ATTEMPTS, RETREATS
    from table_1000.experts.skills.push.motion import PUSH_GAP
    from table_1000.experts.solver import PlanningFailed
    from table_1000.experts.skills.hold.frames import translated
    s = ctx.skills()
    semantic.require(ctx, not s.held, 'table squaring requires empty hands', kind='planning', obj='jar')
    body = ctx.geometry['jar'].bodies[0]
    ctx.note('RM001 steep jar horizontal push plan', tilt_deg=[80., 90.],
             max_push_m=.100, fallback_push_m=[.050, .025], max_rounds=40, position_tolerance_m=.020,
             contact_above_measured_bottom_m=.065,
             original_motion_solver=True, external_collisions_preserved=True)
    for iteration in range(41):
        check(ctx)
        delta = np.asarray(goal) - ctx.pose('jar')[:2, 3]
        error = float(np.linalg.norm(delta))
        ctx.note('table squaring measured', obj='jar', round=iteration,
                 position_m=error, yaw_rad=0.)
        points = _world.object_points(ctx, 'jar')
        semantic.require(ctx, abs(points[:, 2].min()-ctx.scene.TABLE['height']) <= .01,
                         'table squaring needs table support', kind='check', obj='jar')
        if error <= .020:
            return
        semantic.require(ctx, iteration < 40, 'table squaring did not converge', kind='check', obj='jar')
        axis = int(np.argmax(np.abs(delta)))
        direction = np.eye(3)[axis] * np.sign(delta[axis])
        distance = min(.100, float(np.max(np.abs(delta))))
        face = np.r_[push_contact(points, direction[:2]), points[:, 2].min()+.065]
        before = ctx.pose('jar').copy()
        pushed = False
        with _payload_contacts(s, 'jar', body):
            for arm in sorted(s.arms, key=lambda arm: np.linalg.norm((s.shoulder(arm)-face)[:2])):
                solver = s.solvers[arm]
                for tilt in np.radians([80., 90.]):
                    approach = direction*np.cos(tilt)-np.array([0., 0., np.sin(tilt)])
                    closing = np.cross([0., 0., 1.], direction)
                    contact = np.eye(4)
                    contact[:3, :3] = np.c_[approach, closing, np.cross(approach, closing)]
                    contact[:3, 3] = face-approach*s.hands[arm].tip()
                    ctx.sync_model()
                    solver.sync()
                    start = s.model.vector(ctx.spec.arms[arm].finger_targets(0.))
                    status, q_pre = s.model.ik(arm, translated(contact, -direction*PUSH_GAP),
                                             start, world=True, lift=True, attempts=IK_ATTEMPTS)
                    if status != 'Success':
                        continue
                    status, _ = s.model.ik(arm, translated(contact, direction*distance), q_pre,
                                          world=True, lift=True, attempts=IK_ATTEMPTS,
                                          check_collision=False)
                    while status != 'Success' and distance > .025:
                        distance = max(.025, distance/2)
                        status, _ = s.model.ik(arm, translated(contact, direction*distance), q_pre,
                                              world=True, lift=True, attempts=IK_ATTEMPTS,
                                              check_collision=False)
                    if status != 'Success':
                        continue
                    touched = [name for name, part in solver.world.objects.items()
                               if part.body == 'jar:'+body]
                    try:
                        solver.close_gripper()
                        solver.move_to_pose_with_RRTConnect(None, lift=True, goals=[q_pre],
                                                           attempts=CANDIDATE_PLANS)
                        solver.allow(touched)
                        solver.move_to_pose_with_screw(contact)
                        s.move_by_displacement(arm, direction*distance, drag=('jar', body))
                        pushed = True
                        ctx.note('RM001 steep jar push executed', arm=arm, tilt_deg=float(np.degrees(tilt)),
                                 direction=direction.tolist(), distance_m=distance)
                    except PlanningFailed as failure:
                        ctx.note('RM001 steep jar push rejected', arm=arm, reason=failure.reason,
                                 details=failure.details)
                        distance = max(.025, distance/2)
                    finally:
                        solver.allow(touched, False)
                    # A rejected path can still have moved the jar: remeasure
                    # before any new candidate instead of pushing a stale face.
                    if pushed or np.linalg.norm(ctx.pose('jar')[:2, 3]-before[:2, 3]) >= .0002:
                        break
                if pushed or np.linalg.norm(ctx.pose('jar')[:2, 3]-before[:2, 3]) >= .0002:
                    break
        semantic.require(ctx, pushed or np.linalg.norm(ctx.pose('jar')[:2, 3]-before[:2, 3]) >= .0002,
                         'no collision-checked steep jar push', kind='check', obj='jar')
        tcp = solver.tcp()
        try:
            solver.first_reachable([lambda pose=translated(tcp, move): solver.move_to_pose_with_screw(pose)
                                    for move in [-direction*d for d in RETREATS]
                                    + [np.array([0., 0., d]) for d in RETREATS]], lambda move: move())
        except PlanningFailed as failure:
            ctx.note('RM001 steep jar no retreat', reason=failure.reason)
        check(ctx)
        semantic.require(ctx, np.linalg.norm(ctx.pose('jar')[:2, 3]-before[:2, 3]) >= .0002,
                         'table squaring push stalled', kind='check', obj='jar')


def push_loaded_container(ctx, obj, goal, yaw):
    """Push with shared rider ownership and live physical storage checks."""
    from table_1000.experts.skills.push.squaring import square_on_table
    payload = ((*ctx.scene.CLIPS, 'magnet_red') if obj == 'jar' else
               tuple(ctx.scene.SHORT_PENS if obj == 'holder' else ctx.scene.LONG_PENS))
    payload = tuple(item for item in payload if item not in FLAT_PEN_DRAWERS)
    samples = [0]
    def guard(live):
        remaining_wall(live)
        for item in payload:
            semantic.require(live, semantic.storage_result(live, item, obj),
                             'table push lost container payload', kind='check', obj=obj, payload=item)
        samples[0] += 1
    def check(live):
        guard(live)
        delta = np.linalg.norm(live.pose(obj)[:2, 3] - goal)
        angle = 0. if yaw is None else abs(wrapped(yaw-np.arctan2(live.pose(obj)[1,0],live.pose(obj)[0,0])))
        if delta > .020 or angle > np.radians(5.):
            # Nothing is pushed or carried while the empty hands travel.
            # Check the full storage result at both ends; keep the scene's
            # continuous fall observer and every real push's payload observer.
            live.observers.remove(guard)
            try:
                table_push_approach(live, obj, goal)
            finally:
                live.observers.append(guard)
            guard(live)
            live.note('RM001 empty-hand approach payload rechecked', obj=obj,
                      full_storage_before_and_after=True, actual_push_guard_every_frame=True)
    guard(ctx)
    ctx.observers.append(guard)
    try:
        delta = np.asarray(goal)-ctx.pose(obj)[:2, 3]
        if obj == 'jar' and abs(delta[0]) > abs(delta[1]):
            steep_jar_sideways(ctx, goal, check)
        else:
            square_on_table(ctx, obj, goal, yaw, max_rounds=40, max_push=.025, check=check)
        all_on_desk(ctx)
    finally:
        ctx.observers.remove(guard)
        ctx.note('RM001 loaded container table guard', obj=obj, samples=samples[0])


def align_loaded_jar_whole(ctx, where, near):
    """RM001 whole exterior grip after the ordinary need search found none."""
    from table_1000.experts.skills.hold.targets import rim_sections
    previous = getattr(ctx, 'wall_deadline', None)
    deadline = perf_counter() + 600.
    ctx.wall_deadline = min(previous, deadline) if previous is not None else deadline
    s = ctx.skills()
    original = s.target
    try:
        travel_ready(ctx)
        chosen = None
        for stand_off in (.60, .65, .70, .75, .80, .85, .90):
            goal = np.r_[ctx.pose('jar')[:2, 3] + [-.15, -stand_off], np.pi/2]
            if s.base_route(s.base_pose(), goal) is not None:
                chosen = goal
                break
        semantic.require(ctx, chosen is not None, 'no checked jar-relative front stance', kind='planning')
        semantic.require(ctx, checked_approach(ctx, chosen, 'RM001 whole jar front approach'),
                         'jar-relative front stance changed', kind='check')
        whole = original('jar')
        rim_sections(s, 'jar', whole, ctx.geometry['jar'])
        semantic.require(ctx, whole.container, 'whole jar grip requires original open-container geometry', kind='planning')
        def declared(obj, part=None, shape=None, offset=None, body=None):
            target = original(obj, part, shape, offset, body)
            if obj == 'jar' and shape is whole.shape:
                target.container = True
            return target
        s.target = declared
        ctx.note('RM001 whole jar independent plan', obj='jar', front_distance_m=stand_off,
                 lateral_offset_m=-.15,
                 checked_stance=chosen.tolist(), grip='whole original fitted exterior', cone_deg=15.,
                 independent_pick=True, initial_lift_m=.06,
                 original_container_declaration=True, local_budget_s=600.)
        ordinary_ik = s.model.ik
        sampled = [0]
        def measured_ik(arm, pose, *args, **kwargs):
            remaining_wall(ctx)
            result = ordinary_ik(arm, pose, *args, **kwargs)
            if sampled[0] < 32:
                ctx.note('RM001 jar IK sample', arm=arm, status=result[0],
                         lift=kwargs.get('lift'), check_collision=kwargs.get('check_collision', True),
                         pose=np.asarray(pose).tolist(), shoulder=s.shoulder(arm).tolist(),
                         base=s.base_pose().tolist())
                sampled[0] += 1
            remaining_wall(ctx)
            return result
        s.model.ik = measured_ik
        try:
            with ctx.subtask('pick up the complete loaded jar from its exterior'):
                skills.Pick('jar', shape=whole.shape, offset=whole.offset,
                            approach=(0., 0., -1.), max_angle=np.radians(15.),
                            height=.06).run(ctx, base=False)
        finally:
            s.model.ik = ordinary_ik
        with ctx.subtask('place the loaded jar at the authorized near-layout target'):
            skills.Place('jar', where, near=near).run(ctx)
    finally:
        s.target = original
        ctx.wall_deadline = previous


def align_container(ctx, obj):
    if obj == 'cabinet':
        close_drifted_drawers(ctx)
    near = container_placement_goal(ctx, obj)
    if at_destination(ctx, obj) and (obj not in CONTAINER_OFFSETS
            or np.linalg.norm(ctx.pose(obj)[:2, 3] - near[:2]) <= .01):
        if obj == 'basket':
            ctx.note('RM001 authorized basket staging retained', obj=obj,
                     target_xy_m=list(DESTINATIONS[obj]), target_yaw_rad=TARGET_YAWS[obj],
                     authorization='msg_7e4512db2e6b', original_layout5=False)
        return
    if obj == 'cup':
        return push_loaded_container(ctx, obj, destination(ctx, obj)[:2], 0.)
    if obj == 'holder':
        return transfer_loaded_container(ctx, obj)
    if obj in ('jar', 'holder', 'cup'):
        travel_ready(ctx)
        if obj == 'jar' and ctx.pose('cup')[1, 3] < .09:
            buffer_goal = ctx.pose('cup')[:2, 3].copy()
            buffer_goal[1] += .08
            ctx.note('RM001 cup before jar table buffer', goal_xy_m=buffer_goal.tolist(),
                     reason='cup and marker block the actual east-side jar pre-push contact')
            push_loaded_container(ctx, 'cup', buffer_goal, None)
        if obj == 'jar' and ctx.pose('holder')[1, 3] < .11:
            # The holder is the actual exterior blocker in all six planned
            # exterior grips. Its measured north sweep leaves a supported
            # front lane for the round jar, clear of the untouched books.
            buffer_goal = ctx.pose('holder')[:2, 3].copy()
            buffer_goal[1] += .08
            ctx.note('RM001 holder before jar table buffer', goal_xy_m=buffer_goal.tolist(),
                     reason='external highlighter blocks jar approach; holder blocks leftward path')
            push_loaded_container(ctx, 'holder', buffer_goal, None)
        if obj == 'jar' and ctx.pose('jar')[0, 3] > -.4:
            transit = ctx.pose('jar')[:2, 3].copy()
            transit[1] = -.01
            push_loaded_container(ctx, 'jar', transit, None)
            transit[0] = destination(ctx, 'jar')[0]
            push_loaded_container(ctx, 'jar', transit, None)
        if obj in ('holder', 'cup') and ctx.pose(obj)[1, 3] > destination(ctx, obj)[1] + .020:
            # The actual north buffers are beside the staged basket. Leave
            # that row before translating left; do not drag through it.
            transit = ctx.pose(obj)[:2, 3].copy()
            transit[1] = destination(ctx, obj)[1]
            ctx.note('RM001 small container south before lateral', obj=obj,
                     goal_xy_m=transit.tolist(), reason='leave the staged basket footprint before leftward motion')
            push_loaded_container(ctx, obj, transit, None)
        return push_loaded_container(ctx, obj, destination(ctx, obj)[:2], None if obj == 'jar' else 0.)
    where = ({'jar': jar_destination_region, 'basket': basket_destination_region}[obj]
             if obj in ('jar', 'basket') else
             (lambda ctx: destination_region(ctx, obj)) if obj in ('cup', 'holder') else
             _regions.on_region(obj, 'table', near=near))
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
        heading = TARGET_YAWS.get(obj, 0.)
        if obj in ('jar', 'basket'):
            aligned[:3, :3] = [[np.cos(heading), -np.sin(heading), 0.],
                               [np.sin(heading), np.cos(heading), 0.], [0., 0., 1.]]
        return [(pose, spot) for pose, spot in ordinary(target, aligned, *args, **kwargs)
                if np.linalg.norm(pose[:2, 3] - destination(ctx, obj)[:2]) <= POSITION_TOLERANCE
                and abs(wrapped(np.arctan2(pose[1, 0], pose[0, 0]) - heading)) <= np.radians(10.)]
    s.placements = layout_placements
    try:
        if obj == 'jar':
            return align_loaded_jar_whole(ctx, where, near)
        return transfer(ctx, obj, where, near=near, **({'step': .01} if obj in CONTAINER_OFFSETS else {}))
    finally:
        s.placements = ordinary


def transfer_loaded_container(ctx, obj):
    """Pick and replace the whole loaded body using the x1pro finish method."""
    s = ctx.skills()
    ordinary_pick, ordinary_placements = s.pick, s.placements

    def container_pick(targets, *args, **options):
        group = targets if isinstance(targets, (tuple, list)) else [targets]
        for target in group:
            target.container = True
        return ordinary_pick(group, *args, **options)

    def aligned_placements(target, current, *args, **kwargs):
        seed = current.copy()
        seed[:3, :3] = np.eye(3)
        return [(pose, spot) for pose, spot in ordinary_placements(target, seed, *args, **kwargs)
                if np.linalg.norm(pose[:2, 3]-destination(ctx, obj)[:2]) <= .020
                and abs(wrapped(np.arctan2(pose[1, 0], pose[0, 0]))) <= np.radians(5.)]

    def where(live):
        contents = set()
        for held in live.skills().held.values():
            if held['target'].obj == obj:
                contents.update(name.split(':', 1)[0] for name in held.get('riders', ()))
        shapes = _world.scene_shapes(live, exclude=(obj, *sorted(contents)))
        region, blocked = _world.support_region(shapes, 'table', headroom=_regions.headroom(live, obj))
        semantic.require(live, region is not None, 'no support', kind='planning', surface='table')
        frame, low, high = _regions.around(_regions.inset(region, _regions.EDGE), destination(live, obj))
        inverse = np.linalg.inv(frame)
        points = live.geometry[obj].points() + destination(live, obj)
        local = points @ inverse[:3, :3].T + inverse[:3, 3]
        margin = .020 + PLACE_MARGIN
        live.note('RM001 container release excludes recorded riders', obj=obj,
                  contents=sorted(contents), external_collisions_preserved=True)
        return (frame, np.maximum(low, local[:, :2].min(0)-margin),
                np.minimum(high, local[:, :2].max(0)+margin)), world_keepouts(blocked)

    def attempt():
        semantic.require(ctx, not s.held, 'loaded container entry needs empty hands', kind='planning')
        start = np.asarray(ctx.env.qpos(ctx.spec.lift)[0], dtype=float)
        target = np.full_like(start, .70)
        if np.min(start) < .69:
            for arm in s.arms:
                joints = np.repeat(s.model.q[s.model.arm_indices(arm)][None], 2, axis=0)
                _, pairs = s.model.check_path(arm, joints, .01, np.vstack([start, target]))
                semantic.require(ctx, not pairs, 'empty torso rise blocked before container approach', kind='planning',
                                 arm=arm, collisions=pairs)
            count = max(1, int(np.ceil(np.max(abs(target-start))/ctx.spec.lift_velocity_limit/DT)))
            for k in range(1, count+1):
                ctx.step({'lift': (start+(target-start)*k/count)[None]})
            ctx.hold(.3)
            for solver in s.solvers.values():
                solver.sync()
        s.ready()
        table = ctx.scene.TABLE
        goal = np.array([ctx.pose(obj)[0, 3],
                         table['center'][1]-table['size'][1]/2-.20, np.pi/2])
        start_base = s.base_pose().copy()
        route = [goal]
        if s.base_route(start_base, goal) is None:
            route = None
            for distance in (.08, .12, .18, .25, .35):
                retreat, outside = start_base.copy(), goal.copy()
                retreat[:2] -= distance*np.array([np.cos(start_base[2]), np.sin(start_base[2])])
                outside[:2] -= distance*np.array([np.cos(goal[2]), np.sin(goal[2])])
                candidate = [retreat, outside, goal]
                if all(s.base_route(a, b) is not None
                       for a, b in zip([start_base, *candidate], candidate)):
                    route = candidate
                    ctx.note('RM001 loaded container stance retreat route', obj=obj,
                             retreat_m=distance, waypoints=[p.tolist() for p in route])
                    break
        semantic.require(ctx, route is not None, 'no checked loaded container stance route', kind='planning', obj=obj)
        reached = all(checked_approach(ctx, point, 'RM001 loaded container stance',
                                      tolerance=(.020, np.radians(5.))) for point in route)
        actual = s.base_pose().copy()
        ctx.note('RM001 container pre-grasp stance verified', obj=obj, target=goal.tolist(),
                 actual=actual.tolist(), measured_lift=np.asarray(ctx.env.qpos(ctx.spec.lift)[0]).tolist())
        semantic.require(ctx, reached, 'container grasp stance not reached', kind='check', obj=obj)
        # RM001's wider opening can oppose the complete holder walls. A
        # two-millimetre single-wall pinch slid during the measured lift.
        skills.pick(ctx, obj, body=ctx.geometry[obj].bodies[0], base=False,
                    approach=(0., 0., -1.), max_angle=np.radians(45.))
        return skills.Place(obj, where, step=.005,
                            near=destination(ctx, obj)).run(ctx)

    s.pick = container_pick
    s.placements = aligned_placements
    try:
        return skills.retry(ctx, attempt)
    finally:
        s.pick, s.placements = ordinary_pick, ordinary_placements


def jar_destination_region(ctx):
    return container_destination_region(ctx, 'jar')


def basket_destination_region(ctx):
    return container_destination_region(ctx, 'basket')


def container_destination_region(ctx, obj):
    """Bound final placement by the layout footprint and its acceptance tolerance."""
    near = destination(ctx, obj)
    region, blocked = _regions.on_region(obj, 'table', near=near)(ctx)
    frame, low, high = region[:3]
    inverse = np.linalg.inv(frame)
    heading = TARGET_YAWS.get(obj, 0.)
    rotation = np.array([[np.cos(heading), -np.sin(heading), 0.],
                         [np.sin(heading), np.cos(heading), 0.], [0., 0., 1.]])
    points = ctx.geometry[obj].points() @ rotation.T + near
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


def basket_goal_workspace_clear(ctx):
    points = _world.object_points(ctx, 'basket')
    for frame, low, high in container_goal_workspace(ctx):
        inverse = np.linalg.inv(frame)
        local = (points @ inverse[:3, :3].T + inverse[:3, 3])[:, :2]
        if overlaps(local[ConvexHull(local).vertices], np.asarray(low)[:2], np.asarray(high)[:2]):
            return False
    return True


@contextmanager
def basket_push_payload(ctx):
    """Observe actual cargo; common Motion owns all planning declarations."""
    payload = tuple(obj for obj, container in STORAGE_OVERRIDES.items() if container == 'basket')
    samples = [0]
    def guard(live):
        remaining_wall(live)
        table = live.scene.TABLE
        points = _world.object_points(live, 'basket')
        low = np.asarray(table['center']) - np.asarray(table['size']) / 2
        high = np.asarray(table['center']) + np.asarray(table['size']) / 2
        semantic.require(live, np.all(points[:, :2].min(0) >= low)
                         and np.all(points[:, :2].max(0) <= high),
                         'loaded basket overhangs the table during push', kind='check')
        for obj in payload:
            semantic.require(live, semantic.storage_result(live, obj, 'basket'),
                             'basket payload left during table push', kind='check', obj=obj)
        samples[0] += 1
    guard(ctx)
    ctx.observers.append(guard)
    try:
        yield
    finally:
        ctx.observers.remove(guard)
        ctx.note('RM001 basket push payload frame guard', samples=samples[0])


def basket_front_material_patch(ctx, part):
    """A real small wall patch, with the world push line through basket x."""
    geometry = ctx.geometry['basket']
    body = geometry.bodies[0]
    pieces = [data @ transform[:3, :3].T + transform[:3, 3]
              for owner, name, transform, kind, data in geometry.shapes
              if owner == body and name == part and kind == 'convex']
    semantic.require(ctx, bool(pieces), 'basket push part has no convex material', kind='planning', part=part)
    points = max(pieces, key=lambda points: np.prod(np.ptp(points, axis=0)))
    planes = ConvexHull(points).equations
    pose = ctx.pose('basket', body)
    centre = (points.min(0) + points.max(0)) / 2
    semantic.require(ctx, abs(pose[0, 0]) > .5, 'basket front patch needs a front-facing long wall', kind='planning')
    for _ in range(3):
        centre[0] = (ctx.pose('basket')[0, 3] - pose[0, 3]
                     - pose[0, 1]*centre[1] - pose[0, 2]*centre[2]) / pose[0, 0]
        sides = planes[:, 1]
        limits = -(planes[:, 0]*centre[0] + planes[:, 2]*centre[2] + planes[:, 3])
        lower = np.max(limits[sides < -1e-8] / sides[sides < -1e-8])
        upper = np.min(limits[sides > 1e-8] / sides[sides > 1e-8])
        centre[1] = (lower + upper) / 2
    shape = Box((.020, .0004, .010))
    vertices = shape.points() + centre
    slack = -(vertices @ planes[:, :3].T + planes[:, 3]).max()
    semantic.require(ctx, slack >= -1e-9, 'basket push patch leaves actual wall material', kind='planning',
                     part=part, convex_slack_m=float(slack))
    offset = np.eye(4)
    offset[:3, 3] = centre
    ctx.note('RM001 basket real wall push patch', part=part, body=body,
             centre_body_m=centre.tolist(), size_m=list(shape.size),
             convex_slack_m=float(slack), force_line_world_x_m=float((pose @ np.r_[centre, 1.])[0]))
    return shape, offset, body


def buffer_basket_for_containers(ctx):
    """Keep a clear loaded basket still, else slide it north on the table."""
    table_support(ctx, 'basket')
    ctx.note('RM001 authorized container targets', common_shift_m=list(REVISED_TARGET_SHIFT),
             targets={obj: list(point) for obj, point in DESTINATIONS.items()},
             target_yaws=TARGET_YAWS, authorization='msg_7e4512db2e6b',
             reference='a08 actual fourth-push sample1768; 18.148356mm minimum, 1mm planning grid')
    if basket_goal_workspace_clear(ctx):
        ctx.note('RM001 basket staging already clear', moved=False,
                 original_reserved_footprints_clear=True)
        return
    previous = getattr(ctx, 'wall_deadline', None)
    deadline = perf_counter() + 600.
    ctx.wall_deadline = min(previous, deadline) if previous is not None else deadline
    try:
        travel_ready(ctx)
        s, table = ctx.skills(), ctx.scene.TABLE
        start = s.base_pose().copy()
        selected = None
        for stand_off in (.90, .95, 1.00):
            goal = np.array([ctx.pose('basket')[0, 3], table['center'][1] - stand_off, np.pi / 2])
            retreat = start.copy()
            retreat[:2] -= .20 * np.array([np.cos(start[2]), np.sin(start[2])])
            outside = goal.copy()
            outside[1] -= .20
            route = next((route for route in ([goal], [retreat, outside, goal])
                          if all(s.base_route(a, b) is not None
                                 for a, b in zip([start, *route], route))), None)
            if route is not None:
                selected = route
                break
        semantic.require(ctx, selected is not None, 'no checked front route for basket table push', kind='planning')
        ctx.note('RM001 basket north table push plan', stance_kind='basket x; table front relative',
                 front_distance_m=stand_off, heading_rad=np.pi/2,
                 waypoints=[p.tolist() for p in selected], step_distance_m=.08,
                 max_pushes=4, local_budget_s=600.,
                 source='current basket overlaps original jar/holder goals; preserve contents on table')
        for goal in selected:
            semantic.require(ctx, checked_approach(ctx, goal, 'RM001 basket table push approach'),
                             'basket table push route changed', kind='check')
        geometry = ctx.geometry['basket']
        with basket_push_payload(ctx):
            before = ctx.pose('basket')[:3, 3].copy()
            for index in range(4):
                if basket_goal_workspace_clear(ctx):
                    break
                parts = [name for _, name in geometry.parts() if 'long_wall' in name]
                part = min(parts, key=lambda name: np.mean(
                    geometry.points(name) @ ctx.pose('basket')[:3, :3].T + ctx.pose('basket')[:3, 3], axis=0)[1])
                # Native push reports its TCP command, not the achieved basket
                # travel. Close the loop on the real footprint after every push.
                points = _world.object_points(ctx, 'basket')
                distance = min(.08, table['center'][1] + table['size'][1] / 2
                               - .01 - float(points[:, 1].max()))
                semantic.require(ctx, distance > 0., 'no north table reserve for another basket push', kind='planning')
                ctx.note('RM001 basket table push', index=index+1, part=part,
                         direction=[0., 1., 0.], distance_m=distance, no_lift=True)
                with ctx.subtask('slide the loaded basket north to clear final container footprints'):
                    shape, offset, body = basket_front_material_patch(ctx, part)
                    skills.push(ctx, 'basket', part, direction=(0., 1., 0.), distance=distance,
                                shape=shape, offset=offset, body=body)
                table_support(ctx, 'basket')
                for obj in ('glue_stick', 'screwdriver', 'glue_bottle'):
                    semantic.require(ctx, contained(ctx, obj, 'basket'),
                                     'basket payload left during table push', kind='check', obj=obj)
                all_on_desk(ctx)
            ctx.note('RM001 basket staging table displacement',
                     delta_m=(ctx.pose('basket')[:3, 3]-before).tolist(), no_lift=True,
                     reserved_footprints_clear=basket_goal_workspace_clear(ctx))
            basket_staged(ctx)
    finally:
        ctx.wall_deadline = previous


def basket_staged(ctx):
    buffered(ctx, 'basket', keep_out=container_goal_workspace(ctx))


def destination_region(ctx, obj):
    """Constrain placement search to the footprint at the layout-5 destination."""
    near = destination(ctx, obj)
    region, blocked = _regions.on_region(obj, 'table', near=near)(ctx)
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
    bottom = float(_world.object_points(ctx, obj)[:, 2].min())
    semantic.require(ctx, abs(bottom - ctx.scene.TABLE['height']) <= .01,
                     'container not resting on the table', kind='check', obj=obj)


def aligned(ctx, obj):
    table_support(ctx, obj)
    semantic.require(ctx, at_destination(ctx, obj), 'container away from layout 5 destination', kind='check', obj=obj)


def container_done_without_books(ctx, obj):
    aligned(ctx, obj)
    if obj in ('holder', 'cup'):
        for pen in (ctx.scene.SHORT_PENS if obj == 'holder' else ctx.scene.LONG_PENS):
            pen_done(ctx, pen, obj)
    elif obj == 'jar':
        for clip in ctx.scene.CLIPS:
            clip_done(ctx, clip)
        semantic.require(ctx, contained(ctx, 'magnet_red', 'jar'), 'red magnet left the jar', kind='check')
    elif obj == 'cabinet':
        for number in (1, 2, 3):
            drawer_done(ctx, number)


def close_drifted_drawers(ctx):
    """Physically close visible drift before checking the cabinet's final result."""
    for number in (1, 2, 3):
        if not _joints.is_closed(ctx, 'cabinet', joint(number)):
            ctx.note('RM001 recover drawer drift before final cabinet check', drawer=number,
                     opening_m=_joints.opening(ctx, 'cabinet', joint(number)),
                     closed_tolerance_m=_joints.CLOSED)
            close(ctx, number)
    for number in (1, 2, 3):
        drawer_closed(ctx, number)


def settle_final(ctx):
    ctx.hold(1.)
    close_drifted_drawers(ctx)


def finish(ctx):
    """Accept the non-book #95 chain; book placement belongs to issue #106."""
    if _skipped(ctx):
        pending = {item: value['reason'] for item, value in _skipped(ctx).items()}
        ctx.note('unfinished items', items=pending)
        semantic.require(ctx, False, 'skipped items remain incomplete', kind='check', items=pending)
    for number in (1, 2, 3):
        drawer_done(ctx, number)
    for obj in DESTINATIONS:
        container_done_without_books(ctx, obj)
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
        points = _world.object_points(ctx, obj)
        semantic.require(ctx, points[:, 2].min() >= table['height'] - .005
                         and np.all(points[:, :2].min(0) >= low)
                         and np.all(points[:, :2].max(0) <= high), 'object fell or flew off the desk', kind='check', obj=obj)
        if obj not in DESTINATIONS and obj not in UNRESOLVED_BOOKS:
            semantic.require(ctx, not semantic.on(ctx, obj, 'table'),
                             'loose object left in the front work area', kind='check', obj=obj)
    bounds = {obj: (_world.object_points(ctx, obj).min(0),
                    _world.object_points(ctx, obj).max(0)) for obj in NEAT_ROW}
    rear = [bounds[obj][1][1] for obj in NEAT_ROW]
    gaps = [bounds[right][0][0] - bounds[left][1][0]
            for left, right in zip(NEAT_ROW, NEAT_ROW[1:])]
    ctx.note('RM001 neat row measured', rear_edges_m=rear, adjacent_gaps_m=gaps,
             fixed_goals=DESTINATIONS, retained_goals=RETAINED_CONTAINER_GOALS)
    semantic.require(ctx, max(rear)-min(rear) <= .020
                     and all(abs(gap-.040) <= .020 for gap in gaps),
                     'container row or consistent gaps not aligned', kind='check')


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
    STEPS += [optional_step(obj, (f'C: put {obj} flat in drawer {FLAT_PEN_DRAWERS[obj]}'
                                 if obj in FLAT_PEN_DRAWERS else
                                 f'C: stand {obj} in the {container}'),
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
        STEPS += [optional_step('glue_bottle-side', 'B: put glue_bottle in the basket',
                               rest_bottle_on_side, bottle_side_done, obj='glue_bottle'),
                  optional_step('glue_bottle-stable', 'B: check basket bottle stability',
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

# Known front deadlocks get one different strategy: place them on a table patch,
# then make the same final drawer placement. This is not repeated search.
STEPS += [optional_step('drawer3-rescue-open', 'B: open drawer 3 for deadlock rescue',
                       partial(open_storage_drawer, number=3), partial(drawer_opened, number=3))]
for obj in ('dish', 'charger'):
    STEPS += [optional_step(obj, ('B: put deferred dish in drawer 3 after drawer 2 clearing'
                                if obj == 'dish' else f'B: clear and put {obj} in drawer 3'),
                           partial(buffer_then_store, obj=obj, number=3),
                           partial(stored, obj=obj, number=3), obj=obj, phased=True, retries=0)]
STEPS += [optional_step('drawer3-final', 'B: close drawer 3 after deadlock rescue',
                       partial(close, number=3), partial(drawer_done, number=3), retries=0)]

# The remote rests on A5. Leave that support in place and pick the remote itself.
STEPS += [optional_step('drawer2-tail-open', 'B: open drawer 2 for remote and safety rescue',
                        partial(open_storage_drawer, number=2), partial(drawer_opened, number=2)),
          optional_step('remote', 'A: put remote directly in drawer 2',
                       remote_to_drawer, partial(stored, obj='remote', number=2),
                       obj='remote')]
STEPS += [optional_step(obj, f'B: rescue failed {obj} from its safety table patch into drawer 2',
                       partial(store, obj=obj, number=2, open_if_needed=False),
                       partial(stored, obj=obj, number=2), obj=obj)
          for obj in ('tape', 'board_eraser')]
STEPS += [optional_step('drawer2-final', 'B: close drawer 2 after remote',
                       partial(close, number=2), partial(drawer_done, number=2), retries=0)]
# E: stage the basket away from every final container goal, then move the
# remaining containers into layout 5 before returning the basket to its goal.
def defer_eraser(ctx):
    """Keep this blocked object for the less cluttered entry before E."""
    ctx.note('RM001 deferred eraser', object_complete=False,
             retry_before='E: temporarily stage basket clear of final container goals',
             reason='original grasp and push window exceeded forty minutes')
    ctx.hold(.2)


STEPS[2] = Step('A: defer eraser until its loose neighbors are stored',
                defer_eraser, all_on_desk)
# This intermediate close does not claim the deferred eraser is stored.
STEPS[6] = Step(STEPS[6].name, partial(close, number=1),
                partial(drawer_closed, number=1))
STEPS += [optional_step('eraser-tail', 'A: put deferred eraser in drawer 1 before container moves',
                        partial(store, obj='eraser', number=1),
                        partial(stored, obj='eraser', number=1), obj='eraser'),
          optional_step('drawer1-tail', 'A: close and check drawer 1 after deferred eraser',
                        partial(close, number=1), partial(drawer_done, number=1))]
STEPS += [optional_step('basket-staging', 'E: temporarily stage basket clear of final container goals',
                       buffer_basket_for_containers, basket_staged, obj='basket')]
STEPS += [optional_step(obj, f'E: align the {obj}', partial(align_container, obj=obj),
                       partial(container_done_without_books, obj=obj), obj=obj)
          for obj in ('cabinet', 'jar', 'holder', 'cup')]
STEPS += [optional_step('basket', 'E: align the basket after container goals are clear',
                       partial(align_container, obj='basket'),
                       partial(container_done_without_books, obj='basket'), obj='basket')]
STEPS += [Step('F: settle and check every object', settle_final, finish)]



def defer_dish_until_drawer2(ctx):
    """Leave the validated buffer untouched until the existing rescue entry."""
    ctx.note('RM001 deferred dish', object_complete=False,
             retry_at='B: put deferred dish in drawer 3 after drawer 2 clearing',
             reason='drawer research window ended; basket approach also stopped tracking')
    ctx.hold(.2)


dish_index = next(i for i, step in enumerate(STEPS)
                  if step.name == 'B: put dish directly in drawer 3')
STEPS[dish_index] = Step('B: defer dish until drawer 2 loose contents are cleared',
                         defer_dish_until_drawer2, all_on_desk)


def working_edge(ctx):
    """Approach the working edge with the nearest collision-free folded pose."""
    initial = ctx.scene.ROBOTS[ROBOT]
    yaw = initial['yaw']
    s = ctx.skills()
    travel_ready(ctx)
    poses, actual = ctx.poses(), s.model.base.copy()
    selected = None
    try:
        for distance in (.5, .45, .4, .35):
            goal = np.r_[np.asarray(initial['position'])[:2]
                         + distance * np.array([np.cos(yaw), np.sin(yaw)]), yaw]
            s.put_base(goal, poses)
            contacts = s.touching(s.model.q)
            s.model.base = actual.copy()
            for solver in s.solvers.values():
                solver.sync()
            route = s.base_route(s.base_pose(), goal)
            ctx.note('RM001 working edge candidate', relative_forward_m=distance,
                     goal=goal.tolist(), collisions=contacts, route_clear=route is not None)
            print('RM001_WORK_EDGE', distance, contacts, route is not None, flush=True)
            if route is not None:
                selected = goal
                break
    finally:
        s.model.base = actual
        for solver in s.solvers.values():
            solver.sync()
    semantic.require(ctx, selected is not None, 'no checked working edge approach', kind='planning')
    semantic.require(ctx, checked_approach(ctx, selected, 'RM001 working edge'),
                     'working edge route changed', kind='check')


def rm_step(index, step):
    """Bound the whole step and check all scene objects at every checkpoint."""
    def run(ctx):
        previous = getattr(ctx, 'wall_deadline', None)
        deadline = perf_counter() + ITEM_WALL_BUDGET
        ctx.wall_deadline = min(previous, deadline) if previous is not None else deadline
        print('RM001_STEP_START', index, step.name, flush=True)
        original_note = ctx.note
        def progress(message, **details):
            original_note(message, **details)
            if message.startswith(('RM001', 'grasped', 'placed', 'base choice',
                                   'brace approach', 'cabinet braced', 'need unmet',
                                   'missed grasp', 'next placement', 'next push')):
                concise = {key: details[key] for key in ('obj', 'arm', 'reason', 'drawer',
                           'opening_m', 'chosen', 'checked', 'cone_deg', 'target',
                           'finger_gap', 'width', 'held_mm2', 'stage', 'spot') if key in details}
                print('RM001_PROGRESS', index, message, concise, flush=True)
        ctx.note = progress
        try:
            remaining_wall(ctx)
            if index > 55:
                ctx.note('RM001 authorized container targets',
                         targets={obj: list(point) for obj, point in DESTINATIONS.items()},
                         target_yaws=TARGET_YAWS, retained_goals=RETAINED_CONTAINER_GOALS,
                         authorization='msg_0a363761c4b7 / msg_909cb025e037',
                         reference='fixed s54 anchors; common rear edge and 40mm small-container gaps')
            if index == 1:
                working_edge(ctx)
            step.run(ctx)
            remaining_wall(ctx)
        finally:
            ctx.note = original_note
            ctx.wall_deadline = previous
    def check(ctx):
        step.check(ctx)
        semantic.require(ctx, not ctx.skills().held
                         and not any(s.attachments for s in ctx.solvers.values()),
                         'item checkpoint requires empty hands', kind='check', item=step.name)
        all_on_desk(ctx)
    return Step(step.name, run, check)


STEPS = [rm_step(index, step) for index, step in enumerate(STEPS, 1)]


def solve(ctx):
    continuous_no_fall(ctx)
    ctx.observers.append(continuous_no_fall)
    try:
        run_steps(ctx, STEPS)
    finally:
        for event in getattr(ctx, '_observed_falls', {}).values():
            ctx.note('continuous scene fall minimum; checkpoint still requires recovery', **event)
        ctx.observers.remove(continuous_no_fall)
