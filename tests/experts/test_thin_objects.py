from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from table_1000.experts.skills.hold.frames import Box
from table_1000.experts.context import DT
from table_1000.experts.skills.hold.thin import ThinPicking
from table_1000.experts.skills.common import Target
from table_1000.experts.skills.hold.pick import Grasping
from table_1000.experts.skills.hold.sections import classify, exposed_section, landing, supported_overhangs, swept_obstacle, table_edges


@pytest.fixture
def hand():
    # A 30 mm-wide pad with 1 mm surface sampling and a 2 mm tip margin.
    x, z = np.meshgrid(np.arange(-.02, .0201, .001), np.arange(-.015, .0151, .001))
    points = np.c_[x.ravel(), np.zeros(x.size), z.ravel()]
    return SimpleNamespace(min_pad_area=15e-6, pad_spacing=.001, hold_margin=.002, max_gap=.2,
                           tip=lambda gap: .02,
                           inner=lambda gap: [(sign, points + [0., sign * gap / 2, 0.],
                                                np.ones(len(points), dtype=bool)) for sign in (-1, 1)])


@pytest.fixture
def flat_thin(hand):
    class Skill(ThinPicking, Grasping):
        pass

    skill = Skill()
    points = Box((.08, .06, .001)).points()
    target = Target('sheet', ['sheet:body'], 'sheet:body', Box((.08, .06, .001)), np.eye(4), points)
    pose = np.eye(4)
    pose[:3, 3] = [.5, 0., .0005]
    geometry = SimpleNamespace(shapes=[('body', 'body', np.eye(4), 'convex', points)],
                               points=lambda part=None, frame=None: points, parts=lambda: [('body', 'body')],
                               solid=lambda part, **kw: True)
    skill.arms, skill.held, skill.hands = ['right'], {}, {'right': hand}
    skill.ctx = SimpleNamespace(
        geometry={'sheet': geometry}, spec=SimpleNamespace(name='test'),
        scene=SimpleNamespace(TABLE={'center': (.5, 0.), 'size': (1., 1.), 'height': 0.}),
        poses=lambda: {'sheet:body': pose.copy()}, base_transform=lambda: np.eye(4),
        env=SimpleNamespace(objects={}, center_of_mass=lambda body: [pose[:3, 3]]),
        hold=lambda seconds: None, skills=lambda: skill)
    skill.target = lambda *args, **kwargs: target
    skill.solvers = {'right': SimpleNamespace(finger_gap=lambda: .001, tcp=lambda: np.eye(4), attachments={})}
    skill.move_by_displacement = lambda arm, command: pose.__setitem__((2, 3), pose[2, 3] + command[2])
    skill.release = lambda arm: None
    return skill, target, pose


@pytest.mark.parametrize('options', [
    {'point': np.zeros(3)}, {'approach': [0., 0., -1.]}, {'max_angle': 0.},
    {'max_angle': [0., np.pi]},
])
def test_thin_pick_rejects_unsupported_arguments_before_preparation(flat_thin, options):
    from table_1000.experts import skills
    skill, _, _ = flat_thin
    with pytest.raises(ValueError, match='thin-object pick route.*point, approach or max_angle.*remove these arguments'):
        skills.pick(skill.ctx, 'sheet', **options)


def test_thin_pick_defaults_still_use_preparation_and_lift(flat_thin):
    from table_1000.experts import skills
    skill, target, _ = flat_thin
    calls = []
    def grasp_thin(chosen, **options):
        skill.held['right'] = {'target': chosen}
        calls.append((chosen, options))
        return chosen, {'arm': 'right'}
    skill.grasp_thin = grasp_thin
    assert skills.pick(skill.ctx, 'sheet', point=None, approach=None, max_angle=None)['rise'] == pytest.approx(.1)
    assert calls == [(target, {'arm': None, 'height': .1, 'need': None, 'base': True})]


def test_ordinary_pick_keeps_custom_arguments(flat_thin):
    skill, target, _ = flat_thin
    skill.thin_geometry = lambda target: None
    calls = []
    def grasp(*args, **kwargs):
        skill.held['right'] = {'target': target}
        calls.append((args, kwargs))
        return {'arm': 'right'}
    skill.grasp = grasp
    point, approach, angle = [1., 2., 3.], [0., 0., -1.], .4
    assert skill.pick(target, point=point, approach=approach, max_angle=angle)['rise'] == pytest.approx(.1)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:2] == (target, None)
    assert np.array_equal(args[2], point) and np.array_equal(args[3], approach)
    assert kwargs == {'then': (0., 0., .1), 'max_angle': angle, 'need': None, 'base': True}


def test_thin_candidate_hand_preserves_container_opening_margin(flat_thin):
    skill, target, _ = flat_thin
    skill.hands['right'].opening_clearance = .01
    target.container = True
    assert skill.candidate_hand(target, 'right').opening_clearance == .002
    skill._fingertip_candidates = True
    tip = skill.candidate_hand(target, 'right')
    assert tip.opening_clearance == .002 and tip.hold_margin == 0.
    assert skill.hands['right'].opening_clearance == .01


def test_sheet_preserves_reachable_edge_route_before_fingertip(flat_thin):
    skill, target, _ = flat_thin
    skill.thin_geometry = lambda target: SimpleNamespace(kind='sheet', thickness=.0045,
        contact_height=.0047, contact_band=.0047)
    skill.solvers['right'].world = SimpleNamespace(objects={
        'sheet:base': SimpleNamespace(body=target.body),
        'sheet:cap': SimpleNamespace(body=target.body),
        'other:body': SimpleNamespace(body='other:body')})
    skill.ctx.note = lambda *args, **kwargs: None
    result = target, {'arm': 'right'}
    skill.grasp_at_edge = lambda *args: result
    skill.pinch_thin = lambda *args: pytest.fail('the existing edge route already succeeded')
    assert skill.grasp_thin(target) is result


@pytest.mark.parametrize('kind,reason,fallback', [
    ('sheet', 'thin preparation search exhausted', True),
    ('sheet', 'physical pressure failed', False),
    ('plate', 'thin preparation search exhausted', False),
    ('filament', 'thin preparation search exhausted', False),
])
def test_only_sheet_budget_exhaustion_falls_back_without_restarting_edge(flat_thin, kind, reason, fallback):
    from table_1000.experts.context import ExpertFailed, Unreachable
    skill, target, _ = flat_thin
    skill.thin_geometry = lambda target: SimpleNamespace(kind=kind, thickness=.0045,
        contact_height=.0047, contact_band=.0047)
    skill.solvers['right'].world = SimpleNamespace(objects={})
    skill.ctx.note = lambda *args, **kwargs: None
    error = ExpertFailed(None, reason, candidates=960, limit=960)
    need, calls = object(), []
    def edge(*args):
        calls.append('edge')
        raise error
    skill.grasp_at_edge = edge
    def pinch(*args):
        calls.append('pinch')
        if kind == 'filament':
            raise Unreachable(None, 'no fingertip grasp')
        assert args[1:] == ('right', .2, need, False, True)
        return 'pinched'
    skill.pinch_thin = pinch
    if fallback:
        assert skill.grasp_thin(target, 'right', .2, need, False, True) == 'pinched'
        assert calls == ['edge', 'pinch']
    else:
        with pytest.raises(ExpertFailed) as raised:
            skill.grasp_thin(target, 'right', .2, need, False, True)
        assert raised.value is error
        assert calls == (['pinch', 'edge'] if kind == 'filament' else ['edge'])


def test_blocked_plate_reports_edge_and_objects_to_clear(flat_thin):
    from table_1000.experts.context import Unreachable
    skill, target, _ = flat_thin
    skill.thin_geometry = lambda target: SimpleNamespace(kind='plate', thickness=.02,
        contact_height=.02, contact_band=.025)
    skill.solvers['right'].world = SimpleNamespace(objects={})
    skill.ctx.note = lambda *args, **kwargs: None
    def fail(reason, kind=RuntimeError, **details):
        raise kind(None, reason, **details)
    skill.ctx.fail = fail
    def blocked(*args):
        raise Unreachable(None, 'no reachable pressure drag', rejected=[
            {'stage': 'target drag sweep', 'obstacle': 'other:cover',
             'edge': [0., -.5, 0.], 'arm': 'right'},
            {'stage': 'target drag sweep', 'obstacle': 'other:spine',
             'edge': [0., -.5, 0.], 'arm': 'left'}])
    skill.grasp_at_edge = blocked
    with pytest.raises(Unreachable, match='clear the obstacles.*retry pick') as error:
        skill.grasp_thin(target)
    assert error.value.details['edges'] == [{'edge': [0., -.5, 0.], 'clear_objects': ['other']}]


@pytest.mark.parametrize('rejected', [{'grasp IK': 4, 'need': 2}, ['holder:wall']])
def test_plate_edge_keeps_grasp_requirement_failure(flat_thin, rejected):
    from table_1000.experts.skills.common import Unmet
    skill, target, _ = flat_thin
    skill.thin_geometry = lambda target: SimpleNamespace(kind='plate', thickness=.02,
        contact_height=.02, contact_band=.025)
    skill.solvers['right'].world = SimpleNamespace(objects={})
    skill.ctx.note = lambda *args, **kwargs: None
    failure = Unmet(None, 'no reachable grasp', rejected=rejected, blocked_by=['holder:wall'])
    def blocked(*args):
        raise failure
    skill.grasp_at_edge = blocked
    with pytest.raises(Unmet) as error:
        skill.grasp_thin(target)
    assert error.value is failure


def test_prepared_plate_area_uses_opposing_local_faces_with_tip_margin(flat_thin):
    from table_1000.experts.skills.hold.frames import _held_area
    from table_1000.experts.skills.hold.sections import ExposedSection
    skill, target, _ = flat_thin
    faces = [Box((.03, .002, .02)).points() + [0., sign * .009, 0.]
             for sign in (-1., 1.)]
    # A thicker, remote spine must not demand its width under these pads.
    faces.append(Box((.002, .024, .02)).points() + [.04, 0., 0.])
    target.shape = ExposedSection(np.vstack(faces), faces)
    skill.thin_geometry = lambda target: SimpleNamespace(kind='plate')
    assert _held_area(target.shape, np.eye(4), np.eye(4), skill.hands['right'], .02) == 0.
    assert skill.measured_held_area(target, 'right', np.eye(4), np.eye(4), .02) > 5e-6
    remote = np.eye(4)
    remote[2, 3] = .1
    assert skill.measured_held_area(target, 'right', np.eye(4), remote, .02) == 0.


def test_plate_preserves_lookahead_grasp_rejection(flat_thin):
    from table_1000.experts.skills.common import Unmet
    skill, target, _ = flat_thin
    skill.thin_geometry = lambda target: SimpleNamespace(kind='plate', thickness=.02,
        contact_height=.02, contact_band=.025)
    skill.solvers['right'].world = SimpleNamespace(objects={})
    skill.ctx.note = lambda *args, **kwargs: None
    failure = Unmet(None, 'no reachable grasp', rejected={'grasp IK': 0, 'need': 3})
    def rejected(*args):
        raise failure
    skill.grasp_at_edge = rejected
    with pytest.raises(Unmet) as error:
        skill.grasp_thin(target)
    assert error.value is failure


def test_grasp_rejects_unprepared_flat_thin_target(flat_thin):
    from table_1000.experts import skills
    skill, target, _ = flat_thin
    assert skill.unprepared_flat_thin(target)
    with pytest.raises(ValueError, match='flat thin object on the table; use pick, or call slide_to_edge before grasp'):
        skills.grasp(skill.ctx, 'sheet')
    with pytest.raises(ValueError, match='use pick'):
        skill.grasp([target])


@pytest.mark.parametrize('state', ['overhanging', 'held', 'raised', 'upright', 'ordinary', 'thin_fingertip'])
def test_grasp_allows_prepared_held_and_ordinary_targets(flat_thin, state):
    skill, target, pose = flat_thin
    if state == 'overhanging':
        pose[0, 3] = .02  # 20 mm beyond the table, with the COM still supported.
    elif state == 'held':
        skill.held['left'] = {'target': target}
    elif state == 'raised':
        pose[2, 3] += .1
    elif state == 'upright':
        pose[:3, :3] = Rotation.from_euler('y', np.pi / 2).as_matrix()
        pose[2, 3] = .04
    elif state == 'ordinary':
        skill.thin_geometry = lambda target: None
    def reach(*args):
        raise RuntimeError('ordinary grasp search reached')
    skill.reach = reach
    options = {'close': skill.close_thin} if state == 'thin_fingertip' else {}
    with pytest.raises(RuntimeError, match='ordinary grasp search reached'):
        skill.grasp(target, arm='right' if state == 'held' else None, **options)


def test_geometry_classification_is_rotation_invariant_and_rejects_thick_body(hand):
    turn = Rotation.from_euler('xyz', [.2, .8, 1.2]).as_matrix()
    for size, expected in [((.035, .012, .001), 'sheet'),
                           ((.07, .0036, .0036), 'filament'),
                           ((.08, .06, .012), None)]:
        points = Box(size).points() @ turn.T
        result = classify(points, [points], hand)
        assert (result.kind if result else None) == expected
    # Two tilted thin patches form a bent sheet, not a 5 mm solid block.
    patch = Box((.02, .012, .001)).points()
    other = patch @ Rotation.from_euler('y', .2).as_matrix().T + [.02, 0., .002]
    assert classify(np.vstack([patch, other]), [patch, other], hand).kind == 'sheet'


def test_landing_prefers_near_edge_and_preserves_support_and_corners():
    table = {'center': (.69, 0.), 'size': (.8, 1.2), 'height': .755}
    edge = table_edges(table, (0., 0., 0.))[0]
    assert edge.point[0] == pytest.approx(.29)
    points = Box((.08, .06, .001)).points() + [.5, -.15, .7555]
    shift = landing(points, [.5, -.15, .7555], edge, .01)
    assert edge.distances(points + shift).max() == pytest.approx(.01)
    assert edge.distances(np.array([.5, -.15, .7555]) + shift) < 0
    assert landing(points, [.5, -.15, .7555], edge, .05) is None
    wide = Box((.08, 1.3, .001)).points() + [.5, 0., .7555]
    assert landing(wide, [.5, 0., .7555], edge, .01) is None


def test_landing_searches_along_side_edge_and_clamps_footprint_at_corner():
    table = {'center': (.69, 0.), 'size': (.8, 1.2), 'height': .755}
    edge = next(edge for edge in table_edges(table, (0., 0., 0.)) if edge.outward[1] == -1.)
    center = np.array([.7, -.15, .7555])
    points = Box((.08, .06, .001)).points() + center
    shoulder = np.array([.1, -.25, 1.])
    along = (shoulder - edge.point) @ edge.tangent
    shift = landing(points, center, edge, .01, along)
    # The nearest reachable end of the side edge is limited by the actual
    # footprint, rather than retaining the object's distant initial X.
    assert (points + shift)[:, 0].min() == pytest.approx(.29)
    assert edge.distances(points + shift).max() == pytest.approx(.01)
    assert edge.distances(center + shift) < 0.
    along = (np.array([.5, 0., 0.]) - edge.point) @ edge.tangent
    shift = landing(points, center, edge, .01, along)
    assert (center + shift)[0] == pytest.approx(.5)


def test_exposed_patch_uses_horizontal_approach_and_actual_clipped_cross_section():
    table = {'center': (.5, 0.), 'size': (1., 1.), 'height': 0.}
    edge = table_edges(table, (-1., 0., 0.))[0]
    # Rotated square crossing the edge: its exposed end is a triangular prism.
    points = Box((.04, .04, .001)).points() @ Rotation.from_euler('z', np.pi / 4).as_matrix().T
    points += [.02, 0., .0005]
    section, frame = exposed_section(points, edge)
    assert np.allclose(frame[:3, 0], [1., 0., 0.])
    assert np.allclose(frame[:3, 1], [0., 0., 1.])
    assert np.linalg.det(frame[:3, :3]) == pytest.approx(1.)
    world = section.points() @ frame[:3, :3].T + frame[:3, 3]
    assert (edge.distances(world) >= -1e-10).all()
    closing, width, _, _ = next(section.closings(.1))
    assert np.allclose(frame[:3, :3] @ closing, [0., 0., 1.])
    assert width == pytest.approx(.001)
    # A corner inside its bounding box but outside the triangle has no material.
    origin = section.points().max(0) - [.0001, .0005, .0001]
    origin[0] = section.points()[:, 0].min() + .0001
    low, high = section.interval(origin[None], [0., 1., 0.])
    assert low[0] > high[0]
    # Asset-grasp reports serialize vars(shape) after the queries have run.
    import json
    assert json.loads(json.dumps(vars(section))) == {
        'vertices': section.vertices, 'size': section.size, 'pieces': section.pieces}


def test_supported_search_covers_short_final_interval_without_crossing_com():
    depths = supported_overhangs(.0022, .00628, .002)
    # This valid window falls between the last regular grid point and the COM.
    assert any(.00625 < depth < .00628 for depth in depths)
    assert (depths < .00628).all()
    assert not len(supported_overhangs(.0063, .00628, .002))


def test_reobserved_search_revalidates_previous_patch_before_smaller_slivers():
    original = supported_overhangs(.0045, .035, .002)
    depths = supported_overhangs(.0045, .035, .002, preferred=.025)
    assert depths[0] == min(original, key=lambda value: abs(value - .025))
    assert set(original) == set(depths)
    assert (depths < .035).all()
    assert np.all(np.diff(abs(depths - .025)) >= 0.)
    # A changed supported interval clips the hint rather than skipping support.
    clipped = supported_overhangs(.0045, .02, .002, preferred=.025)
    assert clipped[0] < .02 and clipped[0] == max(clipped)


def test_exposed_candidate_positions_include_solid_fragments_across_an_empty_center():
    table = {'center': (.5, 0.), 'size': (1., 1.), 'height': 0.}
    edge = table_edges(table, (-1., 0., 0.))[0]
    pieces = [Box((.015, .006, .001)).points() + [-.01, y, .0005] for y in (-.015, .015)]
    section, _ = exposed_section(np.vstack(pieces), edge, pieces)
    closing, _, _, slides = next(section.closings(.1))
    assert np.array_equal(slides[0], [0., 0., 0.])
    low, high = section.interval(np.asarray(slides), closing)
    assert low[0] > high[0]  # the original short-box sample misses material
    assert (high[-2:] > low[-2:]).all()
    assert np.array_equal(np.sign(np.asarray(slides)[-2:, 2]), [1., -1.])


def test_exposed_ray_cache_tracks_input_values_and_does_not_alias_results():
    from table_1000.experts.skills.hold.checks import CollisionSurface
    from table_1000.experts.skills.hold.sections import ExposedSection
    points = Box((.02, .001, .02)).points()
    section = ExposedSection(points)
    reference = CollisionSurface([(np.eye(4), 'convex', points)])
    origins, direction = np.array([[0., 0., 0.], [.03, 0., 0.]]), np.array([0., 1., 0.])
    expected = reference.interval(origins, direction)
    low, high = section.interval(origins, direction)
    low[:] = 42.
    high[:] = -42.
    actual = section.interval(origins, direction)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    # Reusing an array after moving a ray must query its new value.
    origins[0, 0] = .03
    for a, b in zip(section.interval(origins, direction), reference.interval(origins, direction)):
        np.testing.assert_array_equal(a, b)


def test_exposed_closure_cache_shares_tip_geometry_but_keeps_area_checks(hand, monkeypatch):
    import copy
    import table_1000.experts.skills.hold.frames as grasps
    from table_1000.experts.skills.hold.sections import ExposedSection, _CLOSURES
    _CLOSURES.clear()
    hand._contact_geometry_key = object()
    section = ExposedSection(Box((.02, .001, .02)).points())
    transform, tcp = np.eye(4), np.eye(4)
    uncached, calls = grasps._close_uncached, []
    def record(*args, **kwargs):
        calls.append(1)
        return uncached(*args, **kwargs)
    monkeypatch.setattr(grasps, '_close_uncached', record)
    expected = uncached(section, transform, tcp, hand, .03)
    first = grasps._close(section, transform, tcp, hand, .03)
    assert first is not None
    first[1][:] = 42.
    tip = copy.copy(hand)
    tip.hold_margin = 0.
    actual = grasps._close(section, transform, tcp, tip, .03)
    assert len(calls) == 1  # Changing tip permission does not change closure geometry.
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    tip.min_pad_area = 1.
    assert grasps._close(section, transform, tcp, tip, .03) is None
    assert len(calls) == 2
    # Measured-contact checks deliberately omit the sampled area condition.
    assert grasps._close(section, transform, tcp, tip, .03, check_area=False) is not None
    assert len(calls) == 3
    moved = tcp.copy()
    moved[0, 3] = .05
    assert grasps._close(section, transform, moved, hand, .03) is None
    assert len(calls) == 4
    # Upstream candidate contact layers and measured closures are different queries.
    layered = grasps._close(section, transform, tcp, hand, .03, contact_layer=True)
    reference = uncached(section, transform, tcp, hand, .03, contact_layer=True)
    assert len(calls) == 5
    np.testing.assert_array_equal(layered[1], reference[1])
    grasps._close(section, transform, tcp, hand, .03, contact_layer=True)
    assert len(calls) == 5


def test_translation_sweep_checks_middle_without_bridging_empty_space():
    from table_1000.experts.skills.hold.sections import swept_obstacles
    from table_1000.planning.world import box_object
    transform = np.eye(4)
    transform[0, 3] = .5
    obstacle, pose = box_object('middle', (.1, .1, .1), transform)
    pieces = [Box((.1, .1, .1)).points()]
    assert swept_obstacle(pieces, np.array([1., 0., 0.]), [('middle', obstacle.fcl(pose))]) == 'middle'
    assert swept_obstacles(pieces, np.array([1., 0., 0.]),
        [('first', obstacle.fcl(pose)), ('second', obstacle.fcl(pose))]) == ['first', 'second']
    # The obstacle lies between two separately swept pieces, not in material.
    pieces = [piece + [0., offset, 0.] for piece in pieces for offset in (-.2, .2)]
    assert swept_obstacle(pieces, np.array([1., 0., 0.]), [('middle', obstacle.fcl(pose))]) is None


def test_thin_close_does_not_finish_on_speculative_pad_contact():
    import torch
    data = SimpleNamespace(joint_stiffness=torch.tensor([[50.]]), joint_damping=torch.tensor([[20.]]),
                           joint_effort_limits=torch.tensor([[100.]]))
    side = SimpleNamespace(fingers=['finger'], finger_targets=lambda gap: {'finger': gap})
    calls, events = [], []
    skill = ThinPicking()
    skill.ctx = SimpleNamespace(
        env=SimpleNamespace(robot=SimpleNamespace(joint_names=['finger'], data=data),
                            qpos=lambda names: np.array([[.03]])),
        spec=SimpleNamespace(arms={'arm': side}), step=calls.append,
        note=lambda message, **values: events.append(values))
    skill.solvers = {'arm': SimpleNamespace(finger_gap=lambda: .001)}
    # The ordinary closure diagnostic accepts early PhysX contact generation;
    # actual pad surfaces first touch after 0.8 s of this closure.
    skill.check_closure = lambda *args: {
        'held': True,
        'pad_contacts': {finger: [{'separation_m': .0004 if len(calls) * DT < .8 else -.00001}]
                         for finger in ('left', 'right')}}
    skill.close_thin(None, 'arm', None)
    assert len(calls) * DT == pytest.approx(.8)
    assert events[-1]['held']


def test_closed_loop_drag_reobserves_until_overhang_and_bounds_a_slipping_object():
    from table_1000.experts.context import ExpertFailed
    calls = []
    skill = ThinPicking()
    def fail(reason, **details):
        raise ExpertFailed(None, reason, **details)
    skill.arms, skill.held = ['right'], {}
    skill.ctx = SimpleNamespace(note=lambda *args, **kwargs: None, fail=fail,
        scene=SimpleNamespace(TABLE={'center': (0., 0.), 'size': (1., 1.), 'height': 0.}),
        base_transform=lambda: np.eye(4))
    skill._edge_grasp_available = lambda target, prepared, height: prepared['overhang_m'] >= .01
    searches = []
    def segment(*args, **kwargs):
        calls.append(args[-1])
        searches.append(kwargs['search'])
        return {'edge': None, 'required_overhang_m': .002, 'planned_overhang_m': .02,
                # The second segment passes the area-only bound but still
                # lacks a real table-clear grasp. The third has one even
                # before the old planned depth because its pose changed.
                'overhang_m': .01 if len(calls) >= 3 else .004,
                'remaining_command_m': .04, 'budget_span_m': .02}
    skill._slide_segment = segment
    assert len(skill.slide_to_edge(None)['segments']) == 3
    assert calls[0] is None and all(call is not None for call in calls[1:])
    assert all(search is searches[0] for search in searches)
    calls.clear()
    original = segment
    def slipping(*args, **kwargs):
        return {**original(*args, **kwargs), 'overhang_m': 0.}
    skill._slide_segment = slipping
    with pytest.raises(ExpertFailed, match='insufficient thin object overhang'):
        skill.slide_to_edge(None)
    assert len(calls) == 7  # initial travel 2 + endpoint, then two remaining-travel retries


def test_preparation_search_exhaustion_cannot_restart_via_another_mode():
    from table_1000.experts.context import ExpertFailed
    from table_1000.experts.skills.hold.pick import MAX_IK
    skill = ThinPicking()
    skill.arms, skill.held = ['right'], {}
    def fail(reason, **details):
        raise ExpertFailed(None, reason, **details)
    skill.ctx = SimpleNamespace(note=lambda *args, **kwargs: None, fail=fail,
        scene=SimpleNamespace(TABLE={'center': (0., 0.), 'size': (1., 1.), 'height': 0.}),
        base_transform=lambda: np.eye(4))
    searches = []
    def segment(*args, search):
        searches.append(search)
        for _ in range(search.limit - 1):
            search('right', False)
        search('right', True)
        search('right', True)  # must fail before another IK
    skill._slide_segment = segment
    with pytest.raises(ExpertFailed, match='thin preparation search exhausted') as caught:
        skill.slide_to_edge(None)
    assert len(searches) == 1
    assert caught.value.details['candidates'] == caught.value.details['limit'] == MAX_IK * 4 * 2
    assert searches[0].counts == {'right': {'fixed': MAX_IK * 4 * 2 - 1, 'lift': 1}}


def test_supported_extraction_restores_departure_permissions_when_motion_fails():
    permissions = []
    skill = ThinPicking()
    skill.ctx = SimpleNamespace(env=SimpleNamespace(center_of_mass=lambda body: [np.array([.005, 0., 0.])]))
    skill.held = {'right': {'supports': ['table']}}
    skill.solvers = {'right': SimpleNamespace(
        world=SimpleNamespace(allow=lambda *args: permissions.append(args)),
        allow=lambda names: permissions.append(('departure', names)))}
    def blocked(arm, command):
        skill.held[arm]['supports'] = []
        raise RuntimeError('blocked extraction')
    skill.move_by_displacement = blocked
    result = {'arm': 'right', 'supports': ['table'],
              'preparation': {'edge_outward': [-1., 0., 0.], 'edge_point': [0., 0., 0.]}}
    with pytest.raises(RuntimeError, match='blocked extraction'):
        skill.prepare_thin_lift(SimpleNamespace(body='object:body', names=['object']), result)
    assert (['object'], ['table'], False) in permissions
    assert skill.held['right']['supports'] == ['table']
    assert permissions[-1] == ('departure', ['table'])


@pytest.mark.parametrize('reason', ['joint_limit', 'collision'])
def test_optional_plate_extraction_keeps_vertical_departure(reason):
    from table_1000.experts.solver import PlanningFailed
    skill = ThinPicking()
    skill.thin_geometry = lambda target: SimpleNamespace(kind='plate')
    notes, permissions = [], []
    skill.ctx = SimpleNamespace(
        env=SimpleNamespace(center_of_mass=lambda body: [np.array([.005, 0., 0.])]),
        note=lambda *args, **kwargs: notes.append((args, kwargs)))
    skill.held = {'right': {'supports': ['table']}}
    skill.solvers = {'right': SimpleNamespace(
        world=SimpleNamespace(allow=lambda *args: permissions.append(args)),
        allow=lambda names: permissions.append(('departure', names)))}
    def unavailable(arm, command):
        skill.held[arm]['supports'] = []
        raise PlanningFailed(None, reason, plan={'iterations': 4})
    skill.move_by_displacement = unavailable
    result = {'arm': 'right', 'supports': ['table'],
              'preparation': {'edge_outward': [-1., 0., 0.], 'edge_point': [0., 0., 0.]}}
    skill.prepare_thin_lift(SimpleNamespace(body='object:body', names=['object']), result)
    assert notes[0][0] == ('thin extraction omitted',)
    assert notes[0][1]['details'] == {'plan': {'iterations': 4}}
    assert notes[-1][1]['command_m'] == 0.
    assert skill.held['right']['supports'] == ['table']
    assert permissions[-1] == ('departure', ['table'])


def test_supported_extraction_targets_real_edge_instead_of_negative_collision_allowance():
    center = np.array([.0014, 0., 0.])
    commands, events = [], []
    skill = ThinPicking()
    skill.ctx = SimpleNamespace(
        env=SimpleNamespace(center_of_mass=lambda body: [center.copy()]),
        note=lambda message, **values: events.append(values))
    skill.held = {'right': {'supports': ['table']}}
    skill.solvers = {'right': SimpleNamespace(
        world=SimpleNamespace(allow=lambda *args: None), allow=lambda names: None)}
    def move(arm, command):
        commands.append(command)
        center[:] += command
    skill.move_by_displacement = move
    result = {'arm': 'right', 'supports': ['table'],
              'preparation': {'edge_outward': [-1., 0., 0.], 'edge_point': [0., 0., 0.]}}
    skill.prepare_thin_lift(SimpleNamespace(body='object:body', names=['object']), result)
    # Subtracting the existing 0.5 mm collision penetration allowance would
    # wrongly omit this 1.4 mm extraction as a sub-millimetre waypoint.
    assert np.linalg.norm(commands[0]) == pytest.approx(.0014)
    assert events[-1]['com_outside_m'] > 0.


def test_support_polygon_uses_low_material_instead_of_airborne_outline():
    from table_1000.experts.skills.hold.sections import support_polygon, supported_point
    table = {'center': (.5, 0.), 'size': (1., 1.), 'height': 0.}
    foot = Box((.02, .02, .002)).points() + [.05, 0., .001]
    raised = Box((.10, .02, .002)).points() + [.07, 0., .02]
    polygon = support_polygon([foot, raised], table, .001)
    assert supported_point([.05, 0., .02], polygon)
    assert not supported_point([.10, 0., .02], polygon)


def test_com_support_does_not_imply_support_for_an_outboard_pressure_point():
    from table_1000.experts.skills.hold.sections import support_polygon, supported_point
    table = {'center': (.5, 0.), 'size': (1., 1.), 'height': 0.}
    fragment = Box((.012, .008, .002)).points() + [.0001, 0., .001]
    polygon = support_polygon([fragment], table, .001)
    assert supported_point([.0001, 0., .001], polygon)
    assert not supported_point([-.0002, 0., .001], polygon)
    assert not supported_point([0., 0., .001], polygon)


def test_tip_permission_does_not_turn_edge_patch_into_interior_area(hand):
    import copy
    from table_1000.experts.skills.hold.frames import _contact_area
    tip = copy.copy(hand)
    tip.hold_margin = 0.
    surface = Box((.0015, .001, .016))
    transform = np.eye(4)
    transform[0, 3] = .019
    # The piece reaches the last interior sampling column only. Tip mode can use it,
    # but ranking it against an interior grip must retain the real margin.
    permitted = _contact_area(surface, transform, np.eye(4), tip, .001, section=False)
    interior = _contact_area(surface, transform, np.eye(4), hand, .001, section=False)
    assert permitted >= hand.min_pad_area
    assert interior == 0.


def test_observed_edge_target_uses_exposed_section_without_target_surface():
    from table_1000.experts.skills.hold.frames import Box
    from table_1000.experts.skills.common import Target
    from table_1000.experts.skills.hold.pick import Grasping
    from table_1000.experts.skills.hold.thin import ThinPicking
    from table_1000.experts.skills.hold.sections import Edge, ExposedSection
    from types import SimpleNamespace
    class Skill(ThinPicking, Grasping):
        pass
    skill = Skill()
    points = Box((.04, .02, .001)).points()
    target = Target('object', ['object:body'], 'object:body', Box((.04, .02, .001)),
                    np.eye(4), points)
    skill.ctx = SimpleNamespace(poses=lambda: {'object:body': np.eye(4)},
        geometry={'object': SimpleNamespace(shapes=[('body', 'body', np.eye(4), 'convex', points)])})
    edge = Edge(np.zeros(3), np.array([1., 0., 0.]), np.array([0., 1., 0.]), 1.)
    exposed = skill.exposed_target(target, edge)
    assert isinstance(exposed.shape, ExposedSection)
    assert skill.contact_surface(exposed) is exposed.shape
    assert exposed.body == target.body and exposed.names == target.names


def test_exposed_section_runs_shared_candidate_and_contact_queries(hand, monkeypatch):
    import table_1000.experts.skills.hold.frames as grasps
    from table_1000.experts.skills.hold.sections import ExposedSection
    section = ExposedSection(Box((.02, .001, .02)).points())
    hand.radius = lambda: .1
    hand.moment_radius = lambda: .01
    hand.fcl = lambda *args, **kwargs: None
    hand.opening_clearance, hand.min_hold_area, hand.cost_width = .01, 5e-6, 0.
    hand.pad_alignment = lambda gap: {'edge_only': False, 'pad_alignment_cost': 0.}
    monkeypatch.setattr(grasps, '_collides', lambda *args: False)
    candidate = next(grasps.grasp_candidates(section, np.eye(4), hand, [], [1., 0., 0.],
                                             contact_surface=section))
    assert candidate.info['held_mm2'] > 5.
    support = grasps.contact_resistance(section, np.eye(4), candidate.pose, hand,
                                        candidate.info['contact_gap'], open_gap=candidate.open_gap)
    assert support['contact_radius_m'] > 0.
    assert 0. < support['contact_normal_cosine'] <= 1.
    remote = ExposedSection(section.points(), [section.points(), section.points() + [0., .1, 0.]])
    view = remote.closing_span(np.array([0., 1., 0.]), -.02, .02)
    low, high = view.interval(np.zeros((1, 3)), np.array([0., 1., 0.]))
    assert low[0] == pytest.approx(-.0005) and high[0] == pytest.approx(.0005)


def test_broad_plate_uses_edge_route_but_tall_hollow_container_does_not(hand):
    from dataclasses import replace
    hand.max_gap = .08
    plate = Box((.15, .10, .021)).points()
    assert classify(plate, [plate], hand).kind == 'plate'
    turn = Rotation.from_euler('xyz', [.4, .8, .2]).as_matrix()
    assert classify(plate @ turn.T, [plate @ turn.T], hand).kind == 'plate'
    too_thick = Box((.2, .15, .09)).points()
    assert classify(too_thick, [too_thick], hand) is None
    walls = [Box((.003, .10, .13)).points() + [x, 0., 0.] for x in (-.05, .05)]
    assert classify(np.vstack(walls), walls, hand) is None
