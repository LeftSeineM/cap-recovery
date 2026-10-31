import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from table_1000.experts.skills.hold.checks import CollisionSurface, GAP_TOLERANCE, closure, pad_contacts
from table_1000.experts.skills.hold.frames import Box, Hand
from table_1000.robots import load_spec


def test_authored_hollow_stays_empty_in_closure_rays():
    pieces = []
    for x in (-.04, .04):
        transform = np.eye(4)
        transform[0, 3] = x
        pieces.append((transform, 'convex', Box((.01, .02, .01)).points()))
    surface = CollisionSurface(pieces)
    low, high = surface.interval(np.array([[0., 0., 0.], [.04, 0., 0.]]), np.array([0., 1., 0.]))
    assert low[0] > high[0]
    assert low[1] == pytest.approx(-.01) and high[1] == pytest.approx(.01)
    selected = surface.at_contacts(np.array([[.04, .01, 0.]]))
    assert len(selected.pieces) == 1  # the remote side of a ring cannot inflate this opening


def _flat_test_hand():
    x, z = np.meshgrid(np.arange(-.004, .0041, .001), np.arange(-.004, .0041, .001))
    return types.SimpleNamespace(max_gap=.04, min_pad_area=15e-6, pad_spacing=.001, hold_margin=.002,
        side=types.SimpleNamespace(finger_links=['left', 'right']),
        pad_alignment=lambda gap: {'edge_only': False},
        inner=lambda gap: [(sign, np.c_[x.ravel(), np.full(x.size, sign * gap / 2), z.ravel()],
                           np.ones(x.size, dtype=bool)) for sign in (1., -1.)])


def test_empty_closing_needs_no_open_hand_surface_queries(monkeypatch):
    from table_1000.experts.skills.hold.frames import _close
    hand = _flat_test_hand()
    remote = np.eye(4)
    remote[0, 3] = .1
    surface = CollisionSurface([(remote, 'convex', Box((.01,) * 3).points())])
    gaps, queries = [], []
    inner, interval = hand.inner, surface.interval

    def measured_inner(gap):
        gaps.append(gap)
        return inner(gap)

    def measured_interval(origins, direction):
        queries.append(len(origins))
        return interval(origins, direction)

    hand.inner = measured_inner
    monkeypatch.setattr(surface, 'interval', measured_interval)
    assert _close(surface, np.eye(4), np.eye(4), hand, .04) is None
    assert gaps == [0.]
    assert len(queries) == 2  # one actual-surface query for each closed finger


@pytest.mark.parametrize('edge_width,accepted', [(.0232, True), (.025, False)])
def test_candidate_pad_layer_does_not_exceed_existing_contact_tolerance(edge_width, accepted):
    from table_1000.experts.skills.hold.frames import _close, CONTACT_TOLERANCE
    hand = _flat_test_hand()
    surface = CollisionSurface([
        (np.eye(4), 'convex', Box((.002, edge_width, .002)).points()),
        (np.eye(4), 'convex', Box((.008, .02, .008)).points())])
    # A small protruding edge touches first, before the broad backing face.
    assert _close(surface, np.eye(4), np.eye(4), hand, .04) is None
    result = _close(surface, np.eye(4), np.eye(4), hand, .04, contact_layer=True)
    assert (result is not None) == accepted
    if accepted:
        gap, _ = result
        assert (edge_width - gap) / 2 <= CONTACT_TOLERANCE
    # The opt-in candidate query never changes the measured signal's default.
    assert _close(surface, np.eye(4), np.eye(4), hand, .04) is None


def test_candidate_pad_layer_keeps_first_contact_when_it_already_supports_the_pads():
    from table_1000.experts.skills.hold.frames import _close
    hand = _flat_test_hand()
    surface = CollisionSurface([(np.eye(4), 'convex', Box((.008, .02, .008)).points())])
    old = _close(surface, np.eye(4), np.eye(4), hand, .04)
    new = _close(surface, np.eye(4), np.eye(4), hand, .04, contact_layer=True)
    assert old is not None and new is not None
    assert old[0] == new[0]
    np.testing.assert_array_equal(old[1], new[1])


def test_remote_component_cannot_centre_itself_into_an_empty_hand():
    from table_1000.experts.skills.hold.frames import _close, _closing_surface, contact_resistance
    remote = np.eye(4)
    remote[1, 3] = .14
    surface = CollisionSurface([(remote, 'convex', Box((.02,) * 3).points())])
    hand = _flat_test_hand()
    # Unbounded centring can invent a 140 mm translation despite neither
    # finger being able to reach the component anywhere in its closing.
    assert _close(surface, np.eye(4), np.eye(4), hand, .04) is not None
    limited = _closing_surface(surface, np.eye(4), np.eye(4), hand, .04)
    assert _close(limited, np.eye(4), np.eye(4), hand, .04) is None
    assert contact_resistance(surface, np.eye(4), np.eye(4), hand, .02,
                              open_gap=.04)['contact_radius_m'] == 0.
    # This repair belongs to candidate prediction. The expert's measured
    # closure signal keeps the same formula, gap budget and default behavior.
    signal = closure(hand, surface, np.eye(4), np.eye(4), .02, .04, [], 'object:body')
    assert signal['expected_gap_m'] == pytest.approx(.02, abs=.0005)
    assert signal['reasons'] == ['missing opposing pad contact']


def test_finite_closing_queries_keep_true_faces_and_leave_other_views_unchanged():
    far = np.eye(4)
    far[1, 3] = .14
    surface = CollisionSurface([(far, 'convex', Box((.02,) * 3).points()),
                                (np.eye(4), 'convex', Box((.02, .06, .02)).points())])
    origins, axis = np.array([[0., .13, 0.], [0., 0., 0.]]), np.array([0., 1., 0.])
    old = surface.interval(origins, axis)
    view = surface.closing_span(axis, -.02, .02)
    low, high = view.interval(origins, axis)
    # A solid intersecting the span retains its genuine faces outside it;
    # clipping at the span would manufacture a narrower, false contact gap.
    np.testing.assert_allclose(low, [-.16, -.03], atol=1e-15)
    np.testing.assert_allclose(high, [-.10, .03], atol=1e-15)
    for before, after in zip(old, surface.interval(origins, axis)):
        np.testing.assert_array_equal(before, after)


def test_closing_span_includes_intermediate_finger_positions_and_transformed_frame():
    from table_1000.experts.skills.hold.frames import _closing_surface
    hand = _flat_test_hand()
    inner = hand.inner
    def bending(gap):
        # An intermediate pivot position reaches beyond both endpoint poses.
        return inner(.07 if abs(gap - .01) < 1e-8 else gap)
    hand.inner = bending
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('z', 41, degrees=True).as_matrix()
    transform[:3, 3] = [.5, -.3, .8]
    remote = np.eye(4)
    remote[1, 3] = .03
    surface = CollisionSurface([(remote, 'convex', Box((.01,) * 3).points())])
    view = _closing_surface(surface, transform, transform, hand, .02)
    low, high = view.interval(np.zeros((1, 3)), np.array([0., 1., 0.]))
    assert low[0] == pytest.approx(.025) and high[0] == pytest.approx(.035)


def test_actual_wide_section_only_at_pad_tip_cannot_borrow_bounding_box_area(monkeypatch):
    import table_1000.experts.skills.hold.frames as grasps
    proxy = Box((.04, .02, .02))
    transforms = [np.eye(4), np.eye(4)]
    transforms[0][0, 3], transforms[1][0, 3] = .0195, -.0005
    surface = CollisionSurface([(transforms[0], 'convex', Box((.001, .02, .02)).points()),
                                (transforms[1], 'convex', Box((.039, .002, .02)).points())])
    x, z = np.meshgrid(np.arange(-.02, .0201, .001), np.arange(-.01, .0101, .001))
    def inner(gap):
        return [(sign, np.c_[x.ravel(), np.full(x.size, sign * gap/2), z.ravel()],
                 np.ones(x.size, dtype=bool)) for sign in (1., -1.)]
    hand = types.SimpleNamespace(inner=inner, fcl=lambda *a, **k: None, hold_margin=.002,
                                min_hold_area=5e-6, min_pad_area=15e-6, pad_spacing=.001)
    monkeypatch.setattr(grasps, '_collides', lambda *args: False)
    args = (proxy, np.eye(4), None, np.eye(4), hand, .04, [], .05)
    assert grasps._placed(*args) is not None
    # The wide face at the initial pad tip supplies no interior support.
    assert grasps._contact_area(surface, np.eye(4), np.eye(4), hand, .02, section=False) == 0.
    # Backing off can instead grip the actual narrow component. It must not
    # keep the proxy's wide opening or borrow that proxy's contact area.
    actual = grasps._placed(*args, contact_surface=surface)
    assert actual is not None and actual[1] > 0. and actual[2] < .003


def test_subpart_closure_uses_all_actual_surfaces_on_the_same_body():
    from table_1000.experts.skills.hold.pick import Grasping
    patch = CollisionSurface([(np.eye(4), 'convex', Box((.01, .02, .01)).points())])
    skill = Grasping()
    remote = np.eye(4)
    remote[1, 3] = .03
    geometry = types.SimpleNamespace(_candidate_surfaces={}, _grasp_surfaces={}, shapes=[
        ('body', 'fins', np.eye(4), 'convex', Box((.01, .02, .01)).points()),
        ('body', 'fins', remote, 'convex', Box((.01, .02, .01)).points())])
    skill.ctx = types.SimpleNamespace(geometry={'object': geometry},
        env=types.SimpleNamespace(objects={'object': types.SimpleNamespace(articulated=False)}))
    target = types.SimpleNamespace(obj='object', body='object:body', surface=patch, offset=np.eye(4))
    surface = skill.contact_surface(target)
    assert len(surface.pieces) == 2
    assert np.ptp(surface.points(), axis=0)[1] == pytest.approx(.05)
    # A local patch predicts a narrow pinch, while another fin on the same
    # body already blocks that opening. Neither path fills the air gap.
    from table_1000.experts.skills.hold.frames import _close
    x, z = np.meshgrid(np.arange(-.004, .0041, .001), np.arange(-.004, .0041, .001))
    hand = types.SimpleNamespace(pad_spacing=.001, min_pad_area=15e-6,
        inner=lambda gap: [(sign, np.c_[x.ravel(), np.full(x.size, sign * gap / 2), z.ravel()],
                           np.ones(x.size, dtype=bool)) for sign in (1., -1.)])
    assert _close(patch, np.eye(4), np.eye(4), hand, .04) is not None
    assert _close(surface, np.eye(4), np.eye(4), hand, .04) is None


def test_candidate_contact_surfaces_belong_to_the_held_rigid_body():
    from table_1000.experts.skills.hold.pick import Grasping
    skill = Grasping()
    geometry = types.SimpleNamespace(_candidate_surfaces={}, _grasp_surfaces={}, shapes=[
        ('body', 'barrel', np.eye(4), 'convex', Box((.02, .02, .1)).points()),
        ('cap', 'cap_shell', np.eye(4), 'convex', Box((.04, .04, .04)).points())])
    skill.ctx = types.SimpleNamespace(geometry={'object': geometry}, env=types.SimpleNamespace(
        objects={'object': types.SimpleNamespace(articulated=True, root='body')}))
    target = types.SimpleNamespace(obj='object', body='object:body', surface=None,
        names=['object:barrel', 'object:cap_shell'], offset=np.eye(4))
    surface = skill.contact_surface(target)
    assert len(surface.pieces) == 1
    assert np.allclose(np.ptp(surface.points(), axis=0), [.02, .02, .1])


def test_curved_patch_does_not_have_to_hold_the_global_widest_section():
    from table_1000.experts.skills.hold.frames import _contact_area, _held_area
    remote = np.eye(4)
    remote[2, 3] = .1
    surface = CollisionSurface([(np.eye(4), 'convex', Box((.04, .02, .02)).points()),
                                (remote, 'convex', Box((.04, .06, .02)).points())])
    x, z = np.meshgrid(np.arange(-.02, .0201, .001), np.arange(-.01, .0101, .001))
    hand = types.SimpleNamespace(hold_margin=.002, pad_spacing=.001,
        inner=lambda gap: [(sign, np.c_[x.ravel(), np.full(x.size, sign * gap/2), z.ravel()],
                           np.ones(x.size, dtype=bool)) for sign in (1., -1.)])
    assert _held_area(surface, np.eye(4), np.eye(4), hand, .02) == 0.
    assert _contact_area(surface, np.eye(4), np.eye(4), hand, .02) > 200e-6


@pytest.mark.parametrize('first_section', ['no opposing faces', 'small patch'])
def test_backoff_can_find_a_later_component_after_an_unheld_section(monkeypatch, first_section):
    from table_1000.experts.skills.hold import frames as grasps
    # A compound body need not shrink monotonically along the approach. An
    # early edge can have no opposing face, then a later component can hold.
    frame = np.eye(4)
    frame[0, 3] = -.01
    shape = Box((.04, .02, .02))
    surface = CollisionSurface([(np.eye(4), 'convex', shape.points()),
                                (frame, 'convex', shape.points())])
    hand = types.SimpleNamespace(min_hold_area=15e-6,
        inner=lambda gap: [(1., np.array([[.04, 0., 0.]]), np.array([True]))],
        fcl=lambda *args, **kwargs: None)
    monkeypatch.setattr(grasps, '_collides', lambda *args: False)
    def close(surface, transform, tcp, *args, **kwargs):
        if tcp[0, 3] > -.006 and first_section == 'no opposing faces':
            return None
        return .02, tcp[:3, 3]
    monkeypatch.setattr(grasps, '_close', close)
    monkeypatch.setattr(grasps, '_contact_area', lambda surface, transform, tcp, *args, **kwargs:
                        30e-6 if tcp[0, 3] <= -.006 else 0.)
    result = grasps._placed(shape, np.eye(4), None, np.eye(4), hand, .04, [], .05, surface)
    assert result is not None and result[1] == pytest.approx(.006)


def test_tangent_cylinder_patch_does_not_replace_safe_insertion_depth():
    from table_1000.experts.skills.hold.frames import Cylinder, _contact_area, CONTACT_TOLERANCE
    surface = CollisionSurface([(np.eye(4), 'convex', Cylinder(.009, .1).points())])
    x, z = np.meshgrid(np.arange(.006, .0121, .001), np.arange(-.01, .0101, .001))
    points = np.c_[x.ravel(), np.full(x.size, .0065), z.ravel()]
    hand = types.SimpleNamespace(hold_margin=.002, pad_spacing=.001,
        inner=lambda gap: [(sign, points * [1., sign, 1.], np.ones(x.size, dtype=bool))
                          for sign in (1., -1.)])
    # Both pads have a sizeable tangent contact layer behind their tip
    # margin, but the cylinder's equator (X=0) is outside their footprint.
    low, high = surface.interval(points, np.array([0., 1., 0.]))
    interior = points[:, 0] < points[:, 0].max() - hand.hold_margin
    assert np.count_nonzero(interior & (high >= -CONTACT_TOLERANCE)) * hand.pad_spacing ** 2 > 15e-6
    assert _contact_area(surface, np.eye(4), np.eye(4), hand, .013) == 0.
    # Contact-layer support and a round section's insertion are distinct
    # measurements; flat candidates continue to use the former.
    assert _contact_area(surface, np.eye(4), np.eye(4), hand, .013, section=False) > 15e-6


def test_opposing_components_do_not_count_their_separation_as_patch_thickness():
    from table_1000.experts.skills.hold.frames import _contact_area, _held_area
    pieces = []
    for y in (-.02, .02):
        transform = np.eye(4)
        transform[1, 3] = y
        pieces.append((transform, 'convex', Box((.04, .02, .02)).points()))
    surface = CollisionSurface(pieces)
    x, z = np.meshgrid(np.arange(-.02, .0201, .001), np.arange(-.01, .0101, .001))
    hand = types.SimpleNamespace(hold_margin=.002, pad_spacing=.001,
        inner=lambda gap: [(sign, np.c_[x.ravel(), np.full(x.size, sign * gap/2), z.ravel()],
                           np.ones(x.size, dtype=bool)) for sign in (1., -1.)])
    assert _held_area(surface, np.eye(4), np.eye(4), hand, .06) == 0.
    assert _contact_area(surface, np.eye(4), np.eye(4), hand, .06) > 200e-6


@pytest.mark.parametrize('only_record,checks,opens', [(False, 2, 1), (True, 1, 0)])
def test_asset_closure_observation_does_not_change_expert_default(only_record, checks, opens):
    from table_1000.experts.skills.hold.pick import Grasping
    skill = Grasping()
    skill.arms, skill.held, skill.log = ['right'], {}, []
    skill.constrained = lambda target: False
    skill.dragged = lambda *args: ([], [], [])
    skill.rejected = {key: [] for key in ('grasp IK', 'pre-grasp IK', 'then', 'need')}
    opened, checked = [], []
    solver = types.SimpleNamespace(close_gripper=lambda: None, finger_gap=lambda: .02,
        open_gripper=opened.append, first_reachable=lambda poses: None, allow=lambda *args: None,
        tcp=lambda: np.eye(4), attach=lambda *args: None, carry=lambda *args: None,
        world=types.SimpleNamespace(objects={'part': types.SimpleNamespace(body='object:body')}, allow=lambda *args: None))
    skill.solvers = {'right': solver}
    grasp = types.SimpleNamespace(pose=np.eye(4), width=.02, open_gap=.03, backoff=0., cost=0., info={}, target=None, choice=None)
    skill.reach = lambda *args, **kwargs: ('right', grasp, [])
    def check(*args):
        checked.append(True)
        return {'held': len(checked) > 1, 'reasons': ['no opposing collision surfaces']}
    skill.check_closure = check
    events = []
    skill.ctx = types.SimpleNamespace(geometry={}, spec=types.SimpleNamespace(base=False),
        poses=lambda: {'object:body': np.eye(4)}, note=lambda note, **data: events.append((note, data)))
    kwargs = {'closure_only_record': True} if only_record else {}  # exercise the actual default
    result = skill.grasp(types.SimpleNamespace(obj='object', names=['part'], body='object:body'), **kwargs)
    assert len(checked) == checks and len(opened) == opens
    assert result['regrasps'] == opens
    assert result['closure_check']['held'] == (not only_record)
    assert events[0][1]['only_record'] == only_record


@pytest.mark.parametrize('gap,area,surface_held,held,reason', [
    (.015, 5e-6, True, True, None),
    (.014999, 5e-6, True, False, 'closed past planned contact gap'),
    (.020, 4.75e-6, True, False, 'insufficient held area'),
    (.020, 5e-6, False, False, 'missing opposing pad contact'),
])
def test_expert_closure_retains_main_gap_and_area_checks(monkeypatch, gap, area, surface_held, held, reason):
    import table_1000.experts.skills.hold.checks as checks
    import table_1000.experts.skills.hold.pick as grasping
    skill = grasping.Grasping()
    hand = types.SimpleNamespace(side=types.SimpleNamespace(finger_links=[]))
    skill.hands = {'right': hand}
    skill.solvers = {'right': types.SimpleNamespace(finger_gap=lambda: gap)}
    geometry = types.SimpleNamespace(_candidate_surfaces={}, _grasp_surfaces={}, shapes=[])
    body_pose, tcp, offset, shape = np.eye(4), np.eye(4), np.eye(4), object()
    offset[0, 3] = .03
    skill.ctx = types.SimpleNamespace(scene=types.SimpleNamespace(OBJECTS={'object': {}}),
        geometry={'object': geometry}, tcp=lambda arm: tcp,
        poses=lambda: {'object:body': body_pose},
        env=types.SimpleNamespace(contacts=types.SimpleNamespace(surface_contacts={})))
    monkeypatch.setattr(grasping, 'CollisionSurface', lambda shapes: object())
    monkeypatch.setattr(grasping, 'closure', lambda *args: {
        'held': surface_held, 'reasons': [] if surface_held else ['missing opposing pad contact']})
    def held_area(actual_shape, transform, actual_tcp, actual_hand, actual_gap):
        assert actual_shape is shape and actual_tcp is tcp and actual_hand is hand
        np.testing.assert_array_equal(transform, body_pose @ offset)
        assert actual_gap == gap
        return area
    skill.contact_surface = lambda selected: None
    monkeypatch.setattr(grasping, '_grasp_area', lambda *args, **kwargs: held_area(*args))
    result = skill.check_closure(types.SimpleNamespace(obj='object', body='object:body',
                                shape=shape, offset=offset), 'right',
                                types.SimpleNamespace(open_gap=.04, info={'contact_gap': .020}))
    assert result['held'] is held
    assert result['planned_gap_held'] == (gap >= .020 - grasping.HELD_TOLERANCE)
    assert result['area_held'] == (area >= grasping.MIN_HOLD_AREA)
    assert result['actual_held_area_m2'] == area
    assert result['held_gap_tolerance_m'] == .005 and result['min_hold_area_m2'] == 5e-6
    assert result['reasons'] == ([] if reason is None else [reason])


@pytest.mark.parametrize('offset_x', [0., .001])
def test_primitive_closure_uses_its_supplied_shape_without_asset_geometry(offset_x):
    from table_1000.experts.skills.hold.pick import Grasping
    skill = Grasping()
    skill.hands = {'right': _flat_test_hand()}
    skill.solvers = {'right': types.SimpleNamespace(finger_gap=lambda: .02)}
    offset = np.eye(4)
    offset[0, 3] = offset_x
    skill.ctx = types.SimpleNamespace(scene=types.SimpleNamespace(OBJECTS={}), geometry={},
        tcp=lambda arm: np.eye(4), poses=lambda: {'cube1': np.eye(4)},
        env=types.SimpleNamespace(contacts=types.SimpleNamespace(surface_contacts={})))
    target = types.SimpleNamespace(obj='cube1', body='cube1', shape=Box((.008, .02, .008)), offset=offset)
    result = skill.check_closure(target, 'right',
        types.SimpleNamespace(open_gap=.04, info={'contact_gap': .02}))
    assert result['expected_gap_m'] == pytest.approx(.02, abs=.0005)
    assert result['actual_held_area_m2'] > 0.
    assert not result['held']  # Missing pad contacts are still checked.
    assert 'missing opposing pad contact' in result['reasons']


def test_body_or_tip_contact_cannot_substitute_for_two_pad_contacts():
    def contact(link, surface):
        return {'bodies': ['robot.' + link, 'object:body.part'],
                'colliders': ['/robot/' + link + '/collisions/' + surface, '/object/part'], 'points': []}
    contacts = [contact('left', 'left_pad_7'), contact('right', 'tip')]
    assert not all(pad_contacts(contacts, ['left', 'right'], 'object:body').values())
    contacts.append(contact('right', 'right_pad_7'))
    assert all(pad_contacts(contacts, ['left', 'right'], 'object:body').values())
    assert not any(pad_contacts(contacts, ['left', 'right'], 'other:body').values())


def test_closure_uses_actual_surface_gap_and_geometric_tolerance(monkeypatch):
    import table_1000.experts.skills.hold.checks as checks
    monkeypatch.setattr(checks, '_close', lambda *args, **kwargs: (.020, np.zeros(3)))
    hand = types.SimpleNamespace(side=types.SimpleNamespace(finger_links=['l', 'r']), max_gap=.1,
           pad_alignment=lambda gap: {'edge_only': False})
    rows = [{'bodies': ['robot.' + link, 'object:body.part'],
             'colliders': ['/' + link + '/pad', '/object/part']} for link in ['l', 'r']]
    assert closure(hand, None, None, None, .020-GAP_TOLERANCE, .04, rows, 'object:body')['held']
    assert not closure(hand, None, None, None, .020-GAP_TOLERANCE-.0001, .04, rows, 'object:body')['held']
    monkeypatch.setattr(checks, '_close', lambda *args, **kwargs: None)
    assert not closure(hand, None, None, None, .02, .04, rows, 'object:body')['held']


def test_wide_pivot_opening_has_only_edge_band_without_asset_rules():
    side = load_spec('x1pro_rm001').arms['right']
    hand = Hand.__new__(Hand)
    hand.side, hand.hold_margin = side, .002
    hand.inner = lambda gap: [(1., np.array([[0., 0, 0], [.04, 0, 0]]), np.ones(2, dtype=bool))]
    assert not hand.pad_alignment(.02)['edge_only']
    assert hand.pad_alignment(.165)['edge_only']
    assert hand.pad_alignment(.18)['pad_angle_rad'] > np.radians(60)


def test_tensor_contact_points_identify_pad_surface_and_exclude_tip():
    hand = Hand.__new__(Hand)
    hand.shapes = {'finger': [(np.eye(4), 'box', (.04, .005, .02), True)]}
    assert hand.on_pad('finger', [[.01, .0025, 0.]])
    assert not hand.on_pad('finger', [[.04, .0025, 0.]])


def test_contact_patch_resistance_uses_size_and_normal_alignment():
    from table_1000.experts.skills.hold.frames import contact_resistance
    x, z = np.meshgrid(np.arange(-.012, .0121, .0015), np.arange(-.009, .0091, .0015))
    hand = types.SimpleNamespace(hold_margin=0., pad_spacing=.0015,
        inner=lambda gap: [(s, np.c_[x.ravel(), np.full(x.size, s * gap / 2), z.ravel()],
                           np.ones(x.size, dtype=bool)) for s in (1., -1.)])
    def measure(size, angle=0.):
        frame = np.eye(4)
        frame[:3, :3] = Rotation.from_euler('x', angle, degrees=True).as_matrix()
        surface = CollisionSurface([(frame, 'convex', Box(size).points())])
        return contact_resistance(surface, np.eye(4), np.eye(4), hand, .02)
    broad = measure((.04, .02, .03))
    small = measure((.006, .02, .006))
    tilted = measure((.04, .02, .03), 20.)
    assert broad['contact_radius_m'] > small['contact_radius_m'] > 0.
    assert broad['contact_normal_cosine'] == pytest.approx(1.)
    assert broad['normal_redirection_cost'] == 0.
    assert tilted['contact_normal_cosine'] < broad['contact_normal_cosine']
    assert tilted['normal_redirection_cost'] == pytest.approx(np.tan(np.radians(20.)))


def test_collision_surface_normals_keep_disconnected_actual_faces():
    frame = np.eye(4)
    frame[:3, :3] = Rotation.from_euler('z', 30, degrees=True).as_matrix()
    frame[:3, 3] = [.1, .2, .3]
    surface = CollisionSurface([(frame, 'convex', Box((.02,) * 3).points())])
    points = np.array([[.01, 0., 0.], [0., -.01, 0.]]) @ frame[:3, :3].T + frame[:3, 3]
    np.testing.assert_allclose(surface.normals(points), np.array([[1., 0., 0.], [0., -1., 0.]]) @ frame[:3, :3].T)


def test_candidate_clearance_adds_remote_same_body_colliders_without_filling_gaps():
    from mplib.collision_detection import fcl
    from table_1000.planning.world import geometry, pose
    remote = np.eye(4)
    remote[1, 3] = .03
    box = Box((.01, .02, .01))
    surface = CollisionSurface([(np.eye(4), 'convex', box.points()),
                                (remote, 'convex', box.points())])
    objects, poses = surface.collision_objects(box)
    assert len(objects) == 1  # the selected box already covers the first piece
    assert surface.collision_objects(box)[0] is objects
    target = fcl.FCLObject('target', pose(np.eye(4)),
        [fcl.CollisionObject(geometry(*box.fcl_shape())), *objects], [pose(np.eye(4)), *poses])
    def collides(y):
        frame = np.eye(4)
        frame[1, 3] = y
        probe = fcl.FCLObject('finger', pose(frame), [fcl.CollisionObject(fcl.Box(.002, .002, .002))],
                              [pose(np.eye(4))])
        return fcl.collide(probe, target).is_collision()
    assert collides(.03)   # neighbouring fin stops the open hand
    assert not collides(.015)  # the actual space between fins stays empty


def test_holder_rim_clearance_keeps_the_authored_floor_hull():
    import json
    from pathlib import Path
    from mplib.collision_detection import fcl
    from table_1000.planning.world import pose

    fixture = json.loads((Path(__file__).parent / 'fixtures/holder-clearance-hull.json').read_text())
    frame, points = np.asarray(fixture['transform']), np.asarray(fixture['points'])
    surface = CollisionSurface([(frame, fixture['kind'], points)])
    objects, poses = surface.collision_objects(Box(fixture['enclosing_size']))
    assert len(objects) == 1
    floor = fcl.FCLObject('floor', pose(np.eye(4)), objects, poses)

    def collides(local):
        probe_pose = frame.copy()
        probe_pose[:3, 3] = frame[:3, :3] @ local + frame[:3, 3]
        probe = fcl.FCLObject('probe', pose(probe_pose),
                              [fcl.CollisionObject(fcl.Sphere(.0001))], [pose(np.eye(4))])
        return fcl.collide(probe, floor).is_collision()

    center = points.mean(0)
    normal = np.linalg.svd(points - center, full_matrices=False)[2][-1]
    outside = center + normal * (np.max((points - center) @ normal) + .003)
    assert collides(center)
    assert not collides(outside)


@pytest.mark.parametrize('x,accepted', [(.008, True), (.016, False)])
def test_round_part_measured_area_uses_the_touched_facets_with_the_original_tip_margin(x, accepted):
    from table_1000.experts.skills.hold.frames import Cylinder, MIN_HOLD_AREA, _contact_area, _held_area
    from table_1000.experts.skills.hold.pick import Grasping
    hand = _flat_test_hand()
    pieces = []
    for k in range(12):
        a, b = np.arange(k, k + 2) * 2 * np.pi / 12
        triangle = [[0., 0.], [.011 * np.cos(a), .011 * np.sin(a)],
                    [.011 * np.cos(b), .011 * np.sin(b)]]
        points = np.array([[*p, z] for p in triangle for z in (-.0475, .0475)])
        pieces.append((np.eye(4), 'convex', points))
    surface = CollisionSurface(pieces)
    target = types.SimpleNamespace(shape=Cylinder(.011, .095))
    skill = Grasping()
    skill.hands = {'right': hand}
    skill.contact_surface = lambda selected: surface
    tcp = np.eye(4)
    tcp[0, 3] = x
    # The ideal circle's widest section is beyond the pad-tip margin, while
    # a touched facet can still provide a real interior contact section.
    assert _held_area(target.shape, np.eye(4), tcp, hand, .012) == 0.
    area = skill.measured_held_area(target, 'right', np.eye(4), tcp, .012)
    assert area == _contact_area(surface, np.eye(4), tcp, hand, .012)
    assert (area >= MIN_HOLD_AREA) == accepted
    assert hand.hold_margin == .002 and MIN_HOLD_AREA == 5e-6


def test_recorded_jar_closures_use_the_candidate_finite_sweep(tmp_path):
    import json
    from pathlib import Path
    from table_1000.experts.skills.hold.frames import MIN_HOLD_AREA, _grasp_area
    from table_1000.experts.skills.hold.pick import Grasping
    from table_1000.experts.skills.hold.targets import oriented_shape
    from table_1000.robots.kinematics import RobotModel
    from table_1000.scenes.geometry import AssetGeometry
    root = Path(__file__).resolve().parents[2]
    if not (root / 'assets/objects/storage_box/000000').exists():
        pytest.skip('built storage box asset unavailable')
    geometry = AssetGeometry.of(root / 'assets', 'objects/storage_box/000000')
    hand = Hand(RobotModel(load_spec('x1pro'), tmp_path / 'planning', root), 'right')
    skill = Grasping()
    skill.hands = {'right': hand}
    skill.ctx = types.SimpleNamespace(geometry={'jar': geometry},
        env=types.SimpleNamespace(objects={'jar': types.SimpleNamespace(articulated=False)}))
    pieces = []
    for body, name, transform, kind, data in geometry.shapes:
        if kind == 'convex':
            shape, offset = oriented_shape(data)
            pieces.append((shape, transform @ offset))
    fixture = json.loads((Path(__file__).parent / 'fixtures/jar-finite-closing.json').read_text())
    rows = fixture['closings']
    assert len(rows) == 6
    for row in rows:
        offset, body, tcp = map(np.asarray, (row['target_offset'], row['body_pose'], row['tcp']))
        shape, nearest = min(pieces, key=lambda piece: np.linalg.norm(piece[1] - offset))
        assert np.allclose(nearest, offset)
        target = types.SimpleNamespace(obj='jar', body='jar:' + fixture['body_name'], shape=shape, offset=offset)
        transform = body @ offset
        assert _grasp_area(shape, transform, tcp, hand, row['gap'],
                           surface=skill.contact_surface(target)) == 0.
        area = skill.measured_held_area(target, 'right', transform, tcp, row['gap'], open_gap=row['open_gap'])
        assert area == pytest.approx(row['finite_closing_span_area_m2'])
        skill.held = {'right': {'target': target, 'grasp': {'open_gap': row['open_gap']}}}
        assert skill.measured_held_area(target, 'right', transform, tcp, row['gap']) == area
        skill.ctx.poses = lambda: {target.body: body}
        skill.ctx.tcp = lambda arm: tcp
        skill.solvers = {'right': types.SimpleNamespace(finger_gap=lambda: row['gap'])}
        candidate = types.SimpleNamespace(open_gap=row['open_gap'], info={'contact_gap': row['gap']})
        for held in (False, True):
            skill.pad_closure = lambda *args: {'held': held, 'reasons': [] if held else ['missing opposing pad contact']}
            checked = skill.check_closure(target, 'right', candidate)
            assert checked['held'] is held
            assert checked['area_held'] and checked['planned_gap_held']
        candidate.info['contact_gap'] = row['gap'] + .006
        assert not skill.check_closure(target, 'right', candidate)['held']
    assert MIN_HOLD_AREA == 5e-6 and hand.hold_margin == .002


def test_measured_surface_area_cannot_borrow_support_at_the_pad_tip():
    from table_1000.experts.skills.hold.frames import MIN_HOLD_AREA, _held_area
    from table_1000.experts.skills.hold.pick import Grasping
    hand = _flat_test_hand()
    # The broad contact face reaches only the last millimetre of the pads;
    # a thinner backing body cannot supply opposing support at this gap.
    tip, backing = np.eye(4), np.eye(4)
    tip[0, 3], backing[0, 3] = .0035, -.0005
    surface = CollisionSurface([(tip, 'convex', Box((.001, .02, .008)).points()),
                                (backing, 'convex', Box((.007, .002, .008)).points())])
    target = types.SimpleNamespace(shape=Box((.008, .02, .008)))
    skill = Grasping()
    skill.hands = {'right': hand}
    skill.contact_surface = lambda selected: surface
    assert _held_area(target.shape, np.eye(4), np.eye(4), hand, .02) >= MIN_HOLD_AREA
    assert skill.measured_held_area(target, 'right', np.eye(4), np.eye(4), .02) == 0.
    skill.contact_surface = lambda selected: None
    # Exact primitive targets retain their original widest-section rule.
    assert skill.measured_held_area(target, 'right', np.eye(4), np.eye(4), .02) == \
        _held_area(target.shape, np.eye(4), np.eye(4), hand, .02)


def test_recorded_a4_closure_uses_its_actual_cover_contact_layer(tmp_path):
    from pathlib import Path
    from table_1000.experts.skills.hold.frames import MIN_HOLD_AREA, _held_area
    from table_1000.experts.skills.hold.pick import Grasping
    from table_1000.experts.skills.hold.targets import oriented_shape
    from table_1000.robots.kinematics import RobotModel
    from table_1000.scenes.geometry import AssetGeometry
    root = Path(__file__).resolve().parents[2]
    asset = root / 'assets/objects/notebook/000002'
    if not asset.exists():
        pytest.skip('built notebook asset unavailable')
    geometry = AssetGeometry.of(root / 'assets', 'objects/notebook/000002')
    body = geometry.bodies[0]
    shape, offset = oriented_shape(geometry.points('notebook.closed.pages', body))
    hand = Hand(RobotModel(load_spec('x1pro'), tmp_path / 'planning', root), 'right')
    # Scene 000100 strict A8, the third closed frame. This is the measured
    # TCP in the page frame, after aligning post-action state with obs[i+1].
    tcp = np.array([
        [2.806324179435076e-5, -.9999972384843762, .0023785010272418493, 2.038750103183773e-7],
        [-.5002364078929662, -.002073437790107337, -.8658863413553579, .029166749942483756],
        [.865888884467998, -.0011655040468952348, -.5002351803692557, -.11118144600537183],
        [0., 0., 0., 1.]])
    gap = .015043409075587988
    target = types.SimpleNamespace(obj='notebook', body=f'notebook:{body}', shape=shape, offset=offset)
    skill = Grasping()
    skill.hands = {'right': hand}
    skill.ctx = types.SimpleNamespace(geometry={'notebook': geometry},
        env=types.SimpleNamespace(objects={'notebook': types.SimpleNamespace(articulated=False)}))
    assert _held_area(shape, np.eye(4), tcp, hand, gap) == 0.
    assert skill.measured_held_area(target, 'right', np.eye(4), tcp, gap) == pytest.approx(405e-6)
    # Move this same measured grasp to an edge where only the last 2 mm of
    # the pad sees material. Removing the margin would pass; it must fail.
    import copy
    at_tip = tcp.copy()
    at_tip[:3, 3] -= .062 * tcp[:3, 0]
    assert skill.measured_held_area(target, 'right', np.eye(4), at_tip, gap) == 0.
    loose = copy.copy(hand)
    loose.hold_margin = 0.
    skill.hands = {'right': loose}
    assert skill.measured_held_area(target, 'right', np.eye(4), at_tip, gap) >= MIN_HOLD_AREA
    assert hand.hold_margin == .002 and MIN_HOLD_AREA == 5e-6
