"""Orientation is derived from body geometry, and passed through Place lookahead."""
from table_1000.experts.skills.place import regions
from table_1000.experts.world import geometry as world_geometry

from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from table_1000.experts.skills.place.strict import Placing
from table_1000.experts.skills.place.regions import orientations


def box(size):
    return np.array([[x, y, z] for x in (-size[0]/2, size[0]/2)
                     for y in (-size[1]/2, size[1]/2) for z in (-size[2]/2, size[2]/2)])


@pytest.mark.parametrize('size,mode,axis', [((.012,.14,.012),'upright',1), ((.22,.3,.012),'flat',2)])
def test_placement_turns_body_to_support_normal(size, mode, axis):
    skill = Placing()
    skill.ctx = SimpleNamespace(trace=None)
    current = np.eye(4)
    current[:3,:3] = Rotation.from_euler('xyz', [.7,.4,.2]).as_matrix()
    frame = np.eye(4)
    frame[2,3] = .8
    points = box(size)
    options = skill.placements(SimpleNamespace(points=points), current, (frame,[-.2,-.2],[.2,.2]),
                               [0,0,.8], [], .02, .01, mode)
    assert options
    for pose, _ in options:
        assert abs(pose[2,axis]) == pytest.approx(1)
        world = points @ pose[:3,:3].T + pose[:3,3]
        assert world[:,2].min() == pytest.approx(.8)
        assert (np.abs(world[:,:2]) <= .19 + 1e-9).all()


def test_unsupported_orientation_shape():
    for mode in ('upright','flat','slant'):
        with pytest.raises(ValueError, match='orientation|requires'):
            orientations(box((.05,.05,.05)),np.eye(4),np.eye(4),mode)


def test_flat_yaw_search_includes_the_container_axes():
    skill = Placing()
    skill.ctx = SimpleNamespace(trace=None)
    target = SimpleNamespace(points=box((.245,.148,.012)))
    current = np.eye(4)
    current[:3,:3] = Rotation.from_euler('z',np.radians(7.)).as_matrix()
    options = skill.placements(target,current,(np.eye(4),[-.125,-.076],[.125,.076]),
                               [0,0,0],[],.002,.001,'flat')
    assert options     # 7 + k*15 degrees all miss this narrow feasible yaw band.


def test_insert_keeps_each_body_axis_and_both_ends():
    rotations = orientations(box((.03,.04,.05)), np.eye(4), np.eye(4), 'insert')
    aligned = {(int(np.argmax(np.abs(r[2]))), int(np.sign(r[2,np.argmax(np.abs(r[2]))])))
               for r in rotations}
    assert aligned == {(axis, sign) for axis in range(3) for sign in (-1,1)}


@pytest.mark.parametrize('bounded', [False, True])
def test_inside_resting_height_excludes_upright_bottle_but_allows_horizontal(monkeypatch, bounded):
    """A tall bottle fits a shallow open drawer only on its side."""
    from table_1000.experts.skills import semantic
    frame = np.eye(4)
    frame[:3, :3] = Rotation.from_euler('z', .43).as_matrix()
    frame[:3, 3] = [.37, -.18, .86]
    parts = [('bottom', box((.26, .28, .003)) - [0., 0., .0015])]
    parts += [('wall', box((.003, .28, .08025)) + [x, 0., .040125])
              for x in (-.13, .13)]
    shapes = [world_geometry.Shape('cabinet', 'drawer', name, frame,
                              points @ frame[:3, :3].T + frame[:3, 3])
              for name, points in parts]
    monkeypatch.setattr(world_geometry, 'scene_shapes', lambda *a, **kw: shapes)
    monkeypatch.setattr(regions, 'headroom', lambda *a: .2)
    region, _ = regions.inside_region('bottle', 'cabinet', 'drawer', orientation='insert',
                                      top_tolerance=.005 if bounded else None)(SimpleNamespace())
    target = SimpleNamespace(points=box((.0248, .0248, .0945)))
    skill = Placing()
    skill.ctx = SimpleNamespace(trace=None)
    options = skill.placements(target, frame, region, frame[:3, 3], [], .04, .002,
                               'insert', sweep=.15)
    assert options
    normal = region[0][:3, 2]
    vertical = [abs(normal @ pose[:3, 2]) > .999 for pose, _ in options]
    if bounded:
        assert not any(vertical)
        # The descent starts above the measured rim; only the resting top is
        # bounded, so this is not a fictitious roof blocking every insertion.
        assert region[3]['maximum_height'] == pytest.approx(.08525)
        for pose, _ in options:
            world = target.points @ pose[:3, :3].T + pose[:3, 3]
            assert ((world - region[0][:3, 3]) @ normal).max() <= .08525
    else:
        assert any(vertical)
        assert not all(vertical)
        assert region[3]['release_container'] == 'cabinet'


def test_bounded_drawer_uses_exposed_floor_and_measured_release_opening(monkeypatch):
    from table_1000.experts.skills import semantic
    frame = np.eye(4)
    parts = [('drawer', 'bottom', box((.26, .28, .003)) - [0., 0., .0015]),
             ('drawer', 'side', box((.003, .28, .08)) + [.13, 0., .04]),
             ('housing', 'shelf', box((.30, .20, .01)) + [0., .15, .085]),
             ('housing', 'roof', box((.30, .20, .01)) + [0., .15, .285])]
    shapes = [world_geometry.Shape('cabinet', body, name, frame, points)
              for body, name, points in parts]
    monkeypatch.setattr(world_geometry, 'scene_shapes', lambda *a, **kw: shapes)
    monkeypatch.setattr(regions, 'headroom', lambda *a: .2)
    ctx = SimpleNamespace(poses=lambda: {'tape:body': np.eye(4)})
    region, blocked = regions.inside_region('tape', 'cabinet', 'drawer', orientation='flat',
                                             top_tolerance=.005)(ctx)
    target = SimpleNamespace(obj='tape', body='tape:body', points=box((.06, .06, .02)))
    skill = Placing()
    skill.ctx = ctx
    options = skill.placements(target, np.eye(4), region, [0., .1, 0.], blocked, .02, .002, 'flat')
    assert options
    for pose, _ in options:
        points = target.points @ pose[:3, :3].T + pose[:3, 3]
        assert points[:, 1].max() <= .05 - .002 + 1e-9
    assert region[3]['maximum_release_top'] == pytest.approx(.08)
    release = skill.flat_release_geometry(target, region)
    assert release.wall_height == pytest.approx(.083)
    assert release.maximum == pytest.approx(.06)  # object top stays below the opening


def test_upright_candidates_keep_both_ends_and_screen_the_descent():
    skill = Placing()
    skill.ctx = SimpleNamespace(trace=None)
    target = SimpleNamespace(points=box((.012,.014,.14)))
    region = (np.eye(4),[-.02,-.02],[.02,.02])
    # A bridge above the resting pen does not obstruct its final pose, but
    # does obstruct a vertical descent through that space.
    bridge = box((.1,.1,.01)) + [0,0,.2]
    resting = skill.placements(target,np.eye(4),region,[0,0,0],[bridge],.01,.002,'upright')
    assert {int(np.sign(pose[2,2])) for pose,_ in resting} == {-1,1}
    assert not skill.placements(target,np.eye(4),region,[0,0,0],[bridge],.01,.002,'upright',sweep=.2)


def test_oriented_ik_prefix_alternates_the_insertion_ends():
    skill = Placing()
    skill.ctx = SimpleNamespace(trace=None)
    target = SimpleNamespace(points=box((.012,.014,.14)))
    options = skill.placements(target,np.eye(4),(np.eye(4),[-.02,-.02],[.02,.02]),
                               [0,0,0],[],.01,.002,'upright')
    # A pen initially upright strongly favours keeping that end down, but the
    # other end may be the only one compatible with the fingers above a rim.
    assert {int(np.sign(pose[2,2])) for pose,_ in options[:2]} == {-1,1}


def test_explicit_turns_keeps_the_full_insertion_yaw_grid():
    skill = Placing()
    skill.ctx = SimpleNamespace(trace=None)
    target = SimpleNamespace(points=box((.012,.014,.14)))
    arguments = target, np.eye(4), (np.eye(4),[-.03,-.03],[.03,.03]), [0,0,0], [], .1, 0., 'upright'
    default = skill.placements(*arguments)
    explicit = skill.placements(*arguments, turns=8)
    full = skill.placements(*arguments, turns=24)
    assert len(default) == len(explicit) == 16
    assert all(np.array_equal(a, b) and np.array_equal(x, y)
               for (a, x), (b, y) in zip(default, explicit))
    assert len(full) == 48
    for sign in (-1,1):
        rotations = [pose[:3,:3] for pose,_ in full if np.sign(pose[2,2]) == sign]
        assert len({np.round(r, 12).tobytes() for r in rotations}) == 24


def test_three_dimensional_keepout_leaves_space_under_an_overhang():
    skill = Placing()
    skill.ctx = SimpleNamespace(trace=None)
    target = SimpleNamespace(points=box((.012,.014,.04)))
    region = (np.eye(4),[-.02,-.02],[.02,.02])
    high = box((.1,.1,.01)) + [0,0,.2]
    assert skill.placements(target,np.eye(4),region,[0,0,0],[high],.01,.002,'upright')


@pytest.mark.parametrize('fixture', ['keepout-near-coplanar', 'descent-near-coplanar'])
def test_keepout_hull_handles_settled_nearly_coplanar_vertices(fixture):
    import json
    from pathlib import Path
    from mplib.collision_detection import fcl
    from table_1000.experts.skills.place.strict import convex_keepout
    points = np.array(json.loads((Path(__file__).parent/f'fixtures/{fixture}.json').read_text()))
    obstacle = convex_keepout(points)
    assert fcl.distance(obstacle, obstacle).min_distance <= 0.


def test_upright_inside_uses_same_measured_tilt_and_containment(monkeypatch):
    from table_1000.experts.skills import semantic
    pose = np.eye(4)
    pose[:3,:3] = Rotation.from_euler('y',np.pi/2).as_matrix()
    pose[2,3] = .025
    target = SimpleNamespace(points=box((.14,.01,.01)),body='rod:body')
    ctx = SimpleNamespace(skills=lambda:SimpleNamespace(target=lambda obj:target),
                          poses=lambda:{'rod:body':pose})
    wall = world_geometry.Shape('holder','body','wall',np.eye(4),box((.1,.1,.1)))
    monkeypatch.setattr(world_geometry,'scene_shapes',lambda ctx:[wall])
    monkeypatch.setattr(semantic,'_inside_point',lambda *args:True)
    assert semantic.upright_inside(ctx,'rod','holder')
    pose[:3,:3] = np.eye(3)
    assert not semantic.upright_inside(ctx,'rod','holder')
    pose[:3,:3] = Rotation.from_euler('y',np.pi/2).as_matrix()
    monkeypatch.setattr(semantic,'_inside_point',lambda *args:False)
    assert not semantic.upright_inside(ctx,'rod','holder')


def test_upright_inside_checks_immersed_segment_above_mouth(monkeypatch):
    from table_1000.experts.skills import semantic
    pose = np.eye(4)
    pose[:3,:3] = Rotation.from_euler('y',np.pi/2).as_matrix()
    pose[2,3] = .108
    target = SimpleNamespace(points=box((.15,.01,.01)),body='rod:body')
    ctx = SimpleNamespace(skills=lambda:SimpleNamespace(target=lambda obj:target),
                          poses=lambda:{'rod:body':pose})
    parts = [(box((.1,.1,.003)),[0,0,0])]
    parts += [(box((.003,.1,.1)),[x,0,.05]) for x in (-.05,.05)]
    parts += [(box((.1,.003,.1)),[0,y,.05]) for y in (-.05,.05)]
    shapes = [world_geometry.Shape('holder','body',str(i),np.eye(4),p+offset)
              for i,(p,offset) in enumerate(parts)]
    monkeypatch.setattr(world_geometry,'scene_shapes',lambda ctx:shapes)
    # The body centre is above the rim, but the lower end is well enclosed.
    assert semantic.upright_inside(ctx,'rod','holder')
    pose[2,3] = .17
    assert not semantic.upright_inside(ctx,'rod','holder')
    pose[2,3] = .108
    pose[0,3] = .1
    assert not semantic.upright_inside(ctx,'rod','holder')


def test_stack_region_uses_live_support_top(monkeypatch):
    from table_1000.experts.skills import semantic
    def shape(name, low, high):
        points = box(np.subtract(high,low)) + (np.asarray(high)+low)/2
        return world_geometry.Shape(name,name,name,np.eye(4),points)
    shapes = [shape('basket',[-.2,-.2,0],[.2,.2,.01]),
              shape('basket',[-.2,-.2,0],[-.19,.2,.15]),
              shape('basket',[.19,-.2,0],[.2,.2,.15]),
              shape('basket',[-.2,-.2,0],[.2,-.19,.15]),
              shape('basket',[-.2,.19,0],[.2,.2,.15]),
              shape('lower',[-.15,-.15,.01],[.15,.15,.03]),
              shape('upper',[.3,.3,.01],[.5,.5,.02])]
    monkeypatch.setattr(world_geometry,'scene_shapes',lambda ctx,exclude=():[s for s in shapes if s.obj not in exclude])
    ctx = SimpleNamespace(fail=lambda reason,**options:pytest.fail(reason))
    where = regions.inside_region('upper','basket',support='lower')
    first, _ = where(ctx)
    assert first[0][2,3] == pytest.approx(.03)
    shapes[-2] = shape('lower',[-.15,-.15,.04],[.15,.15,.06])
    second, _ = where(ctx)
    assert second[0][2,3] == pytest.approx(.06)
    assert np.allclose(first[1:],second[1:])


def test_flat_stack_allows_overhang_only_with_com_on_the_support():
    skill = Placing()
    skill.ctx = SimpleNamespace(trace=None)
    target = SimpleNamespace(points=box((.21,.161,.01)))
    frame = np.eye(4)
    support = dict(surface=(frame,[-.1225,-.074],[.1225,.074]),
                   centre_of_mass=np.array([.02,0.,0.]))
    # A5 is wider than slim along one edge, but fits in the container and
    # remains supported when the actual COM, rather than its box centre, fits.
    region = (frame,[-.145,-.10],[.145,.10],support)
    options = skill.placements(target,frame,region,[0,0,0],[],.01,.002,'flat')
    assert options
    for pose,_ in options:
        com = (pose @ np.r_[support['centre_of_mass'],1.])[:2]
        assert (np.abs(com) <= [.1205,.072]).all()
        world = target.points @ pose[:3,:3].T + pose[:3,3]
        assert (np.abs(world[:,:2]) <= [.143,.098]).all()
    support['centre_of_mass'] = np.array([1.,0.,0.])
    assert not skill.placements(target,frame,region,[0,0,0],[],.01,.002,'flat')


def test_flat_stack_uses_container_up_when_lower_body_is_flipped(monkeypatch):
    from table_1000.experts.skills import semantic
    shapes = [world_geometry.Shape('basket','body','floor',np.eye(4),box((.4,.3,.01))),
              world_geometry.Shape('lower','body','cover',np.diag([1.,-1.,-1.,1.]),
                             box((.245,.148,.02)) + [0,0,.015]),
              world_geometry.Shape('upper','body','cover',np.eye(4),box((.21,.161,.01)) + [1,0,0])]
    monkeypatch.setattr(world_geometry,'scene_shapes',lambda ctx,exclude=():[s for s in shapes if s.obj not in exclude])
    monkeypatch.setattr(semantic,'inside',lambda *args:True)
    target = SimpleNamespace(body='upper:body')
    ctx = SimpleNamespace(fail=lambda reason,**kw:pytest.fail(reason),
                          skills=lambda:SimpleNamespace(target=lambda obj:target),
                          poses=lambda:{target.body:np.eye(4)},
                          env=SimpleNamespace(center_of_mass=lambda body:np.array([[.02,0,0]])))
    region,_ = regions.inside_region('upper','basket',support='lower',orientation='flat')(ctx)
    assert region[0][2,3] == pytest.approx(.025)
    assert region[0][2,2] == pytest.approx(1.)
    np.testing.assert_allclose(region[3]['centre_of_mass'],[.02,0,0])
    shapes[1].points[:,2] += .05 * shapes[1].points[:,0]
    shapes[1].__post_init__()
    tilted,_ = regions.inside_region('upper','basket',support='lower',orientation='flat')(ctx)
    assert tilted[0][2,3] == pytest.approx(shapes[1].points[:,2].max())

    # A tilt must not collapse the usable top to its highest grid strip.
    np.testing.assert_allclose(tilted[3]['surface'][1:],region[3]['surface'][1:])


def test_oriented_motion_rejects_lost_hold_before_updating_attachments(monkeypatch):
    from table_1000.experts.skills.place import strict as placing
    monkeypatch.setattr(placing,'_held_area',lambda *args:0.)
    attached = []
    target = SimpleNamespace(obj='rod',body='rod:body',names=['rod:part'],
                             shape=None,offset=np.eye(4))
    solver = SimpleNamespace(tcp=lambda:np.eye(4),finger_gap=lambda:.01,
                             attach=lambda *args:attached.append(args))
    def fail(reason,**details):
        raise RuntimeError(reason)
    skill = Placing()
    skill.ctx = SimpleNamespace(trace=None,poses=lambda:{target.body:np.eye(4)},fail=fail)
    skill.solvers = {'left':solver}
    skill.held = {'left':{'target':target, 'grasp':{'open_gap':.03}}}
    skill.hands = {'left':None}
    skill.pad_closure = lambda *args: {'held':False}
    with pytest.raises(RuntimeError,match='pick it again'):
        skill.lower('left',np.eye(4),np.zeros(3),np.eye(4),.01,.15,False,
                    {'ik_checked':0},'upright')
    assert not attached


def test_oriented_low_projected_area_keeps_measured_opposing_pad_contact(monkeypatch):
    from table_1000.experts.skills.place import strict as placing
    monkeypatch.setattr(placing, '_held_area', lambda *args: 0.)
    attached = []
    target = SimpleNamespace(obj='rod', body='rod:body', names=['rod:part'],
                             shape=None, offset=np.eye(4))
    solver = SimpleNamespace(tcp=lambda:np.eye(4), finger_gap=lambda:.01,
                             attach=lambda *args:attached.append(args),
                             world=SimpleNamespace(objects={'rod:part':SimpleNamespace(body=target.body)}))
    skill = Placing()
    skill.ctx = SimpleNamespace(trace=None, poses=lambda:{target.body:np.eye(4)}, note=lambda *a,**k:None)
    skill.solvers, skill.hands = {'left':solver}, {'left':None}
    skill.held = {'left':{'target':target, 'grasp':{'open_gap':.03}}}
    skill.pad_closure = lambda *args: {'held':True}
    skill.down_goals = lambda *a,**k: []
    assert not skill.lower('left', np.eye(4), np.zeros(3), np.eye(4), .002, .15, False,
                           {'ik_checked':0}, 'upright')
    assert len(attached) == 1


def test_insertion_channel_reports_the_pen_above_the_release(monkeypatch):
    skill = Placing()
    shape = (np.eye(4), 'box', (.01, .01, .01), False)
    skill.hands = {'left':SimpleNamespace(shapes={'finger':[shape]}, frames=lambda gap:{'finger':np.eye(4)})}
    from table_1000.experts.skills.place.strict import convex_keepout
    blocker = box((.015,.015,.015)) + [0,0,.1]
    obstacle = ('other_pen:barrel',convex_keepout(blocker),(blocker.min(0),blocker.max(0)))
    assert skill.insertion_blockers('left',np.eye(4),.2,.02,[obstacle]) == ['other_pen:barrel']
    release = np.eye(4);release[0,3] = .1
    assert skill.insertion_blockers('left',release,.2,.02,[obstacle]) == []
    assert len(skill._insertion_channels) == 1


@pytest.mark.parametrize('maximum, expected', [(.08, .026), (.025, .025)])
def test_flat_place_leaves_release_margin_within_the_maximum(maximum, expected):
    skill = Placing()
    skill.ctx = SimpleNamespace(trace=None)
    skill.held = {'left':{'target':SimpleNamespace(points=box((.21,.16,.01))), 'grasp':{'open_gap':.05}}}
    skill.solvers = {'left':SimpleNamespace(finger_gap=lambda:.02)}
    skill.hands = {'left':object()}
    geometry = SimpleNamespace(minimum=lambda *a,**kw:.024,maximum=maximum,blocked_by=set())
    calls=[]
    def lower(arm,placed,spot,tcp,clearance,above,lift,result,orientation=None):
        calls.append(clearance)
        return clearance >= .024 - 1e-9
    skill.lower=lower
    result=skill.put_down('left',[(np.eye(4),np.zeros(3))],np.eye(4),.02,.08,'flat',geometry)
    assert calls == pytest.approx([expected])
    assert result['release_clearance_m'] == pytest.approx(expected)


def test_insertion_need_targets_live_floor_without_raised_release_candidates(monkeypatch):
    skill = Placing()
    target = SimpleNamespace(points=box((.01,.01,.14)),body='rod:body')
    skill.ctx = SimpleNamespace(trace=None, poses=lambda:{target.body:np.eye(4)})
    skill.target = lambda obj:target
    skill.arms,skill.held=['left'],{}
    expected=np.eye(4);expected[2,3]=.75
    skill.placements=lambda *a,**k:[(expected,np.array([0,0,.75]))]
    skill.restore_need=lambda recipe:recipe
    recipe=skill.need('rod',(np.eye(4),[-.1,-.1],[.1,.1]),orientation='upright')
    assert len(recipe['options']) == 1
    np.testing.assert_array_equal(recipe['options'][0][0],expected)
    assert recipe['clearance'] == pytest.approx(.002)


def test_insertion_depth_uses_measured_end_and_live_floor(monkeypatch):
    from table_1000.experts.skills import semantic
    skill = Placing()
    target = SimpleNamespace(obj='rod', body='rod:body', points=box((.01,.01,.14)))
    floor = np.eye(4);floor[2,3]=.83
    pose = np.eye(4);pose[2,3]=.902
    events=[]
    def fail(reason, **details):
        assert details['recoverable']
        raise RuntimeError(reason)
    skill.ctx=SimpleNamespace(poses=lambda:{target.body:pose},note=lambda *a,**k:events.append(k),fail=fail)
    monkeypatch.setattr(world_geometry,'scene_shapes',lambda ctx,exclude:[])
    monkeypatch.setattr(world_geometry,'support_region',lambda *a,**k:((floor,[-.1,-.1],[.1,.1]),[]))
    skill.check_insertion_depth(target,'holder',None,.002)
    assert events[-1]['floor_gap_m'] == pytest.approx(.002)
    pose[2,3]+=.02
    with pytest.raises(RuntimeError,match='above container floor'):
        skill.check_insertion_depth(target,'holder',None,.002)
    floor[2,3]+=.02
    skill.check_insertion_depth(target,'holder',None,.002)


def test_flat_release_rechecks_measured_bottom_against_live_support(monkeypatch):
    from table_1000.experts.skills import semantic
    from table_1000.experts.context import Unreachable
    skill = Placing()
    target = SimpleNamespace(obj='book', body='book:body', points=box((.21,.16,.01)))
    frame = np.eye(4);frame[2,3]=.8
    pose = np.eye(4);pose[2,3]=.807
    events=[]
    def fail(reason, kind=RuntimeError, **details):
        assert details['recoverable']
        raise kind(None, reason, **details)
    skill.ctx = SimpleNamespace(poses=lambda:{target.body:pose},
        env=SimpleNamespace(center_of_mass=lambda body:(pose[:3,3],0.)),
        note=lambda *a,**k:events.append(k),fail=fail)
    skill.hands={'left':object()};skill.held={'left':{'grasp':{'open_gap':.05}}}
    skill.solvers={'left':SimpleNamespace(tcp=lambda:np.eye(4),finger_gap=lambda:.02)}
    skill.flat_release_geometry=lambda *a:SimpleNamespace(minimum=lambda *a,**k:.002,maximum=.03,blocked_by=set())
    monkeypatch.setattr(regions,'inside_region',lambda *a,**k:
        lambda ctx:((frame,[-.1,-.1],[.1,.1]),[]))
    region = (np.eye(4),[-.1,-.1],[.1,.1],
              dict(flat_container='basket',part=None,support='lower'))
    skill.check_flat_release(target,region,'left',.08)
    assert events[-1]['support_gap_m'] == pytest.approx(.002)
    pose[2,3]+=.04
    with pytest.raises(Unreachable,match='violates measured.*pick again'):
        skill.check_flat_release(target,region,'left',.08)
    frame[2,3]+=.04
    skill.check_flat_release(target,region,'left',.08)
    pose[:3,:3]=Rotation.from_euler('y',np.radians(6.)).as_matrix()
    pose[2,3]=frame[2,3]+.002-(target.points@pose[:3,:3].T)[:,2].min()
    with pytest.raises(Unreachable) as level:
        skill.check_flat_release(target,region,'left',.08)
    assert level.value.details['tilt_deg']>3.
    pose[:3,:3]=np.eye(3);pose[2,3]=frame[2,3]+.007;pose[0,3]=.2
    with pytest.raises(Unreachable) as outside:
        skill.check_flat_release(target,region,'left',.08)
    assert not outside.value.details['com_inside']


def test_insertion_occupancy_ranks_existing_pen_with_three_dimensional_bounds():
    skill = Placing()
    shape = (np.eye(4), 'box', (.01, .01, .01), False)
    skill.hands = {'left': SimpleNamespace(shapes={'finger': [shape]}, frames=lambda gap: {'finger': np.eye(4)})}
    blocker = box((.015, .015, .015)) + [0., 0., .1]
    obstacle = ('existing_pen:cap', None, (blocker.min(0), blocker.max(0)))
    blocked = skill.insertion_occupancy('left', np.eye(4), .02, [(np.eye(4), None)], .2, [obstacle])
    side = np.eye(4); side[0, 3] = .1
    clear = skill.insertion_occupancy('left', side, .02, [(np.eye(4), None)], .2, [obstacle])
    high = np.eye(4); high[2, 3] = -.3
    above = skill.insertion_occupancy('left', high, .02, [(np.eye(4), None)], .2, [obstacle])
    assert blocked > clear == above == 0.
