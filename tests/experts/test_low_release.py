"""A bounded gravity fallback cannot generate high drops or spill carried contents."""
from types import SimpleNamespace

import numpy as np
import pytest

from table_1000.experts.context import ExpertFailed, Unreachable
from table_1000.experts.skills.chain import InHand
from table_1000.experts.skills.common import Target, upright_need
from table_1000.experts.skills.placing import Placing
from table_1000.scenes.geometry import AssetGeometry


def setup_release(monkeypatch, limit=.05):
    from table_1000.experts.skills import semantic
    skill = Placing()
    frame = np.eye(4)
    region = (frame, np.array([-.15, -.02]), np.array([.15, .02]),
              {'maximum_height': .05, 'release_container': 'cabinet', 'release_part': 'drawer'})
    points = np.array([[x, y, z] for x in [-.005,.005] for y in [-.005,.005] for z in [0.,.15]])
    target = SimpleNamespace(obj='tool', body='tool:body', points=points, container=False)
    skill.target = lambda obj: target
    skill.ctx = SimpleNamespace(geometry={'tool':SimpleNamespace(max_drop_m=limit)}, trace=None,
        poses=lambda: {'tool:body':frame}, env=SimpleNamespace(center_of_mass=lambda body:np.array([[0.,0.,.075]])))
    skill.placements = lambda *a, **kw: [(frame, np.zeros(3))]
    shapes = []
    monkeypatch.setattr(semantic, 'scene_shapes', lambda *a, **kw: shapes)
    monkeypatch.setattr(semantic, 'support_region', lambda *a, **kw: ((frame,[-.15,-.15],[.15,.15]), []))
    return skill, target, region, shapes


def test_candidates_obey_default_and_authored_drop_limit(monkeypatch):
    skill, target, region, _ = setup_release(monkeypatch, .013)
    candidates = skill.low_release_options(target, region, np.zeros(3), [], .02, .002)
    assert candidates
    assert all(0. < gap <= .013 for _, _, gap in candidates)
    assert max(gap for _, _, gap in candidates) == .013


def test_fifteen_centimetre_limit_keeps_the_lower_release_levels(monkeypatch):
    skill, target, region, _ = setup_release(monkeypatch, .15)
    candidates = skill.low_release_options(target, region, np.zeros(3), [], .02, .002)
    assert {gap for _, _, gap in candidates} == {.002, .01, .025, .05, .10, .15}


def test_free_fall_before_tipping_cannot_clear_a_nearby_object(monkeypatch):
    skill, target, region, shapes = setup_release(monkeypatch, .15)
    # Outside the sphere at the initial 15 cm pivot, inside its descent sweep.
    shapes.append(SimpleNamespace(obj='loose', points=np.array([[.14,0.,.002],[.145,.005,.007]])))
    assert not skill.low_release_options(target, region, np.zeros(3), [], .02, .002)


def test_elevated_expert_plane_is_not_a_low_drop(monkeypatch):
    skill, target, region, _ = setup_release(monkeypatch)
    elevated = region[0].copy();elevated[2,3] = .3
    assert not skill.low_release_options(target, (elevated, *region[1:]), np.zeros(3), [], .02, .002)


def test_other_objects_in_tipping_envelope_are_excluded(monkeypatch):
    skill, target, region, shapes = setup_release(monkeypatch)
    shapes.append(SimpleNamespace(obj='loose', points=np.array([[.03,0.,.01],[.04,.01,.04]])))
    assert not skill.low_release_options(target, region, np.zeros(3), [], .02, .002)


def test_loaded_container_fails_before_any_motion_or_release():
    skill = Placing()
    skill.solvers = {'left':SimpleNamespace()}
    skill.held = {'left':{'riders':['pen:body']}}
    def fail(reason, exception, **details):
        raise exception('place', reason, **details)
    skill.ctx = SimpleNamespace(fail=fail)
    with pytest.raises(Unreachable, match='contents'):
        skill.low_release('cup', (), 'left', None, (), .02, .01, 'strict no room')


def test_model_drop_limit_defaults_to_fifteen_centimetres():
    assert AssetGeometry({'nodes': []}).max_drop_m == .15
    assert AssetGeometry({'nodes': [], 'max_drop_m': .05}).max_drop_m == .05
    assert AssetGeometry({'nodes': [], 'max_drop_m': .012}).max_drop_m == .012
    assert AssetGeometry({'nodes': [], 'max_drop_m': 0.}).max_drop_m == 0.


@pytest.mark.parametrize('limit', [float('nan'), float('inf'), -.01])
def test_invalid_asset_drop_limits_are_rejected(limit):
    with pytest.raises(ValueError, match='finite nonnegative'):
        AssetGeometry({'nodes': [], 'max_drop_m': limit})


def measured_release(measured_drop):
    skill = Placing()
    pose = np.eye(4)
    target = Target('tool', ['tool:body'], 'tool:body', None, np.eye(4), np.zeros((1, 3)))
    opened, notes = [], []
    def fail(reason, exception=ExpertFailed, **details):
        raise exception('place', reason, **details)
    skill.ctx = SimpleNamespace(geometry={'tool': SimpleNamespace(max_drop_m=.05)},
        scene=SimpleNamespace(OBJECTS=[]),
        poses=lambda: {'tool:body': pose}, fail=fail, spec=SimpleNamespace(base=False),
        note=lambda note, **details: notes.append((note, details)), hold=lambda seconds: None)
    skill.target = lambda obj: target
    skill.solvers = {'left': SimpleNamespace(tcp=lambda: np.eye(4), attach=lambda *args: None,
        world=SimpleNamespace(objects={'tool:body': SimpleNamespace(body='tool:body')}))}
    skill.held = {'left': {'target': target}}
    skill.low_release_options = lambda *args: [(np.eye(4), np.zeros(3), .002)]
    skill.release_floor = lambda region: (region[0], [-1., -1.], [1., 1.])
    def lower(*args):
        pose[2, 3] = measured_drop
        args[-2].update(done=True, spot=np.zeros(3))
        return True
    skill.lower = lower
    skill.release = lambda *args, **kwargs: opened.append(True)
    mock_opening(skill)
    return skill, opened, notes


def mock_opening(skill):
    skill.hands = {'left': object()}
    skill.solvers['left'].finger_gap = lambda: .02
    skill.held['left']['grasp'] = {'open_gap': .04}
    geometry = SimpleNamespace(minimum=lambda *args, **kwargs: 0., blocked_by=set())
    skill.flat_release_geometry = lambda *args: geometry
    return geometry


def test_independent_release_checks_live_opening_before_using_relaxation():
    skill, opened, notes = measured_release(.025)
    geometry = mock_opening(skill)
    def blocked(hand, current, hold, closed, opened, upward, **kwargs):
        assert hand is skill.hands['left'] and closed == .02 and opened == .04
        if current[2, 3] == .002:  # nominal candidate cleared, live hold shifted
            return 0.
        assert current[2, 3] == .025 and hold[2, 3] == .025
        assert upward == 0. and kwargs['backoff'] == 0. and kwargs['heights'] == (0.,)
        kwargs['check_budget']()
        geometry.blocked_by.add('cabinet:drawer.rim')
        return None
    geometry.minimum = blocked
    with pytest.raises(Unreachable, match='no reachable bounded gravity release') as failure:
        skill.low_release('tool', (np.eye(4),), 'left', None, (), .02, .01, 'strict failed')
    assert failure.value.details['last'] == 'bounded gravity release opening is blocked'
    assert not opened and not any(note == 'placement relaxation used' for note, _ in notes)


def test_blocked_opening_skips_a_candidate_and_still_tries_the_next_one():
    skill, opened, notes = measured_release(.025)
    geometry = mock_opening(skill)
    skill.low_release_options = lambda *args: [(np.eye(4), np.zeros(3), gap) for gap in (.002, .025)]
    geometry.minimum = lambda hand, placed, *args, **kwargs: None if placed[2, 3] == .002 else 0.
    planned = []
    lower = skill.lower
    def record(*args):
        planned.append(args[4])
        return lower(*args)
    skill.lower = record
    result = skill.low_release('tool', (np.eye(4),), 'left', None, (), .02, .01, 'strict failed')
    assert planned == [.025] and opened and result['drop_m'] == .025


def test_live_opening_blocker_retains_the_object_and_continues_search():
    skill, opened, notes = measured_release(.025)
    geometry = mock_opening(skill)
    skill.low_release_options = lambda *args: [(np.eye(4), np.zeros(3), gap) for gap in (.002, .025)]
    live = skill.ctx.poses()['tool:body']
    moved = []
    def lower(*args):
        moved.append(args[4])
        live[2, 3] = .01 if len(moved) == 1 else .025
        args[-2].update(done=True, spot=np.zeros(3))
        assert not opened
        return True
    skill.lower = lower
    geometry.minimum = lambda hand, placed, *args, **kwargs: None if placed[2, 3] == .01 else 0.
    result = skill.low_release('tool', (np.eye(4),), 'left', None, (), .02, .01, 'strict failed')
    assert moved == [.002, .025] and opened and result['drop_m'] == .025
    assert any(note == 'next placement' and details['stage'] == 'opening' for note, details in notes)


def test_flat_release_retains_its_strict_level_com_and_clearance_checks():
    skill, opened, notes = measured_release(.025)
    with pytest.raises(Unreachable, match='does not replace flat release checks'):
        skill.low_release('tool', (np.eye(4),), 'left', None, (), .02, .01, 'strict failed', 'flat')
    assert not opened and not notes
    strict = InHand('tool', None, ['left'], recipe={'origin': {'region': (np.eye(4),), 'orientation': 'flat'}})
    assert skill.low_release_need(strict) is None


@pytest.mark.parametrize('floor_z,accepted', [(-.03, False), (.03, False), (-.01, True)])
def test_floor_motion_is_measured_after_the_release_plan(floor_z, accepted):
    skill, opened, notes = measured_release(.025)
    floor = np.eye(4)
    lower = skill.lower
    def moving_floor(*args):
        result = lower(*args)
        floor[2, 3] = floor_z
        return result
    skill.lower = moving_floor
    skill.release_floor = lambda region: (floor, [-1., -1.], [1., 1.])
    if accepted:
        result = skill.low_release('tool', (np.eye(4),), 'left', None, (), .02, .01, 'strict failed')
        assert opened and result['drop_m'] == pytest.approx(.025-floor_z)
        used = next(details for note, details in notes if note == 'placement relaxation used')
        assert used['drop_m'] == pytest.approx(.025-floor_z)
    else:
        with pytest.raises(Unreachable, match='drop limit'):
            skill.low_release('tool', (np.eye(4),), 'left', None, (), .02, .01, 'strict failed')
        assert not opened


def test_missing_live_floor_cannot_authorize_opening():
    skill, opened, _ = measured_release(.025)
    skill.release_floor = lambda region: None
    with pytest.raises(Unreachable, match='no live container floor'):
        skill.low_release('tool', (np.eye(4),), 'left', None, (), .02, .01, 'strict failed')
    assert not opened


def test_live_floor_measurement_preserves_the_original_final_height_frame():
    skill, opened, _ = measured_release(.025)
    floor = np.eye(4);floor[2, 3] = .01
    skill.release_floor = lambda region: (floor, [-1., -1.], [1., 1.])
    region = (np.eye(4), None, None, {'maximum_height': .02})
    with pytest.raises(ExpertFailed, match='above target height'):
        skill.low_release('tool', region, 'left', None, (), .02, .01, 'strict failed')
    assert opened


@pytest.mark.parametrize('measured_drop,upright_ok', [
    (-.001, True), (.051, True), (.025, True), (.025, False),
    (float('nan'), True), (float('inf'), True)])
def test_live_drop_is_checked_before_opening_and_recorded(measured_drop, upright_ok, monkeypatch):
    """A planned gap cannot authorize opening after the payload has shifted."""
    skill, opened, notes = measured_release(measured_drop)
    if not upright_ok:
        from table_1000.experts.skills import semantic
        monkeypatch.setattr(semantic, 'upright_inside', lambda *args: False)
        region = (np.eye(4), None, None, {'upright_container': 'cabinet', 'part': 'drawer'})
        with pytest.raises(ExpertFailed, match='not upright inside'):
            skill.low_release('tool', region, 'left', None, (), .02, .01, 'strict failed', 'upright')
        assert opened
    elif 0. <= measured_drop <= .05:
        result = skill.low_release('tool', (np.eye(4),), 'left', None, (), .02, .01, 'strict failed')
        assert opened and result['drop_m'] == pytest.approx(measured_drop)
        used = next(details for note, details in notes if note == 'placement relaxation used')
        assert used['level'] == 1 and used['drop_m'] == pytest.approx(measured_drop)
    else:
        with pytest.raises(Unreachable, match='drop limit'):
            skill.low_release('tool', (np.eye(4),), 'left', None, (), .02, .01, 'strict failed')
        assert not opened


@pytest.mark.parametrize('maximum,accepted', [(.01, False), (.03, True)])
def test_relaxing_release_height_retains_the_final_resting_height_limit(maximum, accepted):
    skill, opened, _ = measured_release(.025)
    region = (np.eye(4), None, None, {'maximum_height': maximum})
    if accepted:
        skill.low_release('tool', region, 'left', None, (), .02, .01, 'strict failed')
    else:
        with pytest.raises(ExpertFailed, match='above target height'):
            skill.low_release('tool', region, 'left', None, (), .02, .01, 'strict failed')
    assert opened


def test_base_search_covers_late_hanging_poses_with_the_existing_budget():
    skill = Placing()
    pose = np.eye(4);pose[2, 3] = .025
    target = Target('tool', ['tool:body'], 'tool:body', None, np.eye(4), np.zeros((1, 3)))
    def fail(reason, exception, **details):
        raise exception('place', reason, **details)
    skill.ctx = SimpleNamespace(geometry={'tool': SimpleNamespace(max_drop_m=.05)}, trace=None,
        scene=SimpleNamespace(OBJECTS=[]),
        poses=lambda: {'tool:body': pose}, fail=fail, spec=SimpleNamespace(base=True),
        note=lambda *args, **details: None, hold=lambda seconds: None)
    skill.target = lambda obj: target
    skill.solvers = {'left': SimpleNamespace(tcp=lambda: np.eye(4), attach=lambda *args: None,
        world=SimpleNamespace(objects={'tool:body': SimpleNamespace(body='tool:body')}))}
    skill.held = {'left': {'target': target}}
    skill.release_floor = lambda region: (region[0], [-1., -1.], [1., 1.])
    options = []
    for index in range(200):
        placed = np.eye(4)
        if index >= 150:
            placed[:3, :3] = np.diag([1., -1., -1.])
        options.append((placed, np.zeros(3), (.002, .01, .025, .05)[index % 4]))
    skill.low_release_options = lambda *args: options
    searched = []
    def lower(*args):
        if searched:
            args[-2].update(done=True, spot=np.zeros(3))
        return bool(searched)
    skill.lower = lower
    skill.make_room = lambda groups, *args, **kwargs: searched.extend(groups)
    skill.release = lambda *args, **kwargs: None
    mock_opening(skill)
    skill.low_release('tool', (np.eye(4),), 'left', None, (), .02, .01, 'strict failed')
    assert len(searched) == 10
    assert any(group[0][2, 2] < -.99 for group in searched)
    assert any(group[0][2, 2] > .99 for group in searched)


def test_live_contents_are_protected_even_when_part_pick_registered_no_riders(monkeypatch):
    from table_1000.experts.skills import semantic
    skill = Placing()
    skill.solvers = {'left': SimpleNamespace()}
    skill.held = {'left': {'riders': []}}
    def fail(reason, exception, **details):
        raise exception('place', reason, **details)
    skill.ctx = SimpleNamespace(trace=None, fail=fail, scene=SimpleNamespace(OBJECTS=['cup', 'pen']))
    monkeypatch.setattr(semantic, 'inside', lambda ctx, obj, container: obj == 'pen' and container == 'cup')
    with pytest.raises(Unreachable, match='contents'):
        skill.low_release('cup', (), 'left', None, (), .02, .01, 'strict no room')


@pytest.mark.parametrize('convex,container', [(True, False), (False, False), (False, True)])
def test_hanging_release_requests_end_grasps_for_slender_noncontainers(monkeypatch, convex, container):
    from table_1000.experts.skills import targets
    skill, _, region, _ = setup_release(monkeypatch)
    skill.ctx.geometry['tool'].solid = lambda *args, **kwargs: convex
    monkeypatch.setattr(targets, 'rim_sections', lambda s, obj, target, geometry: setattr(target, 'container', container))
    skill.ctx.requirements = []
    skill.insertion_obstacles = lambda *args: []
    strict = InHand('tool', None, ['left'], recipe={'origin': dict(
        region=region, near=np.zeros(3), keep_out=[], step=.02, margin=.002)})
    relaxed = skill.low_release_need(strict)
    assert relaxed is not None
    assert relaxed.hands is None  # independent pickup and placement stances
    assert upright_need(relaxed) == (not container)
    strict.recipe['hanging'] = False
    assert not upright_need(skill.low_release_need(strict))


@pytest.mark.parametrize('already_held', [False, True])
def test_pick_fallback_retains_the_release_hold_constraint_and_never_retries_a_held_object(monkeypatch, already_held):
    from table_1000.experts import skills
    calls = []
    target = SimpleNamespace(rim=False, container=False)
    accepts = lambda arm, hold: arm == 'left'
    need = InHand('tool', accepts, ['left'], recipe={'gravity_release': True, 'hanging': True})
    def pick(*args):
        wanted = args[-2]
        calls.append(wanted)
        if wanted.recipe['hanging']:
            raise Unreachable('pick', 'no reachable grasp')
        assert wanted.accepts is accepts and wanted.arms == ['left']
        return {'arm': 'left'}
    fake = SimpleNamespace(target=lambda obj: target, pick=pick,
                           holding=lambda obj: 'left' if already_held else None)
    monkeypatch.setattr(skills, '_skills', lambda ctx: fake)
    monkeypatch.setattr(skills, '_target_attempts', lambda *args: args[-2]([target]))
    ctx = SimpleNamespace(trace=None, note=lambda *args, **kwargs: None)
    if already_held:
        with pytest.raises(Unreachable):
            skills.pick(ctx, 'tool', need=need)
        assert len(calls) == 1
    else:
        assert skills.pick(ctx, 'tool', need=need) == {'arm': 'left'}
        assert len(calls) == 2 and not calls[-1].recipe['hanging']
