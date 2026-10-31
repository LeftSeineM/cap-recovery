import types

import numpy as np
import pytest

from table_1000.experts.skills.hold.frames import Box, gravity_lever, pinch_normal_force
from table_1000.experts.skills.common import Target
from table_1000.robots import load_spec


@pytest.fixture(autouse=True)
def isolated_placement_cache(monkeypatch):
    """Ordering tests replace geometric candidates and use sentinel hands."""
    from table_1000.experts.skills.hold.pick import Grasping
    monkeypatch.setattr(Grasping, '_grasp_placement_cache', lambda *args, **kwargs: {})


def test_gravity_moment_uses_com_and_clamping_line_not_mass():
    assert gravity_lever(np.zeros(3), np.array([.17, 0, 0]), [0, 1, 0]) == pytest.approx(.17)
    assert gravity_lever(np.zeros(3), np.array([0, .17, 0]), [0, 1, 0]) == 0
    assert gravity_lever(np.zeros(3), np.array([.17, 0, 0]), [0, 0, 1]) == 0


def test_closing_force_uses_existing_drive_and_pivot_virtual_work():
    side = load_spec('x1pro_rm001').arms['right']
    angle = abs(next(iter(side.finger_targets(.02).values())))
    closed = abs(next(iter(side.finger_targets(0.).values())))
    expected = 2 * 50 * closed / .14  # pads parallel at this opening; lever is finger length
    assert angle < 1e-12
    assert pinch_normal_force(side, .02, [50., 50.], [20., 20.]) == pytest.approx(expected)
    assert pinch_normal_force(side, .02, [50., 50.], [1., 1.]) == pytest.approx(2 / .14)
    assert pinch_normal_force(side, 0., [50., 50.], [20., 20.]) == 0.


def test_joint_constraint_does_not_use_free_suspension_center_or_torque():
    from table_1000.experts.skills.hold.pick import Grasping
    skill = Grasping()
    target = types.SimpleNamespace(obj='object', body='object:body')
    structure = types.SimpleNamespace(articulated=False, root='body')
    skill.ctx = types.SimpleNamespace(env=types.SimpleNamespace(
        objects={'object': structure}, center_of_mass=lambda body: np.array([[.3, .2, .1]])))
    assert np.allclose(skill.grasp_center_of_mass(target, None), [.3, .2, .1])
    structure.articulated = True
    assert np.allclose(skill.grasp_center_of_mass(target, None), [.3, .2, .1])  # a free articulation root
    structure.root = 'housing'
    assert skill.grasp_center_of_mass(target, None) is None
    assert skill.constrained(target)


def test_cost_sorted_candidates_do_not_stop_at_an_out_of_angle_candidate(monkeypatch):
    import table_1000.experts.skills.hold.pick as grasping
    from table_1000.experts.skills.hold.frames import Grasp
    skill = grasping.Grasping()
    skill.braced, skill.brace_seed = False, None
    skill.stow = lambda **kwargs: False
    shape = Box((.03,) * 3)
    target = Target('object', ['object:body.part'], 'object:body', shape, np.eye(4), shape.points())
    candidates = [Grasp(np.eye(4), .04, .03, 0., cost,
                         {'approach_deg': angle, 'closest_deg': 0.}, choice=i)
                  for i, (cost, angle) in enumerate(
                      [*[(float(i), 30.) for i in range(grasping.ROOM_PLACES + 1)],
                       (float(grasping.ROOM_PLACES + 2), 0.)])]
    centres = []
    calls = []
    def generated(*args, **kw):
        centres.append(kw['mass_center'])
        limit = kw.get('max_angle')
        calls.append(limit)
        return iter(candidates if limit is None else
                    [item for item in candidates if item.info['approach_deg'] <= np.degrees(limit) + 1e-6])
    monkeypatch.setattr(grasping, 'grasp_candidates', generated)
    side = types.SimpleNamespace(finger_targets=lambda gap: {})
    skill.ctx = types.SimpleNamespace(geometry={}, poses=lambda: {'object:body': np.eye(4)},
        spec=types.SimpleNamespace(arms={'right': side}),
        env=types.SimpleNamespace(objects={'object': types.SimpleNamespace(articulated=False)},
                                 center_of_mass=lambda body: np.zeros((1, 3)), body_mass=lambda body: [.1]))
    skill.hands = {'right': None}
    skill.shoulder = lambda arm: np.array([1., 0., 1.])
    skill.obstacles = lambda *args: ([], [])
    skill.model = types.SimpleNamespace(vector=lambda values: np.zeros(1),
                   ik=lambda *args, **kw: ('Success', np.zeros(1)))
    skill.solvers = {'right': types.SimpleNamespace(sync=lambda: None, allow=lambda *args: None)}
    skip = set()
    choices = list(skill.options(target, ['right'], skip, max_angle=np.radians(15)))
    assert len(choices) == 1 and choices[0][1].info['approach_deg'] == 0.
    assert skip == {('right', grasping.ROOM_PLACES + 1)}  # geometry id survives phase filtering
    assert list(skill.options(target, ['right'], skip, max_angle=np.radians(15))) == []
    assert calls == [np.radians(15)] * 6  # angle probe, held lift, movable lift, twice
    assert len(skill.grasp_paths(target, ['right'], max_angle=np.radians(15))['right']) == 1
    assert np.allclose(centres[-1], [0., 0., 0.])
    # A pressure-drag preflight evaluates a future pose, whose measured COM
    # has not moved yet. Its gravity score must use the predicted COM.
    list(skill.options(target, ['right'], set(), mass_center=[.2, -.1, .4]))
    assert np.allclose(centres[-1], [.2, -.1, .4])
    # A preparation's allowance spans successive virtual-depth streams and
    # stops before launching another IK; default retry ids above are intact.
    charged, ik_calls = [], []
    class Exhausted(Exception):
        pass
    def charge(arm, lift):
        if len(charged) == 2:
            raise Exhausted()
        charged.append((arm, lift))
    skill.model.ik = lambda *args, **kw: (ik_calls.append(args) or ('Success', np.zeros(1)))
    for _ in range(2):
        list(skill.options(target, ['right'], set(), max_angle=np.radians(15), candidate_budget=charge))
    with pytest.raises(Exhausted):
        list(skill.options(target, ['right'], set(), max_angle=np.radians(15), candidate_budget=charge))
    assert len(ik_calls) == 4  # grasp and pre-grasp for each of the two charged candidates
    # A nearby virtual grasp can reuse its arm branch, while the torso of a
    # fixed-lift query and unrelated joints retain the observed configuration.
    skill.model.vector = lambda values: np.array([.1, .2, .3, .4])
    skill.model.arm_indices = lambda arm: [1, 2]
    skill.model.lift_indices = lambda: [0]
    skill.model.ik = lambda arm, pose, q, **kw: (ik_calls.append(q.copy()) or ('Success', q.copy()))
    hint = np.array([1., 2., 3., 4.])
    for lift in (False, True):
        ik_calls.clear()
        tried = {False: grasping.MAX_PLANS if lift else 0, True: 0}
        list(skill.options(target, ['right'], set(), max_angle=np.radians(15), ik_seed=hint, tried=tried))
        np.testing.assert_array_equal(ik_calls[0], [1. if lift else .1, 2., 3., .4])
    # Reusing geometric previews retains angular filtering, both IK checks,
    # supports and the original candidate ids without placing geometry again.
    calls.clear()
    ik_calls.clear()
    skip = set()
    choices = list(skill.options(target, ['right'], skip, max_angle=np.radians(15),
        geometric={'right': (['table'], iter(candidates))}))
    assert len(choices) == 1 and choices[0][2] == ['table']
    assert skip == {('right', grasping.ROOM_PLACES + 1)}
    assert calls == [] and len(ik_calls) == 2
    # A cabinet brace cannot try the movable-lift phase: its fixed-lift pass
    # must retain the full requested angle range instead of dropping the
    # farther candidates normally deferred to the movable-lift pass.
    skill.braced = True
    angled = candidates[-1]
    angled.info['approach_deg'] = 18.
    choices = list(skill.options(target, ['right'], set(), max_angle=np.radians(20),
        geometric={'right': ([], iter([angled]))}))
    assert len(choices) == 1 and choices[0][-1] is False
    skill.braced = False
    angled.info['approach_deg'] = 0.
    candidates.pop()  # no in-angle choice: avoid constructing the full stream
    calls.clear()
    assert list(skill.options(target, ['right'], set(), max_angle=np.radians(15))) == []
    assert calls == [np.radians(15)]


@pytest.mark.parametrize('with_need', [False, True])
def test_wrist_preference_requires_a_declared_next_hold(monkeypatch, with_need):
    import table_1000.experts.skills.hold.pick as grasping
    from table_1000.experts.skills.hold.frames import Grasp
    references = []
    def candidates(*args, **kwargs):
        references.append(kwargs.get('reference_rotation'))
        return iter([Grasp(np.eye(4), .04, .03, 0., 0., {'approach_deg': 0., 'closest_deg': 0.})])
    monkeypatch.setattr(grasping, 'grasp_candidates', candidates)
    skill = grasping.Grasping()
    skill.braced, skill.brace_seed = False, None
    shape = Box((.03,) * 3)
    target = Target('object', ['object:body.part'], 'object:body', shape, np.eye(4), shape.points())
    skill.ctx = types.SimpleNamespace(poses=lambda: {'object:body': np.eye(4)},
        spec=types.SimpleNamespace(arms={'right': types.SimpleNamespace(finger_targets=lambda gap: {})}))
    skill.model = types.SimpleNamespace(tcp=lambda *a, **k: np.eye(4), vector=lambda _: np.zeros(1),
                                        ik=lambda *a, **k: ('Success', np.zeros(1)))
    skill.solvers = {'right': types.SimpleNamespace(sync=lambda: None, allow=lambda *a: None)}
    skill.hands = {'right': None}
    skill.shoulder = lambda arm: np.ones(3)
    skill.contact_surface = lambda target: object()
    skill.grasp_center_of_mass = lambda *a: None
    skill.grasp_weight = lambda target: 0.
    skill.obstacles = lambda *a: ([], [])
    skill.stow = lambda **kw: False
    need = types.SimpleNamespace(arms=['right'], accepts=lambda *a: True, hands=None, recipe=None) if with_need else None
    assert list(skill.options(target, ['right'], set(), need=need))
    assert skill.grasp_paths(target, ['right'], need=need)['right']
    # Only a choice at the current base can use its measured wrist. The
    # base-planning paths describe a future stance and retain the physical
    # ranking regardless of whether the next hold has been declared.
    for reference in references[:-1]:
        if with_need:
            np.testing.assert_array_equal(reference, np.eye(3))
        else:
            assert reference is None
    assert references[-1] is None


def test_parts_merge_by_cost_before_ik_with_distinct_retry_ids(monkeypatch):
    import table_1000.experts.skills.hold.pick as grasping
    from table_1000.experts.skills.hold.frames import Grasp
    skill = grasping.Grasping()
    skill.braced, skill.brace_seed = False, None
    skill.stow = lambda **kwargs: False
    targets = [Target('object', [name], 'object:body', Box((width, .02, .02)), np.eye(4), np.empty((0, 3)))
               for name, width in [('large', .1), ('small', .02)]]
    def generated(shape, *args, **kwargs):
        return iter([Grasp(np.eye(4), .04, .03, 0., shape.size[0],
                           {'approach_deg': 0., 'closest_deg': 0.})])
    monkeypatch.setattr(grasping, 'grasp_candidates', generated)
    skill.ctx = types.SimpleNamespace(geometry={}, poses=lambda: {'object:body': np.eye(4)},
        spec=types.SimpleNamespace(arms={'right': types.SimpleNamespace(finger_targets=lambda gap: {})}),
        env=types.SimpleNamespace(objects={'object': types.SimpleNamespace(articulated=False)},
            center_of_mass=lambda body: np.zeros((1, 3)), body_mass=lambda body: [.1]))
    skill.hands = {'right': None}
    skill.part_load = lambda arm, target, grasp: target.shape.size[0]
    skill.shoulder = lambda arm: np.ones(3)
    skill.obstacles = lambda *args: ([], [])
    skill.model = types.SimpleNamespace(vector=lambda values: np.zeros(1),
        ik=lambda *args, **kwargs: ('Success', np.zeros(1)))
    skill.solvers = {'right': types.SimpleNamespace(sync=lambda: None, allow=lambda *args: None)}
    skip = set()
    choices = list(skill.options(targets, ['right'], skip))
    assert [choice[1].target.names for choice in choices] == [['small'], ['large']]
    assert skip == {('right', 0, 0), ('right', 1, 0)}


def test_empty_held_lift_phase_preserves_later_part_order_and_retry_ids(monkeypatch):
    import table_1000.experts.skills.hold.pick as grasping
    from table_1000.experts.skills.hold.frames import Grasp
    skill = grasping.Grasping()
    skill.braced, skill.brace_seed = False, None
    skill.stow = lambda **kwargs: False
    targets = [Target('object', [str(i)], 'object:body', Box((i + 1., .02, .02)), np.eye(4), np.empty((0, 3)))
               for i in range(2)]
    generated, at_first_lift = [0, 0], []
    def candidates(shape, *args, **kwargs):
        part = int(shape.size[0]) - 1
        for index in range(20 if part else 2):
            if not kwargs.get('_first_angle'):
                generated[part] += 1
            angle = 40. if part else 20. * index
            yield Grasp(np.eye(4), .04, .03, 0., float(index),
                {'approach_deg': angle, 'closest_deg': 40. if part else 0.,
                 'label': (part, index), 'load': .1 if part else 1. if index == 0 else .001})
    monkeypatch.setattr(grasping, 'grasp_candidates', candidates)
    skill.ctx = types.SimpleNamespace(geometry={}, poses=lambda: {'object:body': np.eye(4)},
        spec=types.SimpleNamespace(arms={'right': types.SimpleNamespace(finger_targets=lambda gap: {})}),
        env=types.SimpleNamespace(objects={}))
    skill.hands = {'right': None}
    skill.shoulder = lambda arm: np.ones(3)
    skill.obstacles = lambda *args: ([], [])
    skill.part_load = lambda arm, target, grasp: grasp.info['load']
    def ik(*args, **kwargs):
        if kwargs.get('lift') is True and not at_first_lift:
            at_first_lift.append(generated.copy())
        return ('Failure' if kwargs.get('lift') is False else 'Success'), np.zeros(1)
    skill.model = types.SimpleNamespace(vector=lambda values: np.zeros(1), ik=ik)
    skill.solvers = {'right': types.SimpleNamespace(sync=lambda: None, allow=lambda *args: None)}
    skill.why = lambda *args: None
    skip = set()
    choices = list(skill.options(targets, ['right'], skip))
    assert at_first_lift == [[2, 1]]  # cost heads retain their original phase and retry order
    assert all(choice[-1] for choice in choices)
    assert [choice[1].info['label'] for choice in choices] == [*( (1, i) for i in range(20)), (0, 0), (0, 1)]
    assert skip == {('right', part, index) for part, count in [(0, 2), (1, 20)] for index in range(count)}


@pytest.mark.parametrize('near_arm_reachable', [True, False])
def test_part_cost_heads_are_lazy_for_unused_arm_and_phase(monkeypatch, near_arm_reachable):
    import table_1000.experts.skills.hold.pick as grasping
    from table_1000.experts.skills.hold.frames import Grasp
    skill = grasping.Grasping()
    skill.braced, skill.brace_seed = False, None
    targets = [Target('object', [str(i)], 'object:body', Box((i + 1., .02, .02)), np.eye(4), np.empty((0, 3)))
               for i in range(2)]
    calls = []
    def candidates(shape, transform, hand, *args, **kwargs):
        part = int(shape.size[0]) - 1
        calls.append((hand, part, bool(kwargs.get('_first_angle'))))
        angle = 40. if part else 0.
        yield Grasp(np.eye(4), .04, .03, 0., float(part),
                    {'approach_deg': angle, 'closest_deg': angle})
    monkeypatch.setattr(grasping, 'grasp_candidates', candidates)
    side = types.SimpleNamespace(finger_targets=lambda gap: {})
    skill.ctx = types.SimpleNamespace(poses=lambda: {'object:body': np.eye(4)},
                                     spec=types.SimpleNamespace(arms={'right': side, 'left': side}))
    skill.model = types.SimpleNamespace(vector=lambda _: np.zeros(1), ik=lambda arm, *a, **kw:
        ('Success' if near_arm_reachable or arm == 'left' else 'Failure', np.zeros(1)))
    skill.solvers = {arm: types.SimpleNamespace(sync=lambda: None, allow=lambda *a: None)
                     for arm in ('right', 'left')}
    skill.hands = {arm: arm for arm in ('right', 'left')}
    skill.centre = lambda target: np.zeros(3)
    skill.shoulder = lambda arm: np.ones(3) * (1 if arm == 'right' else 2)
    skill.obstacles = lambda *a: ([], [])
    skill.contact_surface = lambda target: None
    skill.grasp_center_of_mass = lambda *a: None
    skill.grasp_weight = lambda target: 0.
    skill.part_load = lambda *a: 0.
    skill.why = lambda *a: None
    skip = set()
    first = next(skill.options(targets, ['left', 'right'], skip))
    selected = 'right' if near_arm_reachable else 'left'
    assert first[0] == selected and first[1].target.names == ['0']
    assert skip == {(selected, 0, 0)}
    assert {(arm, part) for arm, part, probe in calls if probe} == {
        (arm, part) for arm in ('right', 'left') for part in (0, 1)}
    assert {(arm, part) for arm, part, probe in calls if not probe} == (
        {('right', 0)} if near_arm_reachable else {('right', 0), ('left', 0)})


def test_part_heads_preserve_internal_order_even_when_hidden_load_is_lower():
    from table_1000.experts.skills.hold.frames import Grasp, merge_part_heads
    def grasp(cost, load, closest):
        return Grasp(np.eye(4), .04, .03, 0., cost,
                     {'load': load, 'closest_deg': closest})
    streams = [[grasp(.1, 3., 0.), grasp(.2, .01, 0.)],
               [grasp(.8, 1., 10.), grasp(.9, 2., 10.)]]
    selected = list(merge_part_heads(streams, lambda number, g: g.info['load']))
    assert [(n, i, g.cost) for n, i, g in selected] == [(1, 0, .8), (1, 1, .9), (0, 0, .1), (0, 1, .2)]
    assert all(g.info['closest_deg'] == 0. for n, i, g in selected)
