"""Expert target policy and asset enumeration keep their separate call paths."""
from types import SimpleNamespace

import numpy as np
import pytest

from table_1000.experts import skills
from table_1000.experts.context import Unreachable
from table_1000.experts.skills.hold.frames import Box
from table_1000.experts.skills.common import Target
from table_1000.experts.skills.hold.targets import targets, target_groups
from table_1000.experts.skills.hold.targets import rim_sections


def test_named_object_enumeration_matches_asset_alias_and_explicit_override():
    box = Box((.02,) * 3)
    geometry = SimpleNamespace(parts=lambda: [('body', 'body.a'), ('body', 'body.b')],
        points=lambda *args: box.points(), solid=lambda *args, **kw: True, shapes=[])
    def target(obj, part=None, shape=None, offset=None, body=None):
        return Target(obj, [f'{obj}:{part}'] if part else [f'{obj}:body.a', f'{obj}:body.b'],
                      f'{obj}:body', box if shape is None else shape, np.eye(4), box.points())
    skill = SimpleNamespace(ctx=SimpleNamespace(geometry={'cup': geometry, 'object': geometry}), target=target)
    named = list(targets(skill, 'cup'))
    alias = list(targets(skill))
    assert [t.names for t in named] == [[n.replace('object:', 'cup:') for n in t.names] for t in alias]
    assert [[t.fallback for t in group] for group in target_groups(named)] == [[False, False], [True]]
    assert [t.names for t in targets(skill, 'cup', part='body.a')] == [['cup:body.a']]
    custom = Box((.01,) * 3)
    assert list(targets(skill, 'cup', shape=custom))[0].shape is custom


def test_public_pick_keeps_groups_and_stops_after_execution_failure(monkeypatch):
    box = Box((.02,) * 3)
    parts = Target('cup', ['cup:part'], 'cup:body', box, np.eye(4), box.points(), container=True)
    whole = Target('cup', ['cup:part'], 'cup:body', box, np.eye(4), box.points(), fallback=True, container=True)
    monkeypatch.setattr(skills, 'enumerate_targets', lambda *args: iter([whole, parts]))
    calls = []
    def pick(group, *args):
        calls.append(group)
        raise Unreachable(None, 'no reachable grasp' if group[0] is parts else 'collision')
    skill = SimpleNamespace(pick=pick, target=lambda obj: whole, ctx=SimpleNamespace(geometry={}))
    ctx = SimpleNamespace(skills=lambda: skill)
    with pytest.raises(Unreachable, match='collision'):
        skills.pick(ctx, 'cup')
    assert calls == [[parts], [whole]]


@pytest.mark.parametrize('operation', ['pick', 'grasp'])
def test_ordinary_public_grasp_uses_the_original_whole_target(monkeypatch, operation):
    box = Box((.16, .012, .014))
    whole = Target('pen', ['pen:barrel', 'pen:cap'], 'pen:body', box, np.eye(4), box.points())
    part = Target('pen', ['pen:barrel'], 'pen:body', box, np.eye(4), box.points())
    fitted_whole = Target('pen', whole.names, whole.body, box, np.eye(4), box.points(), fallback=True)
    monkeypatch.setattr(skills, 'enumerate_targets',
                        lambda *args: pytest.fail('ordinary expert enumerated asset part heads'))
    calls = []
    def execute(group, *args):
        calls.append(group)
        return 'whole grip'
    skill = SimpleNamespace(target=lambda obj: whole, ctx=SimpleNamespace(geometry={}), **{operation: execute})
    ctx = SimpleNamespace(skills=lambda: skill)
    assert getattr(skills, operation)(ctx, 'pen') == 'whole grip'
    assert calls == [[whole]]


def test_explicit_public_part_keeps_its_single_target(monkeypatch):
    box = Box((.02,) * 3)
    part = Target('pen', ['pen:cap'], 'pen:body', box, np.eye(4), box.points())
    monkeypatch.setattr(skills, 'enumerate_targets', lambda *args: iter([part]))
    calls = []
    skill = SimpleNamespace(pick=lambda group, *args: calls.append(group) or 'cap grip')
    ctx = SimpleNamespace(skills=lambda: skill)
    assert skills.pick(ctx, 'pen', part='cap') == 'cap grip'
    assert calls == [[part]]


def test_upright_pick_tries_whole_body_frame_before_part_search(monkeypatch):
    from table_1000.experts.skills.chain import InHand
    box = Box((.16, .012, .014))
    whole = Target('pen', ['pen:body'], 'pen:body', box, np.eye(4), box.points())
    part = Target('pen', ['pen:cap'], 'pen:body', box, np.eye(4), box.points())
    calls = []
    monkeypatch.setattr(skills, 'enumerate_targets', lambda *args: iter([part]))
    def pick(group, *args):
        calls.append(group)
        if group == [whole]:
            raise Unreachable(None, 'no reachable grasp')
        return 'part grip'
    skill = SimpleNamespace(target=lambda obj: whole, pick=pick)
    ctx = SimpleNamespace(skills=lambda: skill)
    need = InHand('pen', lambda *args: True, ['right'], recipe={'origin': {'orientation': 'upright'}})
    assert skills.pick(ctx, 'pen', need=need) == 'part grip'
    assert calls == [[whole], [part]]


@pytest.mark.parametrize('closed', [False, True])
def test_upright_container_sections_are_actual_wall_tops_and_keep_divided_cavities(closed):
    from table_1000.scenes.geometry import AssetGeometry
    nodes = [dict(name='body', parent=None, matrix=np.eye(4).tolist(), rigid=True)]
    for index, (center, size) in enumerate([
        ((-.05, 0., .05), (.002, .1, .1)), ((.05, 0., .05), (.002, .1, .1)),
        ((0., -.05, .05), (.1, .002, .1)), ((0., .05, .05), (.1, .002, .1)),
        ((0., 0., .05), (.002, .1, .1)), ((0., 0., 0.), (.1, .1, .002))]):
        transform = np.eye(4)
        transform[:3, 3] = center
        nodes.append(dict(name=f'wall{index}', parent='body', matrix=transform.tolist(),
                          vertices=Box(size).points().tolist(), role='collision', rigid=False, part=True))
    geometry = AssetGeometry(dict(nodes=nodes, joints=[]))
    if closed:
        transform = np.eye(4)
        transform[2, 3] = .1
        nodes.append(dict(name='lid', parent='body', matrix=transform.tolist(),
                          vertices=Box((.102, .102, .002)).points().tolist(),
                          role='collision', rigid=False, part=True))
        geometry = AssetGeometry(dict(nodes=nodes, joints=[]))
    hand = SimpleNamespace(hold_margin=.002, inner=lambda gap: [(None, np.array([[0., 0., 0.], [.025, 0., 0.]]),
                                                                 np.array([True, True]))])
    skill = SimpleNamespace(hands={'left': hand}, arms=['left'],
        ctx=SimpleNamespace(poses=lambda: {'obj:body': np.eye(4)}),
        target=lambda obj, part, shape, offset: SimpleNamespace(shape=shape, offset=offset, part=part))
    whole = SimpleNamespace(body='obj:body')
    result = rim_sections(skill, 'obj', whole, geometry)
    assert whole.container
    if closed:
        assert whole.enclosed and result == []
        return
    assert len(result) == 5
    for target in result:
        points = target.shape.points() @ target.offset[:3, :3].T + target.offset[:3, 3]
        assert points[:, 2].min() == pytest.approx(.075)
        assert points[:, 2].max() == pytest.approx(.1)
        assert target.rim


@pytest.mark.parametrize('width,container', [(.012, False), (.1, True)])
def test_closed_narrow_tube_keeps_ordinary_targets_while_wide_box_is_a_container(width, container):
    from table_1000.scenes.geometry import AssetGeometry
    nodes = [dict(name='body', parent=None, matrix=np.eye(4).tolist(), rigid=True)]
    walls = [((-.08, 0., width / 2), (.001, width, width)),
             ((.08, 0., width / 2), (.001, width, width)),
             ((0., -width / 2, width / 2), (.16, .001, width)),
             ((0., width / 2, width / 2), (.16, .001, width)),
             ((0., 0., 0.), (.16, width, .001)),
             ((0., 0., width), (.16, width, .001))]
    for index, (center, size) in enumerate(walls):
        transform = np.eye(4)
        transform[:3, 3] = center
        nodes.append(dict(name=f'wall{index}', parent='body', matrix=transform.tolist(),
                          vertices=Box(size).points().tolist(), role='collision', rigid=False, part=True))
    geometry = AssetGeometry(dict(nodes=nodes, joints=[]))
    hand = SimpleNamespace(hold_margin=.002,
        inner=lambda gap: [(None, np.array([[0., 0., 0.], [.025, 0., 0.]]), np.array([True, True]))])
    skill = SimpleNamespace(hands={'right': hand}, arms=['right'],
                            ctx=SimpleNamespace(poses=lambda: {'obj:body': np.eye(4)}))
    whole = SimpleNamespace(body='obj:body', container=False, enclosed=False)
    assert rim_sections(skill, 'obj', whole, geometry) == []
    assert whole.container is container
    assert whole.enclosed is container


def test_plate_pick_preserves_chain_unmet_before_preparing_an_edge():
    from table_1000.experts.skills.hold.pick import Grasping
    from table_1000.experts.skills.common import Unmet
    skill = Grasping()
    skill.ctx = SimpleNamespace()
    skill.thin_geometry = lambda target: SimpleNamespace(kind='plate')
    failure = Unmet(None, 'no reachable grasp')
    def unmet(*args, **kwargs):
        raise failure
    skill.grasp = unmet
    skill.grasp_thin = lambda *args, **kwargs: pytest.fail('chain adaptation was bypassed')
    with pytest.raises(Unmet) as result:
        skill.pick(SimpleNamespace(), need=object(), base=False)
    assert result.value is failure


@pytest.mark.parametrize('reason,expected', [('no reachable grasp', ['whole', 'wall']),
                                           ('tracking', ['whole'])])
def test_closed_body_uses_exterior_parts_only_after_unreachable_envelope(reason, expected):
    from table_1000.experts.evals.asset_grasp import pick
    from table_1000.experts.context import ExpertFailed
    box = Box((.02,) * 3)
    whole = Target('object', ['whole'], 'object:body', box, np.eye(4), box.points(),
                   fallback=True, container=True, enclosed=True)
    wall = Target('object', ['wall'], 'object:body', box, np.eye(4), box.points(), container=True)
    calls = []
    def grasp(group, **kwargs):
        calls.append(group[0].names[0])
        raise ExpertFailed(None, reason if len(calls) == 1 else 'tracking')
    skill = SimpleNamespace(ctx=None, grasp=grasp, thin_geometry=lambda target: None)
    with pytest.raises(ExpertFailed, match='tracking'):
        pick(skill, {}, [wall, whole])
    assert calls == expected
