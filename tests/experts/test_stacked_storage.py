"""Stored contents can support stacks, but exterior and lateral contacts cannot."""
from types import SimpleNamespace

import numpy as np

from table_1000.experts.skills import semantic
from table_1000.experts.skills.hold.frames import Box
from table_1000.experts.world import geometry


def shape(obj, size, centre, part='body'):
    return geometry.Shape(obj, 'body', part, np.eye(4), Box(size).points() + centre)


def cavity():
    return [shape('drawer', (.2, .2, .01), (0., 0., 0.), 'floor'),
            *[shape('drawer', (.01, .2, .12), (x, 0., .055), 'side') for x in (-.1, .1)],
            *[shape('drawer', (.2, .01, .12), (0., y, .055), 'side') for y in (-.1, .1)]]


def test_storage_accepts_recursive_stacks_and_reports_immediate_support(monkeypatch):
    shapes = cavity() + [shape('eraser', (.08, .08, .015), (0., 0., .013)),
                         shape('tape', (.04, .04, .012), (0., 0., .027)),
                         shape('clip', (.02, .02, .008), (0., 0., .0375))]
    monkeypatch.setattr(geometry, 'scene_shapes', lambda ctx: shapes)
    events = []
    ctx = SimpleNamespace(note=lambda note, **values: events.append(values),
                          poses=lambda: {s.obj + ':body': np.eye(4) for s in shapes},
                          skills=lambda: SimpleNamespace(target=lambda obj: SimpleNamespace(
                              points=next(s.points for s in shapes if s.obj == obj), body=obj + ':body')))
    for obj, support in [('eraser', 'container'), ('tape', 'eraser'), ('clip', 'tape')]:
        assert semantic.supported_inside(ctx, obj, 'drawer')
        assert semantic.storage_result(ctx, obj, 'drawer')
        assert events[-1]['support'] == support


def test_storage_rejects_a_stack_supported_by_an_exterior_object(monkeypatch):
    # The plank reaches inside, but its own immersed centre is outside the cavity.
    shapes = cavity() + [shape('outside plank', (.18, .05, .02), (.11, 0., .05)),
                         shape('tape', (.02, .02, .012), (.05, 0., .0665))]
    monkeypatch.setattr(geometry, 'scene_shapes', lambda ctx: shapes)
    assert not semantic.supported_inside(SimpleNamespace(), 'outside plank', 'drawer')
    assert not semantic.supported_inside(SimpleNamespace(), 'tape', 'drawer')


def test_storage_rejects_lateral_contact_with_a_stored_object(monkeypatch):
    shapes = cavity() + [shape('block', (.06, .06, .08), (0., 0., .0455)),
                         shape('floating', (.02, .02, .02), (.0405, 0., .05))]
    monkeypatch.setattr(geometry, 'scene_shapes', lambda ctx: shapes)
    assert semantic.supported_inside(SimpleNamespace(), 'block', 'drawer')
    assert not semantic.supported_inside(SimpleNamespace(), 'floating', 'drawer')


def test_touching_floating_objects_do_not_create_their_own_support(monkeypatch):
    shapes = cavity() + [shape('lower', (.03, .03, .01), (0., 0., .05)),
                         shape('upper', (.03, .03, .01), (0., 0., .0595))]
    monkeypatch.setattr(geometry, 'scene_shapes', lambda ctx: shapes)
    for obj in ('lower', 'upper'):
        assert not semantic.supported_inside(SimpleNamespace(), obj, 'drawer')
