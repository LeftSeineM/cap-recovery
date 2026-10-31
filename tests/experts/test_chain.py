from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import numpy as np

from table_1000.experts import skills
from table_1000.experts.skills import chain
from table_1000.experts.skills.chain import Action, InHand, Pick
from table_1000.experts.skills.common import Unmet


class Context:
    def __init__(self):
        self.notes, self.subtasks = [], []
        self.skill_attempts = 1

    def note(self, text, **values):
        self.notes.append(text)

    @contextmanager
    def subtask(self, instruction):
        self.subtasks.append(instruction)
        yield

    def skills(self):
        return SimpleNamespace(arms=['left', 'right'])


class Placing(Action):
    """Needs the pen in the left hand."""

    def __init__(self, log):
        self.log, self.instruction = log, 'put the pen in'

    def requirements(self, ctx):
        return InHand('pen', lambda arm, hold: arm == 'left', ['left'])

    def run(self, ctx, need=None, base=True):
        self.log.append('place')
        return 'placed'


@pytest.fixture
def picking(monkeypatch):
    """skills.pick: the right hand takes the pen unless it is needed in the left."""
    log = []

    def pick(ctx, obj, need=None, base=True, **options):
        if need is not None and 'right' not in need.arms:
            log.append('pick unmet')
            raise Unmet('pick', 'no reachable grasp')
        log.append('pick right')
        return {'arm': 'right'}
    monkeypatch.setattr(skills, 'pick', pick)
    return log


def test_chain_hands_over_when_the_pick_cannot_meet_the_place(picking, monkeypatch):
    def handover(ctx, obj, need=None):
        picking.append('handover')
        return {'arm': need.arms[0]}
    monkeypatch.setattr(skills, 'handover', handover, raising=False)
    ctx = Context()
    results = chain(ctx, [Pick('pen', instruction='pick up the pen'), Placing(picking)])
    # Unmet from here, unmet with the base free (adapt: base, then handover), so picked as it can and handed over.
    assert picking == ['pick unmet', 'pick unmet', 'pick right', 'handover', 'place']
    assert results == [[{'arm': 'right'}, {'arm': 'left'}], 'placed']
    assert ctx.subtasks == ['pick up the pen', 'pick up the pen', 'pick up the pen', 'hand the pen over to the left hand',
                            'put the pen in']


def test_chain_leaves_the_rest_to_the_next_action_when_no_handover_meets_it(picking, monkeypatch):
    def handover(ctx, obj, need=None):
        raise Unmet('handover', 'no handover')
    monkeypatch.setattr(skills, 'handover', handover, raising=False)
    results = chain(Context(), [Pick('pen'), Placing(picking)])
    assert picking == ['pick unmet', 'pick unmet', 'pick right', 'place']
    assert results == [[{'arm': 'right'}], 'placed']


def test_chain_runs_actions_without_needs_as_they_are(picking):
    results = chain(Context(), [Pick('pen'), Pick('cup')])
    assert picking == ['pick right', 'pick right']
    assert results == [{'arm': 'right'}, {'arm': 'right'}]


def test_chain_moves_the_base_where_the_pick_meets_the_place(monkeypatch):
    log = []

    def pick(ctx, obj, need=None, base=True, **options):
        if need is not None and not base:
            log.append('pick unmet')
            raise Unmet('pick', 'no reachable grasp')
        log.append('pick left, the base moved' if need is not None else 'pick right')
        return {'arm': 'left' if need is not None else 'right'}
    monkeypatch.setattr(skills, 'pick', pick)
    results = chain(Context(), [Pick('pen'), Placing(log)])
    assert log == ['pick unmet', 'pick left, the base moved', 'place']
    assert results == [[{'arm': 'left'}], 'placed']


def test_chain_hands_nothing_over_when_the_pick_meets_the_need_after_all(monkeypatch):
    log = []

    def pick(ctx, obj, need=None, base=True, **options):
        if need is not None:
            log.append('pick unmet')
            raise Unmet('pick', 'no reachable grasp')
        log.append('pick left')
        return {'arm': 'left', 'tcp_object': [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]}

    def handover(ctx, obj, need=None):
        log.append('handover')
    monkeypatch.setattr(skills, 'pick', pick)
    monkeypatch.setattr(skills, 'handover', handover, raising=False)
    chain(Context(), [Pick('pen'), Placing(log)])
    assert log == ['pick unmet', 'pick unmet', 'pick left', 'place']


def test_chain_asks_each_need_of_the_state_the_actions_before_leave(picking):
    class Asked(Placing):
        def requirements(self, ctx):
            self.log.append('asked')
            return None
    chain(Context(), [Pick('pen'), Pick('cup'), Asked(picking)])
    assert picking == ['pick right', 'asked', 'pick right', 'place']


@pytest.mark.parametrize('adapt', [('base', 'handover'), ('base',)])
def test_chain_tries_independent_release_hold_after_shared_stance_fails(monkeypatch, adapt):
    log = []
    strict = InHand('pen', lambda *args: False, ['left'], recipe={'origin': {}})
    relaxed = InHand('pen', lambda *args: True, ['left'], recipe={'gravity_release': True})
    ctx = Context()
    ctx.skills = lambda: SimpleNamespace(low_release_need=lambda need: relaxed if need is strict else None)
    def pick(ctx, obj, need=None, base=True, **options):
        log.append((need, base))
        if need is strict:
            raise Unmet('pick', 'no shared stance')
        assert need is relaxed and base
        return {'arm': 'left'}
    class Release(Placing):
        def requirements(self, ctx):
            return strict
    monkeypatch.setattr(skills, 'pick', pick)
    result = chain(ctx, [Pick('pen'), Release(log)], adapt=adapt)
    assert log == [(strict, False), (strict, True), (relaxed, True), 'place']
    assert result == [[{'arm': 'left'}], 'placed']


def test_failed_independent_release_hold_keeps_the_existing_handover_fallback(monkeypatch):
    log = []
    strict = InHand('pen', lambda *args: False, ['left'], recipe={'origin': {}})
    relaxed = InHand('pen', lambda *args: True, ['left'], recipe={'gravity_release': True})
    ctx = Context()
    ctx.skills = lambda: SimpleNamespace(low_release_need=lambda need: relaxed)
    def pick(ctx, obj, need=None, base=True, **options):
        log.append((need, base))
        if need is not None:
            raise Unmet('pick', 'no compatible hold')
        return {'arm': 'left', 'tcp_object': np.eye(4)}
    class Release(Placing):
        def requirements(self, ctx):
            return strict
    monkeypatch.setattr(skills, 'pick', pick)
    chain(ctx, [Pick('pen'), Release(log)])
    assert log == [(strict, False), (strict, True), (relaxed, True), (None, True), 'place']
