"""Actions run in a chain: what one needs of the one before is asked of it, and met otherwise.

An expert says what to do with the objects; :func:`chain` runs the actions in order, each asked
for what the next needs of it (its :meth:`Action.requirements`: an :class:`InHand` for a place,
the hands and holds from which the object is put down). Where an action cannot meet that
(:class:`~table_1000.experts.skills.common.Unmet`), it is done as it can and the need is met
after, the first of ``adapt`` that does: ``'base'`` (the action again with the need and the base
free to move: one stance from which it is done and the next one can be, a pick and the place of
what it holds), ``'handover'`` (done as it can, then the object handed to the other hand, held as
needed; what is still unmet is left to the next action, which may move the base itself).
"""

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np

from table_1000.experts.context import Unreachable
from table_1000.experts.trace import traced
from table_1000.experts.skills.common import Unmet

ADAPT = ('base', 'handover')


@dataclass
class InHand:
    """``obj`` held by one of ``arms`` with a hold that ``accepts(arm, tcp_object)`` takes
    (``tcp_object``: the object's 4x4 pose in the TCP frame); ``hands(arm, tcp_object)``, where
    given, are the world TCP poses the next action takes it to held so (for the base to reach too)."""
    obj: str
    accepts: callable
    arms: list
    hands: callable = None
    recipe: dict = None


class Action:
    """One action of a chain, run in subtask ``instruction`` when given (:class:`Pick` and
    :class:`Place` under :func:`~table_1000.experts.skills.retry`)."""
    instruction = None

    def requirements(self, ctx):
        """What this action needs of the state the one before leaves it (None: nothing)."""
        return None

    @traced(concrete=False, action=True)
    def run(self, ctx, need=None, base=True):
        """Do it, leaving the state as ``need`` asks (else :class:`Unmet`); ``base``: whether the base may move."""
        raise NotImplementedError


class Pick(Action):
    """:func:`~table_1000.experts.skills.pick` of ``obj`` with its keyword ``options``."""

    def __init__(self, obj, *, instruction=None, **options):
        self.obj, self.options, self.instruction = obj, options, instruction

    @traced(concrete=False, action=True)
    def run(self, ctx, need=None, base=True):
        from table_1000.experts import skills
        return skills.retry(ctx, lambda: skills.pick(ctx, self.obj, need=need, base=base, **self.options))


class Place(Action):
    """:func:`~table_1000.experts.skills.place` of ``obj`` in ``region`` with its keyword ``options``;
    ``region`` may be a function ``region(ctx) -> (region, keep_out)`` of the state it is asked in
    (its ``keep_out`` added to the option's)."""

    def __init__(self, obj, region, *, instruction=None, **options):
        self.obj, self.region, self.options, self.instruction = obj, region, options, instruction

    def where(self, ctx):
        """``(region, options)`` in the state now."""
        if not callable(self.region):
            return self.region, self.options
        region, keep_out = self.region(ctx)
        return region, {**self.options, 'keep_out': [*self.options.get('keep_out', ()), *keep_out]}

    def requirements(self, ctx):
        region, options = self.where(ctx)
        return ctx.skills().need(self.obj, region, **{key: value for key, value in options.items() if key != 'arm'})

    @traced(concrete=False, action=True)
    def run(self, ctx, need=None, base=True):
        from table_1000.experts import skills

        def place():
            region, options = self.where(ctx)
            return skills.place(ctx, self.obj, region, **options)
        return skills.retry(ctx, place)


class Handover(Action):
    """``obj`` handed from the hand holding it to the other, held as ``need`` asks."""

    def __init__(self, obj, *, instruction=None):
        self.obj, self.instruction = obj, instruction

    @traced(concrete=False, action=True)
    def run(self, ctx, need=None, base=True):
        from table_1000.experts import skills
        return skills.handover(ctx, self.obj, need=need)


def subtask(ctx, instruction):
    return ctx.subtask(instruction) if instruction else nullcontext()


def chain(ctx, actions, adapt=ADAPT):
    """Run ``actions`` in order, each asked for what the next needs (see the module docstring);
    their results, an adapted action's a list of its own and the adaptation's."""
    results = []
    for index, action in enumerate(actions):
        # Asked of the state the actions before leave, as the next one sees it now.
        need = actions[index + 1].requirements(ctx) if index + 1 < len(actions) else None
        try:
            with subtask(ctx, action.instruction):
                results.append(action.run(ctx) if need is None else action.run(ctx, need, base=False))
            continue
        except Unmet as unmet:
            ctx.note('need unmet', action=type(action).__name__, reason=unmet.reason)
            failure = unmet
        results.append(adapted(ctx, action, need, adapt, failure))
    return results


def adapted(ctx, action, need, adapt, failure):
    """``action`` with ``need`` met by the first of ``adapt`` that does (see the module docstring)."""
    for way in adapt:
        if way == 'handover' and isinstance(action, Pick):
            with subtask(ctx, action.instruction):
                result = action.run(ctx)
            if result['arm'] in need.arms and need.accepts(result['arm'], np.asarray(result['tcp_object'])):
                return [result]     # held as needed after all (from where the pick moved the base)
            to = next((arm for arm in need.arms if arm != result['arm']), None)
            if to is None:
                return [result]
            try:
                with subtask(ctx, f'hand the {action.obj} over to the {to} hand'):
                    return [result, Handover(action.obj).run(ctx, need)]
            except Unmet as unmet:
                ctx.note('need unmet', action='Handover', reason=unmet.reason)
                return [result]     # the next action moves the base where it must
        if way == 'base':
            try:
                with subtask(ctx, action.instruction):
                    return [action.run(ctx, need)]
            except Unreachable as unmet:     # also nothing in reach with the need from anywhere: on as it can
                ctx.note('need unmet', action=type(action).__name__, reason=unmet.reason)
                failure = unmet
    raise failure
