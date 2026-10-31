"""Restore a retry callback's physical and expert state, retaining failed attempts separately."""
from copy import deepcopy
import numpy as np
from table_1000.experts.trace import encode, decode, planner_state, restore


def copy_recorded(tree):
    """Copy append-only recorder containers; recorded tensor values are already clones."""
    if isinstance(tree, dict):
        return {key: copy_recorded(value) for key, value in tree.items()}
    if isinstance(tree, list):
        return list(tree)
    return tree


class RetryState:
    def __init__(self, ctx):
        self.ctx = ctx
        ctx.skills()
        self.physical = ctx.env.capture_retry()
        self.entry = dict(held=encode(ctx.skills().held), reaches=encode(ctx.skills().reaches),
                          planners=planner_state(ctx), current=ctx.current, rng=encode(ctx.rng.bit_generator.state))
        self.worlds = {arm: (dict(s.world.objects), s.world.object_padding.copy()) for arm,s in ctx.solvers.items()}
        self.logs = {arm: len(s.log) for arm,s in ctx.solvers.items()}
        self.skill_log = len(ctx.skills().log)
        self.lengths = {name:len(getattr(ctx,name)) for name in ('events','subtasks','subtask_wall_s','subtask_profiles','checkpoints')}
        self.frames = len(ctx.trace.frames)
        self.record = ctx.trace.problems
        if self.record:
            self.problem_lengths = {name:len(getattr(self.record,name)) for name in ('operations','decisions','events')}
            self.diagnostics = self.record._diagnostics.copy()
        self.requirements = [(need, encode(need.recipe)) for need in ctx.requirements]
        self.control = ctx.trace.stack[-1]
        self.parent_retry = list(getattr(ctx.trace,'retry_path',[]))
        self.attempts = []

    def reset(self, retry, failure):
        ctx, trace = self.ctx, self.ctx.trace
        # The archive owns the failed trajectory and frames; the main episode retains
        # only a continuous prefix followed by the successful/final attempt.
        attempt = ctx.archive_retry(self.control, retry, failure)
        self.attempts.append(attempt)
        ctx.env.restore_retry(self.physical)
        ctx.sync_model()
        for arm, solver in list(ctx.solvers.items()):
            for pair in list(solver.world.permissions):
                names = sorted(pair)
                solver.world.allow(names[:1], names[1:] or names[:1], False)
            solver.world.clear()
            solver.attachments.clear()
            if arm not in self.worlds:
                del ctx.solvers[arm]
                continue
            objects, padding = self.worlds[arm]
            state = self.entry['planners'][arm]
            solver.world.frame = decode(state['frame'])
            for name, obj in objects.items():
                solver.world.padding = padding[name]
                solver.world.add(obj,decode(state['poses'][name]))
            solver.world.padding = state['padding']
            del solver.log[self.logs[arm]:]
        restore(ctx,self.entry,restore_rng=False)
        for need, recipe in self.requirements:
            restored = ctx.skills().restore_need(decode(recipe, ctx))
            vars(need).update(vars(restored))
        ctx.requirements[:] = [need for need, _ in self.requirements]
        del ctx.skills().log[self.skill_log:]
        for name,length in self.lengths.items():
            del getattr(ctx,name)[length:]
        del trace.frames[self.frames:]
        trace.failure_frame = None
        if self.record:
            for name,length in self.problem_lengths.items():
                del getattr(self.record,name)[length:]
            self.record._diagnostics=self.diagnostics.copy()
        trace.retry_path = [*self.parent_retry,self.control,retry]
        streams=[trace.seed,trace.attempt,self.control,trace.repeat,*trace.retry_path]
        ctx.rng.bit_generator.state=np.random.default_rng(np.random.SeedSequence(streams)).bit_generator.state
        trace.frames[self.control]['attempts']=deepcopy(self.attempts)
        ctx.note('retry restored',attempt=retry,reason=failure.reason,archive=attempt['batch'])

    def close(self):
        self.ctx.trace.retry_path = self.parent_retry
