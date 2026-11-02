"""Lightweight observations of existing decisions, solves, checks and recovery.

This observer never evaluates a task predicate or runs a diagnostic solve. Physical
state comes from the simulator adapter, outside the skills' information interface.
Collision geometry is rebuilt from the scene; only planner settings are recorded.
"""
from contextlib import contextmanager
from functools import wraps
import inspect
import time

from .trace import encode, planner_state


def compact(value):
    """Resolve decision inputs without copying collision meshes into lightweight calls."""
    if type(value).__name__ == 'FCLObject':
        return dict(collision_object=value.name, pose=encode(value.pose.to_transformation_matrix()))
    if type(value).__name__ == 'FlatReleaseGeometry':
        return compact(dict(obstacles=value.obstacles, wall_height=value.wall_height,
                            released=value.released, backoff=value.backoff, ceiling=value.ceiling,
                            blocked_by=sorted(value.blocked_by)))
    if isinstance(value, dict):
        return {str(key): compact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [compact(item) for item in (sorted(value, key=repr) if isinstance(value, set) else value)]
    return encode(value)


def clock():
    return time.process_time(), time.perf_counter()


def elapsed(start):
    return dict(zip(('cpu_s', 'wall_s'), (a - b for a, b in zip(clock(), start))))


def measurement(name, value, unit, operator, threshold, result, source='observed'):
    """Describe an already evaluated subcondition, without evaluating it again."""
    return dict(name=name, value=encode(value), unit=unit, operator=operator,
                threshold=encode(threshold), result=bool(result), source=source)


def check(ctx, criterion_id, result, *, kind, family_id=None, measurements=(), **values):
    """Record an existing check. No criterion or threshold belongs to this layer."""
    trace = getattr(ctx, 'trace', None)
    if trace:
        return trace.search.check(criterion_id, result, kind, family_id or criterion_id, measurements, values)


class SearchRecord:
    def __init__(self, trace):
        self.trace, self.ctx = trace, trace.ctx
        self.decisions, self.failures, self.checks = [], [], []
        self.outcomes, self.solves = [], []
        self.stack, self.methods, self.started, self.stage_started = [], {}, {}, {}
        self.branch = 'main'
        self.serial = 0
        self._state_key, self._state = None, None
        self.suspended = False

    def physical(self):
        key = (self.branch, self.ctx.steps)
        if key != self._state_key:
            self._state = encode(self.ctx.env.audit_state())
            self._state_key = key
        return self._state

    def settings(self):
        ctx = self.ctx
        skill = ctx._skills
        return dict(held=encode(skill.held if skill else {}), reaches=encode(skill.reaches if skill else {}),
                    planners=planner_state(ctx), current=ctx.current, rng=encode(ctx.rng.bit_generator.state),
                    targets=encode(ctx.env.targets()))

    def snapshot(self):
        return dict(physical=self.physical(), **self.settings())

    def owner(self):
        return self.trace.stack[-1] if self.trace.stack else None

    def step(self):
        return next((i for i in reversed(self.trace.stack) if self.trace.frames[i]['kind'] == 'step'), None)

    def add_failure(self, kind, reason, *, snapshot=None, costs=None, **values):
        if kind not in ('planning', 'check', 'record'):
            raise ValueError(f'unknown failure kind: {kind}')
        self.serial += 1
        entry = dict(observed_clock=list(clock()), id=self.serial, frame=self.owner(), step=self.step(),
                     branch=self.branch, kind=kind, legacy_error=reason, reason=reason, at=self.ctx.steps,
                     **(costs or dict(cpu_s=0., wall_s=0.)), **values)
        if snapshot is not None:
            entry['state'] = snapshot
        self.failures.append(entry)
        owner = entry['frame']
        while owner is not None:
            frame = self.trace.frames[owner]
            if frame['concrete']:
                frame['state'] = frame['_snapshot'] if '_snapshot' in frame else frame['state']
                break
            owner = frame['parent']
        return entry

    @contextmanager
    def decision(self, name, arguments):
        entry = dict(id=len(self.decisions), parent=self.stack[-1] if self.stack else None, frame=self.owner(),
                     branch=self.branch, step=self.step(), name=name, arguments=compact(arguments), candidate=None, candidate_count=0,
                     stage=None, rejected={}, events=[], enter=self.ctx.steps)
        self.decisions.append(entry)
        self.started[entry['id']] = clock()
        self.stage_started[entry['id']] = self.started[entry['id']]
        with self.active(entry):
            yield entry

    @contextmanager
    def active(self, entry):
        self.stack.append(entry['id'])
        started = clock()
        try:
            yield
        finally:
            cost = elapsed(started)
            for key, value in cost.items():
                entry[key] = entry.get(key, 0.) + value
            entry['exit'] = self.ctx.steps
            self.stack.pop()

    def mark(self, stage, candidate=None, outcome=None, *, kind='planning', **values):
        if not self.stack:
            return
        entry = self.decisions[self.stack[-1]]
        stage_start = self.stage_started[entry['id']]
        if candidate is not None:
            if str(candidate) != entry['candidate']:
                entry['candidate_count'] += 1
            entry['candidate'] = str(candidate)
        entry['stage'] = stage
        event = dict(branch=self.branch, cost_start=list(stage_start), **elapsed(stage_start), at=self.ctx.steps, stage=stage, candidate=entry['candidate'], outcome=outcome, **compact(values))
        entry['events'].append(event)
        if outcome == 'rejected':
            reason = values.get('reason', values.get('status', stage))
            key = f'{stage}: {reason}'
            entry['rejected'][key] = entry['rejected'].get(key, 0) + 1
            # A numerical solve already owns the failure and its cost/state.
            prior = next((f for f in reversed(self.failures)
                          if f.get('decision') == entry['id'] and f['at'] == self.ctx.steps
                          and not f.get('marked')), None)
            if prior:
                prior.update(stage=stage, candidate=entry['candidate'], marked=True)
            else:
                self.add_failure(kind, str(reason), start=entry['enter'],
                                 action_start=entry['enter'], decision=entry['id'], stage=stage,
                                 candidate=entry['candidate'], state_scope='decision',
                                 snapshot=self.snapshot() if kind == 'planning' else None, marked=True,
                                 costs=elapsed(stage_start), cost_start=list(stage_start))
        self.stage_started[entry['id']] = clock()

    def finish(self, entry, result, skill):
        entry['result'] = compact(result)
        if hasattr(skill, 'rejected'):
            entry['rejection_totals'] = dict(skill.rejected)
        empty = (isinstance(result, (list, tuple)) and not result) or result is False
        entry['empty'] = empty
        descendants = {entry['id']}
        for decision in self.decisions:
            if decision['parent'] in descendants:
                descendants.add(decision['id'])
        if empty and not any(f.get('decision') in descendants for f in self.failures):
            self.add_failure('planning', 'zero candidates' if not entry['candidate_count'] else 'all candidates rejected',
                decision=entry['id'], stage=entry['name'], candidate_count=entry['candidate_count'],
                snapshot=self.snapshot(), state_scope='decision',
                costs=elapsed(self.started[entry['id']]), cost_start=list(self.started[entry['id']]))

    def finish_generator(self, entry, yielded, exhausted, skill):
        entry.update(yielded=yielded, exhausted=exhausted)
        if hasattr(skill, 'rejected'):
            entry['rejection_totals'] = dict(skill.rejected)
        if exhausted and yielded == 0:
            self.finish(entry, [], skill)

    def install(self):
        for name in ('ik', 'plan_rrt', 'plan_screw'):
            function = getattr(self.ctx.model, name)
            self.methods[name] = function
            setattr(self.ctx.model, name, self.observe(name, function))

    def close(self):
        for name, function in self.methods.items():
            setattr(self.ctx.model, name, function)

    def observe(self, name, function):
        signature = inspect.signature(function)
        @wraps(function)
        def call(*args, **kwargs):
            if self.suspended:
                return function(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            arguments.pop('check', None)
            settings = self.settings()
            q, base = encode(self.ctx.model.q), encode(self.ctx.model.base)
            started, at = clock(), self.ctx.steps
            result = function(*args, **kwargs)
            cost = elapsed(started)
            status = result[0] if isinstance(result, tuple) else result['status']
            solve = dict(id=len(self.solves), name=name, status=str(status), frame=self.owner(),
                         step=self.step(), branch=self.branch, at=at, cost_start=list(started), **cost)
            self.solves.append(solve)
            if status != 'Success':
                owner = self.decisions[self.stack[-1]] if self.stack else {}
                failure = self.add_failure('planning', str(status), costs=cost,
                    snapshot=dict(physical=self.physical(), **settings), state_scope='operation',
                    decision=owner.get('id'), stage=owner.get('stage'), candidate=owner.get('candidate'),
                    operation=dict(name=name, arguments=encode(arguments), q=q, base=base),
                    criterion_id=f'RobotModel.{name}', family_id='planning.feasibility')
                failure['solve'] = solve['id']
                failure['start'] = at
                failure['cost_start'] = list(started)
            return result
        return call

    def check(self, criterion_id, result, kind, family_id, measurements, values):
        entry = dict(id=len(self.checks), frame=self.owner(), branch=self.branch, at=self.ctx.steps,
                     kind=kind, criterion_id=criterion_id, family_id=family_id, result=bool(result),
                     measurements=list(measurements), **encode(values))
        self.checks.append(entry)
        if not result:
            active = [self.trace.frames[i] for i in self.trace.stack if self.trace.frames[i]['concrete']]
            cost_start = list(active[-1]['_clock']) if active else list(clock())
            return self.add_failure(kind, values.get('legacy_error', criterion_id), criterion_id=criterion_id, family_id=family_id,
                                    check=entry['id'], start=self.ctx.steps, action_start=self.action_start(),
                                    snapshot=self.snapshot() if kind == 'planning' else None,
                                    costs=elapsed(cost_start), cost_start=cost_start)['id']

    def action_start(self):
        concrete = [self.trace.frames[i] for i in self.trace.stack if self.trace.frames[i]['concrete']]
        return concrete[-1]['enter'] if concrete else self.ctx.steps

    def outcome(self, kind, **values):
        entry = dict(id=len(self.outcomes), kind=kind, step=self.step(), branch=self.branch,
                     at=self.ctx.steps, observed_clock=list(clock()), **values)
        self.outcomes.append(entry)
        return entry

    def exception(self, entry, error):
        from table_1000.experts.context import ExpertFailed
        if not isinstance(error, ExpertFailed) or getattr(error, 'search_failure_id', None) is not None:
            return
        descendants = {entry['id']}
        for child in self.trace.frames[entry['id'] + 1:]:
            if child['parent'] in descendants:
                descendants.add(child['id'])
        prior = next((f for f in reversed(self.failures) if (error.kind is None or f['kind'] == error.kind)
                      and f['frame'] in descendants and f['branch'] == self.branch
                      and f['at'] == self.ctx.steps and f['legacy_error'] == error.reason), None)
        if prior is not None:
            error.search_failure_id = prior['id']
            return
        if error.kind is None:
            prior = next((f for f in reversed(self.failures)
                          if f['frame'] in descendants and f['branch'] == self.branch), None)
            if prior is not None:
                error.search_failure_id = prior['id']
        # Exception/control wrappers link to the originating event. Only producers
        # with an explicit kind can introduce a new observation.
        if getattr(error, 'kind', None) is not None:
            active = [self.trace.frames[i] for i in self.trace.stack if self.trace.frames[i]['concrete']]
            cost_start = list(active[-1]['_clock']) if active else list(clock())
            failure = self.add_failure(error.kind, error.reason,
                snapshot=self.snapshot() if error.kind == 'planning' else None,
                state_scope='skill', start=self.ctx.steps, action_start=self.action_start(),
                criterion_id=error.criterion_id, family_id='criterion', details=encode(error.details),
                decision=self.stack[-1] if self.stack else None,
                costs=elapsed(cost_start), cost_start=cost_start)
            error.search_failure_id = failure['id']

    def rollback(self, control, retry, failure, after):
        self.exception(self.trace.frames[control], failure)
        start = self.trace.frames[control]['enter']
        number = getattr(failure, 'search_failure_id', None)
        return self.outcome('rollback', control=control, failure_id=number, reason=failure.reason,
            discarded=[f['id'] for f in self.failures if f['id'] > after
                       and (f['branch'] == self.branch or self.branch.startswith(f['branch'] + '/'))],
            restore=dict(branch=f'{self.branch}/{control}-{retry}', at=start),
            cost_start=list(self.trace.frames[control]['_clock']))

    def next_branch(self, control, retry):
        self.branch = f'{self.branch}/{control}-{retry}'
        self._state_key = None

    def terminal(self, report):
        if report['status'] != 'failed':
            return
        root = next((f for f in reversed(self.trace.frames) if f['kind'] == 'step'), None)
        failure = report['failure']
        self.outcome('step_abandoned', frame=root['id'] if root else None,
                     failure_id=report.get('failure_id'), reason=failure['reason'],
                     start=root['enter'] if root else 0,
                     details=encode(failure.get('details', {})))
        self.outcomes[-1]['step'] = root['id'] if root else None

    def document(self):
        return dict(solver_coverage='all', decisions=self.decisions, solves=self.solves, failures=self.failures, checks=self.checks, outcomes=self.outcomes)


def observed_motion(method):
    """Retain the existing motion wrapper, including its path validation and temporary contacts."""
    @wraps(method)
    def call(solver, kind, function, goals=(), problem=None):
        from table_1000.experts.solver import PlanningFailed
        trace = getattr(solver.ctx, 'trace', None)
        search = getattr(trace, 'search', None)
        if search is None:
            return method(solver, kind, function, goals, problem)
        snapshot, started = search.snapshot(), clock()
        q, base, serial = encode(solver.model.q), encode(solver.model.base), search.serial
        try:
            return method(solver, kind, function, goals, problem)
        except PlanningFailed as error:
            previous = next((f for f in reversed(search.failures) if f['id'] > serial
                             and f.get('operation', {}).get('name') == f'plan_{kind}'), None)
            entry = previous or search.add_failure('planning', error.reason, snapshot=snapshot,
                costs=elapsed(started), state_scope='operation', criterion_id='MotionSolver._plan',
                family_id='planning.feasibility', cost_start=list(started))
            entry.update(state=snapshot, reason=error.reason, legacy_error=error.reason,
                         operation=dict(name='motion', kind=kind, q=q, base=base,
                                        arguments=encode(dict(arm=solver.arm.name, goals=goals, problem=problem))))
            error.search_failure_id = entry['id']
            error.kind = 'planning'
            raise
    return call
