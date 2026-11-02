from dataclasses import dataclass, replace
import numpy as np
from table_1000.experts.skills.hold.frames import Box, OPEN_MARGIN
from table_1000.experts.context import remaining_wall
from table_1000.experts.skills.chain import InHand


@dataclass(frozen=True)
class SampledBox(Box):
    """X1pro table-pen sampling; keep outside common skills until another user exists."""
    contact_step: float = .005

    def __post_init__(self):
        if not np.isfinite(self.contact_step) or self.contact_step <= 0.:
            raise ValueError('contact_step must be a finite positive distance')

    def closings(self, step):
        size, axes = np.asarray(self.size), np.eye(3)
        longest = int(np.argmax(size))
        for k, (closing, width, basis, slides) in enumerate(super().closings(step)):
            if k != longest and size[longest] >= 2 * max(size[i] for i in range(3) if i != longest):
                slides += [axes[longest] * t for t in np.arange(-size[longest] / 2 + self.contact_step,
                                                               size[longest] / 2, self.contact_step)]
            yield closing, width, basis, slides


def free_insertion_need(obj, arms):
    """A replayable hold before an independent, upright mouth release."""
    return InHand(obj, lambda *args: True, list(arms),
                  recipe={'expert_free_insertion': True, 'obj': obj, 'arms': list(arms),
                          'origin': {'orientation': 'upright'}})


def upright_insertion_need(ctx, obj, region, *, check_reach=False, **options):
    if check_reach:
        raise ValueError('use ctx.skills().need for joint pick/place reach')
    if callable(region):
        region, blocked = region(ctx)
    else:
        blocked = ()
    keep_out = [*options.pop('keep_out', ()), *blocked]
    need = ctx.skills().need(obj, region, keep_out=keep_out, orientation='upright', **options)
    ctx.requirements.pop()
    return geometric_insertion_need(ctx, {**need.recipe, 'expert_geometric_insertion': True})


def geometric_insertion_need(ctx, recipe):
    skill = ctx.skills()
    obj, target, options = recipe['obj'], recipe['target'], recipe['options']
    clearance, above = recipe['clearance'], recipe['above']
    obstacles = skill.insertion_obstacles(target)
    verdicts = dict(recipe['verdicts'])
    def accepts(arm, tcp_object):
        remaining_wall(ctx)
        tcp_object = np.asarray(tcp_object)
        key = (arm, tcp_object.dtype.str, tcp_object.shape, tcp_object.tobytes())
        if key not in verdicts:
            closing = target.offset[:3, :3].T @ np.linalg.inv(tcp_object)[:3, 1]
            gap = min(float(np.ptp(target.shape.points() @ closing)) + OPEN_MARGIN, skill.hands[arm].max_gap)
            rejected = set()
            verdicts[key] = False
            for placed, _ in options:
                remaining_wall(ctx)
                release = placed.copy(); release[2, 3] += clearance
                blocked = skill.insertion_blockers(arm, release @ np.linalg.inv(tcp_object), above, gap, obstacles)
                if not blocked:
                    verdicts[key] = True
                    break
                rejected.update(blocked)
            recipe['blocked_by'] = sorted(set(recipe.get('blocked_by', ())) | rejected)
            recipe['verdicts'] = list(verdicts.items())
            ctx.note('insertion hold screened', obj=obj, arm=arm, accepted=verdicts[key],
                     opening_gap_m=gap, blocked_by=sorted(rejected))
        return verdicts[key]
    need = InHand(obj, accepts, recipe['arms'], recipe=recipe)
    ctx.requirements.append(need)
    return need


# This expert owns both custom records. Generic skills have no special cases
# for these types or recipe flags; the source expert is loaded before replay.
from table_1000.experts.audit import trace as _trace
_decode_common = _trace.decode

def _decode_expert(value, ctx=None):
    if isinstance(value, dict) and value.get('$') == 'SampledBox':
        return SampledBox(**_trace.decode(value['value'], ctx))
    if isinstance(value, dict) and value.get('$') == 'InHand':
        recipe = _trace.decode(value['value'], ctx)
        if recipe.get('expert_free_insertion'):
            return free_insertion_need(recipe['obj'], recipe['arms'])
        if recipe.get('expert_geometric_insertion'):
            return geometric_insertion_need(ctx, recipe)
    return _decode_common(value, ctx)

_trace.decode = _decode_expert
