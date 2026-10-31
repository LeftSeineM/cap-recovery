"""Fresh substep bearing signals for pressure moves, separate from contact presence."""

from collections import defaultdict
import math

import numpy as np

from table_1000.experts.context import DT


def material_motion(points, before, after):
    """Largest material displacement, retaining rotation and sideways motion."""
    old = points @ before[:3, :3].T + before[:3, 3]
    new = points @ after[:3, :3].T + after[:3, 3]
    return float(np.linalg.norm(new - old, axis=1).max())


def same_pressure_contact(old, arm, flip, edge, origin, pose, points, tolerance):
    """An empty action is repeatable only at the same contact and material state."""
    return (old['arm'] == arm and old['flip'] == flip
            and np.array_equal(old['edge'], edge)
            and np.linalg.norm(old['origin'] - origin) <= tolerance
            and material_motion(points, old['pose'], pose) <= tolerance)


class PressureLoad:
    """Finger-on-body normal impulse per physics substep.

    ``clock`` returns the current physics time. GPU contact reports skip
    substeps without contact, so ``bearing_now`` requires a load in the
    latest substep rather than in the last reported one; one substep of slack
    covers CPU reports stamped before the step clock advances.
    """

    def __init__(self, body, finger_links, physics_dt, weight, clock):
        self.body = body
        self.clock = clock
        self.fingers = {'robot.' + link for link in finger_links}
        self.dt = physics_dt
        self.substeps = round(DT / physics_dt)
        self.minimum = weight * physics_dt
        self.impulses = defaultdict(float)

    def record(self, time, pair, impulse):
        if any(name in self.fingers for name in pair) and any(
                name == self.body or name.startswith(self.body + '.') for name in pair):
            self.impulses[round(time / self.dt)] += impulse

    def read(self):
        values = list(self.impulses.values())
        loaded = sum(value > self.minimum for value in values)
        result = {'loaded_substeps': loaded, 'required_substeps': self.substeps,
                  'normal_impulse_ns': sum(values),
                  'minimum_substep_impulse_ns': self.minimum,
                  'bearing': sum(values) > self.minimum * self.substeps,
                  'continuous_substep_bearing': loaded >= self.substeps,
                  'bearing_now': (self.latest() > self.minimum and sum(values) > self.minimum * self.substeps),
                  'substep_impulses_ns': list(self.impulses.items())}
        self.impulses.clear()
        return result

    def latest(self):
        now = round(self.clock() / self.dt)
        recent = [step for step in self.impulses if step >= now - 1]
        return self.impulses[max(recent)] if recent else 0.

    def point(self, rows):
        """Current load-weighted contact point, excluding speculative zero-load points."""
        points, weights = [], []
        for row in rows:
            if not (any(name in self.fingers for name in row['bodies']) and any(
                    name == self.body or name.startswith(self.body + '.') for name in row['bodies'])):
                continue
            if 'impulses_ns' in row:
                for point, impulse, normal in zip(row['points'], row['impulses_ns'], row['normals']):
                    weight = abs(float(np.dot(impulse, normal)))
                    if weight > 0.:
                        points.append(point)
                        weights.append(weight)
            elif row.get('source') == 'tensor':
                points.extend(row['points'])
                weights.extend([1.] * len(row['points']))
        return np.average(points, axis=0, weights=weights) if points else None


def correction_path(start, goal, velocity_limits, acceleration_limits):
    """Quintic rest-to-rest joint correction under the existing motion limits."""
    start, goal = np.asarray(start), np.asarray(goal)
    delta = goal - start
    # Max derivatives of 10s³ - 15s⁴ + 6s⁵ are 15/8 and 10/sqrt(3).
    duration = max(float(np.max(15 / 8 * abs(delta) / velocity_limits)),
                   float(np.sqrt(np.max(10 / np.sqrt(3) * abs(delta) / acceleration_limits))))
    count = max(1, math.ceil(duration / DT))
    duration = count * DT
    time = np.arange(count + 1) * DT
    s = (time / duration)[:, None]
    return {'status': 'Success', 'position': start + (10*s**3 - 15*s**4 + 6*s**5) * delta,
            'velocity': (30*s**2 - 60*s**3 + 30*s**4) * delta / duration,
            'acceleration': (60*s - 180*s**2 + 120*s**3) * delta / duration**2,
            'time': time, 'duration': duration}
