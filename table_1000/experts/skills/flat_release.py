"""The lowest collision-free opening and withdrawal of a flat hold."""
from dataclasses import dataclass, field

import numpy as np
from mplib.collision_detection import fcl

from table_1000.experts.grasps import _vertices
from table_1000.planning.world import pose
from table_1000.experts.skills.common import convex_keepout


FLAT_STEP = .002
FLAT_HEADROOM = .06
FLAT_LEVEL = np.radians(3.)


@dataclass
class FlatReleaseGeometry:
    """Named live obstacles and the object left behind when the hand withdraws.

    The measured wall height plus 6 cm bounds the search: actual opening
    clearances of 144–154 mm above a 100.5 mm basket required more than 3 cm.
    Only geometry chooses the minimum; an IK failure cannot raise it.
    An optional measured opening ceiling can further restrict that search.
    """
    obstacles: list
    wall_height: float = 0.
    released: list = field(default_factory=list)
    backoff: float = 0.
    ceiling: float | None = None
    channels: dict = field(default_factory=dict, repr=False)
    withdrawals: dict = field(default_factory=dict, repr=False)
    heights: dict = field(default_factory=dict, repr=False)
    blocked_by: set = field(default_factory=set)

    @property
    def maximum(self):
        height = self.wall_height + FLAT_HEADROOM
        return height if self.ceiling is None else min(height, self.ceiling)

    def minimum(self, hand, placed, tcp_object, closed_gap, open_gap, upward, backoff=None, *, heights=None,
                check_budget=None):
        """Only geometric clearance sets the height; IK cannot raise it.

        Opening, backing out and vertical withdrawal sweep every hand collider.
        Finger joints translate, so endpoint hulls enclose intermediate gaps;
        the midpoint also retains the same enclosure for alternate hands.
        ``heights`` can restrict verification to the measured release height.
        """
        def check():
            if check_budget is not None:
                check_budget()
        def collides(first, second):
            check()
            result = fcl.collide(first, second).is_collision()
            check()
            return result
        check()
        if self.maximum < 0.:
            return None
        release = placed @ np.linalg.inv(tcp_object)
        backoff = self.backoff if backoff is None else backoff
        delta = release[:3, :3].T @ [0., 0., upward]
        gaps = tuple(np.linspace(closed_gap, open_gap, 3))
        heights = tuple(heights) if heights is not None else None
        retreat = np.array([-backoff, 0., 0.])
        key = (id(hand), *np.round(gaps, 6), *np.round(delta, 9), round(backoff, 9))
        cache = (*key, placed.tobytes(), tcp_object.tobytes(), heights)
        if cache in self.heights:
            return self.heights[cache]
        if key not in self.channels:
            frames = [hand.frames(gap) for gap in gaps]
            channels, withdrawals = [], []
            for link, shapes in hand.shapes.items():
                for shape in shapes:
                    check()
                    vertices = _vertices(shape)
                    points = np.vstack([vertices @ fs[link][:3, :3].T + fs[link][:3, 3] for fs in frames])
                    channels.append((convex_keepout(points), points))
                    opened = vertices @ frames[-1][link][:3, :3].T + frames[-1][link][:3, 3]
                    stages = ([np.vstack([opened, opened + retreat])] if backoff else [])
                    stages.append(np.vstack([opened + retreat, opened + retreat + delta]))
                    for swept in stages:
                        check()
                        channel = convex_keepout(swept)
                        channels.append((channel, swept))
                        withdrawals.append(channel)
            check()
            self.channels[key] = channels
            self.withdrawals[key] = withdrawals
        # The released body stays at the release pose, rather than moving
        # upward with its former attachment. Opening starts in intended pad
        # contact; only the fully open withdrawal must clear this body.
        blocked = set()
        for name, points in self.released:
            check()
            obstacle = convex_keepout(points @ tcp_object[:3, :3].T + tcp_object[:3, 3])
            if any(collides(channel, obstacle)
                   for channel in self.withdrawals[key]):
                blocked.add(name)
        check()
        if blocked:
            self.blocked_by.update(blocked)
            self.heights[cache] = None
            return None
        obstacles = self.obstacles
        lows = np.asarray([bounds[0] for _, _, bounds in obstacles]).reshape(-1, 3)
        highs = np.asarray([bounds[1] for _, _, bounds in obstacles]).reshape(-1, 3)
        channels = []
        for channel, points in self.channels[key]:
            check()
            world = points @ release[:3, :3].T + release[:3, 3]
            channels.append((channel, world.min(0), world.max(0),
                             release @ channel.pose.to_transformation_matrix()))
        levels = (np.r_[np.arange(FLAT_STEP, self.maximum - 1e-9, FLAT_STEP), self.maximum]
                  if heights is None else np.asarray(sorted(h for h in heights if 0. <= h <= self.maximum)))
        index, cleared = 0, {}
        while index < len(levels):
            check()
            height, next_index = float(levels[index]), index
            for number, (channel, bottom, top, frame) in enumerate(channels):
                check()
                low, high = bottom + [0., 0., height], top + [0., 0., height]
                nearby = np.flatnonzero(np.all(high >= lows, axis=1) & np.all(low <= highs, axis=1))
                if not len(nearby):
                    continue
                def at(level):
                    translated = frame.copy(); translated[2, 3] += level
                    # FCLObject updates CollisionObject poses in place. Share
                    # immutable geometry, never wrappers from another height.
                    shapes = [fcl.CollisionObject(shape.get_collision_geometry()) for shape in channel.shapes]
                    return fcl.FCLObject('opening channel', pose(translated), shapes, channel.shape_poses)
                body = at(height)
                for other in nearby:
                    pair = number, int(other)
                    if pair in cleared and index >= cleared[pair]:
                        continue
                    name, obstacle, bounds = obstacles[other]
                    if not collides(body, obstacle):
                        continue
                    self.blocked_by.add(name)
                    def hit(at_index):
                        level = levels[at_index]
                        offset = [0., 0., level]
                        return (np.all(top + offset >= bounds[0]) and np.all(bottom + offset <= bounds[1])
                                and collides(at(level), obstacle))
                    # Two convex bodies intersect over one interval along a
                    # translation. Jump only past an interval containing this
                    # height; a separate overhead obstacle is checked again.
                    if hit(len(levels) - 1):
                        self.heights[cache] = None
                        return None
                    lower, upper = index + 1, len(levels) - 1
                    while lower < upper:
                        middle = (lower + upper) // 2
                        if hit(middle):
                            lower = middle + 1
                        else:
                            upper = middle
                    cleared[pair] = lower
                    next_index = max(next_index, lower)
            if next_index == index:
                check()
                self.heights[cache] = height
                return height
            index = next_index
        check()
        self.heights[cache] = None
        return None
