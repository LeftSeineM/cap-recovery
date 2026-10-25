"""Bounded joint-space shortcuts, checked in the current planning world."""

import time
from contextlib import nullcontext
from collections import deque

import numpy as np

from table_1000.common.timing import measure


def shortcut(path, valid, *, budget=2., attempts=256, step=.01, clock=time.perf_counter,
             contact_pairs=None, cost=None):
    """Shorten a polyline without changing its endpoints or accepting an unchecked edge.

    Each accepted edge is sampled at the same maximum joint displacement as
    execution's path validation. The deadline also applies inside an edge check;
    running out of time leaves that edge unchanged. Original edges are retained
    without rechecking them. The first attempt joins the endpoints, followed by
    deterministic random pairs of vertices, then a greedy farthest-successor pass.
    A final local relaxation removes residual turns within the same deadline.
    All passes share the exact collision-state cache. With
    ``contact_pairs``, each initial pair is permitted only until it first
    separates. The complete preceding prefix is checked before extending that
    permission to an interior vertex; ordered samples reject every re-entry,
    including one finger re-entering while another still touches.
    """
    rows = np.asarray(path, dtype=float).copy()
    deadline = clock() + budget
    rng = np.random.default_rng(0)
    random_deadline = clock() + budget * .4
    cache = {}

    def contacts(q):
        key = tuple(q)
        if key not in cache:
            cache[key] = contact_pairs(q) if contact_pairs is not None else not valid(q)
        return cache[key]

    def departure_contacts(first, until):
        if contact_pairs is None or clock() >= until:
            return None
        initial = contacts(rows[0])
        if not initial:
            return None
        if first == 0:
            return initial
        present = contacts(rows[first])
        if not present or not present <= initial:
            return None
        # A retained bend can still be inside the original contact region.
        # Checking only its endpoints could miss leaving and re-entering along
        # the prefix, so use check_path's complete ordered grid here as well.
        remaining = set(initial)
        for start, end in zip(rows[:first], rows[1:first + 1]):
            count = max(1, int(np.ceil(np.max(np.abs(end - start)) / step)))
            for index in range(1, count + 1):
                if clock() >= until:
                    return None
                pairs = contacts(start + (end - start) * (index / count))
                if not pairs or not pairs <= remaining:
                    return None
                remaining &= pairs
        return remaining

    def clear_edge(start, end, until, initial=None):
        count = max(1, int(np.ceil(np.max(np.abs(end - start)) / step)))
        intervals = deque([(0, count)])
        samples = [0, count]
        while intervals:
            low, high = intervals.popleft()
            if high - low > 1:
                middle = (low + high) // 2
                samples.append(middle)
                intervals.extend(((low, middle), (middle, high)))
        if initial is not None:
            samples = range(count + 1)
        leaving = initial
        for index in samples:
            if clock() >= until:
                return False
            q = start + (end - start) * (index / count)
            pairs = contacts(q)
            if initial is not None:
                if not pairs:
                    leaving = set()
                elif not (leaving and pairs <= leaving):
                    return False
                leaving = leaving & pairs
            elif pairs:
                return False
        return True

    for attempt in range(attempts):
        if len(rows) < 3 or clock() >= random_deadline:
            break
        if attempt == 0:
            first, last = 0, len(rows) - 1
        else:
            first, last = sorted(rng.choice(len(rows), 2, replace=False))
        if last - first < 2:
            continue
        initial = departure_contacts(first, random_deadline)
        if clear_edge(rows[first], rows[last], random_deadline, initial):
            rows = np.concatenate((rows[:first + 1], rows[last:]))
    # Spend the remaining bounded time reducing redundant vertices in order.
    # The farthest clear successor removes turns that otherwise slow TOPP.
    first = 0
    while first < len(rows) - 2 and clock() < deadline:
        for last in range(len(rows) - 1, first + 1, -1):
            if clock() >= deadline:
                break
            initial = departure_contacts(first, deadline)
            if clear_edge(rows[first], rows[last], deadline, initial):
                rows = np.concatenate((rows[:first + 1], rows[last:]))
                break
        first += 1
    # Native simplifyMax also smooths bends. Vertex removal alone can leave a
    # jagged route whose spline executes slowly despite a short total length.
    # Move corners toward their neighbours only when both replacement edges
    # are fully checked and local polyline length strictly decreases. When the
    # caller supplies TOPP cost, also require the complete duration to improve.
    best = cost(rows) if cost is not None and len(rows) > 2 and clock() < deadline else None
    for _ in range(4):
        changed = False
        for middle in range(1, len(rows) - 1):
            if clock() >= deadline:
                return rows
            start, corner, end = rows[middle - 1:middle + 2]
            length = np.linalg.norm(corner - start) + np.linalg.norm(end - corner)
            for fraction in (.5, .25, .125):
                candidate = corner + fraction * ((start + end) / 2 - corner)
                shorter = np.linalg.norm(candidate - start) + np.linalg.norm(end - candidate)
                if shorter >= length - 1e-9:
                    continue
                initial = departure_contacts(middle - 1, deadline)
                if not clear_edge(start, candidate, deadline, initial):
                    continue
                continuing = initial if initial is not None and contacts(candidate) else None
                if clear_edge(candidate, end, deadline, continuing):
                    if cost is not None:
                        proposal = rows.copy()
                        proposal[middle] = candidate
                        value = cost(proposal)
                        if clock() >= deadline or best is None or value >= best:
                            continue
                        best = value
                    rows[middle] = candidate
                    changed = True
                    break
        if not changed:
            break
    return rows


class BoundedPlanner:
    """MPlib native planner adapter; approximate solutions need no simplification.

    The Python MPlib planner continues to own goal handling and TOPP. Keeping
    this adapter in the project avoids modifying the installed MPlib package.
    """

    def __init__(self, planner, world, reduction=None, cost=None):
        self.planner, self.world = planner, world
        self.before_simplify = None
        self.reduction = reduction
        self.cost = cost

    def plan(self, *args, simplify=True, **kwargs):
        context = (self.reduction.reduced(args[0], kwargs.get('fixed_joints', ()))
                   if self.reduction is not None else nullcontext())
        with context:
            with measure('rrt_native_solve'):
                status, path = self.planner.plan(*args, simplify=False, **kwargs)
        if status == 'Exact solution' and simplify and kwargs.get('constraint_function') is None:
            path = self.simplify_path(path)
        return status, path

    def simplify_path(self, path):
        """Use the same bounded contact-aware shortcut after raw-route validation."""
        if self.before_simplify is not None:
            self.before_simplify()

        def contacts(q):
            self.world.set_qpos_all(q)
            return {(c.link_name1, c.object_name2, c.link_name2) for c in self.world.check_collision()}

        with measure('rrt_simplify'):
            return shortcut(path, lambda q: not contacts(q), contact_pairs=contacts, cost=self.cost)
