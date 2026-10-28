"""Halfspace ray intersections without per-piece Python array dispatch."""
import numpy as np
from numba import njit


@njit(cache=True)
def possible_pieces(origins, direction, bounds, axis, minimum, maximum, has_span):
    """Conservative batch broad phase, without per-call NumPy temporaries."""
    low, high = origins[0].copy(), origins[0].copy()
    for row in range(1, len(origins)):
        for dim in range(3):
            low[dim] = min(low[dim], origins[row, dim])
            high[dim] = max(high[dim], origins[row, dim])
    possible = np.ones(len(bounds), dtype=np.bool_)
    for piece in range(len(bounds)):
        enter, leave = -np.inf, np.inf
        finite, lower, upper = True, 0., 0.
        for dim in range(3):
            a, b = bounds[piece, 0, dim] - high[dim], bounds[piece, 1, dim] - low[dim]
            if direction[dim] == 0.:
                if a > 0. or b < 0.:
                    possible[piece] = False
            else:
                a, b = a / direction[dim], b / direction[dim]
                enter = max(enter, np.nextafter(min(a, b), -np.inf))
                leave = min(leave, np.nextafter(max(a, b), np.inf))
            finite &= np.isfinite(bounds[piece, 0, dim]) and np.isfinite(bounds[piece, 1, dim])
            if axis[dim] >= 0.:
                lower += bounds[piece, 0, dim] * axis[dim]
                upper += bounds[piece, 1, dim] * axis[dim]
            else:
                lower += bounds[piece, 1, dim] * axis[dim]
                upper += bounds[piece, 0, dim] * axis[dim]
        possible[piece] &= leave >= enter
        if has_span and finite:
            # Outward round the projection too. The exact halfspaces below
            # still decide every contact and retain their original tolerance.
            slack = 8 * np.finfo(np.float64).eps * max(1., abs(lower), abs(upper))
            possible[piece] &= upper + slack >= minimum and lower - slack <= maximum
    return possible


@njit(cache=True)
def _packet_faces(lower, upper, faces, count, planes, slopes, minima, maxima, active, tolerance):
    """Remove constraints redundant throughout a spatial packet, exactly."""
    floor, ceiling = -np.inf, np.inf
    for index in range(count):
        face = faces[index]
        x = lower[0] if planes[face, 0] >= 0. else upper[0]
        y = lower[1] if planes[face, 1] >= 0. else upper[1]
        z = lower[2] if planes[face, 2] >= 0. else upper[2]
        low = (x * planes[face, 0] + z * planes[face, 2]) + y * planes[face, 1] + planes[face, 3]
        x = upper[0] if planes[face, 0] >= 0. else lower[0]
        y = upper[1] if planes[face, 1] >= 0. else lower[1]
        z = upper[2] if planes[face, 2] >= 0. else lower[2]
        high = (x * planes[face, 0] + z * planes[face, 2]) + y * planes[face, 1] + planes[face, 3]
        slope = slopes[face]
        if abs(slope) < 1e-12:
            minima[face], maxima[face] = low, high
            if low > tolerance:
                return 0, False
        else:
            a, b = -low / slope, -high / slope
            minima[face], maxima[face] = min(a, b), max(a, b)
            if slope > 0.:
                ceiling = min(ceiling, maxima[face])
            else:
                floor = max(floor, minima[face])
    if floor > ceiling:
        return 0, False
    kept = 0
    for index in range(count):
        face = faces[index]
        slope = slopes[face]
        if slope >= 1e-12:
            keep = minima[face] <= ceiling
        elif slope <= -1e-12:
            keep = maxima[face] >= floor
        else:
            keep = maxima[face] > tolerance
        if keep:
            active[kept] = face
            kept += 1
    return kept, True


@njit(cache=True)
def intervals(locals, groups, starts, planes, slopes, sphere_indices, spheres,
              possible, span_origins, span_slope, span_low, span_high, has_span,
              tolerance, order):
    count = len(span_origins)
    low, high = np.full(count, np.inf), np.full(count, -np.inf)
    nearest = np.full(count, np.inf)
    leaves, depth = 1, 0
    while leaves * 16 < count:
        leaves *= 2
        depth += 1
    # The cached spatial permutation is reused across closing gaps. Build
    # exact boxes bottom-up for this call's transformed points; no contact
    # coordinates, halfspace arithmetic or output indices are changed.
    lower = np.full((len(locals), 2 * leaves - 1, 3), np.inf)
    upper = np.full((len(locals), 2 * leaves - 1, 3), -np.inf)
    needed = np.zeros(len(locals), dtype=np.bool_)
    for piece in range(len(groups)):
        if possible[piece] and sphere_indices[piece] < 0:
            needed[groups[piece]] = True
    for group in range(len(locals)):
        if not needed[group]:
            continue
        for leaf in range(leaves):
            node = leaves - 1 + leaf
            for index in range(leaf * count // leaves, (leaf + 1) * count // leaves):
                row = order[index]
                for dim in range(3):
                    lower[group, node, dim] = min(lower[group, node, dim], locals[group, row, dim])
                    upper[group, node, dim] = max(upper[group, node, dim], locals[group, row, dim])
        for node in range(leaves - 2, -1, -1):
            for dim in range(3):
                lower[group, node, dim] = min(lower[group, 2 * node + 1, dim], lower[group, 2 * node + 2, dim])
                upper[group, node, dim] = max(upper[group, 2 * node + 1, dim], upper[group, 2 * node + 2, dim])
    minima, maxima = np.empty(len(planes)), np.empty(len(planes))
    active = np.empty(len(planes), dtype=np.int64)
    stack_nodes = np.empty(depth + 2, dtype=np.int64)
    stack_counts = np.empty(depth + 2, dtype=np.int64)
    stack_faces = np.empty((depth + 2, len(planes)), dtype=np.int64)
    for piece in range(len(groups)):
        if not possible[piece]:
            continue
        group, sphere = groups[piece], sphere_indices[piece]
        local = locals[group]
        stack_nodes[0], stack_counts[0] = 0, starts[piece + 1] - starts[piece]
        for index in range(stack_counts[0]):
            stack_faces[0, index] = starts[piece] + index
        top = 1
        while top:
            top -= 1
            node, face_count = stack_nodes[top], stack_counts[top]
            if sphere < 0:
                face_count, hit = _packet_faces(lower[group, node], upper[group, node], stack_faces[top],
                    face_count, planes, slopes, minima, maxima, active, tolerance)
                if not hit:
                    continue
                if node < leaves - 1:
                    # Descendants inherit only constraints that can still
                    # matter. Each pruning step is conservative on its entire
                    # parent box, including the original parallel tolerance.
                    for child in (2 * node + 2, 2 * node + 1):
                        stack_nodes[top], stack_counts[top] = child, face_count
                        stack_faces[top, :face_count] = active[:face_count]
                        top += 1
                    continue
                leaf = node - leaves + 1
                first, last = leaf * count // leaves, (leaf + 1) * count // leaves
            else:
                first, last = 0, count
            for index in range(first, last):
                row = order[index]
                if sphere >= 0:
                    lo, hi = spheres[sphere, 0, row], spheres[sphere, 1, row]
                else:
                    x, y, z = local[row]
                    lo, hi = -np.inf, np.inf
                    for index in range(face_count):
                        face = active[index]
                        # Preserve the original einsum accumulation and each
                        # endpoint division; do not enable fastmath.
                        distance = (x * planes[face, 0] + z * planes[face, 2]) \
                            + y * planes[face, 1] + planes[face, 3]
                        slope = slopes[face]
                        if slope >= 1e-12:
                            hi = min(hi, -distance / slope)
                        elif slope <= -1e-12:
                            lo = max(lo, -distance / slope)
                        elif distance > tolerance:
                            hi = -np.inf
                            break
                        if hi < lo:
                            break
                if hi < lo:
                    continue
                if has_span:
                    origin = span_origins[row]
                    a, b = origin + lo * span_slope, origin + hi * span_slope
                    if max(a, b) < span_low or min(a, b) > span_high:
                        continue
                distance = max(lo, -hi)
                if distance < nearest[row]:
                    low[row], high[row], nearest[row] = lo, hi, distance
    return low, high
