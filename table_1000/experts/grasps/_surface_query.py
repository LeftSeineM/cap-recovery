"""Halfspace ray intersections without per-piece Python array dispatch."""
import numpy as np
from numba import njit

@njit(cache=True)
def intervals(locals, groups, starts, planes, slopes, sphere_indices, spheres,
              possible, span_origins, span_slope, span_low, span_high, has_span, tolerance):
    count = len(span_origins)
    low, high = np.full(count, np.inf), np.full(count, -np.inf)
    nearest = np.full(count, np.inf)
    for piece in range(len(groups)):
        if not possible[piece]:
            continue
        local = locals[groups[piece]]
        for row in range(count):
            sphere = sphere_indices[piece]
            if sphere >= 0:
                lo, hi = spheres[sphere, 0, row], spheres[sphere, 1, row]
            else:
                x, y, z = local[row]
                lo, hi = -np.inf, np.inf
                for face in range(starts[piece], starts[piece + 1]):
                    # Match the three-coordinate NumPy einsum accumulation;
                    # do not use fastmath, contract products, or relax planes.
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
                first, last = origin + lo * span_slope, origin + hi * span_slope
                if max(first, last) < span_low or min(first, last) > span_high:
                    continue
            distance = max(lo, -hi)
            if distance < nearest[row]:
                low[row], high[row], nearest[row] = lo, hi, distance
    return low, high
