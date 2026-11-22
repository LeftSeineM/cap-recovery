"""Dependency-light wrist-to-contact geometry helpers for cloth grasping."""

from __future__ import annotations

import math
from typing import Sequence

ROBO_DOJO_ARX_X5_CLOTH_TIP_OFFSET: tuple[float, float, float] = (0.150, 0.0, 0.0)


def _as_vec3(values: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must have length 3, got {len(values)}")
    vector = (float(values[0]), float(values[1]), float(values[2]))
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain finite coordinates")
    return vector


def normalize_quat_wxyz(quat_wxyz: Sequence[float]) -> tuple[float, float, float, float]:
    """Normalize a quaternion in ``(w, x, y, z)`` order."""
    if len(quat_wxyz) != 4:
        raise ValueError(f"quat_wxyz must have length 4, got {len(quat_wxyz)}")
    w, x, y, z = (float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]))
    norm = math.hypot(w, x, y, z)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("quat_wxyz must have a finite positive norm")
    return (w / norm, x / norm, y / norm, z / norm)


def rotate_vector_wxyz(
    vector_xyz: Sequence[float],
    quat_wxyz: Sequence[float],
) -> tuple[float, float, float]:
    """Rotate a 3D vector by a ``(w, x, y, z)`` quaternion."""
    vx, vy, vz = _as_vec3(vector_xyz, "vector_xyz")
    w, qx, qy, qz = normalize_quat_wxyz(quat_wxyz)

    # t = 2 * (q_vec x v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    # v' = v + w * t + (q_vec x t)
    cx = qy * tz - qz * ty
    cy = qz * tx - qx * tz
    cz = qx * ty - qy * tx

    return (
        vx + w * tx + cx,
        vy + w * ty + cy,
        vz + w * tz + cz,
    )


def contact_to_wrist_xyz(
    contact_xyz: Sequence[float],
    wrist_quat_wxyz: Sequence[float],
    tip_offset_local: Sequence[float],
) -> tuple[float, float, float]:
    """Convert a desired world contact position into a world wrist position.

    Given ``p_contact = p_wrist + R(wrist_quat_wxyz) @ tip_offset_local``,
    returns ``p_wrist = p_contact - R(wrist_quat_wxyz) @ tip_offset_local``.

    Args:
        contact_xyz: Desired world-frame contact point ``(x, y, z)`` in meters.
        wrist_quat_wxyz: Wrist orientation as ``(w, x, y, z)`` (normalized internally).
        tip_offset_local: Contact offset in the wrist-local frame ``(x, y, z)`` in meters.
    """
    cx, cy, cz = _as_vec3(contact_xyz, "contact_xyz")
    ox, oy, oz = rotate_vector_wxyz(tip_offset_local, wrist_quat_wxyz)
    return (cx - ox, cy - oy, cz - oz)
