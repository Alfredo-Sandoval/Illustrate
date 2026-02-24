from __future__ import annotations

import math

import numpy as np


def clearmatrix() -> np.ndarray:
    """Fortran clearmatrix equivalent: identity 4x4 float32."""
    return np.eye(4, dtype=np.float32)


def catenate(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Fortran catenate(m1,m2) semantics.

    This intentionally matches the index ordering from the historical implementation
    instead of using Python's conventional matrix multiply operators.
    """
    out = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            total = 0.0
            for k in range(4):
                total += float(m1[k, j]) * float(m2[i, k])
            out[i, j] = np.float32(total)
    return out


def rotation_x(angle_degrees: float) -> np.ndarray:
    m = clearmatrix()
    angle = -angle_degrees * math.pi / 180.0
    m[1, 1] = np.float32(math.cos(angle))
    m[1, 2] = np.float32(-math.sin(angle))
    m[2, 1] = np.float32(math.sin(angle))
    m[2, 2] = np.float32(math.cos(angle))
    return m


def rotation_y(angle_degrees: float) -> np.ndarray:
    m = clearmatrix()
    angle = -angle_degrees * math.pi / 180.0
    m[0, 0] = np.float32(math.cos(angle))
    m[0, 2] = np.float32(math.sin(angle))
    m[2, 0] = np.float32(-math.sin(angle))
    m[2, 2] = np.float32(math.cos(angle))
    return m


def rotation_z(angle_degrees: float) -> np.ndarray:
    m = clearmatrix()
    angle = -angle_degrees * math.pi / 180.0
    m[0, 0] = np.float32(math.cos(angle))
    m[0, 1] = np.float32(-math.sin(angle))
    m[1, 0] = np.float32(math.sin(angle))
    m[1, 1] = np.float32(math.cos(angle))
    return m


def rotate_xyz(x: float, y: float, z: float, rm: np.ndarray) -> tuple[np.float32, np.float32, np.float32]:
    """Apply rotation like Fortran code: vector dotted with rm columns."""
    rx = np.float32(x * float(rm[0, 0]) + y * float(rm[1, 0]) + z * float(rm[2, 0]))
    ry = np.float32(x * float(rm[0, 1]) + y * float(rm[1, 1]) + z * float(rm[2, 1]))
    rz = np.float32(x * float(rm[0, 2]) + y * float(rm[1, 2]) + z * float(rm[2, 2]))
    return rx, ry, rz
