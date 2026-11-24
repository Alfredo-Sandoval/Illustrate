from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def catenate(m1: NDArray[np.float32], m2: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compose two 4×4 matrices as ``m2 @ m1`` using Fortran accumulation order."""
    out = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            total = 0.0
            for k in range(4):
                total += float(m1[k, j]) * float(m2[i, k])
            out[i, j] = np.float32(total)
    return out


def rotation_x(angle_degrees: float) -> NDArray[np.float32]:
    m = np.eye(4, dtype=np.float32)
    angle = -angle_degrees * math.pi / 180.0
    m[1, 1] = np.float32(math.cos(angle))
    m[1, 2] = np.float32(-math.sin(angle))
    m[2, 1] = np.float32(math.sin(angle))
    m[2, 2] = np.float32(math.cos(angle))
    return m


def rotation_y(angle_degrees: float) -> NDArray[np.float32]:
    m = np.eye(4, dtype=np.float32)
    angle = -angle_degrees * math.pi / 180.0
    m[0, 0] = np.float32(math.cos(angle))
    m[0, 2] = np.float32(math.sin(angle))
    m[2, 0] = np.float32(-math.sin(angle))
    m[2, 2] = np.float32(math.cos(angle))
    return m


def rotation_z(angle_degrees: float) -> NDArray[np.float32]:
    m = np.eye(4, dtype=np.float32)
    angle = -angle_degrees * math.pi / 180.0
    m[0, 0] = np.float32(math.cos(angle))
    m[0, 1] = np.float32(-math.sin(angle))
    m[1, 0] = np.float32(math.sin(angle))
    m[1, 1] = np.float32(math.cos(angle))
    return m


