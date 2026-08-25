"""The local three-factor rule, isolated for bit-level verification.

Canonical operation grouping (frozen; tests assert bitwise identity against
hand-computed traces using exactly this grouping):

    u_A(next) = alpha * u_A + (gamma * error) * z_t
    prediction   = float(numpy.dot(u_A, z_t))

where z_t is the regressor in apical basis (W_A x_t) and error is the
next-token prediction residual e_t = y_t - prediction, gated OFF at decision
points by the caller. No RNG, no hidden state, no soma coupling: u_A is never
read from or written by anything spiking.
"""

from __future__ import annotations

import numpy as np


def apical_lms_update(
    u_a: np.ndarray, z_t: np.ndarray, error: float, alpha: float, gamma: float
) -> np.ndarray:
    """One local apical-LMS step; returns a NEW array (u_A is never mutated)."""
    return alpha * u_a + (gamma * error) * z_t


def predict_scalar(u_a: np.ndarray, z_t: np.ndarray) -> float:
    """Read the scalar next-token prediction out of the apical state."""
    return float(np.dot(u_a, z_t))
