"""Approximate Median Significance (AMS), the physics metric of the 2014 challenge."""
from __future__ import annotations

import numpy as np

B_REG = 10.0  # regularisation term from the challenge definition


def ams(s: float, b: float, b_reg: float = B_REG) -> float:
    """AMS = sqrt(2 * ((s + b + b_reg) * ln(1 + s / (b + b_reg)) - s)).

    s and b are the *weighted* numbers of signal and background events the
    classifier selects. Higher AMS = a more significant discovery signal.
    """
    b = b + b_reg
    return float(np.sqrt(2 * ((s + b) * np.log1p(s / b) - s)))


def ams_curve(y: np.ndarray, score: np.ndarray, w: np.ndarray, fractions: np.ndarray):
    """AMS when selecting the top `fraction` of events ranked by classifier score."""
    order = np.argsort(-score)
    y, w = y[order], w[order]
    s_cum = np.cumsum(w * (y == 1))
    b_cum = np.cumsum(w * (y == 0))
    out = []
    for f in fractions:
        k = max(int(f * len(y)) - 1, 0)
        out.append(ams(s_cum[k], b_cum[k]))
    return np.array(out)


def best_threshold(y, score, w, fractions=None):
    """Pick the selection fraction that maximises AMS (tuned on validation only)."""
    fractions = np.linspace(0.05, 0.40, 71) if fractions is None else fractions
    curve = ams_curve(y, score, w, fractions)
    i = int(np.argmax(curve))
    return float(fractions[i]), float(curve[i]), fractions, curve


def ams_at_fraction(y, score, w, fraction: float) -> float:
    return float(ams_curve(y, score, w, np.array([fraction]))[0])
