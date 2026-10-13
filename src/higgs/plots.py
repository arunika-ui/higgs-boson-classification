"""Figures written to results/."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

COLORS = {"Logistic Regression": "#8a8f98", "Neural Network (MLP)": "#2a78b5", "XGBoost": "#d9662b"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def roc(y, scores: dict, out: Path, aucs: dict):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC {aucs[name]:.3f})", color=COLORS.get(name), lw=2)
    ax.plot([0, 1], [0, 1], "--", color="#bbb", lw=1)
    ax.set(xlabel="Background efficiency (false positive rate)",
           ylabel="Signal efficiency (true positive rate)", title="ROC curve, held-out test set")
    ax.legend(loc="lower right", frameon=False)
    _style(ax)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def ams(curves: dict, out: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (fr, c, best) in curves.items():
        ax.plot(fr * 100, c, label=name, color=COLORS.get(name), lw=2)
        ax.axvline(best * 100, color=COLORS.get(name), lw=0.8, ls=":")
    ax.set(xlabel="Events selected as signal (%)", ylabel="AMS",
           title="Discovery significance vs. selection threshold (validation)")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def importance(names, values, out: Path, top: int = 15):
    order = np.argsort(values)[-top:]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(names)[order], np.array(values)[order], color="#d9662b")
    ax.set(xlabel="Importance (gain)", title=f"XGBoost: top {top} features")
    _style(ax)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def distributions(df, cols, out: Path):
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.4))
    for ax, c in zip(np.atleast_1d(axes), cols):
        v = df[c]
        lo, hi = np.nanpercentile(v, [1, 99])
        bins = np.linspace(lo, hi, 50)
        for lab, name, col in [(0, "Background", "#2a78b5"), (1, "Signal (Higgs)", "#d9662b")]:
            ax.hist(v[df["y"] == lab].dropna(), bins=bins, density=True, histtype="step",
                    lw=2, color=col, label=name)
        ax.set(title=c, yticks=[])
        _style(ax)
    np.atleast_1d(axes)[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
