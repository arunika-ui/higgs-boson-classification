"""Train, tune and evaluate all models; write metrics, figures and the README results table.

Usage:  python -m higgs.train            (downloads the CERN dataset on first run)
        python -m higgs.train --csv path  (use a local copy)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from . import data, features, metrics, models, plots

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def run(csv: Path | None = None, out: Path = RESULTS, update_readme: bool = True, fast: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    df = data.load(csv)
    feats = data.feature_columns(df)
    parts = data.split(df)
    X = {k: features.engineer(v[feats]) for k, v in parts.items()}
    y = {k: v["y"].to_numpy() for k, v in parts.items()}
    w = {k: v["KaggleWeight"].to_numpy() for k, v in parts.items()}

    zoo = models.build()
    if fast:  # used by tests: tiny models on tiny data
        zoo["XGBoost"].set_params(n_estimators=20)
        zoo["Neural Network (MLP)"].set_params(mlpclassifier__max_iter=20)

    report = {"dataset": {k: {"events": int(len(v)), "signal_fraction": round(float(v["y"].mean()), 4)}
                          for k, v in parts.items()},
              "models": {}}
    test_scores, curves = {}, {}
    for name, model in zoo.items():
        t0 = time.time()
        model.fit(X["train"], y["train"])
        secs = time.time() - t0
        s_val = model.predict_proba(X["val"])[:, 1]
        s_test = model.predict_proba(X["test"])[:, 1]
        frac, ams_val, fr, curve = metrics.best_threshold(y["val"], s_val, w["val"])
        curves[name] = (fr, curve, frac)
        test_scores[name] = s_test
        report["models"][name] = {
            "test_auc": round(float(roc_auc_score(y["test"], s_test)), 4),
            "test_ams": round(metrics.ams_at_fraction(y["test"], s_test, w["test"], frac), 3),
            "val_ams": round(ams_val, 3),
            "selected_fraction": round(frac, 3),
            "train_seconds": round(secs, 1),
        }
        print(f"{name:22s} AUC {report['models'][name]['test_auc']:.4f}  "
              f"AMS {report['models'][name]['test_ams']:.3f}  ({secs:.0f}s)")

    best = max(report["models"], key=lambda m: report["models"][m]["test_ams"])
    report["best_model"] = best
    (out / "metrics.json").write_text(json.dumps(report, indent=2))

    aucs = {m: r["test_auc"] for m, r in report["models"].items()}
    plots.roc(y["test"], test_scores, out / "roc.png", aucs)
    plots.ams(curves, out / "ams.png")
    xgb = zoo["XGBoost"]
    plots.importance(list(X["train"].columns), xgb.feature_importances_, out / "feature_importance.png")
    plots.distributions(parts["train"], ["DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_pt_h"],
                        out / "distributions.png")

    table = results_table(report)
    (out / "RESULTS.md").write_text(table + "\n")
    if update_readme:
        _update_readme(table)
    return report


def results_table(report: dict) -> str:
    lines = ["| Model | Test ROC AUC | Test AMS | Events selected | Train time |",
             "| --- | --- | --- | --- | --- |"]
    for name, r in sorted(report["models"].items(), key=lambda kv: kv[1]["test_ams"]):
        mark = " **(best)**" if name == report["best_model"] else ""
        lines.append(f"| {name}{mark} | {r['test_auc']:.4f} | {r['test_ams']:.3f} | "
                     f"{r['selected_fraction'] * 100:.1f}% | {r['train_seconds']:.0f} s |")
    n = report["dataset"]["test"]["events"]
    lines.append(f"\n_Test set: {n:,} held-out events (the challenge's private leaderboard split)._")
    return "\n".join(lines)


def _update_readme(table: str, readme: Path = ROOT / "README.md"):
    if not readme.exists():
        return
    text = readme.read_text()
    new = re.sub(r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)",
                 lambda m: f"{m.group(1)}\n{table}\n{m.group(2)}", text, flags=re.S)
    readme.write_text(new)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()
    run(args.csv)
