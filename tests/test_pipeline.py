import json

import numpy as np

from higgs import data, features, metrics, train


def test_ams_known_values():
    assert metrics.ams(0, 100) == 0.0
    # more signal over the same background is more significant
    assert metrics.ams(50, 100) > metrics.ams(10, 100) > 0


def test_load_marks_missing_and_label(fake_csv):
    df = data.load(fake_csv)
    assert df["DER_mass_MMC"].isna().any()
    assert set(df["y"].unique()) == {0, 1}
    assert len(data.feature_columns(df)) == 30


def test_engineer_adds_physics_features(fake_csv):
    df = data.load(fake_csv)
    X = features.engineer(df[data.feature_columns(df)])
    assert {"dphi_tau_lep", "mmc_missing"} <= set(X.columns)
    assert "PRI_tau_phi" not in X.columns
    assert (X["dphi_tau_lep"].between(0, np.pi)).all()


def test_end_to_end(fake_csv, tmp_path):
    report = train.run(fake_csv, out=tmp_path, update_readme=False, fast=True)
    assert report["best_model"] in report["models"]
    for r in report["models"].values():
        assert 0.5 < r["test_auc"] <= 1.0   # synthetic signal is learnable
    assert json.loads((tmp_path / "metrics.json").read_text())
    for f in ["roc.png", "ams.png", "feature_importance.png", "distributions.png", "RESULTS.md"]:
        assert (tmp_path / f).exists()
