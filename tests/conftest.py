import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

COLS = ["DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h", "DER_deltaeta_jet_jet",
        "DER_mass_jet_jet", "DER_prodeta_jet_jet", "DER_deltar_tau_lep", "DER_pt_tot", "DER_sum_pt",
        "DER_pt_ratio_lep_tau", "DER_met_phi_centrality", "DER_lep_eta_centrality", "PRI_tau_pt",
        "PRI_tau_eta", "PRI_tau_phi", "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi", "PRI_met", "PRI_met_phi",
        "PRI_met_sumet", "PRI_jet_num", "PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi",
        "PRI_jet_subleading_pt", "PRI_jet_subleading_eta", "PRI_jet_subleading_phi", "PRI_jet_all_pt"]


@pytest.fixture
def fake_csv(tmp_path):
    """A small synthetic file with the real dataset's schema (no download needed)."""
    rng = np.random.default_rng(0)
    n = 3000
    y = rng.random(n) < 0.34
    df = pd.DataFrame({c: rng.normal(50, 20, n) for c in COLS})
    df["DER_mass_MMC"] += np.where(y, 75, 0)            # signal peaks near the Higgs mass
    for c in ["PRI_tau_phi", "PRI_lep_phi", "PRI_met_phi"]:
        df[c] = rng.uniform(-np.pi, np.pi, n)
    df.loc[rng.random(n) < 0.15, "DER_mass_MMC"] = -999.0  # missing-value marker
    df.insert(0, "EventId", np.arange(n))
    df["Weight"] = np.where(y, 0.002, 1.0)
    df["Label"] = np.where(y, "s", "b")
    df["KaggleSet"] = rng.choice(["t", "b", "v"], n, p=[0.5, 0.2, 0.3])
    df["KaggleWeight"] = df["Weight"]
    path = tmp_path / "fake.csv"
    df.to_csv(path, index=False)
    return path
