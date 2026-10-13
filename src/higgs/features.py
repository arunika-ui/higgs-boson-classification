"""Feature engineering based on the detector physics."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Momentum-like features are heavily right-skewed; log1p makes them friendlier for linear models.
SKEWED = [
    "DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h",
    "DER_mass_jet_jet", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau",
    "PRI_tau_pt", "PRI_lep_pt", "PRI_met", "PRI_met_sumet",
    "PRI_jet_leading_pt", "PRI_jet_subleading_pt", "PRI_jet_all_pt",
]
# Azimuthal angles are only meaningful relative to each other (detector is symmetric in phi).
PHI = ["PRI_tau_phi", "PRI_lep_phi", "PRI_met_phi"]


def _delta_phi(a: pd.Series, b: pd.Series) -> pd.Series:
    d = np.abs(a - b)
    return np.minimum(d, 2 * np.pi - d)


def engineer(X: pd.DataFrame) -> pd.DataFrame:
    """Return a new frame with physics-motivated features added.

    - missing-value indicators (a feature is undefined when there are too few jets,
      which is itself informative)
    - relative angles between tau, lepton and missing energy instead of absolute phi
    - log transforms of skewed momenta
    """
    X = X.copy()
    if "DER_mass_MMC" in X:
        X["mmc_missing"] = X["DER_mass_MMC"].isna().astype(int)
    if all(c in X for c in PHI):
        X["dphi_tau_lep"] = _delta_phi(X["PRI_tau_phi"], X["PRI_lep_phi"])
        X["dphi_tau_met"] = _delta_phi(X["PRI_tau_phi"], X["PRI_met_phi"])
        X["dphi_lep_met"] = _delta_phi(X["PRI_lep_phi"], X["PRI_met_phi"])
        X = X.drop(columns=PHI)
    for c in SKEWED:
        if c in X:
            X[c] = np.log1p(X[c].clip(lower=0))
    return X
