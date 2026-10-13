"""Download and load the ATLAS Higgs Machine Learning Challenge 2014 dataset.

Source: ATLAS collaboration (2014), "Dataset from the ATLAS Higgs Boson Machine
Learning Challenge 2014", CERN Open Data Portal, DOI:10.7483/OPENDATA.ATLAS.ZBP2.M5T8

Each row is one simulated proton-proton collision event in the ATLAS detector.
Label "s" = signal (H -> tau tau decay), "b" = background (Z -> tau tau, top-pair, W).
Physics weights make the simulated sample match the expected real-world rates.
"""
from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CSV = DATA_DIR / "atlas-higgs-challenge-2014-v2.csv"
MISSING = -999.0  # the dataset's marker for "not defined for this event"


def download(url: str = URL, dest: Path = CSV) -> Path:
    """Fetch the dataset once (~60 MB compressed) and cache it under data/."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    gz = dest.with_suffix(".csv.gz")
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, gz)
    with gzip.open(gz, "rb") as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out)
    gz.unlink()
    return dest


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The 30 physics features: DER_* (derived) and PRI_* (primary/raw)."""
    return [c for c in df.columns if c.startswith(("DER_", "PRI_"))]


def load(path: Path | None = None) -> pd.DataFrame:
    """Load events, convert -999 to NaN and encode the label as 1 = signal, 0 = background."""
    df = pd.read_csv(path or download())
    feats = feature_columns(df)
    df[feats] = df[feats].replace(MISSING, np.nan)
    df["y"] = (df["Label"] == "s").astype(int)
    return df


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Use the challenge's official split so results are comparable with published ones.

    t = training (250k), b = public leaderboard (100k, used here for validation and
    threshold tuning), v = private leaderboard (450k, held-out test).
    The KaggleWeight column is the weight normalised within each subset.
    """
    parts = {"train": "t", "val": "b", "test": "v"}
    return {name: df[df["KaggleSet"] == code].reset_index(drop=True) for name, code in parts.items()}
