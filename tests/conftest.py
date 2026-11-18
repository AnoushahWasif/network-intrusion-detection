import numpy as np
import pandas as pd

from nids.data import CATEGORY, ID, LABEL, NUMERIC

CATS = ["Normal", "Normal", "Normal", "Exploits", "DoS", "Worms"]


def flows(n=6, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.integers(0, 50, n) for c in NUMERIC})
    df["dur"] = rng.random(n)
    df["proto"] = rng.choice(["tcp", "udp"], n)
    df["service"] = "-"
    df["state"] = "FIN"
    df[CATEGORY] = [CATS[i % len(CATS)] for i in range(n)]
    df[LABEL] = (df[CATEGORY] != "Normal").astype(int)
    df.insert(0, ID, range(1, n + 1))
    return df


def to_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()
