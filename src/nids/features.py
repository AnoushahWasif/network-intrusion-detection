"""Day 3: the one preprocessing pipeline every model uses, fitted on training rows only.

- Validation is 20% of the official train file, stratified by attack category. The official test
  file is scored once, at the end.
- Byte and packet counts span many orders of magnitude, so numeric columns go through log1p, then
  scaling. (The loader has already refused negative values.) Trees don't care; logistic
  regression does.
- proto has many rare values. One-hot with min_frequency folds them, and any value the fitting
  rows never had, into one "infrequent" column instead of failing or growing new columns. In a
  column with no rare values, an unseen value encodes as all zeros.
- Anything not listed as a feature (id, attack_cat, label, or a stray column) is dropped.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from nids.data import CATEGORICAL, CATEGORY, FEATURES, ID, LABEL, NUMERIC, TTL

VALID_FRACTION = 0.2
MIN_CATEGORY_COUNT = 20  # rarer proto/service/state values share one column


class LeakageError(RuntimeError):
    """Rows or columns that would let a model see what it's judged on."""


@dataclass(frozen=True)
class Splits:
    fit: pd.DataFrame
    valid: pd.DataFrame


def split_validation(train: pd.DataFrame, fraction=VALID_FRACTION, seed=0) -> Splits:
    fit, valid = train_test_split(
        train, test_size=fraction, stratify=train[CATEGORY], random_state=seed
    )
    return Splits(fit, valid)


def columns(with_ttl: bool = True) -> list[str]:
    return [c for c in FEATURES if with_ttl or c not in TTL]


def preprocessor(with_ttl: bool = True) -> ColumnTransformer:
    numeric = [c for c in NUMERIC if with_ttl or c not in TTL]
    log_scale = Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    one_hot = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=MIN_CATEGORY_COUNT,
        sparse_output=False,
    )
    return ColumnTransformer(
        [("numeric", log_scale, numeric), ("categorical", one_hot, CATEGORICAL)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def check_no_leakage(splits: Splits, with_ttl: bool = True) -> None:
    """Run before every training run. Raises instead of warning."""
    forbidden = {ID, CATEGORY, LABEL} & set(columns(with_ttl))
    if forbidden:
        raise LeakageError(f"{sorted(forbidden)} would be used as features")
    shared = splits.fit.index.intersection(splits.valid.index)
    if len(shared):
        raise LeakageError(f"{len(shared)} rows are in both the fitting and validation sets")
    if splits.fit[ID].duplicated().any() or splits.fit[ID].isin(splits.valid[ID]).any():
        raise LeakageError("an id appears twice across fitting and validation rows")
