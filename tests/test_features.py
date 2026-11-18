import numpy as np
import pandas as pd
import pytest
from conftest import flows

from nids import features
from nids.data import CATEGORY, ID, LABEL, TTL
from nids.features import LeakageError, Splits


@pytest.fixture
def train():
    df = flows(600)
    # 3 common protocols and 2 rare ones (under MIN_CATEGORY_COUNT rows each)
    df["proto"] = ["tcp", "udp", "arp"] * 196 + ["sctp"] * 6 + ["gre"] * 6
    return df


def fitted(rows, with_ttl=True):
    return features.preprocessor(with_ttl).fit(rows)


def test_validation_is_a_stratified_fifth(train):
    s = features.split_validation(train)
    assert len(s.valid) == 120 and len(s.fit) == 480
    share = s.valid[CATEGORY].value_counts(normalize=True)
    expected = train[CATEGORY].value_counts(normalize=True)
    assert np.allclose(share.sort_index(), expected.sort_index(), atol=0.01)


def test_split_is_reproducible(train):
    a, b = features.split_validation(train), features.split_validation(train)
    assert a.valid.index.equals(b.valid.index)
    assert not a.valid.index.equals(features.split_validation(train, seed=1).valid.index)


def test_labels_and_id_never_reach_the_output(train):
    names = list(fitted(train).get_feature_names_out())
    assert not {ID, CATEGORY, LABEL} & set(names)
    assert not any(n.startswith(("attack_cat", "label")) for n in names)


def test_a_stray_column_is_dropped(train):
    out = fitted(train.assign(srcip="10.0.0.1")).get_feature_names_out()
    assert "srcip" not in out


def test_without_ttl_drops_exactly_the_ttl_columns(train):
    with_ttl = set(fitted(train).get_feature_names_out())
    without = set(fitted(train, with_ttl=False).get_feature_names_out())
    assert with_ttl - without == set(TTL)
    assert not set(TTL) & set(features.columns(with_ttl=False))


def test_rare_and_unseen_protocols_share_one_column(train):
    pre = fitted(train)
    names = list(pre.get_feature_names_out())
    proto_cols = [n for n in names if n.startswith("proto_")]
    assert sorted(proto_cols) == ["proto_arp", "proto_infrequent_sklearn", "proto_tcp", "proto_udp"]
    new = train.head(3).assign(proto=["sctp", "never-seen", "tcp"])
    out = pd.DataFrame(pre.transform(new), columns=names)
    assert out["proto_infrequent_sklearn"].tolist() == [1, 1, 0]
    assert out[proto_cols].sum(axis=1).tolist() == [1, 1, 1]


def test_an_unseen_value_without_a_rare_bucket_encodes_as_zeros(train):
    pre = fitted(train)  # service is always "-" in training
    names = list(pre.get_feature_names_out())
    out = pd.DataFrame(pre.transform(train.head(1).assign(service="http")), columns=names)
    assert out[[n for n in names if n.startswith("service_")]].to_numpy().sum() == 0


def test_scaling_is_learned_from_fitting_rows_only(train):
    s = features.split_validation(train)
    pre = fitted(s.fit)
    scaler = pre.named_transformers_["numeric"].named_steps["scale"]
    assert np.allclose(scaler.mean_[0], np.log1p(s.fit["dur"]).mean())
    assert not np.allclose(scaler.mean_[0], np.log1p(train["dur"]).mean())


def test_transforming_other_rows_changes_nothing_learned(train):
    s = features.split_validation(train)
    pre = fitted(s.fit)
    before = pre.transform(s.fit)
    pre.transform(s.valid.assign(sbytes=10**9, proto="weird"))
    assert np.array_equal(before, pre.transform(s.fit))


def test_huge_counts_stay_finite(train):
    pre = fitted(train)
    out = pre.transform(train.head(2).assign(sbytes=10**15, dur=1e6))
    assert np.isfinite(out).all()


def test_leakage_check_passes_on_a_real_split(train):
    features.check_no_leakage(features.split_validation(train))


def test_leakage_check_catches_shared_rows(train):
    s = features.split_validation(train)
    with pytest.raises(LeakageError, match="both"):
        features.check_no_leakage(Splits(s.fit, pd.concat([s.valid, s.fit.head(2)])))


def test_leakage_check_catches_a_reused_id(train):
    s = features.split_validation(train)
    valid = s.valid.copy()
    valid.iloc[0, valid.columns.get_loc(ID)] = s.fit[ID].iloc[0]
    with pytest.raises(LeakageError, match="id"):
        features.check_no_leakage(Splits(s.fit, valid))


def test_leakage_check_catches_a_label_listed_as_a_feature(train, monkeypatch):
    monkeypatch.setattr(features, "FEATURES", [*features.FEATURES, LABEL])
    with pytest.raises(LeakageError, match="label"):
        features.check_no_leakage(features.split_validation(train))
