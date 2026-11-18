import hashlib

import numpy as np
import pandas as pd
import pytest
from conftest import CATS, flows, to_bytes

from nids import data
from nids.data import CATEGORY, FEATURES, ID, LABEL, DataError


def edited(change) -> bytes:
    df = flows().astype(object)  # so a test can put any value in any column
    change(df)
    return to_bytes(df)


def test_a_valid_file_parses():
    df = data.parse(to_bytes(flows()), 6)
    assert len(df) == 6 and set(df[CATEGORY]) == set(CATS)
    assert (df["service"] == "-").all()


def test_byte_order_mark_and_padded_categories_are_fine():
    df = flows()
    df[CATEGORY] = " " + df[CATEGORY] + " "
    out = data.parse(b"\xef\xbb\xbf" + to_bytes(df), 6)
    assert set(out[CATEGORY]) == set(CATS)


def test_na_text_stays_text():
    out = data.parse(edited(lambda d: d.__setitem__("service", "NA")), 6)
    assert (out["service"] == "NA").all()


BAD = {
    "missing column": (lambda d: d.pop("sttl"), "missing \\['sttl'\\]"),
    "extra column": (lambda d: d.__setitem__("srcip", "10.0.0.1"), "extra \\['srcip'\\]"),
    "text in a number": (lambda d: d.__setitem__("sbytes", "abc"), "sbytes"),
    "empty number": (lambda d: d.loc.__setitem__((0, "dur"), ""), "dur"),
    "infinite": (lambda d: d.loc.__setitem__((0, "rate"), np.inf), "rate"),
    "negative": (lambda d: d.loc.__setitem__((0, "dbytes"), -1), "dbytes"),
    "duplicate id": (lambda d: d.loc.__setitem__((1, ID), 1), "duplicate ids"),
    "unknown category": (lambda d: d.loc.__setitem__((0, CATEGORY), "Ransom"), "Ransom"),
    "empty category": (lambda d: d.loc.__setitem__((0, "proto"), " "), "proto"),
    "label 2": (lambda d: d.loc.__setitem__((3, LABEL), 2), "0 or 1"),
    "normal labelled attack": (lambda d: d.loc.__setitem__((0, LABEL), 1), "disagrees"),
    "attack labelled normal": (lambda d: d.loc.__setitem__((3, LABEL), 0), "disagrees"),
}


@pytest.mark.parametrize("name", BAD)
def test_bad_files_are_refused(name):
    change, message = BAD[name]
    with pytest.raises(DataError, match=message):
        data.parse(edited(change), 6)


def test_a_duplicated_header_is_refused():
    text = to_bytes(flows()).decode().splitlines()
    text[0] = text[0].replace("sttl", "dttl")
    with pytest.raises(DataError, match="columns"):
        data.parse("\n".join(text).encode(), 6)


def test_wrong_row_count_points_at_swapped_files():
    with pytest.raises(DataError, match="swapped"):
        data.parse(to_bytes(flows()), 7)


@pytest.mark.parametrize("raw", [b"", b"\xff\xfe\x00garbage", b'a,b\n"unclosed'])
def test_garbage_is_refused(raw):
    with pytest.raises(DataError):
        data.parse(raw, 6)


def test_features_never_include_id_or_labels():
    assert ID not in FEATURES and CATEGORY not in FEATURES and LABEL not in FEATURES
    assert len(FEATURES) == 42 and len(set(FEATURES)) == 42
    assert set(data.TTL) <= set(FEATURES)


@pytest.fixture
def downloaded(tmp_path, monkeypatch):
    """Both files in a raw folder, with small row counts in place of the real ones."""
    monkeypatch.setattr(data, "FILES", {"train": ("train.csv", 6), "test": ("test.csv", 4)})
    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "train.csv").write_bytes(to_bytes(flows(6)))
    (folder / "test.csv").write_bytes(to_bytes(flows(4, seed=1)))
    return folder, tmp_path / "data.sha256"


def test_pin_then_load(downloaded):
    folder, hashes = downloaded
    digests = data.pin(folder, hashes)
    expected = hashlib.sha256((folder / "train.csv").read_bytes()).hexdigest()
    assert digests["train.csv"] == expected
    assert f"{expected}  train.csv" in hashes.read_text()
    assert len(data.load("train", folder, hashes)) == 6
    assert len(data.load("test", folder, hashes)) == 4


def test_loading_before_pinning_is_refused(downloaded):
    folder, hashes = downloaded
    with pytest.raises(DataError, match="no pinned hash"):
        data.load("train", folder, hashes)


def test_an_edited_file_is_refused(downloaded):
    folder, hashes = downloaded
    data.pin(folder, hashes)
    path = folder / "train.csv"
    path.write_bytes(path.read_bytes().replace(b"FIN", b"CON", 1))
    with pytest.raises(DataError, match="doesn't match"):
        data.load("train", folder, hashes)


def test_pinning_twice_is_refused(downloaded):
    folder, hashes = downloaded
    data.pin(folder, hashes)
    with pytest.raises(DataError, match="already pins"):
        data.pin(folder, hashes)


def test_a_broken_file_is_never_pinned(downloaded):
    folder, hashes = downloaded
    (folder / "test.csv").write_bytes(to_bytes(flows(5)))
    with pytest.raises(DataError, match="rows"):
        data.pin(folder, hashes)
    assert not hashes.exists()


def test_a_missing_file_says_how_to_get_it(downloaded):
    folder, hashes = downloaded
    (folder / "test.csv").unlink()
    with pytest.raises(DataError, match="by hand"):
        data.pin(folder, hashes)


def test_oversized_files_are_refused_before_reading(downloaded, monkeypatch):
    folder, hashes = downloaded
    monkeypatch.setattr(data, "MAX_BYTES", 10)
    with pytest.raises(DataError, match="cap"):
        data.pin(folder, hashes)


@pytest.mark.parametrize("line", ["abc  train.csv", "0" * 64, f"{'g' * 64}  train.csv"])
def test_malformed_hash_lines_are_refused(tmp_path, line):
    path = tmp_path / "data.sha256"
    path.write_text(line + "\n")
    with pytest.raises(DataError, match="line 1"):
        data.read_hashes(path)


def test_hash_file_accepts_comments_and_sha256sum_binary_marker(tmp_path):
    path = tmp_path / "data.sha256"
    path.write_text(f"# pinned by hand\n{'A' * 64} *train.csv\n")
    assert data.read_hashes(path) == {"train.csv": "a" * 64}


def test_duplicate_report():
    train = flows(6)
    train = pd.concat([train, train.iloc[[3]].assign(id=99)], ignore_index=True)
    twin = train.iloc[[0]].assign(id=100, attack_cat="DoS", label=1)
    train = pd.concat([train, twin], ignore_index=True)
    test = pd.concat([train.iloc[[1, 1]], flows(2, seed=5)], ignore_index=True)
    report = data.duplicates(train, test)
    assert report["train_rows"] == 8
    assert report["train_repeats"] == 1
    assert report["train_same_features_different_category"] == 2
    assert report["test_repeats"] == 1
    assert report["test_rows_also_in_train"] == 2


def test_cli_usage():
    with pytest.raises(SystemExit, match="usage"):
        data.main(["nonsense"])
