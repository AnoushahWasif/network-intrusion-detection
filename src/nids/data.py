"""Load the official UNSW-NB15 train/test CSVs, checked against hashes recorded in the repo.

    uv run python -m nids.data pin     # once, after downloading the two files by hand
    uv run python -m nids.data check   # hashes, schema, row counts, duplicate report

The official link can't be fetched by a script (decision 2), so the files are put in data/raw/ by
hand. `pin` records their SHA-256 in config/data.sha256 (sha256sum format) after the schema checks
pass; from then on every load is refused unless the bytes match. The hash and the parse use the
same bytes, so the file can't change in between.

These two CSVs have 45 columns: an id, 42 features, attack_cat and label. (The full dataset's
"49 features" include source/destination addresses and times, which these files leave out.)
"""

import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HASH_FILE = ROOT / "config" / "data.sha256"
MAX_BYTES = 200 * 1024 * 1024  # the real files are about 15 MB and 32 MB

# The training file has the larger row count. Checking it catches the two files being swapped.
FILES = {
    "train": ("UNSW_NB15_training-set.csv", 175_341),
    "test": ("UNSW_NB15_testing-set.csv", 82_332),
}

ID, CATEGORY, LABEL = "id", "attack_cat", "label"
CATEGORICAL = ["proto", "service", "state"]
NUMERIC = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", "sload", "dload",
    "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
    "ct_srv_dst", "is_sm_ips_ports",
]  # fmt: skip
FEATURES = [*CATEGORICAL, *NUMERIC]
# Testbed artifacts that nearly give the label away; reported with and without (decision 4).
TTL = ["sttl", "dttl", "ct_state_ttl"]
COLUMNS = {ID, *FEATURES, CATEGORY, LABEL}

NORMAL = "Normal"
CATEGORIES = {
    NORMAL, "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
    "Analysis", "Backdoor", "Shellcode", "Worms",
}  # fmt: skip


class DataError(ValueError):
    """The file isn't the one pinned, or doesn't look like UNSW-NB15."""


def raw_dir() -> Path:
    return Path(os.environ.get("NIDS_DATA_DIR", ROOT / "data")) / "raw"


def read_hashes(hash_file: Path = HASH_FILE) -> dict[str, str]:
    """File name -> SHA-256, from sha256sum-format lines. Missing file means nothing pinned."""
    if not hash_file.exists():
        return {}
    pinned = {}
    for n, line in enumerate(hash_file.read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        digest = parts[0].lower() if parts else ""
        if len(parts) != 2 or len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest):
            raise DataError(f"{hash_file.name} line {n} isn't '<sha256>  <file name>'")
        pinned[parts[1].lstrip("*")] = digest
    return pinned


def _read_bytes(path: Path) -> bytes:
    if not path.exists():
        raise DataError(
            f"{path} is missing. Download the official UNSW-NB15 train/test CSVs by hand "
            "(see README) into that folder."
        )
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise DataError(f"{path.name} is {size:,} bytes, over the {MAX_BYTES:,} cap")
    return path.read_bytes()


def parse(raw: bytes, expected_rows: int) -> pd.DataFrame:
    """Parse and check one CSV. Every problem found is a refusal, not a warning."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataError("not UTF-8 text") from exc
    try:
        # keep_default_na=False: a text cell like "NA" stays text, and an empty numeric cell
        # turns the column into text, which the type check below then refuses.
        df = pd.read_csv(
            io.StringIO(text),
            dtype={c: str for c in [*CATEGORICAL, CATEGORY]},
            keep_default_na=False,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataError(f"not a readable CSV: {exc}") from exc

    header = list(df.columns)
    if len(header) != len(set(header)) or set(header) != COLUMNS:
        missing = sorted(COLUMNS - set(header))
        extra = sorted(set(header) - COLUMNS)
        raise DataError(f"unexpected columns: missing {missing}, extra {extra}")
    if len(df) != expected_rows:
        raise DataError(
            f"{len(df):,} rows, expected {expected_rows:,} (are the train and test files swapped?)"
        )

    for col in [ID, *NUMERIC, LABEL]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataError(f"column {col} has non-numeric or empty values")
    values = df[NUMERIC].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = [c for c in NUMERIC if not np.isfinite(df[c].to_numpy(dtype=float)).all()]
        raise DataError(f"non-finite values in {bad}")
    if (values < 0).any():
        bad = [c for c in NUMERIC if (df[c] < 0).any()]
        raise DataError(f"negative values in {bad}")
    if df[ID].duplicated().any():
        raise DataError("duplicate ids")

    for col in [*CATEGORICAL, CATEGORY]:
        df[col] = df[col].str.strip()
        if (df[col] == "").any():
            raise DataError(f"empty values in {col}")
    unknown = set(df[CATEGORY]) - CATEGORIES
    if unknown:
        raise DataError(f"unknown attack categories {sorted(unknown)}")
    if not df[LABEL].isin([0, 1]).all():
        raise DataError("label must be 0 or 1")
    if ((df[CATEGORY] == NORMAL) != (df[LABEL] == 0)).any():
        raise DataError("label disagrees with attack_cat (Normal must be 0, attacks 1)")
    return df


def load(split: str, folder: Path | None = None, hash_file: Path = HASH_FILE) -> pd.DataFrame:
    """One split, refused unless its bytes match the pinned hash."""
    name, rows = FILES[split]
    pinned = read_hashes(hash_file).get(name)
    if pinned is None:
        raise DataError(f"{name} has no pinned hash; run `uv run python -m nids.data pin` once")
    raw = _read_bytes((folder or raw_dir()) / name)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != pinned:
        raise DataError(f"{name} hash {digest[:12]}... doesn't match the pinned {pinned[:12]}...")
    return parse(raw, rows)


def pin(folder: Path | None = None, hash_file: Path = HASH_FILE) -> dict[str, str]:
    """Record the hashes of the downloaded files, once, and only if they pass every check."""
    if read_hashes(hash_file):
        raise DataError(f"{hash_file.name} already pins files; delete it to pin new ones")
    digests = {}
    for name, rows in FILES.values():
        raw = _read_bytes((folder or raw_dir()) / name)
        parse(raw, rows)
        digests[name] = hashlib.sha256(raw).hexdigest()
    hash_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{digest}  {name}" for name, digest in digests.items()]
    hash_file.write_text("\n".join(lines) + "\n")
    return digests


def duplicates(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, int]:
    """How many flows repeat, on the features plus category (the id always differs)."""
    cols = [*FEATURES, CATEGORY]
    features_only = train.drop_duplicates(cols)
    conflicting = features_only.duplicated(FEATURES, keep=False)
    seen = set(map(tuple, train[cols].itertuples(index=False)))
    return {
        "train_rows": len(train),
        "train_repeats": int(train.duplicated(cols).sum()),
        "train_same_features_different_category": int(conflicting.sum()),
        "test_rows": len(test),
        "test_repeats": int(test.duplicated(cols).sum()),
        "test_rows_also_in_train": int(
            sum(row in seen for row in map(tuple, test[cols].itertuples(index=False)))
        ),
    }


def check() -> dict:
    train, test = load("train"), load("test")
    return {
        "categories": {
            "train": train[CATEGORY].value_counts().to_dict(),
            "test": test[CATEGORY].value_counts().to_dict(),
        },
        "duplicates": duplicates(train, test),
    }


def main(argv: list[str]) -> None:
    if argv == ["pin"]:
        for name, digest in pin().items():
            print(f"pinned {name}: {digest}")
    elif argv == ["check"]:
        print(json.dumps(check(), indent=2))
    else:
        sys.exit("usage: python -m nids.data pin|check")


if __name__ == "__main__":
    main(sys.argv[1:])
