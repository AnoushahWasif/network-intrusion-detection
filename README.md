# network-intrusion-detection

Detecting and classifying attacks in network flow records (UNSW-NB15), measured the way an analyst queue would feel it: how many attacks are caught when only 1% of normal flows may raise an alert.

The point is to check what most intrusion-detection results skip:
- whether the model leans on testbed artifacts (the TTL features);
- how the rare attack classes do;
- whether a model trained on one network still works on another.

**Status:** in progress (day 1 of 9: architecture agreed). Results will go here once there are any.

## Planned approach

1. Load the official UNSW-NB15 train and test files, checked by hash; EDA of attack categories and imbalance
2. One preprocessing pipeline fitted on train only, with leakage checks
3. Binary detection at a fixed false-alarm rate: baselines, random forest, LightGBM, with and without the TTL features
4. Multi-class attack classification with recall per class
5. Anomaly detection trained on normal traffic only vs supervised models
6. SHAP reasons for each alert
7. A Streamlit dashboard that replays test flows as live alerts
8. A cross-dataset test on the NetFlow versions of UNSW-NB15 and CSE-CIC-IDS2018

## Stack

Python 3.12, uv, pandas, scikit-learn, matplotlib, pytest, ruff. CI: GitHub Actions with ruff, pytest, pip-audit and Gitleaks.

## Data

[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) (Moustafa and Slay, 2015 and 2016), free for academic research. The files are downloaded by hand once (the official link can't be scripted), and a script checks their SHA-256 before use. Data and trained models are never committed or re-hosted.

## Quick start

```bash
uv sync
make lint test
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT
