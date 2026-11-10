"""Kaggle late-submission files (docs/plan.md §6.5) from existing Jigsaw test predictions; no model is re-run.

Writes outputs/submissions/{run}.csv in sample_submission.csv format (all 153,164 test ids). Rows that
test_labels.csv marks -1 are not scored by Kaggle, so they get the sample file's 0.5. Only runs fixed in advance are
submitted (no selection by Kaggle score).

Run: .venv/Scripts/python src/make_submission.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import LABELS, OUTPUTS, PROCESSED, RAW

RUNS = ["tfidf_lr_e0", "deberta_e2", "ensemble_e0", "ensemble_e2"]  # fixed in advance (plan §6.5)

sample = pd.read_csv(RAW / "jigsaw" / "sample_submission.csv")
labels = pd.read_csv(RAW / "jigsaw" / "test_labels.csv")
test = pd.read_parquet(PROCESSED / "test_jigsaw.parquet")
assert sample["id"].equals(labels["id"]), "sample_submission and test_labels ids differ"
scored = labels[(labels[LABELS] != -1).all(axis=1)]
assert set(scored["id"]) == set(test["id"]), "test_jigsaw is not exactly the scored test rows"
metrics = pd.read_csv(OUTPUTS / "metrics_long.csv")
out_dir = OUTPUTS / "submissions"
out_dir.mkdir(exist_ok=True)

for run in RUNS:
    preds = pd.read_parquet(OUTPUTS / run / "preds_test_jigsaw.parquet")
    assert np.array_equal(np.sort(preds["row"]), np.arange(len(test))), f"{run}: predictions do not cover test_jigsaw"
    probs = preds.set_index(test["id"].to_numpy()[preds["row"]])[LABELS]
    sub = sample.copy()
    sub[LABELS] = probs.reindex(sub["id"]).to_numpy()
    sub[LABELS] = sub[LABELS].fillna(0.5)  # unscored rows only (every scored id has a prediction, asserted above)
    assert ((sub[LABELS] >= 0) & (sub[LABELS] <= 1)).all().all(), run
    path = out_dir / f"{run}.csv"
    sub.to_csv(path, index=False, float_format="%.9g")  # float32 round-trips

    # Self-check: re-read the file and score the scored rows; must equal metrics_long.csv (evaluate.py).
    back = pd.read_csv(path).set_index("id").loc[scored["id"]]
    auc = np.mean([roc_auc_score(scored[l].to_numpy() == 1, back[l]) for l in LABELS])
    ref = metrics.query("run == @run and split == 'test_jigsaw'")["mean_roc_auc"].item()
    assert abs(auc - ref) < 1e-6, f"{run}: file scores {auc:.6f}, results.md has {ref:.6f}"
    print(f"Wrote {path} ({len(sub):,} rows; local mean ROC-AUC on scored rows {auc:.5f} = results.md)")
