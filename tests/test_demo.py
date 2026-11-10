"""demo.py reproduces the evaluated ensemble_e2 probabilities (docs/plan.md §8).

Needs the local bert/roberta/deberta_e2 checkpoints, outputs/ensemble_e2 and data/processed; skipped otherwise.
Run with: .venv/Scripts/python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import demo
from common import LABELS, OUTPUTS, PROCESSED

NEEDED = [OUTPUTS / run / ckpt for run, ckpt in demo.CHECKPOINTS.items()] + [
    OUTPUTS / "ensemble_e2" / "preds_test_jigsaw.parquet", PROCESSED / "test_jigsaw.parquet"]


@unittest.skipUnless(all(p.exists() for p in NEEDED), "needs local checkpoints, ensemble_e2 predictions and data")
class DemoMatchesEvaluation(unittest.TestCase):
    def test_probabilities(self):
        test = pd.read_parquet(PROCESSED / "test_jigsaw.parquet")
        preds = pd.read_parquet(OUTPUTS / "ensemble_e2" / "preds_test_jigsaw.parquet").set_index("row")
        # First two positives per label, two all-negative rows and the longest text (truncated at 256 tokens);
        # one row is positive for four labels, so 11 distinct rows remain.
        rows = [i for l in LABELS for i in test.index[test[l] == 1][:2]]
        rows += list(test.index[test[LABELS].sum(axis=1) == 0][:2]) + [test["text"].str.len().idxmax()]
        rows = list(dict.fromkeys(rows))
        models = demo.load_models()
        got = np.array([demo.predict(test.loc[r, "text"], models) for r in rows])
        np.testing.assert_allclose(got, preds.loc[rows, LABELS].to_numpy(), atol=1e-4)


if __name__ == "__main__":
    unittest.main()
