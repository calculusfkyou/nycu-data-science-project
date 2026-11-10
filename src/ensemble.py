"""Comparison method (docs/plan.md §6.7): equal-weight ensemble of the three Transformers per setting.

For each setting whose BERT, RoBERTa and DeBERTa runs (ratio 1, seed 42) all have predictions, averages their
probabilities and writes outputs/ensemble_{setting}/ in the same format as a training run, so evaluate.py scores it
like any other run. One natural configuration only (all three members, weights 1/3 each); nothing is tuned on test.

Run: .venv/Scripts/python src/ensemble.py [--outputs outputs/smoke]   (then src/evaluate.py)
Re-run it whenever a member is re-run, before evaluate.py; otherwise evaluate.py scores the stale ensemble.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import EVAL_SPLITS, LABELS, OUTPUTS, SEED, environment, save_predictions, write_json

MEMBERS = ["bert", "roberta", "deberta"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--outputs", type=Path, default=OUTPUTS, help="folder holding one sub-folder per run")
outdir = ap.parse_args().outputs

for setting in ["e0", "e1", "e2"]:
    dirs = [outdir / f"{m}_{setting}" for m in MEMBERS]
    if not all((d / f"preds_{s}.parquet").exists() for d in dirs for s in EVAL_SPLITS):
        print(f"skip {setting}: not all of {MEMBERS} have predictions")
        if (outdir / f"ensemble_{setting}").exists():
            print(f"WARNING: {outdir / f'ensemble_{setting}'} is left from an earlier run and may be stale; "
                  "delete it before running evaluate.py.")
        continue
    cfgs = [json.loads((d / "config.json").read_text(encoding="utf-8")) for d in dirs]
    # Members must share the data (same files, same fingerprints) and the main-experiment settings.
    assert all(c["data"] == cfgs[0]["data"] for c in cfgs), f"{setting}: members trained/evaluated on different data"
    assert all(c["seed"] == SEED and c.get("ratio", 1.0) == 1.0 for c in cfgs), f"{setting}: not main-experiment runs"
    smoke = {c["smoke"] for c in cfgs}
    assert len(smoke) == 1, f"{setting}: mixes smoke and full runs"

    run = f"ensemble_{setting}"
    run_dir = outdir / run
    run_dir.mkdir(exist_ok=True)
    for s in EVAL_SPLITS:
        preds = [pd.read_parquet(d / f"preds_{s}.parquet").set_index("row").sort_index() for d in dirs]
        assert all(p.index.equals(preds[0].index) for p in preds), f"{setting}/{s}: members cover different rows"
        probs = np.mean([p[LABELS].to_numpy(dtype="float64") for p in preds], axis=0)
        save_predictions(run_dir, s, preds[0].index.to_numpy(), probs)
    write_json(run_dir / "config.json", {
        "run": run, "model": "ensemble", "setting": setting, "seed": SEED, "ratio": 1.0, "smoke": smoke.pop(),
        "members": [d.name for d in dirs], "weights": [1 / len(dirs)] * len(dirs), "combination": "mean probability",
        "member_git_commits": [c.get("git_commit") for c in cfgs],
        "data": cfgs[0]["data"],
        **environment(["numpy", "pandas", "pyarrow"]),
    })
    print(f"Wrote {run_dir} ({' + '.join(d.name for d in dirs)})")
