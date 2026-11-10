"""Phase 4 evaluation (docs/plan.md §6). Reads prediction files only; never re-runs a model.

Run: .venv/Scripts/python src/evaluate.py [--outputs outputs/smoke]

Writes {outputs}/results.md, metrics_long.csv, val_curves.png, error_samples_{model}.csv and bootstrap_ci.csv.
Section 9 adds threshold-0.5 metrics (precision, recall, MCC, micro F1, Hamming loss, subset accuracy) with references.
Refinement (plan §7 items 1-2) is included: paired bootstrap CI of Δ ROC-AUC and val-tuned per-label thresholds.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

from common import (EVAL_SPLITS, F1_THRESHOLD, LABELS, OUTPUTS, PROCESSED, SEED, SPLIT_LABELS, compute_metrics, fingerprint,
                    load_train, md_table)

MODEL_ORDER = ["tfidf_lr", "bert", "roberta", "deberta", "ensemble"]  # ensemble: src/ensemble.py (plan §6.7)
DELTAS = [("e1", "e0", "E1 − E0（hard augmentation）"), ("e2", "e0", "E2 − E0（soft augmentation）"),
          ("e2", "e1", "E2 − E1（soft vs. hard）")]
FOCUS_LABELS = ["threat", "identity_hate"]  # plan §6.4: read 20 E0 false negatives each
ERROR_MODELS = ["deberta"]
N_SAMPLES = 20
def _mazari(macro, per_label) -> dict:
    """Mazari et al. (2024) Table 3 in % -> fractions keyed like metrics_long.csv.
    macro = (P, R, F1, MCC) Macro Average; per_label = {label: (P, R, F1, MCC)}."""
    d = dict(zip(["macro_precision", "macro_recall", "macro_f1", "macro_mcc"], np.array(macro) / 100))
    for l, prf in per_label.items():
        d |= dict(zip([f"precision/{l}", f"recall/{l}", f"f1/{l}", f"mcc/{l}"], np.array(prf) / 100))
    return d


# plan §6.5: references that report threshold-based metrics, keyed like metrics_long.csv columns. A missing key is
# shown as "—"; a string is shown verbatim.
THRESHOLD_REFERENCES = [
    ("Mazari BERT-base", _mazari((49.63, 77.65, 59.83, 60.62), {
        "toxic": (47.97, 92.82, 63.25, 65.52), "severe_toxic": (32.78, 65.12, 43.61, 43.79),
        "obscene": (61.62, 82.23, 70.45, 67.89), "threat": (56.36, 58.77, 57.54, 55.62),
        "insult": (53.24, 83.37, 64.98, 67.78), "identity_hate": (45.77, 83.57, 59.15, 63.12)})),
    ("Mazari BERT-3_Models (W-AVG)", _mazari((58.50, 67.78, 62.18, 61.64), {
        "toxic": (56.86, 89.43, 69.52, 67.03), "severe_toxic": (41.23, 40.33, 40.77, 44.15),
        "obscene": (60.50, 80.76, 69.18, 69.04), "threat": (57.40, 60.66, 58.99, 57.44),
        "insult": (67.04, 76.51, 71.46, 68.48), "identity_hate": (67.96, 58.99, 63.16, 63.68)})),
    ("Rahman LR + TF-IDF/word", {"subset_accuracy": "0.898（accuracy，定義不明）"}),
]
N_BOOT = 1000  # plan §7 item 1: paired bootstrap resamples of Jigsaw test
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)  # plan §7 item 2
# Mean ROC-AUC values as listed in plan §6.5. Different setups, so a range, not a ranking.
REFERENCES = [
    ("Rahman et al. (2023)", "baseline 參考：傳統 ML", "0.973"),
    ("Mazari et al. (2024)", "SOTA 參考：BERT + Bi-LSTM/Bi-GRU ensemble", "0.9863"),
    ("Kaggle 第一名", "private leaderboard，4,539 隊", "0.98856"),
]


def threshold_metrics(y, hit, labels) -> dict:
    """Plan §6.1 supplementary metrics at threshold 0.5 (y, hit: boolean n × labels), mostly for §6.5 comparisons.
    Macro values are plain means over labels, matching Mazari et al. Table 3's macro average."""
    out = {}
    for i, l in enumerate(labels):
        out[f"precision/{l}"] = precision_score(y[:, i], hit[:, i], zero_division=0)
        out[f"recall/{l}"] = recall_score(y[:, i], hit[:, i], zero_division=0)
        out[f"mcc/{l}"] = matthews_corrcoef(y[:, i], hit[:, i])
    for name in ["precision", "recall", "mcc"]:
        out[f"macro_{name}"] = float(np.mean([out[f"{name}/{l}"] for l in labels]))
    out["micro_f1"] = f1_score(y, hit, average="micro", zero_division=0)
    out["hamming_loss"] = hamming_loss(y, hit)
    out["subset_accuracy"] = accuracy_score(y, hit)  # exact match of all labels
    return out


def load_runs(outdir: Path, truth: dict) -> pd.DataFrame:
    local_fp = {s: fingerprint(df) for s, df in truth.items()}
    train_fp = {}  # (setting, ratio) -> fingerprint of the local training set, computed once
    rows = []
    for cfg_path in sorted(outdir.glob("*/config.json")):
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        run_dir = cfg_path.parent
        if not all((run_dir / f"preds_{s}.parquet").exists() for s in EVAL_SPLITS):
            print(f"skip {run_dir.name}: predictions incomplete (still training?)")
            continue
        # The training set must match too, otherwise E1/E2 may hold a different external sample on another machine.
        key = (cfg["setting"], cfg.get("ratio", 1.0))
        if key not in train_fp:
            train_fp[key] = fingerprint(load_train(*key))
        assert cfg["data"]["train"]["fingerprint"] == train_fp[key], (
            f"{run_dir.name}: trained on different data than data/processed (train fingerprint mismatch)")
        for s in EVAL_SPLITS:
            assert cfg["data"][s]["fingerprint"] == local_fp[s], (
                f"{run_dir.name}: {s} was built from different data than data/processed (fingerprint mismatch)")
            preds = pd.read_parquet(run_dir / f"preds_{s}.parquet")
            if not cfg["smoke"]:
                assert np.array_equal(np.sort(preds["row"]), np.arange(len(truth[s]))), f"{run_dir.name}: {s} rows"
            labels = SPLIT_LABELS[s]
            m = compute_metrics(truth[s][labels].iloc[preds["row"]], preds[labels], labels)
            m |= threshold_metrics(truth[s][labels].iloc[preds["row"]].to_numpy() >= 0.5,
                                   preds[labels].to_numpy() >= F1_THRESHOLD, labels)
            if not cfg["smoke"]:  # a NaN would silently turn the mean into a mean over fewer labels
                assert not any(np.isnan(v) for v in m.values()), f"{run_dir.name}: NaN metric on {s}"
            rows.append({"run": cfg["run"], "model": cfg["model"], "setting": cfg["setting"],
                         "ratio": cfg.get("ratio", 1.0), "seed": cfg["seed"], "split": s, **m})
    df = pd.DataFrame(rows)
    df["model_rank"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return df.sort_values(["model_rank", "setting", "ratio", "seed", "split"]).drop(columns="model_rank")


def fmt(x, signed=False):
    return "—" if pd.isna(x) else (f"{x:+.4f}" if signed else f"{x:.4f}")


def main_runs(metrics):
    return metrics[(metrics["ratio"] == 1.0) & (metrics["seed"] == SEED)]


def tables(metrics) -> list[str]:
    md = []
    test, val = metrics[metrics["split"] == "test_jigsaw"], metrics[metrics["split"] == "val"].set_index("run")
    md += ["## 1. 主表：Jigsaw official labeled test", ""]
    md += md_table(["Run", "Model", "Setting", "Val mean ROC-AUC", "Mean ROC-AUC", "Mean PR-AUC", "Macro F1"],
                   [[r.run, r.model, r.setting, fmt(val.loc[r.run, "mean_roc_auc"]), fmt(r.mean_roc_auc),
                     fmt(r.mean_pr_auc), fmt(r.macro_f1)] for r in test.itertuples()]
                   + [[f"*{name}*", desc, "—", "—", auc, "—", "—"] for name, desc, auc in REFERENCES])
    md += ["斜體列是參考值（plan §6.5），評估設定和本專案不同，只作為範圍參考，不是同一基準下的排名："
           "Rahman 用自己的切分；Mazari 在官方 labeled test（63,977 列）上評估，但 ensemble 權重依 test 表現設定；"
           "Kaggle private leaderboard 用的是官方 test 中的 private 部分，本表的 test 是全部已標註的列。"
           "Precision、Recall、MCC、Micro F1、Hamming loss、accuracy 和文獻的比較見第 9 節。"
           "ensemble_* 是 BERT、RoBERTa、DeBERTa 三個同 setting run 的等權重機率平均（plan §6.7，src/ensemble.py）。", ""]

    md += ["## 2. Per-label：Jigsaw official labeled test", ""]
    for key, name in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]:
        md += [f"**{name}**", ""]
        md += md_table(["Run", *LABELS], [[r["run"], *(fmt(r[f"{key}/{l}"]) for l in LABELS)]
                                          for _, r in test.iterrows()])

    delta_md = []
    main = main_runs(test).set_index(["model", "setting"])
    for model in [m for m in MODEL_ORDER if m in main.index.get_level_values(0)]:
        pairs = [(a, b, title) for a, b, title in DELTAS if (model, a) in main.index and (model, b) in main.index]
        if not pairs:
            continue
        delta_md += [f"**{model}**", ""]
        for key, name in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]:
            mean_key = f"mean_{key}"
            delta_md += md_table([f"Δ {name}", "Mean", *LABELS],
                                 [[title, fmt(main.loc[(model, a), mean_key] - main.loc[(model, b), mean_key], True),
                                   *(fmt(main.loc[(model, a), f"{key}/{l}"] - main.loc[(model, b), f"{key}/{l}"],
                                         True) for l in LABELS)] for a, b, title in pairs])
    if delta_md:
        md += ["## 3. Δ 表（Jigsaw test；主實驗 runs：ratio 1、seed 42）", "", *delta_md]

    civ_labels = SPLIT_LABELS["test_civil"]
    civ = metrics[metrics["split"] == "test_civil"]
    md += ["## 4. Cross-dataset：Civil test（五類）", ""]
    md += md_table(["Run", "Mean ROC-AUC", "Mean PR-AUC", "Macro F1"],
                   [[r["run"], fmt(r["mean_roc_auc"]), fmt(r["mean_pr_auc"]), fmt(r["macro_f1"])]
                    for _, r in civ.iterrows()])
    for key, name in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]:
        md += [f"**{name}**", ""]
        md += md_table(["Run", *civ_labels], [[r["run"], *(fmt(r[f"{key}/{l}"]) for l in civ_labels)]
                                              for _, r in civ.iterrows()])
    return md


def val_curves(outdir: Path, path: Path) -> bool:
    fig, ax = plt.subplots(figsize=(9, 5))
    drawn = False
    for log_path in sorted(outdir.glob("*/log_history.json")):
        log = [e for e in json.loads(log_path.read_text(encoding="utf-8")) if "eval_mean_roc_auc" in e]
        if log:
            ax.plot([e["epoch"] for e in log], [e["eval_mean_roc_auc"] for e in log], marker="o",
                    label=log_path.parent.name)
            drawn = True
    if drawn:
        ax.set(xlabel="Epoch", ylabel="Val mean ROC-AUC", title="Validation mean ROC-AUC during training")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return drawn


def error_analysis(outdir: Path, metrics, test: pd.DataFrame) -> list[str]:
    """Plan §6.4 at threshold 0.5 on Jigsaw test: FP/FN per setting, E0 -> E2 flips, E0 false-negative samples."""
    md = []
    main = main_runs(metrics[metrics["split"] == "test_jigsaw"])
    for model in ERROR_MODELS:
        runs = main[main["model"] == model].set_index("setting")["run"]
        if not {"e0", "e2"} <= set(runs.index):
            continue
        preds = {s: pd.read_parquet(outdir / run / "preds_test_jigsaw.parquet").set_index("row").sort_index()
                 for s, run in runs.items()}
        rows = preds["e0"].index.to_numpy()  # all rows for real runs, the 1% subset for smoke runs
        assert all(np.array_equal(p.index, rows) for p in preds.values()), f"{model}: settings cover different rows"
        sub = test.iloc[rows]
        y = sub[LABELS].to_numpy() >= 0.5
        pred = {s: p[LABELS].to_numpy() for s, p in preds.items()}
        hit = {k: v >= 0.5 for k, v in pred.items()}
        md += [f"## 5. 錯誤分析：{model}（Jigsaw test，threshold 0.5）", ""]
        md += md_table(["Setting", *(f"{l} FP / FN" for l in LABELS)],
                       [[s, *(f"{(h[:, i] & ~y[:, i]).sum():,} / {(~h[:, i] & y[:, i]).sum():,}"
                              for i in range(len(LABELS)))] for s, h in sorted(hit.items())])
        e0, e2 = hit["e0"], hit["e2"]
        md += ["**E0 → E2 預測翻轉**", ""]
        md += md_table(["Flip", *LABELS], [
            [name, *(f"{(mask[:, i]).sum():,}" for i in range(len(LABELS)))] for name, mask in [
                ("FN → TP", ~e0 & e2 & y), ("TP → FN", e0 & ~e2 & y),
                ("TN → FP", ~e0 & e2 & ~y), ("FP → TN", e0 & ~e2 & ~y)]])
        samples = []
        for l in FOCUS_LABELS:
            i = LABELS.index(l)
            fn = np.flatnonzero(~e0[:, i] & y[:, i])
            pick = np.random.default_rng(SEED).choice(fn, size=min(N_SAMPLES, len(fn)), replace=False)
            samples.append(pd.DataFrame({"label": l, "row": rows[pick], "id": sub["id"].to_numpy()[pick],
                                         "text": sub["text"].to_numpy()[pick], "prob_e0": pred["e0"][pick, i],
                                         "prob_e2": pred["e2"][pick, i]}))
        out = outdir / f"error_samples_{model}.csv"
        samples = pd.concat(samples)
        samples.to_csv(out, index=False, encoding="utf-8-sig")
        counts = "、".join(f"{l} {n} 筆" for l, n in samples.groupby("label", sort=False).size().items())
        md += [f"E0 false negatives（每類最多 {N_SAMPLES} 筆，seed {SEED}；實際 {counts}）已存到 `{out.name}`，"
               "供人工閱讀。", ""]
    return md


def read_preds(outdir: Path, run: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(outdir / run / f"preds_{split}.parquet").set_index("row").sort_index()


def auc_parts(y, s):
    """Precompute each row's tie group (groups ordered by score) once, so each bootstrap ROC-AUC is O(n)."""
    _, group = np.unique(s, return_inverse=True)
    return group, y


def weighted_auc(parts, w):
    """ROC-AUC with row weights w (bootstrap counts): P(score_pos > score_neg) + 0.5 * P(tie)."""
    group, y = parts
    neg, pos = np.bincount(group, w * ~y), np.bincount(group, w * y)
    below = np.cumsum(neg) - neg  # negative weight strictly below each tie group
    with np.errstate(invalid="ignore"):  # NaN if a resample has one class (only on tiny smoke subsets)
        return (pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum())


def bootstrap_ci(outdir: Path, metrics, test: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Plan §7 item 1: paired bootstrap 95% CI of Δ ROC-AUC on Jigsaw test (same resampled rows for both settings)."""
    md, rows_out = [], []
    main = main_runs(metrics[metrics["split"] == "test_jigsaw"])
    for model in [m for m in MODEL_ORDER if m in set(main["model"])]:
        runs = main[main["model"] == model].set_index("setting")["run"]
        pairs = [(a, b, title) for a, b, title in DELTAS if {a, b} <= set(runs.index)]
        if not pairs:
            continue
        preds = {s: read_preds(outdir, run, "test_jigsaw") for s, run in runs.items()}
        rows = next(iter(preds.values())).index.to_numpy()
        assert all(np.array_equal(p.index, rows) for p in preds.values()), f"{model}: settings cover different rows"
        y = test.iloc[rows][LABELS].to_numpy() >= 0.5
        parts = {(s, i): auc_parts(y[:, i], p[LABELS].to_numpy()[:, i]) for s, p in preds.items()
                 for i in range(len(LABELS))}
        ones = np.ones(len(rows))
        full = {k: weighted_auc(v, ones) for k, v in parts.items()}
        for (s, i), auc in full.items():  # self-check against sklearn on the unresampled data
            assert abs(auc - roc_auc_score(y[:, i], preds[s][LABELS[i]])) < 1e-9, (model, s, LABELS[i])
        rng = np.random.default_rng(SEED)  # same resamples for every model
        boot = {k: np.empty(N_BOOT) for k in parts}
        for b in range(N_BOOT):
            w = np.bincount(rng.integers(0, len(rows), len(rows)), minlength=len(rows))
            for k, v in parts.items():
                boot[k][b] = weighted_auc(v, w)
        cells = []
        for a, b, title in pairs:
            per_label = [(boot[(a, i)] - boot[(b, i)], full[(a, i)] - full[(b, i)]) for i in range(len(LABELS))]
            mean = (np.mean([d for d, _ in per_label], axis=0), np.mean([f for _, f in per_label]))
            row_cells = [title]
            for name, (dist, point) in zip(["Mean", *LABELS], [mean, *per_label]):
                lo, hi = np.nanpercentile(dist, [2.5, 97.5])
                excl = lo > 0 or hi < 0
                rows_out.append({"model": model, "delta": f"{a}-{b}", "label": name, "point": point,
                                 "ci_low": lo, "ci_high": hi, "excludes_zero": excl})
                row_cells.append(f"{point:+.4f} [{lo:+.4f}, {hi:+.4f}]{' *' if excl else ''}")
            cells.append(row_cells)
        md += [f"**{model}**", ""] + md_table(["Δ ROC-AUC", "Mean", *LABELS], cells)
    if md:
        md = [f"## 7. Bootstrap 95% CI：Δ ROC-AUC（Jigsaw test，paired，{N_BOOT:,} 次，seed {SEED}）", "",
              "格式：點估計 [2.5%, 97.5%]；`*` 表示 CI 不含 0。只反映測試資料的抽樣誤差，不含訓練隨機性（單一 seed）。",
              ""] + md
    return md, pd.DataFrame(rows_out)


def f1_grid(prob, y):
    """F1 at every THRESHOLD_GRID value (columns) for one label."""
    hit = prob[:, None] >= THRESHOLD_GRID
    tp = (hit & y[:, None]).sum(0)
    return 2 * tp / np.maximum(2 * tp + (hit & ~y[:, None]).sum(0) + (~hit & y[:, None]).sum(0), 1)


def tuned_thresholds(outdir: Path, metrics, truth: dict) -> list[str]:
    """Plan §7 item 2: per label, pick the F1-best threshold on val (lowest one on ties), apply it to Jigsaw test.
    Reported separately; the 0.5 results stay the main ones."""
    rows_f1, rows_t = [], []
    test = main_runs(metrics[metrics["split"] == "test_jigsaw"])
    at_half = int(np.flatnonzero(THRESHOLD_GRID == F1_THRESHOLD)[0])
    for _, r in test.iterrows():
        val_p, test_p = read_preds(outdir, r["run"], "val"), read_preds(outdir, r["run"], "test_jigsaw")
        y_val = truth["val"].iloc[val_p.index][LABELS].to_numpy() >= 0.5
        y_test = truth["test_jigsaw"].iloc[test_p.index][LABELS].to_numpy() >= 0.5
        cells, f1s = [], []
        for i, l in enumerate(LABELS):
            test_f1 = f1_grid(test_p[l].to_numpy(), y_test[:, i])
            assert abs(test_f1[at_half] - r[f"f1/{l}"]) < 1e-9, (r["run"], l)  # same F1 as compute_metrics at 0.5
            k = int(np.argmax(f1_grid(val_p[l].to_numpy(), y_val[:, i])))
            f1s.append(test_f1[k])
            cells.append(f"{THRESHOLD_GRID[k]:.2f} / {test_f1[k]:.4f}")
        tuned = float(np.mean(f1s))
        rows_f1.append([r["run"], fmt(r["macro_f1"]), fmt(tuned), fmt(tuned - r["macro_f1"], True)])
        rows_t.append([r["run"], *cells])
    md = ["## 8. Per-label threshold（val 上選 F1 最高的 threshold，套用到 Jigsaw test）", "",
          f"搜尋 {THRESHOLD_GRID[0]:.2f}–{THRESHOLD_GRID[-1]:.2f}（step 0.01），val 上 F1 並列最高時取最小的 threshold。"
          "另外報告，不取代 threshold 0.5 的主結果。", ""]
    md += md_table(["Run", "Macro F1（0.5）", "Macro F1（val 選）", "Δ"], rows_f1)
    md += ["**各 label：threshold / test F1**", ""] + md_table(["Run", *LABELS], rows_t)
    return md


def threshold_table(metrics, test: pd.DataFrame) -> list[str]:
    """Plan §6.1 supplementary metrics at threshold 0.5 on Jigsaw test, with §6.5 references that report them."""
    runs = main_runs(metrics[metrics["split"] == "test_jigsaw"])
    cols = ["macro_precision", "macro_recall", "macro_f1", "micro_f1", "macro_mcc", "hamming_loss", "subset_accuracy"]
    y = test[LABELS].to_numpy() >= 0.5
    zero = threshold_metrics(y, np.zeros_like(y), LABELS) | {"macro_f1": 0.0}
    md = ["## 9. Threshold 0.5 指標與文獻比較（Jigsaw test）", ""]
    md += md_table(["Run", "Macro P", "Macro R", "Macro F1", "Micro F1", "Macro MCC", "Hamming loss",
                    "Subset accuracy"],
                   [[r["run"], *(fmt(r[c]) for c in cols)] for _, r in runs.iterrows()]
                   + [["*全部預測為 0*", *(fmt(zero[c]) for c in cols)]]
                   + [[f"*{name}*", *(ref_cell(ref.get(c)) for c in cols)] for name, ref in THRESHOLD_REFERENCES])
    md += ["Macro 是六個 label 的平均。斜體列是參考：「全部預測為 0」由 test 標籤直接算出；Mazari et al. (2024) 的數字"
           "取自 Table 3（官方 test，原文未說明 threshold，max length 128）；Rahman et al. (2023) 只報告 accuracy"
           "（Table 4，LR + TF-IDF/word；原文未說明是 subset 還是 per-label accuracy，最高為 SGD + Hashing/word 的 0.907），"
           "而本表 test 有約九成留言六個 label 全為 0，accuracy 幾乎無法區分模型好壞。", ""]
    md += ["**各 label：Precision / Recall / F1**", ""]
    md += md_table(["Run", *LABELS],
                   [[r["run"], *(f"{r[f'precision/{l}']:.4f} / {r[f'recall/{l}']:.4f} / {r[f'f1/{l}']:.4f}"
                                 for l in LABELS)] for _, r in runs.iterrows()]
                   + [[f"*{name}*", *(f"{ref[f'precision/{l}']:.4f} / {ref[f'recall/{l}']:.4f} / {ref[f'f1/{l}']:.4f}"
                                      for l in LABELS)]
                      for name, ref in THRESHOLD_REFERENCES if f"f1/{LABELS[0]}" in ref])
    md += ["**各 label：MCC**", ""]
    md += md_table(["Run", *LABELS, "Macro"],
                   [[r["run"], *(fmt(r[f"mcc/{l}"]) for l in LABELS), fmt(r["macro_mcc"])] for _, r in runs.iterrows()]
                   + [[f"*{name}*", *(fmt(ref[f"mcc/{l}"]) for l in LABELS), fmt(ref["macro_mcc"])]
                      for name, ref in THRESHOLD_REFERENCES if f"mcc/{LABELS[0]}" in ref])
    return md


def ref_cell(v) -> str:
    return "—" if v is None else v if isinstance(v, str) else fmt(v)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs", type=Path, default=OUTPUTS, help="folder holding one sub-folder per run")
    outdir = p.parse_args().outputs
    truth = {s: pd.read_parquet(PROCESSED / f"{s}.parquet") for s in EVAL_SPLITS}
    metrics = load_runs(outdir, truth)
    assert len(metrics), f"no finished runs under {outdir}"
    metrics.to_csv(outdir / "metrics_long.csv", index=False)

    md = ["# Results", "", f"Generated by `src/evaluate.py` from `{outdir.name}/`; do not edit by hand.", "",
          "Metrics follow plan §6.1; F1 uses threshold 0.5.", ""]
    md += tables(metrics)
    md += error_analysis(outdir, metrics, truth["test_jigsaw"])
    if val_curves(outdir, outdir / "val_curves.png"):
        md += ["## 6. 訓練曲線", "", "![val curves](val_curves.png)", ""]
    ci_md, ci = bootstrap_ci(outdir, metrics, truth["test_jigsaw"])
    md += ci_md
    if len(ci):
        ci.to_csv(outdir / "bootstrap_ci.csv", index=False)
    md += tuned_thresholds(outdir, metrics, truth)
    md += threshold_table(metrics, truth["test_jigsaw"])
    (outdir / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Evaluated {metrics['run'].nunique()} runs. Wrote {outdir / 'results.md'}")


if __name__ == "__main__":
    main()
