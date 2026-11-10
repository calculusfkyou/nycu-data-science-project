# seed-42 主實驗結果

9 個 Transformer runs（BERT／RoBERTa／DeBERTa-v3 × E0／E1／E2），seed 42，batch 16，3 epochs，RTX 5070（bf16），加上本機 CPU 跑的 TF-IDF + LR baseline（只有 E0）。本資料夾的 `results.md`、`metrics_long.csv`、`val_curves.png` 是 `src/evaluate.py` 對這 10 個 runs 的原始輸出；`analysis.md` 是依這些結果撰寫的錯誤分析與解讀（plan §6.4、§6.6）。

## 程式版本

**Transformer runs**
- 每個 run 的 `config.json` 記錄的 commit 是 `17a4da8-dirty`。
- 實際執行的程式是 `17a4da8` 加上 `src/train.py` 的 `_save` 修正，修正內容和 commit `40478f0` 的 `src/train.py` 完全相同。
- `40478f0` 的 parent 是 `a446dac`，它是 `17a4da8` 重新 commit 的版本，兩者的 tree 完全相同。
- 證據在 release 的 `outputs/execution/`：
  - `source.patch`：執行時未 commit 的修改，就是 `40478f0` 對 `src/train.py` 的 diff。
  - `queue.status.json`：每個 run 前檢查的 `src/train.py` SHA256（CRLF）為 `95A90249A3F79BC0EC822F3FA1A881A13750A92E5122456FD20C1D9D4E910E46`。

**Baseline（`tfidf_lr_e0`）**
- `config.json` 記錄的 commit 是 `49dcf61-dirty`：在 `49dcf61` 上執行，未 commit 的修改就是之後 commit `f97e648` 的 `src/baseline.py`（另存 `model.joblib`）。
- 預測檔和 `17a4da8` 版 `baseline.py` 的輸出逐 byte 相同（C=2，val mean ROC-AUC 0.98587）。

之後的 runs 一律從乾淨的 commit 執行，`git_commit` 不應帶 `-dirty`。

## 完整輸出

- Release：https://github.com/a1105516/nycu-data-science-project/releases/tag/experiments-seed42-2026-11-08
- 內含每個 run 的 prediction parquets、config／runtime／log、執行紀錄，以及 `results.md` 第 5 節提到的 `error_samples_deberta.csv`（含原始留言，不放進 repo）。
- 不含 checkpoint（依 `docs/run_guide.md` 的估計，每個 run 2.6–4.2GB），保留在執行者的電腦上。
- 不含 TF-IDF baseline；baseline 的輸出在本機 `outputs/tfidf_lr_e0/`（gitignored）。

## 解讀注意

只有單一 seed，三個模型 E0／E1／E2 之間的 |Δ Mean ROC-AUC| 都 < 0.002。依 `docs/plan.md` 第 7 節，要先做 bootstrap CI，再視結果做 DeBERTa 多 seed，才能下結論。
