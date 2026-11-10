# seed-42 結果分析（Evaluation §6.4、§6.6）

依據：
- 同資料夾的 `results.md`（10 個 runs 和 3 個 ensemble，Jigsaw official labeled test 63,978 列）
- release 內的 `error_samples_deberta.csv`（含原始留言）
- `data/processed/data_summary.md`（各訓練集的正例數）
- `docs/plan.md` §6.5（文獻與 Kaggle 參考值）
- `results.md` 第 7、8 節與 `bootstrap_ci.csv`（Refinement：bootstrap CI、per-label threshold）
- `results.md` 第 9 節（threshold 0.5 指標與文獻比較）

**結論的適用範圍。** 每個設定只有一個 seed（42），E0／E1／E2 之間的 |Δ Mean ROC-AUC| 都 < 0.002。Bootstrap 95% CI（第 1.7 節）只反映測試資料的抽樣誤差；多 seed 決定不做（plan §7），所以「CI 不含 0」只代表在 seed 42 下差異超出測試抽樣誤差，不能推論到其他 seed。

---

## 1. 解讀（§6.6）

### 1.1 Transformer 與 baseline
- 三個 Transformer 的 Mean ROC-AUC（0.9843–0.9864）都高於 TF-IDF + LR baseline（0.9801），差距 +0.004～+0.006；Mean PR-AUC 差距 +0.01～+0.07。
- Baseline 0.9801 高於 Rahman et al. (2023) 的 0.973。單一模型中最好的 run（BERT E2，0.9864）和 Mazari et al. (2024) 的 0.9863 相當，但 Mazari 依 test 表現設定 ensemble 權重，不是同一基準。和 Kaggle 第一名（private LB 0.98856）的同一基準比較見第 1.8 節。
- **Threshold 0.5 指標**（`results.md` 第 9 節；Mazari 未說明 threshold，僅供參考）：
  - BERT E0 和 Mazari 的 BERT-base 同為 bert-base-uncased、lr 2e-5、3 epochs（max length 256 vs 128、batch 16 vs 32），結果相近：macro P／R／F1 為 0.480／0.776／0.585，對 0.496／0.777／0.598。
  - BERT E2 的 macro F1 0.626、macro MCC 0.624，略高於 Mazari 最佳 AUC 的 ensemble（0.622、0.616）。
  - Subset accuracy：所有模型（0.851–0.893）都低於「全部預測為 0」的 0.902，Rahman 的 accuracy 0.898 也落在這附近，所以 accuracy 不適合比較模型。Hamming loss 也一樣由負例主導：BERT E0 的 0.0376 幾乎等於「全部預測為 0」的 0.0378。Hamming loss 最低的是 TF-IDF baseline（0.025），因為它在 0.5 下預測的正例最少（13,990 筆，Transformer 為 21,405–25,531 筆；test 實際正例 14,498 筆）。

### 1.2 RQ1：加入 Civil 外部資料有沒有幫助
- **ROC-AUC**：三個模型都是 E1、E2 ≥ E0，但 Mean 的增幅只有 +0.0005～+0.0019，都低於 0.002。
- **RoBERTa、DeBERTa 的增益集中在 threat**：E2 − E0 的 threat ROC-AUC 為 +0.0072、+0.0087，是這兩個模型各 label 中最大的。BERT 各 label 的 Δ 都 ≤ 0.003，threat 只有 +0.0010（E1 − E0 為 +0.0000）。外部資料讓 threat 正例從 430（E0）增加到 755（E1 hard）或 soft mass 1,809（E2）。
- **PR-AUC 不一致**：E1 讓 RoBERTa（−0.032）和 DeBERTa（−0.010）的 Mean PR-AUC 下降，BERT 上升（+0.016）；E2 則三個模型都上升（+0.008～+0.033）。
- 小結：外部資料對排序能力（ROC-AUC）有小幅、方向一致的正面效果；以 hard label 加入時，precision–recall 的表現不穩定。依 bootstrap CI，E2 − E0 的 Mean 在三個模型都不含 0，E1 − E0 只有 BERT 不含 0（第 1.7 節）。

### 1.3 RQ2：hard label 與 soft label
- 在 Mean 指標上 E2 ≥ E1：E2 − E1 的 Mean ROC-AUC 三個模型都 ≥ 0（+0.0001／+0.0007／+0.0013），Mean PR-AUC 都 > 0（+0.017～+0.042）。
- 差異最大的是 threat 的 PR-AUC（E2 − E1：+0.047／+0.136／+0.118）。一個可能的解釋（推測，未驗證）是 soft label 保留了 score 介於 0 和 0.5 之間的留言，hard label 會把它們都變成 0：threat 的 soft mass 1,809 是 hard 正例 755 的 2.4 倍。
- per-label 有例外，例如 E2 − E1 的 PR-AUC：DeBERTa severe_toxic −0.060、BERT toxic −0.017、RoBERTa toxic −0.016；ROC-AUC：BERT toxic −0.0021、RoBERTa toxic −0.0011、DeBERTa severe_toxic −0.0008。Civil test 的 Macro F1 也是 E1 較高（見 1.5）。
- 小結：在相同外部樣本下，soft label 的 Mean 指標一致地優於或等於 hard label，但差距很小，且不是每個 label 都成立。依 bootstrap CI，E2 − E1 的 Mean 在 RoBERTa、DeBERTa 不含 0，BERT 含 0（第 1.7 節）。

### 1.4 模型比較
- DeBERTa-v3 沒有明顯勝出：Jigsaw test 上 BERT E2 0.9864、DeBERTa E2 0.9862、RoBERTa E2 0.9861，差距 < 0.0005。
- 單一模型中 DeBERTa E2 的 val 分數最高（0.9923），但 val 只用於 checkpoint 選擇，不作為模型比較的依據。

### 1.5 Cross-dataset（Civil test，五類）
- 三個 Transformer 的 E0 在 Civil test 只有 0.927–0.944，E1／E2 提高到 0.971–0.981。
- 這個提升**不能當成泛化能力的證據**：E1／E2 的訓練資料包含 Civil train，Civil test 對它們是同分布資料。它真正顯示的是只用 Jigsaw 訓練時的 domain gap：三個 Transformer 的 E0 從 Jigsaw test 到 Civil test 下降 0.04–0.06，TF-IDF baseline 下降 0.086（Jigsaw 六類對 Civil 五類，標籤定義也不同，僅供參考）。
- E2 的 ROC-AUC 三個模型都高於 E1，但 Macro F1（threshold 0.5）反而是 E1 較高，可能和校準差異有關（推測，見 1.6）。

### 1.6 Threshold 與校準
- 加入外部資料會改變各 label 輸出機率的分布。以 DeBERTa 為例，E0 → E2 在 threshold 0.5 下：
  - FP 增加：threat 207 → 326、severe_toxic 283 → 625、insult 1,657 → 2,306
  - FP 減少：toxic 5,628 → 4,934、obscene 2,552 → 2,093
- 因此 threshold 0.5 的 Macro F1 在不同 setting 之間不能直接比較，排序能力要看 ROC-AUC／PR-AUC。per-label threshold 的結果見第 1.7 節。

### 1.7 Refinement：bootstrap CI 與 per-label threshold
數字見 `results.md` 第 7、8 節。

**Bootstrap 95% CI（Δ ROC-AUC，paired，1,000 次）**

| Mean Δ ROC-AUC | BERT | RoBERTa | DeBERTa |
|---|---|---|---|
| E1 − E0 | +0.0007 [+0.0002, +0.0011] * | +0.0011 [−0.0000, +0.0029] | +0.0005 [−0.0003, +0.0015] |
| E2 − E0 | +0.0008 [+0.0004, +0.0012] * | +0.0019 [+0.0008, +0.0035] * | +0.0018 [+0.0007, +0.0035] * |
| E2 − E1 | +0.0001 [−0.0002, +0.0004] | +0.0007 [+0.0003, +0.0012] * | +0.0013 [+0.0006, +0.0023] * |

`*` 表示 CI 不含 0。
- **E2（soft）相對 E0 的增益最穩定**：三個模型的 Mean 都不含 0。E1（hard）相對 E0 只有 BERT 不含 0。
- **soft 優於 hard（Mean，RoBERTa、DeBERTa，seed 42）**：RoBERTa、DeBERTa 的 E2 − E1 不含 0，BERT 含 0；per-label 的反向差異見下一點。
- **threat**：E2 − E0 三個模型都不含 0，但區間很寬（DeBERTa +0.0087 [+0.0019, +0.0181]），反映 test 只有 211 個正例。
- **反向的差異也有**：BERT、RoBERTa 的 toxic E2 − E1 為負且不含 0；DeBERTa 的 insult E1 − E0、severe_toxic E2 − E1 也是。外部資料不是對每個 label 都有幫助。
- 三個單一模型的 63 個 per-label／Mean CI 中有 33 個不含 0（加上 ensemble 共 84 個、41 個不含 0）。每個 CI 是單獨計算的，沒有做多重比較校正，個別 label 的結果要保守看待。

**Per-label threshold（val 上選 F1 最高的 threshold）**
- Transformer 的 threshold 大多落在 0.4–0.7，Macro F1 的變化只有 −0.004～+0.019；baseline +0.0215。所以 threshold 0.5 對 Transformer 大致合適。
- DeBERTa E2 的增益最大（0.5942 → 0.6135），E0、E1 只有 +0.005、+0.008。調整後 DeBERTa 三個 setting 的 Macro F1 為 0.6015／0.6012／0.6135，E2 在 threshold 0.5 下 F1 偏低的情況（第 1.6 節）有一部分來自 threshold。
- 單一模型中，調整後的 Macro F1 最高仍是 BERT E2（0.6236）；ensemble 見第 1.8 節。

### 1.8 比較方法：ensemble（plan §6.7）
文獻數字的評估設定不同（第 1.1 節），所以另外在同一套 test 上加入 SOTA／比賽常用的 ensemble：每個 setting 把 BERT、RoBERTa、DeBERTa 的機率等權重平均。ensemble 是在看過單一模型的結果後才加入的事後分析，不屬於原本規劃的 RQ 檢定；但只用唯一的自然設定（三個 Transformer 全用、等權重），沒有比較其他組合或權重，也沒有用 test 調整。

| Run | Val ROC-AUC | Test ROC-AUC | Mean PR-AUC | Macro F1 | Macro MCC |
|---|---|---|---|---|---|
| ensemble_e0 | 0.9927 | 0.9874 | 0.6862 | 0.6238 | 0.6215 |
| ensemble_e1 | 0.9926 | 0.9875 | 0.6755 | 0.6135 | 0.6187 |
| ensemble_e2 | 0.9934 | 0.9877 | 0.6997 | 0.6270 | 0.6294 |
| 最佳單一模型（val 選：DeBERTa E2） | 0.9923 | 0.9862 | 0.6638 | 0.5942 | 0.6043 |
| *Mazari BERT-3_Models (AVG)*（等權重） | — | 0.9856 | — | 0.6051 | 0.6027 |
| *Mazari BERT-3_Models (W-AVG)*（權重依 test 設定） | — | 0.9863 | — | 0.6218 | 0.6164 |
| *Kaggle 第一名*（private LB） | — | 0.98856 | — | — | — |

- **和參考值的位置**：ensemble_e2 的 test ROC-AUC 0.9877 高於 Mazari 同樣等權重的 BERT-3_Models (AVG) 0.9856，也高於用 test 設定權重的 W-AVG 0.9863；macro F1、MCC 也略高於 Mazari（Mazari 未說明 threshold）。
- **Kaggle late submission（和 leaderboard 對照）**：上表的 test 分數用的是全部計分列；為了和 leaderboard 在同一計分方式下對照，把 4 個事先決定的 run 提交為 late submission（結果見 plan §6.5）。Kaggle late submission 只用來和 leaderboard 對照分數的位置。本專案的設定（資料前處理只做 strip、切分方式、模型與超參數、外部資料）都依文獻調查決定，沒有為比賽調整；和當年參賽者的條件也不同（使用比賽後才有的 BERT、RoBERTa、DeBERTa-v3 與 Civil Comments），所以名次不代表方法優劣。
  - Private LB：ensemble_e2 0.98788、ensemble_e0 0.98746、DeBERTa E2 0.98634、TF-IDF baseline 0.98055。
  - ensemble_e2 比 Kaggle 第一名（0.98856）低 0.0007；放進 private leaderboard 快照（4,539 隊）約為第 4–5 名。最佳單一模型 DeBERTa E2 約為第 600–872 名，baseline 約為第 2,381–2,389 名。
  - 外部資料在 ensemble 上的差異，在 private LB 上也是同方向：E2 − E0 = +0.0004（public +0.0002）。private LB 的列是本機 test 的子集，已包含在 bootstrap CI 裡，所以這不是獨立驗證，只確認換成 Kaggle 的計分方式後方向不變。
- **ensemble 本身的效果**：ensemble_e2 比最佳單一模型高 +0.0015（val 選的 DeBERTa E2），比 test 上最好的單一模型 BERT E2 高 +0.0014（點估計，未做 CI）。
- **外部資料在 ensemble 上的增益變小**：bootstrap CI（`results.md` 第 7 節）
  - E1 − E0：+0.0001 [−0.0002, +0.0004]，含 0
  - E2 − E0：+0.0004 [+0.0001, +0.0006]，不含 0
  - E2 − E1：+0.0002 [−0.0000, +0.0006]，含 0
- 單一模型的 E2 − E0 是 +0.0008～+0.0019，ensemble 只剩 +0.0004。外部資料帶來的改善有一部分和 ensemble 的效果重疊（推測，未驗證）。threat 仍是增益最明顯的 label（E2 − E0 +0.0009 [+0.0002, +0.0018]）。
- 小結：ensemble 的 test ROC-AUC 高於 Mazari 的報告值（評估設定仍不完全相同，Mazari 依 test 表現設定權重），在 Kaggle private LB 上的分數也接近前幾名，但設定依文獻決定、條件和參賽者不同，名次只是對照，只能說具競爭力；外部資料（soft label）在 ensemble 上仍有小幅但 CI 不含 0 的增益，hard label 則沒有。

---

## 2. 錯誤分析：DeBERTa（§6.4）

### 2.1 FP／FN 與 E0 → E2 翻轉（threshold 0.5）
數字見 `results.md` 第 5 節。
- **threat**：FN 56 → 46（E1）→ 25（E2）。FN → TP 34 筆，TP → FN 只有 3 筆，是改善最明確的 label。
- **identity_hate**：FN 175 → 203 → 173，幾乎沒變；FN → TP 48、TP → FN 46，翻轉互相抵消。
- **insult**：FN 668 → 460，但 FP 增加 649，等於整體往「多預測正例」的方向移動。
- **obscene**：E2 的 FN 增加（474 → 643，TP → FN 230），E2 對 obscene 變得保守。
- **toxic**：FP 減少 694（FP → TN 1,251），E2 的 toxic 預測比 E0 精準。

### 2.2 severe_toxic：E1 的外部樣本幾乎全是負例
- E1 加入 143,613 筆 Civil 留言，其中 severe_toxic 正例只有 1 筆（正例數 1,435 → 1,436），訓練資料的正例比例從 1.0% 降到約 0.5%。原本預期 E1 的 severe_toxic 會變差。
- 實際上沒有一致的下降：
  - ROC-AUC E1 − E0：BERT +0.0004、RoBERTa +0.0005、DeBERTa +0.0005
  - PR-AUC E1 − E0：BERT −0.008、RoBERTa +0.003、DeBERTa +0.031
  - DeBERTa 在 threshold 0.5 下反而預測更多正例：FN 210 → 117、FP 283 → 587
- 可能的原因（推測，未驗證）是 severe_toxic 和 toxic、insult 高度共現，而這兩類在 E1 的正例大約翻倍（toxic 13,765 → 25,222、insult 7,089 → 15,462）。test 只有 367 個 severe_toxic 正例，需要 bootstrap CI 才能判斷這些 Δ 是否超出抽樣誤差。

### 2.3 人工閱讀：E0 的 false negatives
從 DeBERTa E0 在 test 上的 FN 中，以 seed 42 各隨機抽 20 筆（threat 共 56 筆 FN、identity_hate 共 175 筆），閱讀全文並比較 E0 和 E2 的機率。為避免在 repo 中重複仇恨內容，以下只描述模式並以 `row`（`test_jigsaw.parquet` 的列號）指出樣本，原文見 release 的 `error_samples_deberta.csv`。

**閱讀者：Claude（AI 助理）。** 分類有主觀成分，建議組員複核，特別是標為邊界樣本或可能標註問題的列。

#### threat（20 筆）
E2 讓 19 筆的機率上升、13 筆跨過 0.5；機率中位數 0.24 → 0.59。

| 模式 | 筆數 | E2 ≥ 0.5 | 說明與樣本 row |
|---|---|---|---|
| 短句、直接的殺害字眼，沒有上下文 | 3 | 3 | E2 0.60–0.78。58593、10340 的 E0 < 0.003（例如 10340 只有兩個字「kill it」），55357 為 0.24 |
| 威脅夾在大量髒話與辱罵中 | 7 | 6 | E0 0.16–0.49，模型把它當成 insult／obscene；E2 多數跨過 0.5。3873、62878、40453、4185、21519、33526；未跨過：48803（0.36） |
| 針對族群的暴力號召 | 4 | 3 | 6871、31140、50988 跨過；61975（長文、大量拼字錯誤）0.11 未跨過 |
| 難例 | 6 | 1 | 威脅在長文中段或後段：27725（669 字元，前段是印尼文遮字髒話，威脅句在最後）、58973（威脅在中段，後接連結）、27406（威脅在最後，唯一跨過，0.54）；41379（2,912 字元，開頭一句想殺人的話，其餘是歌詞）；58024（幫派用語）；33376 是自殺意念而非威脅他人，可能是標註問題，E2 反而下降（0.42 → 0.38） |

#### identity_hate（20 筆）
E2 讓 15 筆的機率上升、5 筆下降，只有 6 筆跨過 0.5；機率中位數 0.21 → 0.25。

| 模式 | 筆數 | E2 ≥ 0.5 | 說明與樣本 row |
|---|---|---|---|
| 蔑稱（多為性傾向）當一般辱罵使用，沒有明確針對族群 | 10 | 4 | 和單純的 insult 很難區分，E2 對這類忽高忽低。跨過：43084、11173、32464、4868；下降：61393、29868；其餘未跨過：52761、46676 仍低，32175（0.010 → 0.483）、47850（0.424）接近 0.5。邊界樣本：47850 針對一個宗教團體並帶有「die」；43084 沒有蔑稱，是性內容的描述 |
| 提到族群或國籍，但不含蔑稱 | 6 | 2 | 跨過：58873、57157；仍低：8780、36146、58833（重複的國家名稱）、27270 |
| 可能是標註問題或後設討論 | 4 | 0 | 50731 只是詢問卡通角色的性傾向；5326 在討論條目是否有偏見；39294 在討論某個詞是否帶有種族歧視；43894 在解釋俚語的意思 |

#### 小結
- threat 的改善在樣本層級也看得到：E0 漏掉的明確威脅，多數在 E2 被找回，和 2.1 的翻轉統計一致。E2 仍漏掉 7 筆：2 筆威脅藏在長文中後段（27725、58973），其餘是歌詞長文（41379）、幫派用語（58024）、自殺意念（33376），以及 48803、61975（E2 0.36、0.11）。
- identity_hate 的錯誤主要來自「蔑稱當一般辱罵」和 insult 難以區分（這類只有 4／10 被 E2 找回），以及疑似標註問題，和 identity_hate 的 Δ 很小一致。
- identity_hate 有 4 筆可能是標註問題或後設討論，模型給低機率未必是錯；threat 的 33376 也屬於這類。

---

## 3. 其他限制
前面已提到單一 seed、Civil test 對 E1／E2 是同分布資料、Macro F1 依賴 threshold；另外：
- **rare label**：test 中 threat 只有 211 個正例、severe_toxic 367 個、identity_hate 712 個，per-label Δ 容易受少數樣本影響。
- **人工閱讀樣本少**：每類 20 筆、單一閱讀者，只能說明錯誤類型，不能估計各類型的比例。
- **更多資料 vs. 更多 steps**：E1／E2 的訓練資料和 optimizer steps 都是 E0 的兩倍（plan §1.5），比例消融決定不做，增益無法歸因於其中之一。
