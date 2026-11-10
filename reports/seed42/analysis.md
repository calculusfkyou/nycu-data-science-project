# seed-42 結果分析（Evaluation §6.4、§6.6）

依據：
- 同資料夾的 `results.md`（10 個 runs，Jigsaw official labeled test 63,978 列）
- release 內的 `error_samples_deberta.csv`（含原始留言）
- `data/processed/data_summary.md`（各訓練集的正例數）
- `docs/plan.md` §6.5（文獻與 Kaggle 參考值）

**所有結論都是暫定的。** 每個設定只有一個 seed，E0／E1／E2 之間的 |Δ Mean ROC-AUC| 都 < 0.002，依 plan §7 要先做 bootstrap CI（Refinement 第 1 項）並觸發 DeBERTa 多 seed（第 3 項），才能判斷差異是否可靠。

---

## 1. 解讀（§6.6）

### 1.1 Transformer 與 baseline
- 三個 Transformer 的 Mean ROC-AUC（0.9843–0.9864）都高於 TF-IDF + LR baseline（0.9801），差距 +0.004～+0.006；Mean PR-AUC 差距 +0.01～+0.07。
- Baseline 0.9801 高於 Rahman et al. (2023) 的 0.973。最好的 run（BERT E2，0.9864）和 Mazari et al. (2024) 的 0.9863 相當，但 Mazari 依 test 表現設定 ensemble 權重，不是同一基準；比 Kaggle 第一名（private LB 0.98856，大型 ensemble）低約 0.002。

### 1.2 RQ1：加入 Civil 外部資料有沒有幫助
- **ROC-AUC**：三個模型都是 E1、E2 ≥ E0，但 Mean 的增幅只有 +0.0005～+0.0019，都低於 0.002。
- **RoBERTa、DeBERTa 的增益集中在 threat**：E2 − E0 的 threat ROC-AUC 為 +0.0072、+0.0087，是這兩個模型各 label 中最大的。BERT 各 label 的 Δ 都 ≤ 0.003，threat 只有 +0.0010（E1 − E0 為 +0.0000）。外部資料讓 threat 正例從 430（E0）增加到 755（E1 hard）或 soft mass 1,809（E2）。
- **PR-AUC 不一致**：E1 讓 RoBERTa（−0.032）和 DeBERTa（−0.010）的 Mean PR-AUC 下降，BERT 上升（+0.016）；E2 則三個模型都上升（+0.008～+0.033）。
- 小結：外部資料對排序能力（ROC-AUC）有小幅、方向一致的正面效果；以 hard label 加入時，precision–recall 的表現不穩定。

### 1.3 RQ2：hard label 與 soft label
- 在 Mean 指標上 E2 ≥ E1：E2 − E1 的 Mean ROC-AUC 三個模型都 ≥ 0（+0.0001／+0.0007／+0.0013），Mean PR-AUC 都 > 0（+0.017～+0.042）。
- 差異最大的是 threat 的 PR-AUC（E2 − E1：+0.047／+0.136／+0.118）。一個可能的解釋（推測，未驗證）是 soft label 保留了 score 介於 0 和 0.5 之間的留言，hard label 會把它們都變成 0：threat 的 soft mass 1,809 是 hard 正例 755 的 2.4 倍。
- per-label 有例外，例如 E2 − E1 的 PR-AUC：DeBERTa severe_toxic −0.060、BERT toxic −0.017、RoBERTa toxic −0.016；ROC-AUC：BERT toxic −0.0021、RoBERTa toxic −0.0011、DeBERTa severe_toxic −0.0008。Civil test 的 Macro F1 也是 E1 較高（見 1.5）。
- 小結：在相同外部樣本下，soft label 的 Mean 指標一致地優於或等於 hard label，但差距很小，且不是每個 label 都成立。

### 1.4 模型比較
- DeBERTa-v3 沒有明顯勝出：Jigsaw test 上 BERT E2 0.9864、DeBERTa E2 0.9862、RoBERTa E2 0.9861，差距 < 0.0005。
- DeBERTa E2 的 val 分數最高（0.9923），但 val 只用於 checkpoint 選擇，不作為模型比較的依據。

### 1.5 Cross-dataset（Civil test，五類）
- 三個 Transformer 的 E0 在 Civil test 只有 0.927–0.944，E1／E2 提高到 0.971–0.981。
- 這個提升**不能當成泛化能力的證據**：E1／E2 的訓練資料包含 Civil train，Civil test 對它們是同分布資料。它真正顯示的是只用 Jigsaw 訓練時的 domain gap：三個 Transformer 的 E0 從 Jigsaw test 到 Civil test 下降 0.04–0.06，TF-IDF baseline 下降 0.086（Jigsaw 六類對 Civil 五類，標籤定義也不同，僅供參考）。
- E2 的 ROC-AUC 三個模型都高於 E1，但 Macro F1（threshold 0.5）反而是 E1 較高，可能和校準差異有關（推測，見 1.6）。

### 1.6 Threshold 與校準
- 加入外部資料會改變各 label 輸出機率的分布。以 DeBERTa 為例，E0 → E2 在 threshold 0.5 下：
  - FP 增加：threat 207 → 326、severe_toxic 283 → 625、insult 1,657 → 2,306
  - FP 減少：toxic 5,628 → 4,934、obscene 2,552 → 2,093
- 因此 threshold 0.5 的 Macro F1 在不同 setting 之間不能直接比較，排序能力要看 ROC-AUC／PR-AUC。per-label threshold（Refinement 第 2 項）可以把校準差異和排序能力分開。

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

下一步見 plan §7 第 1–3 項（bootstrap CI、per-label threshold、DeBERTa 多 seed）。
