# Toxic Comment Multi-label Classification

NYCU 數據科學專題 · Group A · Team 5（吳傢澂、歐冠廷、吳承羲）

本專題針對 Jigsaw Toxic Comment Classification 的六類 toxicity labels（toxic、severe_toxic、obscene、threat、insult、identity_hate）建立 multi-label classification 模型，並研究兩個問題：
- 加入 Civil Comments 作為外部資料，能不能提升分類表現？
- 外部資料用 hard label 或 soft label，結果會不會不同？

完整的執行計畫見 [docs/plan.md](docs/plan.md)，它是所有設計決策的依據。修改資料處理、訓練、評估的行為或產出時，請在同一個 commit 更新 plan.md 的對應章節。

## 實驗設定

| 代號 | 訓練資料 | 外部資料的標籤 |
|---|---|---|
| E0 | Jigsaw train split | — |
| E1 | Jigsaw + Civil Comments（1:1） | Hard：score ≥ 0.5 → 1 |
| E2 | Jigsaw + Civil Comments（1:1） | Soft：保留原始 0–1 score |

- 模型：TF-IDF + Logistic Regression（baseline，只跑 E0），以及 BERT、RoBERTa、DeBERTa-v3（主要分析對象）。
- 主要指標：Mean Column-wise ROC-AUC，在 Jigsaw official labeled test 上計算。

## 進度

| 階段 | 狀態 | 程式 |
|---|---|---|
| Data Understanding | ✅ | EDA，在 repo 外完成 |
| Data Preparation | ✅ | `src/prepare_data.py` |
| Data Modeling | ✅ baseline 與 9 個 seed-42 runs 完成 | `src/baseline.py`、`src/train.py` |
| Evaluation | ✅ 結果表與分析完成（[results](reports/seed42/results.md)、[analysis](reports/seed42/analysis.md)），結論以 Refinement 的 bootstrap CI 為準 | `src/evaluate.py` |
| Refinement | ✅ bootstrap CI、per-label threshold（多 seed 與比例消融決定不做） | `src/evaluate.py` |
| Deployment（選做） | ✅ 本機 Gradio demo（未對外部署） | `src/demo.py` |

## 環境

- Python 3.12
- 訓練需要 NVIDIA GPU。PyTorch 要先單獨安裝 CUDA 版本，再裝其他套件。

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

在另一台電腦跑實驗的完整步驟見 [docs/run_guide.md](docs/run_guide.md)。

## 資料

原始資料不放進 git，需要自行下載，並放到下列位置：

```
datasets/
├── jigsaw/                 # Kaggle: Jigsaw Toxic Comment Classification Challenge
│   ├── train.csv
│   ├── test.csv
│   └── test_labels.csv
└── civil_comments/         # Hugging Face: google/civil_comments
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

接著產生處理後的資料：

```bash
.venv/Scripts/python src/prepare_data.py
```

執行時間約 30 秒，產出寫到 `data/processed/`（同樣不放進 git）：

| 檔案 | 內容 |
|---|---|
| `train_e0/e1/e2.parquet` | 三種設定的訓練資料；`source` 欄位標示每列來自 Jigsaw 或 Civil |
| `val.parquet` | Jigsaw 的 10% validation（multi-label stratified，seed 42） |
| `test_jigsaw.parquet` | Jigsaw official labeled test（63,978 筆） |
| `test_civil.parquet` | Civil test，以 ≥ 0.5 二值化，只含五類，用於 cross-dataset 評估 |
| `civil_pool_perm.parquet` | 打亂後的 augmentation pool，原本給 Refinement 的比例消融使用（已決定不做，檔案保留） |
| `data_summary.md` | 筆數、overlap、各 label 正例數，以及驗收結果 |

處理規則：
- 文字清理只做 strip。
- 所有 label 存成 float32。
- 所有隨機步驟固定 seed 42。重跑結果完全相同。
- 腳本結尾會用 assert 檢查 train 和 val／test 之間沒有 leakage。

## 專案結構

```
├── datasets/          # 原始資料（gitignored）
├── data/processed/    # prepare_data.py 產出（gitignored）
├── outputs/           # 訓練與評估產出（gitignored）
├── reports/seed42/    # seed-42 主實驗結果（evaluate.py 輸出）
├── tests/             # 回歸測試（checkpoint 存讀、demo）
├── space/             # Hugging Face Space 設定（需要 PRO，未部署）
├── docs/
│   ├── plan.md        # 執行計畫
│   └── run_guide.md   # 實驗執行指南
├── src/
│   ├── common.py      # 路徑、label 定義、clean()、metrics、共用工具
│   ├── prepare_data.py
│   ├── baseline.py    # TF-IDF + LR（CPU）
│   ├── train.py       # BERT／RoBERTa／DeBERTa-v3（GPU）
│   ├── ensemble.py    # 三個 Transformer 的等權重 ensemble（比較方法）
│   ├── evaluate.py    # 讀 predictions，產出 results.md
│   ├── make_submission.py  # Kaggle late submission 檔
│   └── demo.py        # Gradio demo（ensemble_e2）
└── requirements.txt
```

## 模型與評估

```bash
.venv/Scripts/python src/baseline.py
```

```bash
.venv/Scripts/python src/train.py --model deberta --setting e0
```

```bash
.venv/Scripts/python src/ensemble.py
```

```bash
.venv/Scripts/python src/evaluate.py
```

- `train.py` 的完整參數與執行順序見 [docs/run_guide.md](docs/run_guide.md)。
- `ensemble.py` 把三個 Transformer 同 setting 的預測等權重平均，寫到 `outputs/ensemble_{setting}/`，作為和 SOTA／比賽方法比較用的 ensemble。任何 run 重跑後，要先重跑 `ensemble.py` 再跑 `evaluate.py`。
- `evaluate.py` 讀 `outputs/*/preds_*.parquet`（含 ensemble），產出 `outputs/results.md`。
- `make_submission.py` 把 4 個事先決定的 run 轉成 Kaggle 提交格式，寫到 `outputs/submissions/`，並檢查本機分數和 `metrics_long.csv`（即 `results.md` 的數字）一致。

回歸測試。checkpoint 存讀的測試用 CPU，不需要資料或網路；`test_demo.py` 需要本機的三個 E2 checkpoint、`outputs/ensemble_e2` 和 `data/processed`，缺少時自動略過：

```bash
.venv/Scripts/python -m unittest discover -s tests -v
```

## Demo

輸入一則英文留言，顯示 `ensemble_e2`（BERT、RoBERTa、DeBERTa-v3 的 E2 平均）估計的六類機率和判定結果，設計見 [docs/plan.md](docs/plan.md) 第 8 節。

本機執行。`outputs/` 沒有 checkpoint 時，會先從 GitHub release 下載（約 1.5GB）：

```bash
.venv/Scripts/python src/demo.py
```

然後打開 http://127.0.0.1:7860 。

**部署到 Hugging Face Spaces（目前沒有部署）**：Gradio Space 需要 Hugging Face PRO 訂閱，免費帳號只能建靜態 Space，所以目前只在本機執行。`space/` 保留了 Space 設定，訂閱 PRO 之後可以照下面的步驟部署，模型會在 Hugging Face 的伺服器上跑，任何人用網址就能使用：

1. 到 huggingface.co 註冊帳號。
2. 登入。指令會開瀏覽器授權；不行的話，到 Settings → Access Tokens 建一個 Write 權限的 token，用 `--token` 傳入：

   ```bash
   .venv/Scripts/hf auth login
   ```

3. 建立 Space，再依序上傳程式、成績表和 Space 設定。`USER` 換成你的帳號名稱：

   ```bash
   .venv/Scripts/hf repos create USER/toxic-comment-demo --type space --sdk gradio
   ```

   ```bash
   .venv/Scripts/hf upload USER/toxic-comment-demo src src --repo-type space --include demo.py --include common.py
   ```

   ```bash
   .venv/Scripts/hf upload USER/toxic-comment-demo reports/seed42/metrics_long.csv reports/seed42/metrics_long.csv --repo-type space
   ```

   ```bash
   .venv/Scripts/hf upload USER/toxic-comment-demo space . --repo-type space
   ```

4. 打開 `https://huggingface.co/spaces/USER/toxic-comment-demo`，等建置和 checkpoint 下載完成，大約幾分鐘。

之後 `demo.py` 或 `common.py` 有修改，重跑第 3 步的上傳指令即可。Space 閒置一段時間會休眠，使用前先打開一次。
