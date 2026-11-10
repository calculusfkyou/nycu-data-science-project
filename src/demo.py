"""Phase 6 (docs/plan.md §8): Gradio demo of the ensemble_e2 model, locally or on Hugging Face Spaces.

ensemble_e2 = mean probability of bert_e2, roberta_e2 and deberta_e2 (plan §6.7), the best run on val.
Inference matches evaluation: clean(), each model's own tokenizer, max_length 256, fp32, sigmoid per label.
Checkpoints come from outputs/{run}/ if present, otherwise from the public GitHub release (as on Spaces).

Run: .venv/Scripts/python src/demo.py   (then open http://127.0.0.1:7860)
"""
import tarfile
import urllib.request

import gradio as gr
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import F1_THRESHOLD, LABELS, OUTPUTS, ROOT, clean

MAX_LENGTH = 256  # train.py HPARAMS["max_length"]
RELEASE = "https://github.com/calculusfkyou/nycu-data-science-project/releases/download/demo-models-seed42"
CHECKPOINTS = {"bert_e2": "bert_e2_best_step-035900", "roberta_e2": "roberta_e2_best_step-035900",
               "deberta_e2": "deberta_e2_best_step-035900"}
SCORES = ROOT / "reports" / "seed42" / "metrics_long.csv"
SCORE_ROWS = [("TF-IDF + LR baseline", "tfidf_lr_e0"), ("只用 Jigsaw（ensemble_e0）", "ensemble_e0"),
              ("加入 Civil Comments（ensemble_e2，本 demo）", "ensemble_e2")]
KAGGLE_FIRST = ("Kaggle 第一名（private leaderboard）", "0.98856")  # evaluate.py REFERENCES
EXAMPLES = [  # written for the demo, not taken from the datasets
    "Thanks for adding the sources, the article reads much better now.",
    "You are a complete idiot, stop editing this page.",
    "I know where you live and I will come and kill you.",
    "What the hell is this crap? Nobody wants to read it.",
]


def checkpoint_dir(run):
    path = OUTPUTS / run / CHECKPOINTS[run]
    if not path.exists():
        tmp = OUTPUTS / run / "download"
        print(f"Downloading {CHECKPOINTS[run]} from the GitHub release ...")
        url = f"{RELEASE}/{CHECKPOINTS[run]}.tar.gz"
        with urllib.request.urlopen(url, timeout=60) as r, tarfile.open(fileobj=r, mode="r|gz") as t:
            t.extractall(tmp, filter="data")
        (tmp / CHECKPOINTS[run]).rename(path)  # only a complete extraction becomes the checkpoint folder
        tmp.rmdir()
    return path


def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for run in CHECKPOINTS:
        path = checkpoint_dir(run)
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32, local_files_only=True)
        assert [model.config.id2label[i] for i in range(len(LABELS))] == LABELS, f"{run}: label order differs"
        models.append((tok, model.to(device).eval()))
    return models


@torch.inference_mode()
def predict(text, models):
    """Six probabilities for one comment, averaged over the three models (as src/ensemble.py)."""
    text = clean(pd.Series([text])).iloc[0]
    probs = []
    for tok, model in models:
        enc = tok(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(model.device)
        probs.append(torch.sigmoid(model(**enc).logits.float())[0].cpu().numpy())
    return np.mean(probs, axis=0, dtype="float64")


def score_table():
    m = pd.read_csv(SCORES).query("split == 'test_jigsaw'").set_index("run")["mean_roc_auc"]
    rows = [(name, f"{m[run]:.4f}") for name, run in SCORE_ROWS] + [KAGGLE_FIRST]
    return "\n".join(["| 方法 | Mean ROC-AUC |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows])


def build_app(models):
    def run(text):
        if not text.strip():
            return None, "請輸入一則英文留言。"
        p = predict(text, models)
        hits = [l for l, x in zip(LABELS, p) if x >= F1_THRESHOLD]
        verdict = f"判定（機率 ≥ {F1_THRESHOLD}）：**{'、'.join(hits) if hits else '沒有任何一類 toxicity'}**"
        return pd.DataFrame({"label": LABELS, "probability": p}), verdict

    with gr.Blocks(title="Toxic Comment Demo", analytics_enabled=False) as app:
        gr.Markdown(
            "# 英文留言 toxicity 偵測\n"
            "輸入一則英文留言，模型估計它屬於下列六類 toxicity 的機率，一則留言可以同時屬於多類："
            "toxic、severe_toxic、obscene、threat、insult、identity_hate。\n\n"
            "本專案測試加入外部資料 Civil Comments 能否讓模型更好。這個 demo 用的是 BERT、RoBERTa、DeBERTa-v3 "
            "三個模型的平均，訓練資料是 Jigsaw 加上 Civil Comments（soft label）。")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="英文留言", lines=5)
                button = gr.Button("判斷", variant="primary")
            with gr.Column():
                plot = gr.BarPlot(x="label", y="probability", y_lim=[0, 1], sort=LABELS, label="六類機率")
                verdict = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=text, outputs=[plot, verdict], fn=run, run_on_click=True, cache_examples=False,
                    label="範例留言（點一下直接判斷）")
        gr.Markdown("### 成績：Jigsaw 官方 test 的 Mean ROC-AUC\n" + score_table())
        button.click(run, inputs=text, outputs=[plot, verdict])
    return app


if __name__ == "__main__":
    build_app(load_models()).launch()
