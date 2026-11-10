---
title: Toxic Comment Demo
emoji: 💬
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: src/demo.py
pinned: false
---

# Toxic Comment Demo

NYCU Data Science term project (Group A, Team 5): multi-label toxicity classification of English comments
(toxic, severe_toxic, obscene, threat, insult, identity_hate).

The model is the equal-weight ensemble of BERT, RoBERTa and DeBERTa-v3 trained on Jigsaw plus Civil Comments
with soft labels (`ensemble_e2`). On startup the app downloads the three checkpoints (about 1.5 GB) from the
project's GitHub release, so the first start after the Space sleeps takes a few minutes.

Code, experiments and results: https://github.com/calculusfkyou/nycu-data-science-project
