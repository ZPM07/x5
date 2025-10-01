import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import os
import re
import joblib
import numpy as np
import pandas as pd

from ast import literal_eval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# Общий классификатор
# ========================
class NERClassifier(nn.Module):
    def __init__(self, base_model_name, num_labels=3):
        super().__init__()
        self.base = AutoModel.from_pretrained(base_model_name)
        hidden = self.base.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.cls = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(out.last_hidden_state)
        logits = self.cls(x)
        return logits

# ========================
# Универсальная функция инференса
# ========================
def infer_ner_model(samples, model_dir, model_name, base_model="cointegrated/rubert-tiny2", max_len=64):
    # Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = NERClassifier(base_model).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(model_dir, f"{model_name}.pt"), map_location=DEVICE))
    model.eval()

    results = []

    for item in samples:
        text = item["text"]
        words = text.split()

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=max_len
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(encoding["input_ids"], encoding["attention_mask"])
            probs = F.softmax(logits, dim=-1)

        word_ids = encoding.word_ids(batch_index=0)
        current_word = None
        curr_1, curr_2 = [], []
        word_probs_1, word_probs_2 = [], []

        for w_id, p in zip(word_ids, probs.squeeze().cpu().numpy()):
            if w_id is None:
                continue
            if w_id != current_word:
                if current_word is not None:
                    word_probs_1.append(float(np.max(curr_1)))
                    word_probs_2.append(float(np.max(curr_2)))
                current_word = w_id
                curr_1, curr_2 = [p[1]], [p[2]]
            else:
                curr_1.append(p[1])
                curr_2.append(p[2])

        if curr_1:
            word_probs_1.append(float(np.max(curr_1)))
            word_probs_2.append(float(np.max(curr_2)))

        results.append({
            f"{model_name}_proba_1": word_probs_1,
            f"{model_name}_proba_2": word_probs_2,
        })

    return results

VOLUME_RE = re.compile(r"(?<!\w)(\d+[.,]?\d*)\s?(л|л\.|литр\w*|мл|ml|г|кг|шт|уп\w*|пак\w*|бут\w*)(?!\w)", re.IGNORECASE)
PERCENT_RE = re.compile(r"(?<!\w)(\d+[.,]?\d*)\s?%|процент\w*", re.IGNORECASE)

def rule_spans(text):
    return [(m.start(), m.end(), "VOLUME") for m in VOLUME_RE.finditer(text)] + \
           [(m.start(), m.end(), "PERCENT") for m in PERCENT_RE.finditer(text)]

def apply_priors(text, words, probs, beta=2.0):
    offsets, char_idx = [], 0
    for w in words:
        start, end = text.find(w, char_idx), text.find(w, char_idx) + len(w)
        offsets.append((start, end))
        char_idx = end + 1

    for s, e, label in rule_spans(text):
        for i, (w_start, w_end) in enumerate(offsets):
            if max(s, w_start) < min(e, w_end):
                if label == "VOLUME":
                    probs[i][4] += beta
                elif label == "PERCENT":
                    probs[i][2] += beta
    return probs

def extract_features(row, words):
    n = len(words)
    features = []
    for i, word in enumerate(words):
        get_probs = lambda idx: [
            row['brand_probs'][idx],
            row['type_probs'][idx],
            row['percent_probs'][idx],
            row['volume_probs'][idx],
            row['o_probs'][idx],
        ] if 0 <= idx < n else [0]*5

        feat = get_probs(i) + [i, len(word)] + get_probs(i-1) + [i > 0] + get_probs(i+1) + [i < n-1]
        features.append(feat)
    return np.array(features, dtype=np.float32)

def annotate_sample(text, words, labels):
    annotations, prev_label, char_idx = [], None, 0
    for word, label in zip(words, labels):
        start, end = text.find(word, char_idx), text.find(word, char_idx) + len(word)
        bio = 'O' if label == 'O' else ('I-' + label if prev_label == label else 'B-' + label)
        annotations.append((start, end, bio))
        prev_label, char_idx = label, end + 1
    return annotations

def infer_ensemble(models, df, le):
    all_annots = []
    for _, row in df.iterrows():
        text, words = row['sample'], row['sample'].split()
        X = extract_features(row, words)
        probs = np.mean([m.predict_proba(X) for m in models], axis=0)
        probs = apply_priors(text, words, probs, beta=1.0)
        labels = le.inverse_transform(probs.argmax(axis=1))
        all_annots.append(annotate_sample(text, words, labels))
    df['annotation'] = all_annots
    return df

def predict(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col != 'sample':
            df[col] = df[col].apply(literal_eval)

    models = [joblib.load(f"xgb/xgb_fold{fold}.joblib") for fold in range(1, 6)]
    le = joblib.load("xgb/label_encoder.joblib")

    return infer_ensemble(models, df, le)

df = pd.read_csv("submission.csv")
samples = [{"text": row["sample"]} for _, row in df.iterrows()]

model_names = ["brand", "type", "percent", "volume", "O"]
all_outputs = []

for name in model_names:
    model_dir = f"models/{name}" 
    model_file = f"{name}_model"
    out = infer_ner_model(samples, model_dir, model_name=name)
    all_outputs.append(pd.DataFrame(out))

for df_out in all_outputs:
    df = pd.concat([df, df_out], axis=1)

result = predict(df)
result[['sample', 'annotation']].to_csv("xgboost_test_ensemble.csv", sep=";", index=False)