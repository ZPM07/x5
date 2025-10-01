import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from ast import literal_eval

from src.models.NER_classifier import NERClassifier
from src.utils.paths import WEIGHTS_DIR

DEVICE = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "cointegrated/rubert-tiny2"
VOLUME_RE = re.compile(r"(?<!\w)(\d+[.,]?\d*)\s?(л|л\.|литр\w*|мл|ml|г|кг|шт|уп\w*|пак\w*|бут\w*)(?!\w)", re.IGNORECASE)
PERCENT_RE = re.compile(r"(?<!\w)(\d+[.,]?\d*)\s?%|процент\w*", re.IGNORECASE)


def run_ner_pipeline(
    samples: List[Dict[str, Any]],
    models: Dict[str, torch.nn.Module],
    tokenizer,
    max_len: int = 64
) -> pd.DataFrame:
    all_results = []

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

        sample_result = {"sample": text}

        for model_name, model in models.items():
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
                    curr_1, curr_2 = [p[1]], [p[2] if len(p) > 2 else 0.0]
                else:
                    curr_1.append(p[1])
                    curr_2.append(p[2] if len(p) > 2 else 0.0)

            if curr_1:
                word_probs_1.append(float(np.max(curr_1)))
                word_probs_2.append(float(np.max(curr_2)))

            sample_result[f"{model_name}_proba_1"] = word_probs_1
            sample_result[f"{model_name}_proba_2"] = word_probs_2

        all_results.append(sample_result)

    return pd.DataFrame(all_results)

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

def run_boost_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col != 'sample':
            df[col] = df[col].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)

    xgboost_folder = os.path.join(WEIGHTS_DIR, "xgboost")
    models = [joblib.load(os.path.join(xgboost_folder, f"xgb_fold{fold}.joblib")) for fold in range(1, 6)]
    le = joblib.load(os.path.join(xgboost_folder, "label_encoder.joblib"))

    return infer_ensemble(models, df, le)

def initialize_models():

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    models = {
        "brand": NERClassifier(base_model_name=BASE_MODEL, num_labels=3, use_dropout=False),
        "type": NERClassifier(base_model_name=BASE_MODEL, num_labels=3, use_dropout=True),
        "volume": NERClassifier(base_model_name=BASE_MODEL, num_labels=3, use_dropout=False),
        "percent": NERClassifier(base_model_name=BASE_MODEL, num_labels=3, use_dropout=False),
        "o": NERClassifier(base_model_name=BASE_MODEL, num_labels=2, use_dropout=False),
    }

    for name, model in models.items():
        model_path = os.path.join(WEIGHTS_DIR, name, "model.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        state_dict = torch.load(model_path, map_location=DEVICE)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            if new_k.startswith("base."):
                new_k = new_k.replace("base.", "base_model.")
            if new_k.startswith("cls."):
                new_k = new_k.replace("cls.", "classifier.")
            new_state_dict[new_k] = v

        model.load_state_dict(new_state_dict)
        model.to(DEVICE)
        model.eval()

    return models, tokenizer


if __name__ == "__main__":
    df = pd.read_csv("data/raw/test.csv", delimiter=";")

    samples = [{"text": row["sample"]} for _, row in df.iterrows()]
    models, tokenizer = initialize_models()

    ner_output = run_ner_pipeline(samples=samples, models=models, tokenizer=tokenizer)

    submit = run_boost_pipeline(ner_output)
    submit[['sample', 'annotation']].to_csv("xgboost_test_ensemble.csv", sep=";", index=False)