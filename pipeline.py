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
from src.utils.rules import apply_priors
from src.utils.paths import WEIGHTS_DIR

DEVICE = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "cointegrated/rubert-tiny2"

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

            if model_name != 'o':
                sample_result[f"{model_name}_proba_1"] = word_probs_1
                sample_result[f"{model_name}_proba_2"] = word_probs_2
            else:
                sample_result[f"{model_name}_probs"] = word_probs_1

        all_results.append(sample_result)

    return pd.DataFrame(all_results)

def extract_features(row, words):
    n = len(words)
    features = []
    for i, word in enumerate(words):
        get_probs = lambda idx: [
            row['brand_proba_1'][idx],
            row['brand_proba_2'][idx],
            row['type_proba_1'][idx],
            row['type_proba_2'][idx],
            row['percent_proba_1'][idx],
            row['percent_proba_2'][idx],
            row['volume_proba_1'][idx],
            row['volume_proba_2'][idx],
            row['o_probs'][idx],
        ] if 0 <= idx < n else [0]*9

        current = get_probs(i)
        prev = get_probs(i-1)
        next_ = get_probs(i+1)
        has_prev = 1 if i > 0 else 0
        has_next = 1 if i < n-1 else 0

        feat = current + [i, len(word)] + prev + [has_prev] + next_ + [has_next]
        features.append(feat)
    return np.array(features, dtype=np.float32)

def annotate_sample(text, words, labels):
    annotations = []
    char_idx = 0
    for word, label in zip(words, labels):
        start = text.find(word, char_idx)
        end = start + len(word)
        char_idx = end + 1
        annotations.append((start, end, label))
    return annotations

def infer_ensemble(models, df, le):
    all_annotations = []
    for _, row in df.iterrows():
        text = row['sample']
        words = text.split()

        X = extract_features(row, words)

        all_probs = [model.predict_proba(X) for model in models]
        mean_probs = np.mean(all_probs, axis=0)

        mean_probs = apply_priors(text, words, mean_probs, beta=1.0)

        pred_indices = mean_probs.argmax(axis=1)
        pred_labels = [str(label) for label in le.inverse_transform(pred_indices)]

        annotations = annotate_sample(text, words, pred_labels)
        all_annotations.append(annotations)

    df['annotation'] = all_annotations
    return df

def run_boost_pipeline(df, models_dir):

    for col in df.columns:
        if col != 'sample':
            df[col] = df[col].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)

    models_dir = os.path.join(WEIGHTS_DIR, "xgboost")
    models = [joblib.load(os.path.join(models_dir, f"xgb_fold{fold}.joblib")) for fold in range(1, 6)]
    le = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))

    df = infer_ensemble(models, df, le)
    return df

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

def process_example(df: pd.DataFrame):

    samples = [{"text": row["sample"]} for _, row in df.iterrows()]
    models, tokenizer = initialize_models()

    ner_output = run_ner_pipeline(samples=samples, models=models, tokenizer=tokenizer)
    submit = run_boost_pipeline(ner_output, models_dir=WEIGHTS_DIR)

    return submit

if __name__ == "__main__":

    df = pd.read_csv("data/raw/test.csv", delimiter=";")
    submit = process_example(df)

    submit[['sample', 'annotation']].to_csv("data/processed/xgboost_test_ensemble.csv", sep=";", index=False)