import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import os
import re
import joblib
import numpy as np
import unicodedata
from typing import List, Tuple, Any

from models.NER_classifier import NERClassifier
from utils.rules import apply_priors
from utils.paths import WEIGHTS_DIR


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "cointegrated/rubert-tiny2"


class Predictor:
    def __init__(self):
        self.tokenizer = None
        self.ner_models = None
        self.boost_models = None
        self.label_encoder = None

        self._initialize_ner_models()
        self._initialize_boost_models()


    def _initialize_ner_models(self):
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

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
            models[name] = model

        self.ner_models = models


    def _initialize_boost_models(self):
        models_dir = os.path.join(WEIGHTS_DIR, "xgboost")
        self.boost_models = [
            joblib.load(os.path.join(models_dir, f"xgb_fold{fold}.joblib"))
            for fold in range(1, 6)
        ]
        self.label_encoder = joblib.load(os.path.join(models_dir, "label_encoder.joblib"))


    def _remove_accents_latin_only(self, text: str) -> str:
        def repl(ch):
            if 'A' <= ch <= 'Z' or 'a' <= ch <= 'z':
                decomp = unicodedata.normalize('NFD', ch)
                return ''.join(c for c in decomp if unicodedata.category(c) != 'Mn')
            return ch
        return ''.join(repl(ch) for ch in text)


    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = text.replace("ë", "ё").replace("Ë", "Ё")
        text = self._remove_accents_latin_only(text)
        text = re.sub(r"\\n|\\t", "", text)
        text = ''.join(
            ch for ch in text
            if unicodedata.category(ch)[0] != 'S' or ch == '№'
        )
        text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
        text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ№!&0-9%\-–—,.'’_/ +]", "", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()


    def _run_ner_pipeline(self, text: str, max_len: int = 64) -> dict:
        words = text.split()
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=max_len
        ).to(DEVICE)

        result = {"sample": text}

        for model_name, model in self.ner_models.items():
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
                result[f"{model_name}_proba_1"] = word_probs_1
                result[f"{model_name}_proba_2"] = word_probs_2
            else:
                result[f"{model_name}_probs"] = word_probs_1

        return result


    def _extract_features_for_text(self, ner_result: dict) -> np.ndarray:
        words = ner_result["sample"].split()
        n = len(words)
        features = []

        def get_probs(idx):
            if 0 <= idx < n:
                return [
                    ner_result['brand_proba_1'][idx],
                    ner_result['brand_proba_2'][idx],
                    ner_result['type_proba_1'][idx],
                    ner_result['type_proba_2'][idx],
                    ner_result['percent_proba_1'][idx],
                    ner_result['percent_proba_2'][idx],
                    ner_result['volume_proba_1'][idx],
                    ner_result['volume_proba_2'][idx],
                    ner_result['o_probs'][idx],
                ]
            else:
                return [0.0] * 9

        for i, word in enumerate(words):
            current = get_probs(i)
            prev = get_probs(i - 1)
            next_ = get_probs(i + 1)
            has_prev = 1 if i > 0 else 0
            has_next = 1 if i < n - 1 else 0

            feat = current + [i, len(word)] + prev + [has_prev] + next_ + [has_next]
            features.append(feat)

        return np.array(features, dtype=np.float32)


    def _annotate_text(
        self,
        original_text: str,
        cleaned_words: List[str],
        labels: List[str]
    ) -> List[Tuple[int, int, str]]:
        """
        Сопоставляет аннотации из cleaned_words с позициями в original_text.
        """
        orig_words = original_text.split()
        annotations = []

        if len(cleaned_words) != len(orig_words):
            char_idx = 0
            for word, label in zip(cleaned_words, labels):
                start = original_text.find(word, char_idx)
                if start == -1:
                    continue
                end = start + len(word)
                char_idx = end
                annotations.append((start, end, label))
        else:
            char_idx = 0
            for orig_word, label in zip(orig_words, labels):
                start = original_text.find(orig_word, char_idx)
                if start == -1:
                    start = char_idx
                end = start + len(orig_word)
                char_idx = end + 1
                annotations.append((start, end, label))

        return annotations


    def predict(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Predict NER annotations for a single input text.

        Parameters:
        -----------
        text : str
            Input text string to annotate.

        Returns:
        --------
        List[Tuple[int, int, str]]
            List of annotations as (start_char_index, end_char_index, label).
        """
        if not isinstance(text, str):
            text = ""

        cleaned_text = self._clean_text(text)

        if not cleaned_text:
            return []

        ner_result = self._run_ner_pipeline(cleaned_text)

        X = self._extract_features_for_text(ner_result)

        all_probs = [model.predict_proba(X) for model in self.boost_models]
        mean_probs = np.mean(all_probs, axis=0)

        words = cleaned_text.split()
        mean_probs = apply_priors(cleaned_text, words, mean_probs, beta=1.0)

        pred_indices = mean_probs.argmax(axis=1)
        pred_labels = [str(label) for label in self.label_encoder.inverse_transform(pred_indices)]

        annotations = self._annotate_text(original_text=text, cleaned_words=words, labels=pred_labels)

        return annotations