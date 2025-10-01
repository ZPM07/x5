import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Dict, Any

def ner_pipeline(samples: List[Dict[str, Any]], models: Dict[str, torch.nn.Module], tokenizer, max_len: int = 64) -> pd.DataFrame:
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
        )

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
