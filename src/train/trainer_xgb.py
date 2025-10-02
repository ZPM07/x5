import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, classification_report
from ast import literal_eval
import joblib
import os
from utils.paths import WEIGHTS_DIR


class Trainer_xgb:
    def __init__(self, data_path, output_dir=WEIGHTS_DIR/"xgboost", n_splits=4, random_state=42):
        self.data_path = data_path
        self.output_dir = output_dir
        self.n_splits = n_splits
        self.random_state = random_state
        self.le = LabelEncoder()
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def words_with_offsets(text):
        words = []
        pos = 0
        for w in text.split():
            start = text.find(w, pos)
            end = start + len(w)
            words.append((w, start, end))
            pos = end
        return words

    def _load_and_prepare_data(self):
        data = pd.read_csv(self.data_path)
        for col in data.columns:
            if col == 'sample':
                continue
            data[col] = data[col].apply(lambda x: literal_eval(x))

        X_list, y_list = [], []

        for _, row in data.iterrows():
            words_offsets = self.words_with_offsets(row['sample'])
            n_words = len(words_offsets)

            for i, (word, start, end) in enumerate(words_offsets):
                probs = [
                    row['brand_proba_1'][i],
                    row['brand_proba_2'][i],
                    row['type_proba_1'][i],
                    row['type_proba_2'][i],
                    row['percent_proba_1'][i],
                    row['percent_proba_2'][i],
                    row['volume_proba_1'][i],
                    row['volume_proba_2'][i],
                    row['o_probs'][i],
                ]

                word_pos = i
                word_len = len(word)

                if i > 0:
                    prev_probs = [
                        row['brand_proba_1'][i-1],
                        row['brand_proba_2'][i-1],
                        row['type_proba_1'][i-1],
                        row['type_proba_2'][i-1],
                        row['percent_proba_1'][i-1],
                        row['percent_proba_2'][i-1],
                        row['volume_proba_1'][i-1],
                        row['volume_proba_2'][i-1],
                        row['o_probs'][i-1],
                    ]
                    has_prev = 1
                else:
                    prev_probs = [0] * 9
                    has_prev = 0

                if i < n_words - 1:
                    next_probs = [
                        row['brand_proba_1'][i+1],
                        row['brand_proba_2'][i+1],
                        row['type_proba_1'][i+1],
                        row['type_proba_2'][i+1],
                        row['percent_proba_1'][i+1],
                        row['percent_proba_2'][i+1],
                        row['volume_proba_1'][i+1],
                        row['volume_proba_2'][i+1],
                        row['o_probs'][i+1],
                    ]
                    has_next = 1
                else:
                    next_probs = [0] * 9
                    has_next = 0

                features = probs + [word_pos, word_len] + prev_probs + [has_prev] + next_probs + [has_next]
                X_list.append(features)

            # метки
            labels = []
            for word, start, end in words_offsets:
                label = 'O'
                for ann_start, ann_end, tag in row['annotation']:
                    if start >= ann_start and end <= ann_end:
                        label = tag
                        break
                labels.append(label)

            y_list.extend(labels)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)

        self.X = X
        self.y = y

    def _encode_labels(self):
        self.y_enc = self.le.fit_transform(self.y)
        print("Classes in LabelEncoder:", self.le.classes_)
        print("Unique labels after transform:", np.unique(self.y_enc))

    def _compute_sample_weights(self):
        classes, counts = np.unique(self.y_enc, return_counts=True)
        class_weights = {cls: 1.0 / cnt for cls, cnt in zip(classes, counts)}
        mean_w = np.mean(list(class_weights.values()))
        class_weights = {cls: w / mean_w for cls, w in class_weights.items()}
        self.sample_weight = np.array([class_weights[label] for label in self.y_enc])

    def train(self):

        self._load_and_prepare_data()
        self._encode_labels()
        self._compute_sample_weights()

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        macro_f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X), 1):
            print(f"\n===== FOLD {fold} =====")

            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y_enc[train_idx], self.y_enc[val_idx]
            w_train = self.sample_weight[train_idx]

            clf = XGBClassifier(
                objective="multi:softprob",
                num_class=len(np.unique(self.y_enc)),
                eval_metric="mlogloss",
                n_estimators=300,
                learning_rate=0.1,
                max_depth=3,
                random_state=self.random_state,
                n_jobs=-1,
            )

            clf.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], verbose=False)

            y_pred = clf.predict(X_val)

            print(f"\nFold {fold} classification report:\n")
            print(classification_report(
                y_val,
                y_pred,
                labels=np.arange(len(self.le.classes_)),
                target_names=self.le.classes_
            ))

            macro_f1 = f1_score(y_val, y_pred, average='macro')
            print(f"Fold {fold} Macro F1-score: {macro_f1:.4f}")
            macro_f1_scores.append(macro_f1)

            joblib.dump(clf, os.path.join(self.output_dir, f"xgb_fold{fold}.joblib"))
            print(f"Fold {fold} model saved to {os.path.join(self.output_dir, f'xgb_fold{fold}.joblib')}")

        print(f"\nAverage Macro F1 over {self.n_splits} folds: {np.mean(macro_f1_scores):.4f}")
        joblib.dump(self.le, os.path.join(self.output_dir, "label_encoder.joblib"))
        print("LabelEncoder saved.")

        self.macro_f1_scores = macro_f1_scores


if __name__ == "__main__":
    trainer = Trainer_xgb(data_path="train_ensamble.csv")
    trainer.train()