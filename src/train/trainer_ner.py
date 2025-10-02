import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

from models.NER_classifier import NERClassifier
from datasets.NER_dataset import NERDataset
from utils.paths import WEIGHTS_DIR

class TrainerNER:
    def __init__(self, model_name: str, train_data: list, val_data: list,
                 base_model="cointegrated/rubert-tiny2",
                 batch_size=64, epochs=5, lr=5e-5, max_len=30, device=None,
                 output_dir=WEIGHTS_DIR):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model = base_model
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.max_len = max_len
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # параметры по именам моделей
        model_params = {
            "brand": {"num_labels": 3, "use_dropout": False},
            "type": {"num_labels": 3, "use_dropout": True},
            "volume": {"num_labels": 3, "use_dropout": False},
            "percent": {"num_labels": 3, "use_dropout": False},
            "o": {"num_labels": 2, "use_dropout": False},
        }

        if model_name not in model_params:
            raise ValueError(f"Unknown model name '{model_name}', available: {list(model_params.keys())}")

        self.model_name = model_name
        self.num_labels = model_params[model_name]["num_labels"]
        self.use_dropout = model_params[model_name]["use_dropout"]

        # общий токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        # датасеты и загрузчики
        self.train_loader = DataLoader(
            NERDataset(train_data, self.tokenizer, self.max_len),
            batch_size=self.batch_size, shuffle=True)

        self.val_loader = DataLoader(
            NERDataset(val_data, self.tokenizer, self.max_len),
            batch_size=self.batch_size, shuffle=False)

        # модель
        self.model = NERClassifier(self.base_model, self.num_labels, self.use_dropout)
        self.model.to(self.device)

        # оптимизатор и лосс
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def train(self):
        best_f1 = 0.0

        for epoch in range(self.epochs):
            # training
            self.model.train()
            loop = tqdm(self.train_loader, leave=False, desc=f"Train {self.model_name} Epoch {epoch+1}")
            for batch in loop:
                self.optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))
                loss.backward()
                self.optimizer.step()

                loop.set_postfix(loss=loss.item())

            # validation
            self.model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in self.val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    logits = self.model(input_ids, attention_mask)
                    preds = torch.argmax(logits, dim=-1)

                    mask = labels != -100
                    all_preds.extend(preds[mask].cpu().numpy())
                    all_labels.extend(labels[mask].cpu().numpy())

            all_labels_np = np.array(all_labels)
            all_preds_np = np.array(all_preds)

            f1_per_label = f1_score(all_labels_np, all_preds_np,
                                    labels=list(range(self.num_labels)),
                                    average=None)
            f1_macro = f1_score(all_labels_np, all_preds_np,
                                labels=list(range(self.num_labels)),
                                average='macro')

            print(f"Epoch {epoch+1} | F1 per label: {f1_per_label} | Macro F1: {f1_macro:.4f}")

            # save best model
            if f1_macro > best_f1:
                best_f1 = f1_macro
                save_path = os.path.join(self.output_dir, self.model_name)
                os.makedirs(save_path, exist_ok=True)
                self.tokenizer.save_pretrained(save_path)
                torch.save(self.model.state_dict(), os.path.join(save_path, "model.pt"))
                print(f"  -> Saved best model at Epoch {epoch+1} with Macro F1={best_f1:.4f}")

        print("Training finished.")

