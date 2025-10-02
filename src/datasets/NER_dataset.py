from torch.utils.data import Dataset
import torch

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        labels = self.data[idx]['labels']

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            is_split_into_words=False
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        word_ids = encoding.word_ids(batch_index=0)
        token_labels = []
        for word_id in word_ids:
            if word_id is None or word_id >= len(labels):
                token_labels.append(-100)
            else:
                token_labels.append(labels[word_id])
        token_labels = torch.tensor(token_labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": token_labels
        }