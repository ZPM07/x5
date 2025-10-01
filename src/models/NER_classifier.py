import torch.nn as nn
from transformers import AutoModel

class NERClassifier(nn.Module):
    def __init__(self, base_model_name: str, num_labels: int, use_dropout: bool = False):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size
        self.dropout = nn.Dropout(0.1) if use_dropout else nn.Identity()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(x)
        return logits

