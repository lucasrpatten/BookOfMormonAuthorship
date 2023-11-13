from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from transformers import RobertaModel

class SiameseAuthorshipModel(nn.Module):
    def __init__(
        self,
        similarity_threshold: float = 3.0,
        roberta_model: str = "roberta-large",
    ):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model)
        self.fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.relu = nn.ReLU()
        self.similarity_threshold = similarity_threshold

    def forward(self, inputs1, inputs2):
        input1_ids = inputs1["input_ids"].squeeze(1)
        input1_mask = inputs1["attention_mask"].squeeze(1)
        input2_ids = inputs2["input_ids"].squeeze(1)
        input2_mask = inputs2["attention_mask"].squeeze(1)
        output1 = self.roberta(input1_ids, attention_mask=input1_mask).last_hidden_state.mean(dim=1)
        output2 = self.roberta(input2_ids, attention_mask=input2_mask).last_hidden_state.mean(dim=1)
        output1 = self.fc(output1)
        output2 = self.fc(output2)
        cosine_similarity = nn.functional.cosine_similarity(output1, output2, dim=1)
        return cosine_similarity

class SiameseAuthorshipNoTraining(SiameseAuthorshipModel):
    def __init__(self, similarity_threshold: float = 3, roberta_model: str = "roberta-large-mnli"):
        super().__init__(similarity_threshold, roberta_model)
        self.fc.apply(self.init_weights)
    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.ones_(layer.weight.data)