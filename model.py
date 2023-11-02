from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel


class BaseAuthorshipModel(nn.Module, ABC):
    def __init__(self, similarity_threshold: float = 3.0, roberta_model: str = "roberta-large-mnli"):
        super(ABC, self).__init__()
        super(nn.Module, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model)
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_model)
        self.similarity_threshold = similarity_threshold
        self.linear1 = nn.Linear(self.roberta.config.hidden_size, 512) # type: ignore
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(512, 256)

        self.sequential = nn.Sequential(
            self.linear1,
            self.relu1,
            self.dropout1,
            self.linear2
        )

        self.cos_similarity = nn.functional.cosine_similarity

    def get_avg_cls_embedding(self, input_text: str, max_length: int = 512) -> torch.Tensor:
        chunks = [
            input_text[i : i + max_length]
            for i in range(0, len(input_text), max_length)
        ]

        embeddings_sum = torch.zeros(self.roberta.config.hidden_size)  # type: ignore
        segment_count = 0

        for chunk in chunks:
            inputs = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
            )

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Run the chunk through RoBERTa
            outputs = self.roberta(input_ids, attention_mask=attention_mask)  # type: ignore
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # Extract CLS embedding

            embeddings_sum += cls_embedding
            segment_count += 1
        average_embedding = embeddings_sum / segment_count
        return average_embedding

    def forward(self, inputs: tuple[str, str], max_length: int = 512):
        text1, text2 = inputs
        author1_embed = self.get_author1_embedding(text1)
        author2_embed = self.get_avg_cls_embedding(text2)
        author1_processed = self.sequential(author1_embed)
        author2_processed = self.sequential(author2_embed)
        cos_similarity = torch.cosine_similarity(author1_processed, author2_processed, dim=1)
        if cos_similarity =< self.cos_similarity:
            return 1
        else:
            return 0

    @abstractmethod
    def get_author1_embedding(self, text: str) -> torch.Tensor:
        pass

class ProductionAuthorshipModel(BaseAuthorshipModel):
    def __init__(self, author1_embedding: torch.Tensor, roberta_model: str = "roberta-large-mnli"):
        super().__init__(roberta_model)
        self._author1_embedding = author1_embedding

    def get_author1_embedding(self, _):
        return self._author1_embedding


class TrainingAuthorshipModel(BaseAuthorshipModel):
    def __init__(self, similarity_threshold: float = 3.0, roberta_model: str = "roberta-large-mnli"):
        super().__init__(roberta_model)

    def get_author1_embedding(self, text: str) -> torch.Tensor:
        return self.get_avg_cls_embedding(text)
