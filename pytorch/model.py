from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

class SiameseAuthorshipModel(nn.Module):
    def __init__(
        self,
        similarity_threshold: float = 3.0,
        roberta_model: str = "roberta-large-mnli",
    ):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model)
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_model)
        self.similarity_threshold = similarity_threshold

    def get_cls_embedding(self, text: str, max_length: int = 512) -> torch.Tensor:
        tokenized = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        output = self.roberta(input_ids, attention_mask=attention_mask)  # type: ignore
        cls_embedding = output.last_hidden_state[:, 0, :]
        return cls_embedding

    def forward(self, inputs: tuple[str, str]):
        text1, text2 = inputs
        author1_embed = self.get_cls_embedding(text1)
        author2_embed = self.get_cls_embedding(text2)
        print(author1_embed.requires_grad)
        print(author2_embed.requires_grad)
        

        cos_similarity = torch.cosine_similarity(author1_embed, author2_embed, dim=1)
        similarity_threshold = self.similarity_threshold * torch.ones_like(
            cos_similarity
        )
        predictions = (cos_similarity <= similarity_threshold).to(torch.float)

        return predictions
