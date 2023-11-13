import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel

roberta = RobertaModel.from_pretrained("roberta-base")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

inputs1 = tokenizer(
    "And it came to pass that he was obedient unto the word of the Lord, wherefore he did as the Lord commanded him.",
    padding="max_length",
    max_length=512,
    return_tensors="pt",
    truncation=True,
)

inputs2 = tokenizer(
    "And after they had been received unto baptism, and were wrought upon and cleansed by the power of the Holy Ghost, they were numbered among the people of the church of Christ; and their names were taken, that they might be remembered and nourished by the good word of God, to keep them in the right way, to keep them continually watchful unto prayer, relying alone upon the merits of Christ, who was the author and the finisher of their faith.",
    padding="max_length",
    max_length=512,
    return_tensors="pt",
    truncation=True,
)

out1 = roberta(**inputs1)
out2 = roberta(**inputs2)

print(nn.functional.cosine_similarity(out1.last_hidden_state.mean(dim=1), out2.last_hidden_state.mean(dim=1), dim=1))
