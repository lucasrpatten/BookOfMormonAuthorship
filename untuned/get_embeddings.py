from numpy import int16
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel
import os
import pandas as pd
import numpy as np

roberta = RobertaModel.from_pretrained("roberta-large")
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

bom_order = [
    "1 Nephi",
    "2 Nephi",
    "Jacob",
    "Enos",
    "Jarom",
    "Omni",
    "Words of Mormon",
    "Mosiah",
    "Alma",
    "Heleman",
    "3 Nephi",
    "Mormon",
    "Ether",
    "Moroni",
]
BOM_PATH = "../dataset/bom/"
LONGEST_VERSE = 174
df = pd.DataFrame(
    columns=["book", "chapter", "verse", "embedding", "cos_sim"]
)
np.set_printoptions(threshold=np.inf)
verse_num = 0
previous_embed = None
for book in bom_order:
    book_path = os.path.join(BOM_PATH, book)
    for chapter in sorted(os.listdir(book_path), key=int):
        print(f"{book} {chapter}")
        chapter_path = os.path.join(book_path, chapter)
        for verse in sorted(
            os.listdir(chapter_path), key=lambda x: int(x.split(".")[0])
        ):
            verse_path = os.path.join(chapter_path, verse)
            with open(verse_path, "r", encoding="utf-8") as file:
                verse_text = file.read()
            tokenized = tokenizer(
                verse_text,
                padding="max_length",
                max_length=LONGEST_VERSE,
                return_tensors="pt",
                truncation=True,
            )
            embed = roberta(**tokenized).last_hidden_state.mean(dim=1)  # type: ignore
            df.loc[verse_num] = [
                book, int(chapter), int(verse.split(".")[0]), embed.detach().numpy(), None
            ]
            if previous_embed is not None:
                cos_sim = nn.functional.cosine_similarity(embed, previous_embed, dim=1)  # pylint: disable=E1102
                df.at[verse_num - 1, "cos_sim"] = float(
                    cos_sim.detach()
                )
            previous_embed = embed
            verse_num += 1
        df.to_csv("embeddings.csv")
