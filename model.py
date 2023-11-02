import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaModel


tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
model = RobertaModel.from_pretrained('roberta-large-mnli')

