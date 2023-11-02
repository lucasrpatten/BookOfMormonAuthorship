import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaModel

ROBERTA_MODEL='roberta-large-mnli'
tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
roberta = RobertaModel.from_pretrained(ROBERTA_MODEL)
roberta.config.output_hidden_states = True