import torch
import torch.nn as nn


model_name = "roberta.large.mnli"
roberta = torch.hub.load('pytorch/fairseq', model_name)