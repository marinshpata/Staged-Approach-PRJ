import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

class ChildTransformer(nn.Module):
    def __init__(self, vocab_size=50257, n_embd=128, n_layer=4, n_head=4): #change the number of layers and heads here
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        )
        self.transformer = GPT2LMHeadModel(config)

    def forward(self, input_ids, labels=None):
        outputs = self.transformer(input_ids=input_ids, labels=labels)
        return outputs

#https://huggingface.co/docs/transformers/model_doc/gpt2 and https://pytorch.org/docs/stable/generated/torch.nn.Module.html