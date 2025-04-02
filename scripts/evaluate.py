from models.child_transformer import ChildTransformer
from scripts.tokenizer import SimpleWordTokenizer
import torch
import math
import random
import torch
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_transformer(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ChildTransformer(vocab_size=ckpt["vocab_size"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    tokenizer = SimpleWordTokenizer()
    tokenizer.word2idx = ckpt["word2idx"]
    tokenizer.idx2word = ckpt["idx2word"]
    tokenizer.vocab_size = ckpt["vocab_size"]

    return model, tokenizer

def compute_perplexity(ckpt_path, test_file="data/test.txt"):  #inspired from https://huggingface.co/docs/transformers/en/perplexity
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_transformer(ckpt_path, device)

    lines = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    total_loss = 0.0
    total_tokens = 0
    for line in lines:
        token_ids = tokenizer.encode(line)
        if len(token_ids) < 2:
            continue
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        seq_len = len(token_ids)
        total_loss += loss.item() * seq_len
        total_tokens += seq_len

    if total_tokens == 0:
        ppl = float('inf')
    else:
        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)

    print(f"[{ckpt_path}] perplexity on {test_file} = {ppl:.3f}")
    return ppl
