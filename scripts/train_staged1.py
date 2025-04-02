import os
import random
import torch
import torch.optim as optim

from models.child_transformer import ChildTransformer
from scripts.tokenizer import SimpleWordTokenizer
from common.utils import (
    load_lines,
    collate_fn,
    encode_lines,
    train_on_encoded_lines
)

def train_staged(data_dir="data/", stage_steps=None, epochs_per_stage=1, batch_size=8, lr=1e-3):
    if stage_steps is None:
        stage_steps = {"A": 500, "B": 700, "C": 700, "D": 700}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    linesA = load_lines(os.path.join(data_dir, "step1.txt"))
    linesB = load_lines(os.path.join(data_dir, "step2.txt"))
    linesC = load_lines(os.path.join(data_dir, "step3.txt"))
    linesD = load_lines(os.path.join(data_dir, "step4.txt"))
    all_lines = linesA + linesB + linesC + linesD

    tokenizer = SimpleWordTokenizer()
    tokenizer.build_vocab(all_lines)
    vocab_size = tokenizer.vocab_size
    print(f"[Staged] Vocab size = {vocab_size}")

    model = ChildTransformer(vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("--- Training Step 1 ---")
    encodedA = encode_lines(linesA, tokenizer)
    train_on_encoded_lines(encodedA, model, optimizer, device, stage_steps["A"], batch_size)

    print("--- Training Step 2 ---")
    encodedB = encode_lines(linesB, tokenizer)
    train_on_encoded_lines(encodedB, model, optimizer, device, stage_steps["B"], batch_size)

    print("--- Training Step 3 ---")
    encodedC = encode_lines(linesC, tokenizer)
    train_on_encoded_lines(encodedC, model, optimizer, device, stage_steps["C"], batch_size)

    print("--- Training Step 4 ---")
    encodedD = encode_lines(linesD, tokenizer)
    train_on_encoded_lines(encodedD, model, optimizer, device, stage_steps["D"], batch_size)

    ckpt = {
        "model_state": model.state_dict(),
        "vocab_size": vocab_size,
        "word2idx": tokenizer.word2idx,
        "idx2word": tokenizer.idx2word
    }
    torch.save(ckpt, "staged_transformer.pth")
    print("[Staged] Model saved as staged_transformer.pth")
