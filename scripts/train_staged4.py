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
    train_on_encoded_lines,
    freeze_first_n_layers,
    unfreeze_all
)

def train_staged4(
    data_dir="data/",
    stage_steps=None,
    batch_size=8,
    lr=1e-3,
    freeze_n_layers=2
):
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
    print(f"[Staged4.0] partial freezing => Vocab size={vocab_size}")

    model = ChildTransformer(vocab_size=vocab_size).to(device)

    def train_stage(lines_list, steps):
        dataset_encoded = encode_lines(lines_list, tokenizer)
        if not dataset_encoded:
            print("No lines => skipping stage.")
            return
        local_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        train_on_encoded_lines(
            lines_encoded=dataset_encoded,
            model=model,
            optimizer=local_optimizer,
            device=device,
            steps=steps,
            batch_size=batch_size
        )

    print("--- Step 1 (unfreeze all) ---")
    unfreeze_all(model)
    train_stage(linesA, stage_steps["A"])

    print(f"--- Step 2 (freeze first {freeze_n_layers}) ---")
    freeze_first_n_layers(model, freeze_n_layers)
    train_stage(linesB, stage_steps["B"])

    print("--- Step 3 (still freeze) ---")
    train_stage(linesC, stage_steps["C"])

    print("--- Step 4 (still freeze) ---")
    train_stage(linesD, stage_steps["D"])

    ckpt = {
        "model_state": model.state_dict(),
        "vocab_size": vocab_size,
        "word2idx": tokenizer.word2idx,
        "idx2word": tokenizer.idx2word
    }
    torch.save(ckpt, "staged4_transformer.pth")
    print("[Staged4.0] Model saved as staged4_transformer.pth")
