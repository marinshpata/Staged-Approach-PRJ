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
    sample_frac
)

def train_staged2(
    data_dir="data/",
    stage_steps=None,
    batch_size=8,
    lr=1e-3,
    replay_frac=0.1
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
    print(f"[Staged2.0] Vocab size = {vocab_size}")

    model = ChildTransformer(vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 1: Train on linesA alone
    print("--- Step 1: training on A alone ---")
    datasetA = encode_lines(linesA, tokenizer)
    train_on_encoded_lines(
        lines_encoded=datasetA,
        model=model,
        optimizer=optimizer,
        device=device,
        steps=stage_steps["A"],
        batch_size=batch_size
    )


    # 2: linesB + 10% of A
    print("--- Step 2: training on B + replay from A ---")
    linesA_replay = sample_frac(linesA, replay_frac)
    stageB_lines = linesB + linesA_replay
    datasetB = encode_lines(stageB_lines, tokenizer)
    train_on_encoded_lines(
        lines_encoded=datasetB,
        model=model,
        optimizer=optimizer,
        device=device,
        steps=stage_steps["B"],
        batch_size=batch_size
    )

    # Combine A + B for future replay
    linesAB = linesA + linesB

    # Step 3: linesC + 10% of (A+B)
    print("--- Step 3: training on C + replay from A+B ---")
    linesAB_replay = sample_frac(linesAB, replay_frac)
    stageC_lines = linesC + linesAB_replay
    datasetC = encode_lines(stageC_lines, tokenizer)
    train_on_encoded_lines(
        lines_encoded=datasetC,
        model=model,
        optimizer=optimizer,
        device=device,
        steps=stage_steps["C"],
        batch_size=batch_size
    )

    # Combine A + B + C
    linesABC = linesA + linesB + linesC

    # Step 4: linesD + 10% of (A+B+C)
    print("--- Step 4: training on D + replay from A+B+C ---")
    linesABC_replay = sample_frac(linesABC, replay_frac)
    stageD_lines = linesD + linesABC_replay
    datasetD = encode_lines(stageD_lines, tokenizer)
    train_on_encoded_lines(
        lines_encoded=datasetD,
        model=model,
        optimizer=optimizer,
        device=device,
        steps=stage_steps["D"],
        batch_size=batch_size
    )

    ckpt = {
        "model_state": model.state_dict(),
        "vocab_size": vocab_size,
        "word2idx": tokenizer.word2idx,
        "idx2word": tokenizer.idx2word
    }
    torch.save(ckpt, "staged2_transformer.pth")
    print("[Staged2.0] Model saved as staged2_transformer.pth")
