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

def train_staged3(
    data_dir="data/",
    stage_steps=None,
    batch_size=8,
    lr=1e-3,
    replay_frac_global=0.1,
    replay_frac_immediate=0.2
):

    #replay_frac_global from the global history
    #replay_frac_immediate from the most recent stage
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
    print(f"[Staged3.0] Weighted replay => Vocab size={vocab_size}")

    model = ChildTransformer(vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Step 1 => just linesA
    print("--- Step 1 ---")
    datasetA = encode_lines(linesA, tokenizer)
    train_on_encoded_lines(
        lines_encoded=datasetA,
        model=model,
        optimizer=optimizer,
        device=device,
        steps=stage_steps["A"],
        batch_size=batch_size
    )

    # Step 2 => linesB + replay_frac_global(A) + replay_frac_immediate(A)
    linesA_global = sample_frac(linesA, replay_frac_global)
    linesA_immediate = sample_frac(linesA, replay_frac_immediate)
    linesB_plus = linesB + linesA_global + linesA_immediate
    print("--- Step 2 ---")
    datasetB = encode_lines(linesB_plus, tokenizer)
    train_on_encoded_lines(
        lines_encoded=datasetB,
        model=model,
        optimizer=optimizer,
        device=device,
        steps=stage_steps["B"],
        batch_size=batch_size
    )

    linesAB = linesA + linesB

    # Step 3 => linesC + replay_frac_global(AB) + replay_frac_immediate(B)
    linesAB_global = sample_frac(linesAB, replay_frac_global)
    linesB_immediate = sample_frac(linesB, replay_frac_immediate)
    linesC_plus = linesC + linesAB_global + linesB_immediate
    print("--- Step 3 ---")
    datasetC = encode_lines(linesC_plus, tokenizer)
    train_on_encoded_lines(
        lines_encoded=datasetC,
        model=model,
        optimizer=optimizer,
        device=device,
        steps=stage_steps["C"],
        batch_size=batch_size
    )

    linesABC = linesA + linesB + linesC

    # Step4 => linesD + replay_frac_global(ABC) + replay_frac_immediate(C)
    linesABC_global = sample_frac(linesABC, replay_frac_global)
    linesC_immediate = sample_frac(linesC, replay_frac_immediate)
    linesD_plus = linesD + linesABC_global + linesC_immediate
    print("--- Step 4")
    datasetD = encode_lines(linesD_plus, tokenizer)
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
    torch.save(ckpt, "staged3_transformer.pth")
    print("[Staged3.0] Model saved as staged3_transformer.pth")
