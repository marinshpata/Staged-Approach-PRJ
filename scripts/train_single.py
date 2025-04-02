import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.child_transformer import ChildTransformer
from scripts.tokenizer import SimpleWordTokenizer
from common.utils import load_lines, collate_fn, SingleDataset

def train_single(data_dir="data/", epochs=3, batch_size=8, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    linesA = load_lines(os.path.join(data_dir, "step1.txt"))
    linesB = load_lines(os.path.join(data_dir, "step2.txt"))
    linesC = load_lines(os.path.join(data_dir, "step3.txt"))
    linesD = load_lines(os.path.join(data_dir, "step4.txt"))
    all_lines = linesA + linesB + linesC + linesD

    tokenizer = SimpleWordTokenizer()
    tokenizer.build_vocab(all_lines)
    vocab_size = tokenizer.vocab_size
    print(f"[SingleShot] Vocab size = {vocab_size}")

    dataset = SingleDataset(all_lines, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = ChildTransformer(vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        num_batches = 0
        for batch_input in dataloader:
            batch_input = batch_input.to(device)
            outputs = model(batch_input, labels=batch_input)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"[SingleShot] Epoch {epoch}/{epochs}, Loss={avg_loss:.4f}")

    ckpt = {
        "model_state": model.state_dict(),
        "vocab_size": vocab_size,
        "word2idx": tokenizer.word2idx,
        "idx2word": tokenizer.idx2word
    }
    torch.save(ckpt, "single_shot_transformer.pth")
    print("[SingleShot] Model saved as single_shot_transformer.pth")
