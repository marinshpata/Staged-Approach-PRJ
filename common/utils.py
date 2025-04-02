import os
import random
import torch
from torch.utils.data import Dataset

def load_lines(path):
    lines = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for l in f:
                l = l.strip()
                if l:
                    lines.append(l)
    return lines

def collate_fn(batch): # https://huggingface.co/docs/transformers/main_classes/data_collator - motivated by default data collator
    max_len = max(len(x) for x in batch)
    padded = []
    for seq in batch:
        padded.append(seq + [0]*(max_len - len(seq)))  # 0 => <pad>
    return torch.tensor(padded, dtype=torch.long)

class SingleDataset(Dataset):
    def __init__(self, lines, tokenizer):
        self.tokenizer = tokenizer
        self.examples = []
        for line in lines:
            token_ids = self.tokenizer.encode(line)
            self.examples.append(token_ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def encode_lines(lines, tokenizer):
    return [tokenizer.encode(l) for l in lines]

def sample_frac(lines, fraction):
    count = int(len(lines) * fraction)
    return random.sample(lines, count) if count > 0 else []

def train_on_encoded_lines(
    lines_encoded,
    model,
    optimizer,
    device,
    steps,
    batch_size=8,
    print_every=100
):
    if not lines_encoded:
        print("No lines => skipping this stage.")
        return

    for step in range(steps):
        # sample a batch from lines_encoded
        batch = random.sample(lines_encoded, min(batch_size, len(lines_encoded)))
        batch_input = collate_fn(batch).to(device)

        # forward pass
        outputs = model(batch_input, labels=batch_input)
        loss = outputs.loss

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % print_every == 0:
            print(f"  step {step + 1}/{steps} loss={loss.item():.4f}")

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True

def freeze_first_n_layers(model, n): #https://discuss.huggingface.co/t/freeze-lower-layers-with-auto-classification-model/11386
    unfreeze_all(model)  # unfreeze everything first, then re-freeze
    for name, param in model.named_parameters():
        if "transformer.h." in name:
            layer_str = name.split("h.")[1].split(".")[0]
            layer_idx = int(layer_str)
            if layer_idx < n:
                param.requires_grad = False

# tutorials and examples like these https://pytorch.org/tutorials/beginner/data_loading_tutorial.html, https://github.com/huggingface/transformers/tree/main/examples have been very helpful