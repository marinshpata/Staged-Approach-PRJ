import os
from scripts.train_single import train_single
from scripts.train_staged1 import train_staged
from scripts.train_staged2 import train_staged2
from scripts.train_staged3 import train_staged3
from scripts.train_staged4 import train_staged4
from scripts.train_staged5 import train_staged5
from scripts.evaluate import set_seed
from scripts.evaluate import compute_perplexity

def main():
    set_seed(24062003)

    # 1) single-shot
    print("--- Train Single-Shot Baseline ---")
    train_single(data_dir="data/", epochs=3, batch_size=8, lr=1e-3)
    single_ckpt = "single_shot_transformer.pth"

    # 2) staged (naive)
    print("--- Train Staged (Naive) ---")
    stage_steps = {"A":500, "B":700, "C":700, "D":700} #final choice was made after testing various allocations, 500/700 better balance that reflects increasing complexity and avoids overfitting. Would be useful to investigate further in future iterations, perhaps using a pipeline similar to the one in chapter 7.
    train_staged(data_dir="data/", stage_steps=stage_steps, batch_size=8, lr=1e-3)
    staged_ckpt = "staged_transformer.pth"

    # 3) staged2 => partial replay
    print("--- Train Staged2 (Partial Replay) ---")
    train_staged2(data_dir="data/", stage_steps=stage_steps, batch_size=8, lr=1e-3, replay_frac=0.1)
    staged2_ckpt = "staged2_transformer.pth"

    # 4) staged3 => weighted replay
    print("--- Train Staged3 (Weighted Replay) ---")
    train_staged3(data_dir="data/", stage_steps=stage_steps, batch_size=8, lr=1e-3,
                  replay_frac_global=0.1, replay_frac_immediate=0.2)
    staged3_ckpt = "staged3_transformer.pth"

    # 5) staged4 => partial freezing
    print("--- Train Staged4 (Partial Freezing) ---")
    train_staged4(data_dir="data/", stage_steps=stage_steps, batch_size=8, lr=1e-3,
                  freeze_n_layers=2)
    staged4_ckpt = "staged4_transformer.pth"

    # 6) staged5 => progressive unfreezing + replay
    print("--- Train Staged5 (Progressive Unfreezing + Replay) ---")
    train_staged5(data_dir="data/", stage_steps=stage_steps, batch_size=8, lr=1e-3,
                  freeze_n_layers=2, replay_frac=0.1)
    staged5_ckpt = "staged5_transformer.pth"

    # perplexity  eval
    test_file = os.path.join("data", "test.txt")
    print(f"\n--- Evaluate Perplexity on {test_file} ---")
    compute_perplexity(single_ckpt, test_file=test_file)
    compute_perplexity(staged_ckpt, test_file=test_file)
    compute_perplexity(staged2_ckpt, test_file=test_file)
    compute_perplexity(staged3_ckpt, test_file=test_file)
    compute_perplexity(staged4_ckpt, test_file=test_file)
    compute_perplexity(staged5_ckpt, test_file=test_file)

if __name__ == "__main__":
    main()
