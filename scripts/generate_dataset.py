#!/usr/bin/env python3
import argparse
from src.bio.simulated_dataset import save_dataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="datasets/sim_eeg")
    p.add_argument("--subjects", type=int, default=3)
    p.add_argument("--epochs", type=int, default=5)
    args = p.parse_args()
    out = save_dataset(path=args.path, n_subjects=args.subjects, epochs_per_subject=args.epochs)
    print(f"Dataset saved to: {out}")

if __name__ == "__main__":
    main()
