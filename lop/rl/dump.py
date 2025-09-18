import pickle
from itertools import chain
from pathlib import Path

import numpy as np


def get_param_performance(path):
    with open(path, "rb+") as f:
        print(f)
        data = pickle.load(f)

    rewards = np.array(data["rewards"])
    pos = np.array(data["pos"])
    output_dir = path.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}.npz"
    print(rewards.shape)
    np.savez_compressed(output_path, rewards=rewards, pos=pos)
    print(f"Saved to {output_path}")


def main():
    paths = list(chain(
        Path("data/foragax").rglob("*.log"),
        Path("data/foragax-sweep").rglob("*.log"),
    ))

    for path in paths:
        get_param_performance(path)


if __name__ == "__main__":
    main()
