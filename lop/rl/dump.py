import argparse
import os
import pickle

import numpy as np
import yaml


def get_param_performance(runs, data_dir=""):
    per_param_setting_performance = []
    for idx in runs:
        file = data_dir + str(idx)
        if file[0] == "d":
            file = "" + file
        try:
            with open(file, "rb+") as f:
                print(f)
                data = pickle.load(f)
        except Exception:
            with open(file + ".log", "rb+") as f:
                print(f)
                data = pickle.load(f)

        rewards = np.array(data["rewards"])
        pos = np.array(data["pos"])
        output_path = data_dir + f"data/{idx}.npz"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(rewards.shape)
        np.savez_compressed(output_path, rewards=rewards, pos=pos)
        print(f"Saved to {output_path}")

        per_param_setting_performance.append(rewards)

    return per_param_setting_performance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=False, type=str, default="foragax")
    parser.add_argument("--all", required=False, type=bool, default=True)

    args = parser.parse_args()
    env = args.env
    plot_all = args.all

    cfg_file = f"cfg/{env}/std.yml"
    cfg_file1 = f"cfg/{env}/cbp.yml"
    cfg_file2, cfg_file3 = "", ""
    if plot_all:
        cfg_file2 = f"cfg/{env}/ns.yml"
        cfg_file3 = f"cfg/{env}/l2.yml"

    cfg_files = [cfg_file, cfg_file1, cfg_file2, cfg_file3]
    cfgs = []
    for file in cfg_files:
        if file not in {cfg_file, cfg_file1}:
            continue
        cfgs.append(yaml.safe_load(open(file)))
        if "label" not in cfgs[-1].keys():
            cfgs[-1]["label"] = ""

    runs = list(range(20, 25))
    for idx, cfg in enumerate(cfgs):
        get_param_performance(data_dir=cfg["dir"], runs=runs)


if __name__ == "__main__":
    main()
