import argparse
import os
from pathlib import Path

import numpy as np


def get_args():
    parser = argparse.ArgumentParser("Split .pt files into sub-directory of soft-links")
    parser.add_argument("pt_files_dir", help="posterior scp file")
    parser.add_argument("store_files_dir", help="where to save split files")
    parser.add_argument("--nj", default=10, type=int, help="Number of splits")

    return parser.parse_args()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == "__main__":
    config = get_args()

    feature_files = [os.path.join(config.pt_files_dir, f) for f in os.listdir(config.pt_files_dir) if f.endswith('.pt')]
    feature_files = list(chunks(feature_files, int(np.round(len(feature_files) / config.nj))))
    if len(feature_files) > config.nj:
        # Merge the last two elements and delete last one
        feature_files[-2] = feature_files[-1] + feature_files[-2]
        feature_files = feature_files[:-1]

    for i in range(1, config.nj + 1):
        path = os.path.join(config.store_files_dir, str(i))
        if os.path.isdir(path):
            os.system("rm -r " + path)  # Remove the folder and make fresh symlinks
        Path(path).mkdir(parents=True)
        for j in feature_files[i - 1]:
            file_basename = os.path.basename(j)
            os.symlink(j, os.path.join(path, file_basename))
