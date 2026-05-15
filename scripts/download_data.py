from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


DATASET = "bittlingmayer/amazonreviews"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Amazon Reviews Kaggle dataset.")
    parser.add_argument("--target", default="data/raw", help="Directory for downloaded dataset files.")
    args = parser.parse_args()

    target = Path(args.target).resolve()
    target.mkdir(parents=True, exist_ok=True)

    kaggle = shutil.which("kaggle")
    if not kaggle:
        raise SystemExit(
            "Kaggle CLI was not found. Install it with `pip install kaggle`, then configure "
            "your Kaggle API token before running this script."
        )

    command = [
        kaggle,
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        str(target),
        "--unzip",
    ]
    subprocess.run(command, check=True)
    print(f"Dataset downloaded to {target}")


if __name__ == "__main__":
    main()
