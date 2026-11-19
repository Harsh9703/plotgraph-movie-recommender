"""Script to download and extract MovieLens-1M dataset."""

import argparse
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile
import yaml


def download_movielens_1m(output_dir: str = "data/raw", config_path: str = "configs/data.yaml") -> None:
    """Download MovieLens-1M zip archive and extract to target directory."""
    raw_dir = Path(output_dir)
    target_dir = raw_dir / "ml-1m"

    ratings_file = target_dir / "ratings.dat"
    movies_file = target_dir / "movies.dat"

    if ratings_file.is_file() and movies_file.is_file():
        print(f"[INFO] MovieLens-1M already exists at {target_dir.resolve()}. Skipping download.")
        return

    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    if Path(config_path).is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            url = cfg.get("ml1m", {}).get("url", url)

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_dest = raw_dir / "ml-1m.zip"

    print(f"[INFO] Downloading MovieLens-1M from {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(zip_dest, "wb") as out_file:
        shutil.copyfileobj(response, out_file)

    print(f"[INFO] Extracting {zip_dest} to {raw_dir}...")
    with zipfile.ZipFile(zip_dest, "r") as zip_ref:
        zip_ref.extractall(raw_dir)

    # Clean up zip
    if zip_dest.is_file():
        zip_dest.unlink()

    print(f"[SUCCESS] MovieLens-1M extracted successfully to {target_dir.resolve()}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MovieLens-1M dataset.")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Directory to store extracted ml-1m data.")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to config file.")
    args = parser.parse_args()

    download_movielens_1m(output_dir=args.output_dir, config_path=args.config)
