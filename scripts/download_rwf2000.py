"""Downloads RWF-2000 and lays it out the way this project expects it.

IMPORTANT - please read before running this.

As of this writing, the dataset's official repository
(https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)
no longer distributes the video files itself: its README now says
"Due to the privacy requirements, video files are currently not available
on this website." The Agreement Sheet + email process that used to hand out
a download link is no longer documented there.

Two ways to get the data, in order of preference:

1. OFFICIAL (slower, may not respond): email the corresponding author,
   Ming Cheng, at ming.cheng@dukekunshan.edu.cn, explain you need the
   dataset for a university project, and ask whether access is still
   possible. This is the only channel with the original authors' blessing.

2. COMMUNITY MIRROR (what this script does): downloads a single
   `RWF-2000.tar.gz` archive from a community-hosted copy on Hugging Face
   (https://huggingface.co/datasets/DanJoshua/RWF-2000). This is fast and
   needs no registration, but note that the original RWF-2000 license
   restricts redistribution without the SMIIP Lab's approval - this mirror
   exists without documented approval. It is common practice in the
   research community to use such mirrors for non-commercial academic work,
   but you should still cite the original paper (see README.md) and should
   not treat this as a fully "authorised" distribution channel.

Usage:
    python scripts/download_rwf2000.py
    python scripts/download_rwf2000.py --target-dir ./data/RWF-2000
"""

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

ARCHIVE_URL = "https://huggingface.co/datasets/DanJoshua/RWF-2000/resolve/main/RWF-2000.tar.gz"
CLASS_NAMES = ("Fight", "NonFight")


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as file_handle, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=destination.name
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            file_handle.write(chunk)
            progress_bar.update(len(chunk))


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    extraction_tmp = target_dir.parent / "_rwf2000_extract_tmp"

    if extraction_tmp.exists():
        shutil.rmtree(extraction_tmp)

    extraction_tmp.mkdir(parents=True)

    print(f"Extracting {archive_path} ...")

    counters = {
        ("train", "Fight"): 0,
        ("train", "NonFight"): 0,
        ("val", "Fight"): 0,
        ("val", "NonFight"): 0,
    }

    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():

            if not member.isfile():
                continue

            parts = Path(member.name).parts

            # Expected structure:
            # RWF-2000/train/Fight/video.avi
            # RWF-2000/train/NonFight/video.avi
            # RWF-2000/val/Fight/video.avi
            # RWF-2000/val/NonFight/video.avi

            if len(parts) < 3:
                continue

            split = parts[-3]
            class_name = parts[-2]

            if split not in {"train", "val"}:
                continue

            if class_name not in CLASS_NAMES:
                continue

            destination_dir = extraction_tmp / split / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)

            # Generate a safe sequential filename.
            key = (split, class_name)
            counters[key] += 1

            output_name = f"video_{counters[key]:04d}.avi"
            destination = destination_dir / output_name

            source = archive.extractfile(member)

            if source is None:
                print(f"Warning: could not read {member.name}")
                continue

            with source, open(destination, "wb") as output:
                shutil.copyfileobj(
                    source,
                    output,
                    length=1024 * 1024,
                )

    # Move extracted dataset to final destination.
    target_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        source_split = extraction_tmp / split
        destination_split = target_dir / split

        if destination_split.exists():
            shutil.rmtree(destination_split)

        shutil.move(
            str(source_split),
            str(destination_split),
        )

    shutil.rmtree(extraction_tmp)

    print(
        f"Extracted: "
        f"train/Fight={counters['train', 'Fight']}, "
        f"train/NonFight={counters['train', 'NonFight']}, "
        f"val/Fight={counters['val', 'Fight']}, "
        f"val/NonFight={counters['val', 'NonFight']}"
    )


def verify_layout(target_dir: Path) -> bool:
    all_ok = True
    for split in ("train", "val"):
        for class_name in CLASS_NAMES:
            class_dir = target_dir / split / class_name
            count = len(list(class_dir.glob("*"))) if class_dir.exists() else 0
            status = "OK" if count > 0 else "MISSING"
            if count == 0:
                all_ok = False
            print(f"  [{status}] {split}/{class_name}: {count} files")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract the RWF-2000 dataset.")
    parser.add_argument("--target-dir", type=str, default="./data/RWF-2000")
    parser.add_argument(
        "--keep-archive", action="store_true", help="Do not delete the downloaded .tar.gz afterwards."
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    archive_path = Path("./RWF-2000.tar.gz")

    print(__doc__.split("Usage:")[0])

    if not archive_path.exists():
        print(f"Downloading RWF-2000 (~12.3 GB) from the Hugging Face mirror ...")
        download_file(ARCHIVE_URL, archive_path)
    else:
        print(f"{archive_path} already exists, skipping download.")

    extract_archive(archive_path, target_dir)

    print("\nVerifying dataset layout:")
    layout_ok = verify_layout(target_dir)

    if not args.keep_archive:
        archive_path.unlink()
        print(f"Removed {archive_path}")

    if layout_ok:
        print(f"\nDone. Dataset ready at: {target_dir.resolve()}")
    else:
        print(
            f"\nExtraction finished but some class folders are empty - inspect "
            f"{target_dir.resolve()} manually before training.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
