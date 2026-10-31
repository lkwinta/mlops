"""Download ABO originals and save paths to images of at least 1000 x 1000 px."""

import argparse
from pathlib import Path
import tarfile
from urllib.request import urlretrieve

import pandas as pd

# The small archive contains thumbnails of at most 256 px, so use originals.
DATASET_PATH = Path("data/abo-images-original.tar")
DATASET_URL = (
    "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-original.tar"
)


def download_and_extract(data_dir: Path) -> None:
    """Reuse completed downloads and extractions on subsequent runs."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / DATASET_PATH.name
    metadata_path = data_dir / "images/metadata/images.csv.gz"
    extraction_marker = data_dir / ".abo-images-original.extracted"

    if (
        extraction_marker.is_file()
        and metadata_path.is_file()
        and (data_dir / "images/original").is_dir()
    ):
        return

    if not archive_path.is_file():
        print(f"Downloading {DATASET_URL} (about 110 GB)...", flush=True)
        partial_path = archive_path.with_suffix(".tar.part")
        urlretrieve(DATASET_URL, partial_path)
        partial_path.replace(archive_path)

    print(f"Extracting {archive_path}...", flush=True)
    with tarfile.open(archive_path) as archive:
        archive.extractall(data_dir, filter="data")

    if not metadata_path.is_file() or not (data_dir / "images/original").is_dir():
        raise FileNotFoundError(
            "The archive is missing ABO image metadata or originals."
        )
    extraction_marker.touch()


def extract_image_paths(metadata_path: Path, images_dir: Path) -> list[Path]:
    """Filter original image dimensions and keep only existing local files."""
    metadata = pd.read_csv(
        metadata_path,
        usecols=["path", "width", "height"],
        dtype={"path": "string"},
    )
    width = pd.to_numeric(metadata["width"], errors="coerce")
    height = pd.to_numeric(metadata["height"], errors="coerce")
    relative_paths = (
        metadata.loc[(width >= 1000) & (height >= 1000), "path"]
        .dropna()
        .drop_duplicates()
    )

    images_dir = images_dir.resolve()
    valid_paths = []
    for relative_path in relative_paths:
        if not relative_path.strip() or Path(relative_path).is_absolute():
            continue
        image_path = (images_dir / relative_path).resolve()
        if image_path.is_relative_to(images_dir) and image_path.is_file():
            valid_paths.append(image_path)

    return valid_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATASET_PATH.parent)
    args = parser.parse_args()

    download_and_extract(args.data_dir)
    image_paths = extract_image_paths(
        args.data_dir / "images/metadata/images.csv.gz",
        args.data_dir / "images/original",
    )
    output_path = args.data_dir / "image_paths.txt"
    with output_path.open("w", encoding="utf-8") as output:
        for image_path in image_paths:
            output.write(f"{image_path}\n")
    print(f"Saved {len(image_paths)} valid image paths to {output_path}.")


if __name__ == "__main__":
    main()
