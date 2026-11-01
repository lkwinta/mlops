from pathlib import Path
import tarfile
from urllib.request import urlretrieve
from tqdm import tqdm

import polars as pl

DATA_DIR = Path("./data")
DATASET_URL = (
    "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar"
)


def download_and_extract() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    archive_path = DATA_DIR / "abo-images-small.tar"

    if (DATA_DIR / "images/small").is_dir():
        return

    if not archive_path.is_file():
        print(f"Downloading {DATASET_URL}...", flush=True)
        with tqdm(desc="Downloading", unit="B", unit_scale=True) as progress:

            def reporthook(block_count, block_size, total_size):
                progress.total = total_size if total_size >= 0 else None
                downloaded = block_count * block_size
                if total_size >= 0:
                    downloaded = min(downloaded, total_size)
                progress.update(downloaded - progress.n)

            urlretrieve(DATASET_URL, archive_path, reporthook=reporthook)

    print(f"Extracting {archive_path}...", flush=True)
    with tarfile.open(archive_path) as archive:
        archive.extractall(DATA_DIR, filter="data")


def extract_image_paths(metadata_path: Path, images_dir: Path) -> list[Path]:
    metadata = pl.read_csv(
        metadata_path,
        columns=["path", "width", "height"],
    )
    relative_paths = (
        metadata.filter((pl.col("width") >= 1000) & (pl.col("height") >= 1000))
        .get_column("path")
        .drop_nulls()
        .unique(maintain_order=True)
    )
    images_dir = images_dir.resolve()
    return [images_dir / path for path in relative_paths]


def main() -> None:
    download_and_extract()
    image_paths = extract_image_paths(
        DATA_DIR / "images/metadata/images.csv.gz",
        DATA_DIR / "images/small",
    )
    output_path = DATA_DIR / "image_paths.txt"
    with output_path.open("w", encoding="utf-8") as output:
        for image_path in image_paths:
            output.write(f"{image_path}\n")
    print(f"Saved {len(image_paths)} image paths to {output_path}.")


if __name__ == "__main__":
    main()
