from itertools import batched, islice

import joblib
from sqlalchemy.orm import Session
from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from models.img import Img

MAX_IMAGES = 100000  # Adjust it for your needs
BATCH_SIZE = joblib.cpu_count(only_physical_cores=True)
IMAGE_PATHS_FILE = "./data/image_paths.txt"


def insert_images(engine: Engine, images: list[Img]):
    with Session(engine) as session:
        for image in images:
            session.add(image)

        session.commit()


def vectorize_images(engine, model, image_paths):
    with tqdm(total=MAX_IMAGES) as pbar:
        for images_paths_batch in batched(islice(image_paths, MAX_IMAGES), BATCH_SIZE):
            images = [Image.open(path) for path in images_paths_batch]

            # calculate embeddings
            embeddings = model.encode(images)

            # create Img instances for all images in batch
            imgs = [
                Img(image_path=path, embedding=embedding)
                for path, embedding in zip(images_paths_batch, embeddings)
            ]

            # insert all batch images
            insert_images(engine, imgs)

            # update pbar
            pbar.update(len(imgs))


def main():
    with open(IMAGE_PATHS_FILE, "r") as f:
        image_paths = [line.strip() for line in f]

    model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
    db_url = URL.create(
        drivername="postgresql+psycopg",
        username="postgres",
        password="password",
        host="localhost",
        port=5555,
        database="similarity_search_service_db",
    )

    engine = create_engine(db_url)

    vectorize_images(engine, model, image_paths)


if __name__ == "__main__":
    main()
