import matplotlib.pyplot as plt
from PIL import Image
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from sqlalchemy import URL, create_engine, select

from models.img import Img


class ImageSearch:
    def __init__(self, engine, model):
        self.engine = engine
        self.model = model

    def __call__(self, image_description: str, k: int):
        found_images = self.find_similar_images(image_description, k)
        # display images
        self.display_images(found_images)

    def find_similar_images(self, image_description: str, k: int):
        image_embedding = self.model.encode([image_description])[0]

        # remember about session and commit
        query = (
            # write query to find K images with highest cosine distance
            select(Img)
            .order_by(Img.embedding.cosine_distance(image_embedding))
            .limit(k)
        )
        with Session(self.engine) as session:
            result = session.execute(query).scalars().all()
        return result

    def display_images(self, images):
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        if len(images) == 1:
            axes = [axes]

        for i, img in enumerate(images):
            img_path = img.image_path
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(f"Image {i + 1}")

        plt.show()


def main():
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

    image_search = ImageSearch(engine, model)

    while True:
        image_description = input("Enter image description: ")
        image_search(image_description, 5)


if __name__ == "__main__":
    main()
