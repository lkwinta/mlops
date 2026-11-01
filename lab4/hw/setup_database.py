from sqlalchemy.engine import URL
from sqlalchemy import create_engine
from models.img import Img
from models import Base

from sqlalchemy import delete
from sqlalchemy.orm import Session


def clear_images(engine):
    with Session(engine) as session:
        session.execute(delete(Img))
        session.commit()


def main():
    db_url = URL.create(
        drivername="postgresql+psycopg",
        username="postgres",
        password="password",
        host="localhost",
        port=5555,
        database="similarity_search_service_db",
    )

    engine = create_engine(db_url)

    clear_images(engine)
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    main()
