import os
import csv
import time
import zipfile
import logging
import tempfile
import urllib.request
from typing import List, Dict

from service import Service
from embeddings import EmbeddingService
from milvus import MilvusClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("movielens_test_train")

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR_NAME = "ml-latest-small"


def download_movielens(dest_dir: str) -> str:
    zip_path = os.path.join(dest_dir, "ml-latest-small.zip")
    if not os.path.exists(zip_path):
        logger.info("Downloading MovieLens (ml-latest-small)...")
        urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
    extract_dir = os.path.join(dest_dir, DATA_DIR_NAME)
    if not os.path.exists(extract_dir):
        logger.info("Extracting MovieLens zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
    return extract_dir


def load_movies(movies_csv_path: str) -> Dict[int, Dict]:
    movies: Dict[int, Dict] = {}
    with open(movies_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                movie_id = int(row["movieId"])  # movieId
            except Exception:
                continue
            title = row.get("title", "")
            movies[movie_id] = {
                "id": movie_id,
                "name": title,
                "description": "",
            }
    return movies


def load_ratings(ratings_csv_path: str, max_rows: int | None = None) -> List[Dict]:
    interactions: List[Dict] = []
    with open(ratings_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            try:
                user_id = int(row["userId"])  # userId
                item_id = int(row["movieId"])  # movieId
                rating = float(row["rating"])  # rating
                ts = row.get("timestamp", "")
            except Exception:
                continue
            interactions.append({
                "user_id": user_id,
                "item_id": item_id,
                "rating": rating,
                "timestamp": ts,
            })
            if max_rows is not None and idx + 1 >= max_rows:
                break
    return interactions


def upsert_content_products(client: MilvusClient, movies: Dict[int, Dict]):
    embedding = EmbeddingService()
    ids: List[int] = []
    names: List[str] = []
    sparse_list = []
    dense_list = []

    texts: List[str] = []
    movie_ids: List[int] = []
    for movie_id, movie in movies.items():
        name = movie.get("name", "")
        desc = movie.get("description", "")
        text = f"{name} {desc}".strip() or "movie"
        movie_ids.append(movie_id)
        texts.append(text)

    logger.info(f"Encoding {len(texts)} movie titles for content embeddings...")


    # MGTE returns sparse as a csr_array per row; Milvus AnnSearchRequest expects single sparse per row
    # We will pass as-is column-wise via upsert
    
    ids = movie_ids
    names = [movies[mid]["name"] for mid in movie_ids]
    descriptions = ["" for _ in movie_ids]
    brands = ["" for _ in movie_ids]
    categories = ["" for _ in movie_ids]
    is_actives = [True for _ in movie_ids]
    ratings = [0.0 for _ in movie_ids]
    skus = [[] for _ in movie_ids]
    specifications = [{} for _ in movie_ids]

    encoded_texts = embedding.embed_texts(texts)
    sparse_list = [enc["sparse"] for enc in encoded_texts]
    dense_list = [enc["dense"] for enc in encoded_texts]

    entities = [
        ids,
        names,
        descriptions,
        brands,
        categories,
        is_actives,
        ratings,
        skus,
        specifications,
        sparse_list,
        dense_list,
    ]
    logger.info("Upserting content_products (this may take a while)...")
    client.upsert_content_products(entities)
    client.content_products_collection.flush()
    logger.info("Upserted content_products")


def main():
    start = time.time()
    data_dir = download_movielens("./tmp")
    movies_csv = os.path.join(data_dir, "movies.csv")
    ratings_csv = os.path.join(data_dir, "ratings.csv")

    movies = load_movies(movies_csv)
    interactions = load_ratings(ratings_csv)
    logger.info(f"Loaded movies: {len(movies)}, interactions: {len(interactions)}")

    service = Service(milvus_host="localhost", milvus_port=19530)

    # Ensure content_products is populated for the items that appear in interactions
    movie_ids_in_data = set(inter["item_id"] for inter in interactions)
    movies_subset = {mid: movies[mid] for mid in movie_ids_in_data if mid in movies}
    upsert_content_products(service.client, movies_subset)

    # Ingest interactions and train
    logger.info("Ingesting interactions into service training buffer...")
    service.ingest_training_data(interactions)

    logger.info("Training CF model...")
    history = service.train_cf_model(epochs=20, batch_size=512)
    logger.info(f"Training done. Keys: {list(history.keys()) if isinstance(history, dict) else 'n/a'}")

    # Try recommendations for a sample user
    sample_user = interactions[0]["user_id"] if interactions else 1
    recs = service.recommend(sample_user, limit=5)
    logger.info(f"Sample recommendations for user {sample_user}: {recs}")

    logger.info(f"Completed in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
