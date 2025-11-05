import os
import argparse
import zipfile
import urllib.request
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from cf_model import CFModel


MOVIELENS_SMALL_URL = (
    "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
)
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_movielens_small(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    filename = MOVIELENS_SMALL_URL.split("/")[-1]
    zip_path = os.path.join(data_dir, filename)
    extracted_dir = os.path.join(data_dir, filename.split(".")[0])

    if os.path.isdir(extracted_dir) and os.path.isfile(
        os.path.join(extracted_dir, "ratings.csv")
    ):
        return extracted_dir

    if not os.path.isfile(zip_path):
        urllib.request.urlretrieve(MOVIELENS_SMALL_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    return extracted_dir


def load_and_prepare_small(
    data_dir: str, test_size: float, val_size: float, random_state: int
):
    ml_dir = download_movielens_small(data_dir)
    ratings_path = os.path.join(ml_dir, "ratings.csv")

    df = pd.read_csv(ratings_path)

    # Map userId and movieId to contiguous indices
    unique_users = df["userId"].unique()
    unique_items = df["movieId"].unique()

    user_id_to_index = {uid: idx for idx, uid in enumerate(sorted(unique_users))}
    item_id_to_index = {iid: idx for idx, iid in enumerate(sorted(unique_items))}

    df["user_idx"] = df["userId"].map(user_id_to_index)
    df["item_idx"] = df["movieId"].map(item_id_to_index)

    # Ratings are 0.5..5.0. Model outputs in [-1, 1] via tanh.
    # Normalize ratings to [-1, 1]: (r - 3.0) / 2.0
    df["score"] = (df["rating"] - 3.0) / 2.0

    # Train/Val/Test split (stratification by user optional; here random split)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=random_state
    )

    X_train_users = train_df["user_idx"].values.astype(np.int32)
    X_train_items = train_df["item_idx"].values.astype(np.int32)
    y_train = train_df["score"].values.astype(np.float32)

    X_val_users = val_df["user_idx"].values.astype(np.int32)
    X_val_items = val_df["item_idx"].values.astype(np.int32)
    y_val = val_df["score"].values.astype(np.float32)

    X_test_users = test_df["user_idx"].values.astype(np.int32)
    X_test_items = test_df["item_idx"].values.astype(np.int32)
    y_test = test_df["score"].values.astype(np.float32)

    n_users = len(user_id_to_index)
    n_items = len(item_id_to_index)

    return (
        X_train_users,
        X_train_items,
        y_train,
        X_val_users,
        X_val_items,
        y_val,
        X_test_users,
        X_test_items,
        y_test,
        n_users,
        n_items,
    )


def download_movielens_1m(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    filename = MOVIELENS_1M_URL.split("/")[-1]
    zip_path = os.path.join(data_dir, filename)
    extracted_dir = os.path.join(data_dir, "ml-1m")

    if os.path.isdir(extracted_dir) and os.path.isfile(
        os.path.join(extracted_dir, "ratings.dat")
    ):
        return extracted_dir

    if not os.path.isfile(zip_path):
        urllib.request.urlretrieve(MOVIELENS_1M_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    return extracted_dir


def load_and_prepare_1m(
    data_dir: str, test_size: float, val_size: float, random_state: int
):
    ml_dir = download_movielens_1m(data_dir)
    ratings_path = os.path.join(ml_dir, "ratings.dat")

    # ratings.dat format: UserID::MovieID::Rating::Timestamp
    df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
    )

    unique_users = df["userId"].unique()
    unique_items = df["movieId"].unique()

    user_id_to_index = {uid: idx for idx, uid in enumerate(sorted(unique_users))}
    item_id_to_index = {iid: idx for idx, iid in enumerate(sorted(unique_items))}

    df["user_idx"] = df["userId"].map(user_id_to_index)
    df["item_idx"] = df["movieId"].map(item_id_to_index)

    # Ratings are 1..5 in ML-1M. Normalize to [-1, 1]: (r - 3) / 2
    df["score"] = (df["rating"] - 3.0) / 2.0

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=random_state
    )

    X_train_users = train_df["user_idx"].values.astype(np.int32)
    X_train_items = train_df["item_idx"].values.astype(np.int32)
    y_train = train_df["score"].values.astype(np.float32)

    X_val_users = val_df["user_idx"].values.astype(np.int32)
    X_val_items = val_df["item_idx"].values.astype(np.int32)
    y_val = val_df["score"].values.astype(np.float32)

    X_test_users = test_df["user_idx"].values.astype(np.int32)
    X_test_items = test_df["item_idx"].values.astype(np.int32)
    y_test = test_df["score"].values.astype(np.float32)

    n_users = len(user_id_to_index)
    n_items = len(item_id_to_index)

    return (
        X_train_users,
        X_train_items,
        y_train,
        X_val_users,
        X_val_items,
        y_val,
        X_test_users,
        X_test_items,
        y_test,
        n_users,
        n_items,
    )


def denormalize_scores(normalized_scores: np.ndarray) -> np.ndarray:
    # Map from [-1, 1] back to rating scale 0.5..5.0: r = 2*x + 3
    return 2.0 * normalized_scores + 3.0


def evaluate_rmse_mae(
    y_true_norm: np.ndarray, y_pred_norm: np.ndarray
) -> Tuple[float, float]:
    # Compute metrics in original rating scale for interpretability
    y_true = denormalize_scores(y_true_norm)
    y_pred = denormalize_scores(y_pred_norm)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return rmse, mae


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CFModel on MovieLens (ml-latest-small)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store/download MovieLens",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ml-small",
        choices=["ml-small", "ml-1m"],
        help="Which MovieLens dataset to use",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=128, help="Embedding dimension"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--l2-reg", type=float, default=1e-6, help="L2 regularization")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--test-size", type=float, default=0.1, help="Test split size fraction"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation split size fraction (of train)",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--verbose", type=int, default=1, help="Keras verbosity (0,1,2)"
    )
    args = parser.parse_args()

    if args.dataset == "ml-1m":
        loader = load_and_prepare_1m
        dataset_name = "MovieLens 1M"
    else:
        loader = load_and_prepare_small
        dataset_name = "MovieLens Small"

    (
        X_train_users,
        X_train_items,
        y_train,
        X_val_users,
        X_val_items,
        y_val,
        X_test_users,
        X_test_items,
        y_test,
        n_users,
        n_items,
    ) = loader(
        data_dir=args.data_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    model = CFModel(
        n_users=n_users,
        n_products=n_items,
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        l2_reg=args.l2_reg,
        dropout_rate=args.dropout,
    )
    model.build_model()

    # Concatenate train and val for Keras validation_split to work as desired, or train on train and validate with explicit data.
    # We pass explicit validation via x=[..], y=.. and use validation_split=0 while emulating early stopping via callbacks set inside CFModel.

    model.train(
        user_ids=X_train_users,
        product_ids=X_train_items,
        scores=y_train,
        validation_data=([X_val_users, X_val_items], y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
        shuffle=True,
    )

    y_pred_test = model.predict(
        user_ids=X_test_users,
        product_ids=X_test_items,
        batch_size=args.batch_size,
    )

    rmse, mae = evaluate_rmse_mae(y_test, y_pred_test)

    print(f"\nEvaluation on {dataset_name}:")
    print(
        f"Users: {n_users}, Items: {n_items}, Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}"
    )
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")


if __name__ == "__main__":
    main()
