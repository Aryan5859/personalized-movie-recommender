import os
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy


DATA_DIR = "data"
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")


def load_movielens(ratings_path=RATINGS_PATH, movies_path=MOVIES_PATH):
    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        raise FileNotFoundError(
            "Missing data files.\n"
            "Expected:\n"
            f" - {ratings_path}\n"
            f" - {movies_path}\n"
            "Download MovieLens 'ml-latest-small' and place ratings.csv/movies.csv in ./data/"
        )

    ratings = pd.read_csv(ratings_path)  # userId,movieId,rating,timestamp
    movies = pd.read_csv(movies_path)    # movieId,title,genres

    # Basic cleanup
    ratings = ratings[["userId", "movieId", "rating"]].copy()
    movies = movies[["movieId", "title", "genres"]].copy()

    return ratings, movies


def train_svd(ratings_df, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, test_size=0.2, random_state=42):
    reader = Reader(rating_scale=(ratings_df.rating.min(), ratings_df.rating.max()))
    data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)

    trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)

    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=random_state
    )
    model.fit(trainset)

    preds = model.test(testset)
    rmse = accuracy.rmse(preds, verbose=False)

    return model, rmse, trainset


def get_user_seen_movies(ratings_df, user_id):
    return set(ratings_df.loc[ratings_df["userId"] == user_id, "movieId"].unique())


def recommend_top_n(model, trainset, ratings_df, movies_df, user_id, n=10, min_ratings=50):
    """
    Recommend top-N movies for a given user by predicting ratings for unseen movies.

    min_ratings: filters out very obscure movies (cold-start-ish) to make recs better.
    """
    seen = get_user_seen_movies(ratings_df, user_id)

    # Popularity filter: keep movies with enough ratings
    counts = ratings_df.groupby("movieId")["rating"].count()
    candidate_movies = counts[counts >= min_ratings].index.values

    # Unseen candidates
    candidates = [mid for mid in candidate_movies if mid not in seen]

    # If the user is new or has seen almost everything in candidates, fallback
    if len(candidates) == 0:
        # fallback to top-rated popular
        popular = (
            ratings_df.groupby("movieId")
            .agg(cnt=("rating", "count"), avg=("rating", "mean"))
            .query("cnt >= @min_ratings")
            .sort_values(["avg", "cnt"], ascending=False)
            .head(n)
            .reset_index()
        )
        out = popular.merge(movies_df, on="movieId", how="left")[["movieId", "title", "genres", "avg", "cnt"]]
        out.rename(columns={"avg": "score", "cnt": "num_ratings"}, inplace=True)
        return out

    # Predict ratings for candidates
    scored = []
    for mid in candidates:
        est = model.predict(user_id, mid).est
        scored.append((mid, est))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:n]

    rec_df = pd.DataFrame(top, columns=["movieId", "score"])
    rec_df = rec_df.merge(movies_df, on="movieId", how="left")

    # add popularity info
    rec_df["num_ratings"] = rec_df["movieId"].map(counts).fillna(0).astype(int)
    return rec_df[["movieId", "title", "genres", "score", "num_ratings"]]


def build_item_embeddings(model, trainset):
    """
    Surprise SVD stores item factors in model.qi aligned to trainset item indices.
    We'll build:
      - movieId list aligned to qi rows
      - embeddings matrix
    """
    # trainset.to_raw_iid(inner_iid) gives raw movieId
    raw_movie_ids = [int(trainset.to_raw_iid(i)) for i in range(trainset.n_items)]
    embeddings = model.qi  # shape: (n_items, n_factors)
    return raw_movie_ids, embeddings


def similar_movies(model, trainset, movies_df, movie_id, top_k=10):
    raw_ids, emb = build_item_embeddings(model, trainset)

    if movie_id not in set(raw_ids):
        raise ValueError("movie_id not found in training set (maybe too few ratings). Try another movieId.")

    idx = raw_ids.index(movie_id)
    target_vec = emb[idx:idx+1]

    sims = cosine_similarity(target_vec, emb).ravel()
    # exclude itself
    sims[idx] = -1.0

    top_idx = np.argsort(-sims)[:top_k]
    results = []
    for j in top_idx:
        mid = raw_ids[j]
        results.append((mid, float(sims[j])))

    df = pd.DataFrame(results, columns=["movieId", "similarity"]).merge(movies_df, on="movieId", how="left")
    return df[["movieId", "title", "genres", "similarity"]]


def main():
    parser = argparse.ArgumentParser(description="Personalized Movie Recommendation using SVD (MovieLens).")
    parser.add_argument("--user", type=int, default=1, help="User ID to recommend for")
    parser.add_argument("--n", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--min_ratings", type=int, default=50, help="Filter movies with fewer ratings")
    parser.add_argument("--similar_movie", type=int, default=None, help="MovieId to find similar movies")
    parser.add_argument("--factors", type=int, default=100, help="SVD latent factors")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    args = parser.parse_args()

    ratings, movies = load_movielens()

    print(f"[INFO] Loaded ratings: {ratings.shape}, movies: {movies.shape}")
    print("[INFO] Training SVD model...")
    model, rmse, trainset = train_svd(
        ratings,
        n_factors=args.factors,
        n_epochs=args.epochs
    )
    print(f"[RESULT] Test RMSE: {rmse:.4f}")

    print("\n=== Top Recommendations ===")
    recs = recommend_top_n(
        model=model,
        trainset=trainset,
        ratings_df=ratings,
        movies_df=movies,
        user_id=args.user,
        n=args.n,
        min_ratings=args.min_ratings
    )
    print(recs.to_string(index=False))

    if args.similar_movie is not None:
        print("\n=== Similar Movies ===")
        sims = similar_movies(model, trainset, movies, args.similar_movie, top_k=args.n)
        print(sims.to_string(index=False))


if __name__ == "__main__":
    main()