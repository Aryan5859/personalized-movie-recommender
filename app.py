from pathlib import Path
import html
import io
import zipfile

import numpy as np
import pandas as pd
import requests
import streamlit as st
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

st.set_page_config(page_title="MovieMind V4 | Hybrid recommendations", page_icon="🎬", layout="wide")
DATA_DIR = Path(__file__).resolve().parent / "data" / "ml-latest-small"
POSTER_BASE = "https://image.tmdb.org/t/p/w342"

st.markdown("""
<style>
.stApp {background:#0b1020;color:#f8fafc}
.block-container {max-width:1380px;padding-top:1.5rem}
.hero {padding:12px 0 16px}
.hero h1 {font-size:clamp(2.5rem,5vw,3.8rem);margin:0}
.hero p {color:#a8b3c7;font-size:1.05rem}
.movie-card {background:#151c30;border:1px solid #29354c;border-radius:12px;
             padding:12px;margin:6px 0 20px;min-height:130px}
.movie-title {font-size:1rem;font-weight:700;min-height:2.8rem}
.movie-genres {color:#a8b3c7;font-size:.82rem;min-height:2.4rem}
.movie-score {color:#fbbf24;font-weight:700;margin-top:7px}
.poster-fallback {height:310px;max-width:210px;background:#1f2937;
                 border-radius:12px;display:flex;align-items:center;
                 justify-content:center;text-align:center;color:#a8b3c7}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Preparing MovieLens dataset...")
def ensure_dataset():
    required_files = ("movies.csv", "ratings.csv", "links.csv")
    if all((DATA_DIR / filename).is_file() for filename in required_files):
        return

    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    response = requests.get(url, timeout=90)
    response.raise_for_status()

    # Extract only the public CSV files needed by the application.
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        contents = {
            filename: archive.read(f"ml-latest-small/{filename}")
            for filename in required_files
        }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, content in contents.items():
        (DATA_DIR / filename).write_bytes(content)


@st.cache_data(show_spinner=False)
def load_data():
    ensure_dataset()
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    links = pd.read_csv(DATA_DIR / "links.csv")
    movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    return movies, ratings

@st.cache_resource(show_spinner="Training SVD model...")
def train_model(_ratings):
    reader = Reader(rating_scale=(0.5, 5.0))
    dataset = Dataset.load_from_df(_ratings[["userId", "movieId", "rating"]], reader)
    model = SVD(n_factors=100, n_epochs=20, random_state=42)
    model.fit(dataset.build_full_trainset())
    return model

@st.cache_data(ttl=7*24*3600, show_spinner=False)
def fetch_poster(tmdb_id, _api_key):
    if pd.isna(tmdb_id) or not _api_key:
        return None
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}",
            params={"api_key": _api_key, "language": "en-US"}, timeout=6)
        if response.status_code != 200:
            return None
        path = response.json().get("poster_path")
        return POSTER_BASE + path if path else None
    except (requests.RequestException, ValueError, TypeError):
        return None

@st.cache_data(show_spinner=False)
def recommendations(user_id, _model, _ratings, _movies):
    rated = set(_ratings.loc[_ratings.userId == user_id, "movieId"])
    candidates = _movies.loc[~_movies.movieId.isin(rated)].copy()
    candidates["predicted_rating"] = [
        _model.predict(int(user_id), int(mid)).est for mid in candidates.movieId
    ]
    return candidates.sort_values("predicted_rating", ascending=False)

@st.cache_data(show_spinner=False)
def similar_movies(movie_id, _model, _movies):
    trainset = _model.trainset
    try:
        target_idx = trainset.to_inner_iid(int(movie_id))
    except ValueError:
        return pd.DataFrame()
    vectors = _model.qi
    target = vectors[target_idx]
    denom = np.linalg.norm(vectors, axis=1) * np.linalg.norm(target)
    scores = (vectors @ target) / np.maximum(denom, 1e-12)
    matches = []
    for idx in np.argsort(scores)[::-1]:
        if idx != target_idx:
            matches.append((int(trainset.to_raw_iid(int(idx))), float(scores[idx])))
        if len(matches) == 12:
            break
    return pd.DataFrame(matches, columns=["movieId", "similarity"]).merge(
        _movies, on="movieId", how="left")

@st.cache_data(show_spinner=False)
def genre_vectors(_movies):
    tags = _movies.genres.fillna("").str.split("|")
    vocabulary = sorted({g for row in tags for g in row if g != "(no genres listed)"})
    matrix = np.array([[float(g in row) for g in vocabulary] for row in tags], dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)

@st.cache_data(show_spinner=False)
def hybrid_similar(movie_id, _model, _movies, genre_weight):
    trainset = _model.trainset
    try:
        target_idx = trainset.to_inner_iid(int(movie_id))
    except ValueError:
        return pd.DataFrame()
    latent = _model.qi
    target = latent[target_idx]
    svd_sim = (latent @ target) / np.maximum(
        np.linalg.norm(latent, axis=1) * np.linalg.norm(target), 1e-12)
    svd_sim = (svd_sim + 1.0) / 2.0
    vectors = genre_vectors(_movies)
    movie_to_row = {int(mid): i for i, mid in enumerate(_movies.movieId)}
    source_row = movie_to_row.get(int(movie_id))
    if source_row is None:
        return pd.DataFrame()
    raw_ids = [int(trainset.to_raw_iid(i)) for i in range(trainset.n_items)]
    candidate_rows = [movie_to_row.get(mid) for mid in raw_ids]
    genre_sim = np.array([float(vectors[r] @ vectors[source_row]) if r is not None else 0.0
                          for r in candidate_rows])
    combined = (1 - genre_weight) * svd_sim + genre_weight * genre_sim
    combined[target_idx] = -np.inf
    best = np.argsort(combined)[::-1][:12]
    return pd.DataFrame({"movieId": [raw_ids[i] for i in best],
                         "similarity": [float(combined[i]) for i in best]}).merge(
                             _movies, on="movieId", how="left")

@st.cache_data(show_spinner=False)
def hybrid_recommendations(user_id, _model, _ratings, _movies, genre_weight):
    ranked = recommendations(user_id, _model, _ratings, _movies).copy()
    history = _ratings[_ratings.userId == user_id].merge(
        _movies[["movieId", "genres"]], on="movieId", how="left")
    if history.empty:
        return ranked
    all_tags = sorted({g for gs in _movies.genres.fillna("") for g in gs.split("|")
                       if g != "(no genres listed)"})
    # Center ratings against the user's own mean to reflect genre preference.
    preferences = {g: 0.0 for g in all_tags}
    totals = {g: 0 for g in all_tags}
    avg = float(history.rating.mean())
    for rec in history.itertuples():
        for g in str(rec.genres).split("|"):
            if g in preferences:
                preferences[g] += float(rec.rating) - avg
                totals[g] += 1
    preferences = {g: preferences[g] / totals[g] if totals[g] else 0.0 for g in all_tags}
    def genre_bonus(gs):
        vals = [preferences[g] for g in str(gs).split("|") if g in preferences]
        return float(np.mean(vals)) if vals else 0.0
    ranked["genre_bonus"] = ranked.genres.map(genre_bonus)
    ranked["hybrid_score"] = ((1 - genre_weight) * ranked.predicted_rating
                               + genre_weight * np.clip(ranked.predicted_rating + ranked.genre_bonus, 0.5, 5))
    return ranked.sort_values("hybrid_score", ascending=False)

@st.cache_data(show_spinner="Evaluating SVD on held-out ratings...")
def evaluate(_ratings, threshold=4.0, k=10):
    reader = Reader(rating_scale=(0.5, 5.0))
    dataset = Dataset.load_from_df(_ratings[["userId", "movieId", "rating"]], reader)
    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)
    evaluation_model = SVD(n_factors=100, n_epochs=20, random_state=42)
    evaluation_model.fit(trainset)
    predictions = evaluation_model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    # Precision/recall among each user's held-out rated items (not all unseen movies).
    from collections import defaultdict
    by_user = defaultdict(list)
    for uid, iid, true_r, estimated_r, _ in predictions:
        by_user[uid].append((estimated_r, true_r))
    precision, recall = [], []
    for values in by_user.values():
        values.sort(key=lambda item: item[0], reverse=True)
        top = values[:k]
        relevant_in_top = sum(true >= threshold for _, true in top)
        total_relevant = sum(true >= threshold for _, true in values)
        if top:
            precision.append(relevant_in_top / len(top))
        if total_relevant:
            recall.append(relevant_in_top / total_relevant)
    return {"rmse": rmse, "mae": mae,
            "precision": float(np.mean(precision)) if precision else 0.0,
            "recall": float(np.mean(recall)) if recall else 0.0,
            "n_test": len(predictions), "n_train": trainset.n_ratings,
            "users_evaluated": len(by_user)}

def filter_genre(frame, genre):
    if genre == "All genres":
        return frame
    return frame.loc[frame.genres.fillna("").str.split("|").apply(lambda tags: genre in tags)]

def render_movies(frame, score_col=None, score_label="", api_key=""):
    if frame.empty:
        st.info("No movies match your selection.")
        return
    for start in range(0, len(frame), 3):
        columns = st.columns(3, gap="medium")
        for col, (_, movie) in zip(columns, frame.iloc[start:start+3].iterrows()):
            with col:
                poster = fetch_poster(movie.get("tmdbId"), api_key)
                if poster:
                    st.image(poster, width=210)
                else:
                    st.markdown('<div class="poster-fallback">🎬 No poster available</div>',
                                unsafe_allow_html=True)
                title = html.escape(str(movie.title))
                genres = html.escape(str(movie.genres).replace("|", " · "))
                score = ""
                if score_col is not None and pd.notna(movie.get(score_col)):
                    score = (f'<div class="movie-score">⭐ {html.escape(score_label)}: '
                             f'{float(movie[score_col]):.2f}</div>')
                st.markdown(
                    f'<div class="movie-card"><div class="movie-title">{title}</div>'
                    f'<div class="movie-genres">{genres}</div>{score}</div>',
                    unsafe_allow_html=True)

try:
    movies, ratings = load_data()
except (FileNotFoundError, KeyError, zipfile.BadZipFile, requests.RequestException, OSError) as exc:
    st.error(f"Could not load MovieLens data: {exc}")
    st.stop()

try:
    api_key = st.secrets.get("TMDB_API_KEY", "")
except (FileNotFoundError, KeyError):
    api_key = ""
if api_key == "YOUR_TMDB_API_KEY":
    api_key = ""
model = train_model(ratings)
all_genres = sorted({g for text in movies.genres.dropna() for g in text.split("|")
                     if g != "(no genres listed)"})
genres = ["All genres"] + all_genres

st.markdown('<div class="hero"><h1>🎬 MovieMind</h1>'
            '<p>Explore movies and get personalized picks powered by SVD.</p></div>',
            unsafe_allow_html=True)
if not api_key:
    st.warning("TMDB key missing. Movie recommendations still work, but posters will be unavailable.")

with st.sidebar:
    st.title("⚙️ Controls")
    users = sorted(int(x) for x in ratings.userId.unique())
    user_id = st.selectbox("MovieLens User", users, index=users.index(15) if 15 in users else 0)
    n_recs = st.slider("Recommendations to show", 3, 21, 9, step=3)
    st.caption("User IDs refer to anonymous MovieLens users, not app accounts.")
    genre_weight = st.slider("Genre weight in hybrid model", 0.0, 1.0, 0.35, 0.05)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 For You", "🔎 Similar Movies", "🍿 Browse Movies", "⭐ Rating History", "📊 About the Model", "📈 Evaluation"
])

with tab1:
    st.subheader(f"Top picks for User {user_id}")
    chosen_genre = st.selectbox("Filter recommendations by genre", genres, key="rec_genre")
    ranked = filter_genre(hybrid_recommendations(user_id, model, ratings, movies, genre_weight), chosen_genre)
    st.caption("Predicted ratings are on a 0.5–5.0 scale. Already rated movies are excluded.")
    render_movies(ranked.head(n_recs), "hybrid_score", "Hybrid score / 5", api_key)

with tab2:
    st.subheader("Discover similar movies")
    trained_ids = {int(model.trainset.to_raw_iid(i)) for i in range(model.trainset.n_items)}
    options = movies.loc[movies.movieId.isin(trained_ids)].sort_values("title")
    lookup = dict(zip(options.movieId, options.title))
    selected = st.selectbox("Search for a movie", options.movieId.tolist(),
                            format_func=lambda mid: lookup[mid])
    st.caption("Similarity blends SVD rating patterns with shared genres. Adjust genre weight in the sidebar.")
    render_movies(hybrid_similar(selected, model, movies, genre_weight), "similarity", "Hybrid similarity", api_key)

with tab3:
    st.subheader("Browse the MovieLens catalog")
    query = st.text_input("Search movie titles", placeholder="e.g. Batman, Toy Story, Matrix")
    left, right = st.columns(2)
    with left:
        browse_genre = st.selectbox("Genre", genres, key="browse_genre")
    with right:
        sort_by = st.selectbox("Sort movies", ["Title A–Z", "Most rated", "Newest first"])
    counts = ratings.groupby("movieId").size().rename("rating_count")
    catalog = movies.join(counts, on="movieId")
    catalog["rating_count"] = catalog.rating_count.fillna(0).astype(int)
    catalog = filter_genre(catalog, browse_genre)
    if query.strip():
        catalog = catalog.loc[catalog.title.str.contains(query.strip(), case=False, regex=False, na=False)]
    if sort_by == "Most rated":
        catalog = catalog.sort_values(["rating_count", "title"], ascending=[False, True])
    elif sort_by == "Newest first":
        catalog = catalog.assign(year=pd.to_numeric(catalog.title.str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce"))
        catalog = catalog.sort_values(["year", "title"], ascending=[False, True], na_position="last")
    else:
        catalog = catalog.sort_values("title")
    st.caption(f"{len(catalog):,} matching movies. Showing the first 12.")
    render_movies(catalog.head(12), api_key=api_key)

with tab4:
    st.subheader(f"Rating history for User {user_id}")
    history = ratings.loc[ratings.userId == user_id].merge(movies, on="movieId", how="left")
    a, b, c = st.columns(3)
    a.metric("Movies rated", f"{len(history):,}")
    b.metric("Average rating", f"{history.rating.mean():.2f} / 5")
    c.metric("Highest rating", f"{history.rating.max():.1f} / 5")
    st.write("#### Rating distribution")
    distribution = history.rating.value_counts().sort_index()
    st.bar_chart(distribution.rename_axis("Rating").rename("Number of ratings"))
    st.write("#### Previously rated movies")
    history_sort = st.selectbox("Order history by", ["Highest rated", "Lowest rated", "Most recent"])
    if history_sort == "Highest rated":
        history = history.sort_values(["rating", "timestamp"], ascending=[False, False])
    elif history_sort == "Lowest rated":
        history = history.sort_values(["rating", "timestamp"], ascending=[True, False])
    else:
        history = history.sort_values("timestamp", ascending=False)
    st.dataframe(history[["title", "genres", "rating"]].rename(columns={
        "title": "Movie", "genres": "Genres", "rating": "Your rating / 5"
    }), hide_index=True, use_container_width=True)

with tab5:
    st.subheader("How MovieMind works")
    a, b, c = st.columns(3)
    a.metric("Users", f"{ratings.userId.nunique():,}")
    b.metric("Movies", f"{movies.movieId.nunique():,}")
    c.metric("Ratings", f"{len(ratings):,}")
    st.markdown("""
**Algorithm:** Hybrid SVD collaborative filtering (Surprise) + genre preference  
**Latent factors:** 100  
**Training epochs:** 20  
**Rating scale:** 0.5–5.0  
**Random seed:** 42

The model learns user and movie factors from historical ratings to predict unseen ratings.
The similar-movies feature blends cosine similarity of learned movie vectors with genre overlap.
The For You ranking blends predicted ratings with genre preferences derived from each user’s rating history.

**Earlier CLI evaluation:** Test RMSE ≈ **0.8807** on a train/test split.
This web app trains a separate model on the full dataset, so this is **not** an
out-of-sample RMSE measurement of the currently fitted model.
""")

with tab6:
    st.subheader("Held-out model evaluation")
    st.caption("Fixed random 80/20 rating split (seed 42). Evaluation uses a separate SVD model, not the full-data app model.")
    threshold = st.selectbox("Relevant rating threshold", [3.5, 4.0, 4.5, 5.0], index=1)
    k = st.selectbox("Top K (held-out items per user)", [5, 10, 20], index=1)
    if st.button("Run evaluation", type="primary"):
        metrics = evaluate(ratings, threshold=threshold, k=k)
        a, b, c, d = st.columns(4)
        a.metric("RMSE", f"{metrics['rmse']:.4f}")
        b.metric("MAE", f"{metrics['mae']:.4f}")
        c.metric(f"Precision@{k}", f"{metrics['precision']:.3f}")
        d.metric(f"Recall@{k}", f"{metrics['recall']:.3f}")
        st.caption(f"Training ratings: {metrics['n_train']:,} · Test ratings: {metrics['n_test']:,} · Users: {metrics['users_evaluated']:,}")
        st.info("Precision and recall rank only each user's held-out rated movies. They are not full-catalog top-K evaluation or hybrid-model metrics. The hybrid weighting is a heuristic and has not been independently validated here.")
    else:
        st.info("Click Run evaluation to train and evaluate a separate SVD model.")

st.divider()
st.caption("MovieMind · MovieLens data · This product uses the TMDB API but is not endorsed or certified by TMDB.")
