
# Personalized Movie Recommendation System

A machine learning project that recommends movies based on a user's rating history using collaborative filtering and matrix factorization.

## Features

- Personalized top-N movie recommendations
- Collaborative filtering using SVD
- Movie similarity search using learned embeddings
- RMSE evaluation on a held-out test set
- Popularity-based candidate filtering
- Command-line interface for recommendations

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Scikit-surprise
- MovieLens dataset

## How It Works

The system learns patterns from user-movie ratings.

1. Load MovieLens ratings and movie metadata.
2. Split the ratings into training and testing data.
3. Train an SVD matrix factorization model.
4. Predict ratings for movies the user has not rated.
5. Return the highest-scoring movies.
6. Use learned movie embeddings to find similar movies.

### Architecture

```text
MovieLens Dataset
       |
       v
Data Preprocessing
       |
       v
SVD Matrix Factorization
       |
       +------------------+
       |                  |
       v                  v
Top-N Recommendations   Similar Movies
       |
       v
Evaluation (RMSE)
```

## Project Structure

```text
personalized-movie-recommender/
├── data/
│   ├── ratings.csv
│   └── movies.csv
├── models/
├── recommender_svd.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/personalized-movie-recommender.git
cd personalized-movie-recommender
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download MovieLens ml-latest-small and place ratings.csv and movies.csv inside the data folder.

## Usage

Run the recommender for a user:

```bash
python recommender_svd.py --user 15 --n 10
```

Find similar movies:

```bash
python recommender_svd.py --similar_movie 1 --n 10
```

Train with custom parameters:

```bash
python recommender_svd.py --user 15 --factors 100 --epochs 20
```

## Evaluation

The model is evaluated using Root Mean Squared Error (RMSE) on a held-out test set.

RMSE measures the difference between predicted and actual ratings. Lower values indicate smaller prediction errors.

## Future Improvements

- Build a Streamlit web interface
- Add movie posters and search
- Add hybrid recommendations using movie genres
- Improve cold-start recommendations for new users
- Compare SVD with other recommendation algorithms

## Author

Aryan Tiwari

GitHub: https://github.com/Aryan5859