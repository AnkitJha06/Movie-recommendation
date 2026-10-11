from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="Movie Recommendation API",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model_files"

movie_model = joblib.load(MODEL_DIR / "movie_model.joblib")
movie_matrix = joblib.load(MODEL_DIR / "movie_matrix.joblib")
movie_to_index = joblib.load(MODEL_DIR / "movie_to_index.joblib")
movie_ids = joblib.load(MODEL_DIR / "movie_ids.joblib")

movies_data = pd.read_csv(MODEL_DIR / "movies.csv")


def search_similar_movies(movie_id: int, n: int = 10) -> pd.DataFrame:
    if movie_id not in movie_to_index:
        return pd.DataFrame(
            columns=["movieId", "title", "genres", "similarity"]
        )

    movie_index = movie_to_index[movie_id]

    distances, indices = movie_model.kneighbors(
        movie_matrix[movie_index],
        n_neighbors=min(n + 1, len(movie_ids))
    )

    similar = pd.DataFrame({
        "movieId": [movie_ids[index] for index in indices[0]],
        "similarity": 1 - distances[0]
    })

    similar = similar[similar["movieId"] != movie_id].head(n)

    return similar.merge(
        movies_data,
        on="movieId",
        how="left",
        validate="one_to_one"
    )[
        ["movieId", "title", "genres", "similarity"]
    ]


@app.get("/health")
def health_check():
    return {
        "health": "OK",
        "model_loaded": True,
        "movies_available": len(movie_ids)
    }


@app.get("/recommend/id/{movie_id}")
def recommend_by_id(movie_id: int, n: int = 10):
    recommendations = search_similar_movies(movie_id, n)

    if recommendations.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Movie ID {movie_id} was not found."
        )

    return {
        "movie_id": movie_id,
        "recommendations": recommendations.to_dict(orient="records")
    }

@app.get("/movies")
def get_movies(limit: int = 10):
    return movies_data[
        ["movieId", "title", "genres"]
    ].head(limit).to_dict(orient="records")


@app.get("/recommend/title/{movie_title}")
def recommend_by_title(movie_title: str, n: int = 10):
    matches = movies_data[
        movies_data["title"].str.contains(
            movie_title,
            case=False,
            na=False
        )
    ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No movie found matching '{movie_title}'."
        )

    selected_movie = matches.iloc[0]

    recommendations = search_similar_movies(
        int(selected_movie["movieId"]),
        n
    )

    return {
        "selected_movie": {
            "movieId": int(selected_movie["movieId"]),
            "title": selected_movie["title"],
            "genres": selected_movie["genres"]
        },
        "recommendations": recommendations.to_dict(orient="records")
    }