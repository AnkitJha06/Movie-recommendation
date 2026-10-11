from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model_files"


@st.cache_resource
def load_model_files():
    movie_model = joblib.load(MODEL_DIR / "movie_model.joblib")
    movie_matrix = joblib.load(MODEL_DIR / "movie_matrix.joblib")
    movie_to_index = joblib.load(MODEL_DIR / "movie_to_index.joblib")
    movie_ids = joblib.load(MODEL_DIR / "movie_ids.joblib")
    movies_data = pd.read_csv(MODEL_DIR / "movies.csv")

    return (
        movie_model,
        movie_matrix,
        movie_to_index,
        movie_ids,
        movies_data
    )


movie_model, movie_matrix, movie_to_index, movie_ids, movies_data = (
    load_model_files()
)


def get_recommendations(movie_id: int, n: int = 5) -> pd.DataFrame:
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

    similar = similar[
        similar["movieId"] != movie_id
    ].head(n)

    return similar.merge(
        movies_data,
        on="movieId",
        how="left",
        validate="one_to_one"
    )[
        ["movieId", "title", "genres", "similarity"]
    ]


st.title("🎬 Movie Recommendation System")
st.write(
    "Enter a movie title to receive recommendations from the trained KNN model."
)

movie_title = st.text_input(
    "Movie Title",
    placeholder="For example: Iron Man"
)

no_of_recommendations = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("Give Recommendation", type="primary"):
    searched_title = movie_title.strip()

    if not searched_title:
        st.warning("Please enter a movie title.")

    else:
        matches = movies_data[
            movies_data["title"].str.contains(
                searched_title,
                case=False,
                na=False
            )
        ]

        if matches.empty:
            st.error(f"No movie found matching '{searched_title}'.")

        else:
            selected_movie = matches.iloc[0]
            recommendations = get_recommendations(
                int(selected_movie["movieId"]),
                no_of_recommendations
            )

            st.subheader("Selected Movie")
            st.success(
                f'{selected_movie["title"]} '
                f'— {selected_movie["genres"].replace("|", ", ")}'
            )

            st.subheader("Recommended Movies")

            for rank, movie in enumerate(recommendations.to_dict("records"), 1):
                similarity_percent = movie["similarity"] * 100
                genres = movie["genres"].replace("|", ", ")

                st.markdown(f"### {rank}. {movie['title']}")
                st.write(f"**Genres:** {genres}")
                st.progress(
                    float(movie["similarity"]),
                    text=f"Similarity: {similarity_percent:.1f}%"
                )
                st.divider()