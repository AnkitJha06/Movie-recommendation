import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.title("Movie Recommendation ")
st.write("Enter a movie title")

movie_title = st.text_input(
    "Movie Title",
    placeholder="Eg:- Avengers"
)

no_of_recommendation = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("Give Recommendation", type="primary"):
    recom_title = movie_title.strip()

    if not recom_title:
        st.warning("Please enter movie title")
    else:
        try:
            response = requests.get(
                f"{API_URL}/recommend/title/{recom_title}",
                params={"n": no_of_recommendation},
                timeout=15
            )

            if response.status_code == 404:
                message = response.json().get(
                    "detail",
                    "Movie not found."
                )
                st.error(message)

            elif response.status_code != 200:
                st.error(
                    f"API error: {response.status_code}. "
                    "Check that the FastAPI server is running correctly."
                )


            else:
                data = response.json()
                selected_movie = data["selected_movie"]
                recommendations = data["recommendations"]

                st.subheader("Selected Movie")
                st.success(
                    f'{selected_movie["title"]} '
                    f'— {selected_movie["genres"].replace("|", ", ")}'
                )

                st.subheader("Recommended Movies")

                for rank, movie in enumerate(recommendations, start=1):
                    similarity_percent = movie["similarity"] * 100
                    genres = movie["genres"].replace("|", ", ")

                    st.markdown(
                        f"### {rank}. {movie['title']}"
                    )
                    st.write(f"**Genres:** {genres}")
                    st.progress(
                        float(movie["similarity"]),
                        text=f"Similarity: {similarity_percent:.1f}%"
                    )
                    st.divider()

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the FastAPI server. "
                "Start the backend first, then try again."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The API took too long to respond. Please try again."
            )

        except requests.exceptions.RequestException as error:
            st.error(f"Request failed: {error}")