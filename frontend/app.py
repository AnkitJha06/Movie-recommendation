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

            