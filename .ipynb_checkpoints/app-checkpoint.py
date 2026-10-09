import streamlit as st
import pandas as pd
import numpy as np
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# ---------- Styling ----------
st.markdown("""
<style>
.main {
    background-color: #f7f8fc;
}
.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(135deg, #1f2937, #374151);
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    margin-bottom: 8px;
}
.card {
    padding: 18px;
    border-radius: 14px;
    background: white;
    border: 1px solid #e5e7eb;
    margin-bottom: 14px;
}
.similarity {
    font-weight: 700;
}
.small {
    color: #6b7280;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# ---------- Load and prepare model ----------
@st.cache_resource(show_spinner="Preparing recommendation model...")
def load_model():

    file_path = "books_1.Best_Books_Ever.csv"
    books = pd.read_csv(file_path, low_memory=False)

    required = [
        "title", "author", "genres",
        "description", "rating",
        "publisher", "language"
    ]

    books = books[required].copy()

    text_columns = [
        "title", "author", "genres",
        "description", "publisher", "language"
    ]

    for col in text_columns:
        books[col] = books[col].fillna("").astype(str)

    books["title"] = books["title"].str.strip()
    books = books[books["title"] != ""].copy()
    books = books.drop_duplicates(
        subset="title"
    ).reset_index(drop=True)

    def parse_genres(value):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return " ".join(str(x) for x in parsed)
            return str(value)
        except:
            return str(value)

    books["genres_text"] = books["genres"].apply(parse_genres)

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    books["combined_text"] = (
        books["title"].apply(clean_text) + " " +
        books["author"].apply(clean_text) + " " +
        books["genres_text"].apply(clean_text) + " " +
        books["description"].apply(clean_text)
    )

    books["title_normalized"] = (
        books["title"]
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = tfidf.fit_transform(books["combined_text"])

    knn = NearestNeighbors(
        n_neighbors=6,
        metric="cosine",
        algorithm="brute"
    )

    knn.fit(tfidf_matrix)

    return books, tfidf_matrix, knn


books, tfidf_matrix, knn = load_model()


# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>📚 Book Recommendation System</h1>
    <p>Content-Based Recommendation using TF-IDF + KNN</p>
</div>
""", unsafe_allow_html=True)


# ---------- Search ----------
st.subheader("🔎 Find Similar Books")

book_titles = books["title"].tolist()

selected_book = st.selectbox(
    "Select a book",
    book_titles,
    index=book_titles.index("The Hunger Games")
    if "The Hunger Games" in book_titles else 0
)

num_recommendations = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)

recommend = st.button(
    "✨ Recommend Books",
    type="primary",
    use_container_width=True
)


# ---------- Recommendation ----------
if recommend:

    book_index = books[
        books["title"] == selected_book
    ].index[0]

    distances, indices = knn.kneighbors(
        tfidf_matrix[book_index],
        n_neighbors=num_recommendations + 1
    )

    st.markdown("---")

    selected = books.iloc[book_index]

    st.subheader("📖 Selected Book")

    st.markdown(f"""
    <div class="card">
        <h3>{selected['title']}</h3>
        <p><b>Author:</b> {selected['author']}</p>
        <p><b>Genres:</b> {selected['genres_text']}</p>
        <p class="small">{selected['description'][:500]}...</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader(f"⭐ Top {num_recommendations} Similar Books")

    rank = 1

    for distance, index in zip(distances[0], indices[0]):

        if index == book_index:
            continue

        book = books.iloc[index]
        similarity = (1 - distance) * 100

        st.markdown(f"""
        <div class="card">
            <h3>#{rank} {book['title']}</h3>
            <p><b>Author:</b> {book['author']}</p>
            <p><b>Genres:</b> {book['genres_text']}</p>
            <p class="similarity">
                Similarity: {similarity:.2f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

        rank += 1

        if rank > num_recommendations:
            break


# ---------- About ----------
with st.expander("ℹ️ About this project"):
    st.write("""
    This is a Content-Based Book Recommendation System.

    The system combines book title, author, genres and description,
    converts the text into TF-IDF vectors, and uses K-Nearest Neighbors
    with cosine distance to find similar books.

    Technologies:
    - Python
    - Pandas
    - Scikit-learn
    - TF-IDF
    - KNN
    - Cosine Distance
    - Streamlit
    """)
