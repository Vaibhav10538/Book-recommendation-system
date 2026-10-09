#python -m streamlit run app_hybrid.py

import streamlit as st
import pandas as pd
import numpy as np
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

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

    # Normalize ratings to 0-1 for the hybrid score
    books["rating"] = pd.to_numeric(
        books["rating"], errors="coerce"
    ).fillna(0)

    rating_scaler = MinMaxScaler()
    books["rating_score"] = rating_scaler.fit_transform(
        books[["rating"]]
    ).ravel()

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
    <p>Hybrid Recommendation using TF-IDF + KNN + Rating</p>
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


# ---------- Hybrid recommendation function ----------
def recommend_books(book_title, n_recommendations=5):

    query = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        str(book_title).lower()
    )
    query = re.sub(r"\s+", " ", query).strip()

    matches = books[
        books["title_normalized"] == query
    ]

    if matches.empty:
        matches = books[
            books["title_normalized"].str.contains(
                query,
                na=False,
                regex=False
            )
        ]

    if matches.empty:
        return None, None

    book_index = matches.index[0]

    distances, indices = knn.kneighbors(
        tfidf_matrix[book_index],
        n_neighbors=n_recommendations + 1
    )

    recommendations = []

    for distance, index in zip(distances[0], indices[0]):

        if index == book_index:
            continue

        # Content similarity from cosine distance
        content_similarity = 1 - distance

        # Normalized rating
        rating_score = books.iloc[index]["rating_score"]

        # Hybrid score: 70% content + 30% rating
        final_score = (
            0.70 * content_similarity
            + 0.30 * rating_score
        )

        recommendations.append({
            "Rank": len(recommendations) + 1,
            "Book Title": books.iloc[index]["title"],
            "Author": books.iloc[index]["author"],
            "Genres": books.iloc[index]["genres_text"],
            "Content Similarity": f"{content_similarity * 100:.2f}%",
            "Rating": f"{books.iloc[index]['rating']:.2f}",
            "Final Score": final_score
        })

        if len(recommendations) == n_recommendations:
            break

    result = pd.DataFrame(recommendations)

    # Final ranking is based on the hybrid score
    result = result.sort_values(
        "Final Score",
        ascending=False
    ).reset_index(drop=True)

    result["Rank"] = range(1, len(result) + 1)

    result["Final Score"] = (
        result["Final Score"] * 100
    ).round(2).astype(str) + "%"

    return result, books.iloc[book_index]


# ---------- Recommendation ----------
if recommend:

    result, selected = recommend_books(
        selected_book,
        num_recommendations
    )

    if result is None:
        st.error("❌ Book not found! Please enter another book title.")
    else:
        st.markdown("---")
        st.markdown("## 📚 HYBRID BOOK RECOMMENDATION SYSTEM")

        st.markdown("### 🔎 Input Book:")
        st.write(f"**{selected['title']}**")

        st.markdown("### ✍️ Author:")
        st.write(f"**{selected['author']}**")

        st.markdown("---")
        st.markdown(f"### ⭐ TOP {num_recommendations} RECOMMENDED BOOKS")

        # Same columns and values as the Jupyter result DataFrame.
        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        st.markdown("### 💡 Recommendation Method:")
        st.write("TF-IDF + KNN + Cosine Similarity + Rating")
        st.caption("Final Score = 70% Content Similarity + 30% Rating")

        st.markdown("---")
        with st.expander("📖 Selected Book Details"):
            st.write(f"**Title:** {selected['title']}")
            st.write(f"**Author:** {selected['author']}")
            st.write(f"**Genres:** {selected['genres_text']}")
            description = str(selected["description"])
            if description:
                st.write(
                    f"**Description:** {description[:1000]}"
                    + ("..." if len(description) > 1000 else "")
                )


# ---------- About ----------
with st.expander("ℹ️ About this project"):
    st.write("""
    This is a Hybrid Book Recommendation System.

    The system combines book title, author, genres and description,
    converts the text into TF-IDF vectors, and uses K-Nearest Neighbors
    with cosine similarity to find similar books.

    Book ratings are normalized and combined with content similarity.
    The final score uses 70% content similarity and 30% rating.

    Technologies:
    - Python
    - Pandas
    - Scikit-learn
    - TF-IDF
    - KNN
    - Cosine Distance
    - Streamlit
    """)
