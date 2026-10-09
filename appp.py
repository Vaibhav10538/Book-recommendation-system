from flask import Flask, render_template, request, jsonify

import pandas as pd
import ast
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ============================================
# FLASK APPLICATION
# ============================================

app = Flask(__name__)


# ============================================
# LOAD DATASET
# ============================================

print("Loading dataset...")

books = pd.read_csv(
    "books_1.Best_Books_Ever.csv",
    low_memory=False
)

print("Dataset loaded successfully!")


# ============================================
# SELECT REQUIRED COLUMNS
# ============================================

required_columns = [
    "title",
    "author",
    "genres",
    "description",
    "rating",
    "publisher",
    "language"
]

books = books[required_columns].copy()


# ============================================
# HANDLE MISSING VALUES
# ============================================

text_columns = [
    "title",
    "author",
    "genres",
    "description",
    "publisher",
    "language"
]

for column in text_columns:

    books[column] = (
        books[column]
        .fillna("")
        .astype(str)
    )


# ============================================
# CLEAN TITLES
# ============================================

books["title"] = books["title"].str.strip()

books = books[
    books["title"] != ""
].copy()


# Remove duplicate books

books = books.drop_duplicates(
    subset="title"
).reset_index(drop=True)


# ============================================
# PROCESS GENRES
# ============================================

def parse_genres(value):

    try:

        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):

            return " ".join(
                str(x)
                for x in parsed
            )

        return str(value)

    except:

        return str(value)


books["genres_text"] = (
    books["genres"]
    .apply(parse_genres)
)


# ============================================
# TEXT CLEANING FUNCTION
# ============================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================
# COMBINE BOOK INFORMATION
# ============================================

books["combined_text"] = (

    books["title"].apply(clean_text)
    + " "
    + books["author"].apply(clean_text)
    + " "
    + books["genres_text"].apply(clean_text)
    + " "
    + books["description"].apply(clean_text)

)


# ============================================
# NORMALIZE TITLES
# ============================================

books["title_normalized"] = (

    books["title"]

    .str.lower()

    .str.replace(
        r"[^a-z0-9\s]",
        " ",
        regex=True
    )

    .str.replace(
        r"\s+",
        " ",
        regex=True
    )

    .str.strip()

)


# ============================================
# TF-IDF
# ============================================

print("Creating TF-IDF matrix...")

tfidf = TfidfVectorizer(

    stop_words="english",

    max_features=30000,

    ngram_range=(1, 2)

)


tfidf_matrix = tfidf.fit_transform(

    books["combined_text"]

)


print("TF-IDF completed!")


# ============================================
# KNN MODEL
# ============================================

print("Training KNN model...")


knn = NearestNeighbors(

    n_neighbors=6,

    metric="cosine",

    algorithm="brute"

)


knn.fit(tfidf_matrix)


print("KNN model ready!")


# ============================================
# RECOMMENDATION FUNCTION
# ============================================

def recommend_books(
    book_title,
    n_recommendations=5
):

    # -----------------------------
    # Normalize user input
    # -----------------------------

    query = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        str(book_title).lower()
    )

    query = re.sub(
        r"\s+",
        " ",
        query
    ).strip()


    # -----------------------------
    # Exact title match
    # -----------------------------

    matches = books[
        books["title_normalized"] == query
    ]


    # -----------------------------
    # Partial match
    # -----------------------------

    if matches.empty:

        matches = books[
            books["title_normalized"]
            .str.contains(
                query,
                na=False,
                regex=False
            )
        ]


    # -----------------------------
    # Book not found
    # -----------------------------

    if matches.empty:

        return None


    # -----------------------------
    # Get book index
    # -----------------------------

    book_index = matches.index[0]


    # -----------------------------
    # Find nearest books
    # -----------------------------

    distances, indices = knn.kneighbors(

        tfidf_matrix[book_index],

        n_neighbors=n_recommendations + 1

    )


    recommendations = []


    # -----------------------------
    # Generate recommendations
    # -----------------------------

    for distance, index in zip(

        distances[0],

        indices[0]

    ):

        # Skip input book

        if index == book_index:

            continue


        # Cosine distance
        # Convert to similarity

        similarity = 1 - distance


        recommendations.append({

            "Rank":
                len(recommendations) + 1,

            "Book Title":
                books.iloc[index]["title"],

            "Author":
                books.iloc[index]["author"],

            "Genres":
                books.iloc[index]["genres_text"],

            "Similarity":
                f"{similarity * 100:.2f}%"

        })


        if len(recommendations) == n_recommendations:

            break


    # ========================================
    # SELECTED BOOK INFORMATION
    # ========================================

    selected_book = books.iloc[book_index]


    return {

        "selected": {

            "title":
                selected_book["title"],

            "author":
                selected_book["author"],

            "genres":
                selected_book["genres_text"],

            "description":
                selected_book["description"]

        },

        "recommendations":
            recommendations

    }


# ============================================
# HOME PAGE
# ============================================

@app.route("/")
def home():

    book_titles = (
        books["title"]
        .tolist()
    )

    return render_template(

        "index.html",

        books=book_titles

    )


# ============================================
# RECOMMENDATION API
# ============================================

@app.route(
    "/recommend",
    methods=["POST"]
)
def recommend():

    # Get JSON data
    data = request.get_json()


    # Get book title

    book_title = data.get(
        "book_title",
        ""
    )


    # Get number of recommendations

    count = int(
        data.get(
            "count",
            5
        )
    )


    # Keep between 3 and 5

    count = max(
        3,
        min(count, 5)
    )


    # Generate recommendations

    result = recommend_books(

        book_title,

        count

    )


    # Book not found

    if result is None:

        return jsonify({

            "success": False,

            "message":
                "Book not found! Please enter another book title."

        }), 404


    # Send result to frontend

    return jsonify({

        "success": True,

        "selected":
            result["selected"],

        "recommendations":
            result["recommendations"]

    })


# ============================================
# RUN APPLICATION
# ============================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("📚 BOOK RECOMMENDATION SYSTEM")
    print("=" * 60)
    print("Server running at:")
    print("http://127.0.0.1:5000")
    print("=" * 60)
    print()

    app.run(
        debug=True
    )