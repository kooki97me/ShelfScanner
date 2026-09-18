import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "processed" / "books_cleaned.csv"
MODEL_DIR = BASE_DIR / "model"

TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
MATRIX_FILE = MODEL_DIR / "tfidf_matrix.joblib"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading book dataset...")

df = pd.read_csv(DATA_FILE)

df = df.fillna("")

print(f"Books loaded: {len(df)}")


# ============================================================
# TF-IDF MODEL
# ============================================================

if TFIDF_FILE.exists() and MATRIX_FILE.exists():

    print("Loading saved TF-IDF model...")

    vectorizer = joblib.load(TFIDF_FILE)
    tfidf_matrix = joblib.load(MATRIX_FILE)

else:

    print("Creating TF-IDF model...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2
    )

    tfidf_matrix = vectorizer.fit_transform(
        df["combined_text"].astype(str)
    )

    joblib.dump(vectorizer, TFIDF_FILE)
    joblib.dump(tfidf_matrix, MATRIX_FILE)

    print("TF-IDF model saved.")


print("Recommendation model ready!")


# ============================================================
# GENRE HELPERS
# ============================================================

def get_genres(value):
    """
    Convert genre text into a clean set.
    """

    if not value:
        return set()

    text = str(value)

    text = re.sub(r"[\[\]'\"{}()]", "", text)

    parts = re.split(r"[,;|]", text)

    genres = set()

    for part in parts:

        part = part.strip().lower()

        if part:
            genres.add(part)

    return genres


def genre_similarity(g1, g2):

    if not g1 or not g2:
        return 0.0

    intersection = len(g1.intersection(g2))
    union = len(g1.union(g2))

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# TITLE NORMALIZATION
# ============================================================

def normalize_title(title):

    if not title:
        return ""

    title = str(title).lower()

    title = re.sub(
        r"[^a-z0-9\s]",
        " ",
        title
    )

    title = re.sub(
        r"\s+",
        " ",
        title
    )

    return title.strip()


# ============================================================
# BOOK RECOMMENDATION
# ============================================================

def recommend_books(
    book_title,
    top_n=5,
    min_rating=0,
    genre_filter=None
):

    if not book_title:
        return pd.DataFrame()

    query = normalize_title(book_title)

    if not query:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Find matching book
    # --------------------------------------------------------

    normalized_titles = df["title"].apply(normalize_title)

    exact_matches = normalized_titles == query

    if exact_matches.any():

        book_index = df[exact_matches].index[0]

    else:

        contains_matches = normalized_titles.str.contains(
            query,
            regex=False,
            na=False
        )

        if not contains_matches.any():

            # Try partial words
            words = query.split()

            if not words:
                return pd.DataFrame()

            mask = normalized_titles.apply(
                lambda x: all(word in x for word in words)
            )

            if not mask.any():
                return pd.DataFrame()

            book_index = df[mask].index[0]

        else:

            book_index = df[contains_matches].index[0]

    # --------------------------------------------------------
    # TF-IDF similarity
    # --------------------------------------------------------

    position = df.index.get_loc(book_index)

    similarity_scores = cosine_similarity(
        tfidf_matrix[position],
        tfidf_matrix
    ).flatten()

    # --------------------------------------------------------
    # Candidate dataframe
    # --------------------------------------------------------

    results = df.copy()

    results["semantic_score"] = similarity_scores

    # Remove selected book
    results = results[
        results.index != book_index
    ].copy()

    # --------------------------------------------------------
    # Genre similarity
    # --------------------------------------------------------

    source_genres = get_genres(
        df.loc[book_index, "genres"]
    )

    results["genre_score"] = results["genres"].apply(
        lambda x: genre_similarity(
            source_genres,
            get_genres(x)
        )
    )

    # --------------------------------------------------------
    # Rating score
    # --------------------------------------------------------

    results["rating_score"] = pd.to_numeric(
        results["avg_rating"],
        errors="coerce"
    ).fillna(0)

    results["rating_score"] = (
        results["rating_score"] / 5
    )

    # --------------------------------------------------------
    # Final hybrid score
    # --------------------------------------------------------

    results["score"] = (
        results["semantic_score"] * 0.65
        + results["genre_score"] * 0.25
        + results["rating_score"] * 0.10
    )

    # --------------------------------------------------------
    # Rating filter
    # --------------------------------------------------------

    if min_rating and min_rating > 0:

        results = results[
            results["avg_rating"] >= min_rating
        ]

    # --------------------------------------------------------
    # Genre filter
    # --------------------------------------------------------

    if genre_filter:

        filter_value = str(
            genre_filter
        ).strip().lower()

        results = results[
            results["genres"].str.lower().str.contains(
                filter_value,
                regex=False,
                na=False
            )
        ]

    # --------------------------------------------------------
    # Remove duplicate titles
    # --------------------------------------------------------

    results = results.drop_duplicates(
        subset=["title", "author"]
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results = results.sort_values(
        "score",
        ascending=False
    )

    return results.head(top_n)


# ============================================================
# SMART SEARCH
# ============================================================

def smart_search(
    query,
    top_n=10,
    min_rating=0,
    genre_filter=None
):

    if not query:
        return pd.DataFrame()

    query = str(query).strip()

    if not query:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Convert natural-language query into TF-IDF vector
    # --------------------------------------------------------

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    results = df.copy()

    results["search_score"] = scores

    # --------------------------------------------------------
    # Rating filter
    # --------------------------------------------------------

    if min_rating and min_rating > 0:

        results = results[
            pd.to_numeric(
                results["avg_rating"],
                errors="coerce"
            ).fillna(0) >= min_rating
        ]

    # --------------------------------------------------------
    # Genre filter
    # --------------------------------------------------------

    if genre_filter:

        filter_value = str(
            genre_filter
        ).strip().lower()

        results = results[
            results["genres"].str.lower().str.contains(
                filter_value,
                regex=False,
                na=False
            )
        ]

    # --------------------------------------------------------
    # Remove duplicate books
    # --------------------------------------------------------

    results = results.drop_duplicates(
        subset=["title", "author"]
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results = results.sort_values(
        "search_score",
        ascending=False
    )

    return results.head(top_n)


# ============================================================
# SEARCH BY TITLE
# ============================================================

def search_books(query, top_n=10):

    if not query:
        return pd.DataFrame()

    query = str(query).strip().lower()

    if not query:
        return pd.DataFrame()

    title_mask = df["title"].str.lower().str.contains(
        query,
        regex=False,
        na=False
    )

    author_mask = df["author"].str.lower().str.contains(
        query,
        regex=False,
        na=False
    )

    results = df[
        title_mask | author_mask
    ].copy()

    results = results.drop_duplicates(
        subset=["title", "author"]
    )

    return results.head(top_n)


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("SHELFSCANNER - CONTENT-BASED AI RECOMMENDER")
    print("=" * 70)

    test_book = "Harry Potter and the Half-Blood Prince"

    print(
        f"\nRecommendations for: {test_book}\n"
    )

    recommendations = recommend_books(
        test_book,
        top_n=5
    )

    for i, (_, book) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        print(
            f"{i}. {book['title']} "
            f"| Score: {book['score']:.3f}"
        )

    print("\n" + "=" * 70)
    print("SMART SEARCH TEST")
    print("=" * 70)

    search_query = "fantasy magic adventure"

    print(
        f"\nSearch query: {search_query}\n"
    )

    search_results = smart_search(
        search_query,
        top_n=5
    )

    for i, (_, book) in enumerate(
        search_results.iterrows(),
        start=1
    ):

        print(
            f"{i}. {book['title']} "
            f"| Score: {book['search_score']:.3f}"
        )

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)