import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json
import re


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ShelfScanner",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #F5F2E8;
        color: #293028;
    }

    [data-testid="stSidebar"] {
        background: #E7ECD9;
    }

    [data-testid="stSidebar"] * {
        color: #293028 !important;
    }

    h1, h2, h3, h4 {
        color: #293028 !important;
    }

    p, label {
        color: #293028;
    }

    .subtitle {
        color: #4F5949 !important;
        font-size: 17px;
        margin-bottom: 20px;
    }

    .book-card {
        background: #FFFDF8;
        border: 1px solid #C8D1B5;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 3px 10px rgba(70, 80, 55, 0.08);
    }

    .book-title {
        font-size: 21px;
        font-weight: 700;
        color: #293028;
        margin-bottom: 5px;
    }

    .book-author {
        font-size: 15px;
        color: #66754E;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .book-description {
        color: #4F5949;
        line-height: 1.6;
        font-size: 14px;
    }

    .genre-tag {
        display: inline-block;
        background: #E7ECD9;
        color: #4F5949;
        padding: 5px 10px;
        border-radius: 15px;
        margin: 3px;
        font-size: 12px;
    }

    .section-box {
        background: #FFFDF8;
        border: 1px solid #C8D1B5;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }

    .footer {
        margin-top: 50px;
        padding: 25px 10px;
        text-align: center;
        border-top: 1px solid #C8D1B5;
        color: #687060 !important;
        font-size: 13px;
    }

    .about-box {
        background: #E7ECD9;
        border-left: 5px solid #71805A;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }

    div[data-testid="stMetric"] {
        background: #FFFDF8 !important;
        border: 1px solid #C8D1B5 !important;
        border-radius: 12px !important;
        padding: 10px !important;
    }

    div[data-testid="stMetric"] label {
        color: #4F5949 !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #4F5949 !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #293028 !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #4F5949 !important;
    }

    textarea {
        color: #293028 !important;
        background-color: #FFFDF8 !important;
    }

    input {
        color: #293028 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "favorites" not in st.session_state:
    st.session_state.favorites = []

if "recommendations" not in st.session_state:
    st.session_state.recommendations = pd.DataFrame()

if "smart_results" not in st.session_state:
    st.session_state.smart_results = pd.DataFrame()


# =========================================================
# HELPERS
# =========================================================

def clean_text(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def normalize_text(value):
    text = clean_text(value).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_isbn(value):
    text = clean_text(value)

    if text.lower() in ["nan", "none", "null"]:
        return ""

    text = re.sub(r"[^0-9xX]", "", text)

    return text


def normalize_book_id(value):
    text = clean_text(value)

    if not text:
        return ""

    if text.endswith(".0"):
        text = text[:-2]

    return text.lower().strip()


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data(show_spinner=False)
def load_data():

    processed_path = Path("processed") / "books_cleaned.csv"
    raw_path = Path("data") / "Goodreads Books.csv"

    if not processed_path.exists():
        st.error(
            "Processed dataset not found. Please run preprocess.py first."
        )
        st.stop()

    df = pd.read_csv(processed_path)

    # -----------------------------------------------------
    # Clean processed dataset
    # -----------------------------------------------------

    text_columns = [
        "bookId",
        "title",
        "author",
        "series",
        "description",
        "genres",
        "characters",
        "language"
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str)

    numeric_columns = [
        "num_pages",
        "num_ratings",
        "avg_rating"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            ).fillna(0)

    # -----------------------------------------------------
    # Add ISBN / ISBN13 from original dataset
    # -----------------------------------------------------

    df["isbn"] = ""
    df["isbn13"] = ""

    if raw_path.exists():

        try:

            raw = pd.read_csv(
                raw_path,
                usecols=[
                    "bookId",
                    "title",
                    "author",
                    "isbn",
                    "isbn13"
                ]
            )

            raw = raw.fillna("")

            raw["bookId_key"] = raw["bookId"].apply(
                normalize_book_id
            )

            raw["title_key"] = raw["title"].apply(
                normalize_text
            )

            raw["author_key"] = raw["author"].apply(
                normalize_text
            )

            # Remove duplicate keys
            raw_id_map = (
                raw[
                    raw["bookId_key"] != ""
                ]
                .drop_duplicates(
                    subset=["bookId_key"]
                )
                .set_index("bookId_key")
            )

            raw_title_author_map = (
                raw[
                    (raw["title_key"] != "")
                    & (raw["author_key"] != "")
                ]
                .drop_duplicates(
                    subset=["title_key", "author_key"]
                )
            )

            # First: match using bookId
            for index in df.index:

                book_id = normalize_book_id(
                    df.at[index, "bookId"]
                )

                if book_id in raw_id_map.index:

                    isbn = clean_isbn(
                        raw_id_map.loc[book_id, "isbn"]
                    )

                    isbn13 = clean_isbn(
                        raw_id_map.loc[book_id, "isbn13"]
                    )

                    df.at[index, "isbn"] = isbn
                    df.at[index, "isbn13"] = isbn13

            # Second: fallback using title + author
            for index in df.index:

                if (
                    clean_isbn(df.at[index, "isbn"])
                    or
                    clean_isbn(df.at[index, "isbn13"])
                ):
                    continue

                title_key = normalize_text(
                    df.at[index, "title"]
                )

                author_key = normalize_text(
                    df.at[index, "author"]
                )

                matches = raw_title_author_map[
                    (
                        raw_title_author_map["title_key"]
                        == title_key
                    )
                    &
                    (
                        raw_title_author_map["author_key"]
                        == author_key
                    )
                ]

                if not matches.empty:

                    first_match = matches.iloc[0]

                    df.at[index, "isbn"] = clean_isbn(
                        first_match["isbn"]
                    )

                    df.at[index, "isbn13"] = clean_isbn(
                        first_match["isbn13"]
                    )

        except Exception:
            pass

    return df


df = load_data()


# =========================================================
# GENRE SYSTEM
# =========================================================

GENRE_KEYWORDS = {

    "Romance": [
        "romance",
        "romantic",
        "love",
        "relationship",
        "love story"
    ],

    "Thriller": [
        "thriller",
        "suspense",
        "psychological thriller",
        "crime thriller"
    ],

    "Mystery": [
        "mystery",
        "detective",
        "murder mystery",
        "crime"
    ],

    "Fantasy": [
        "fantasy",
        "magic",
        "magical",
        "fairy",
        "dragons",
        "supernatural"
    ],

    "Science Fiction": [
        "science fiction",
        "sci-fi",
        "scifi",
        "space",
        "dystopian"
    ],

    "Horror": [
        "horror",
        "ghost",
        "vampire",
        "haunted",
        "dark horror"
    ],

    "Young Adult": [
        "young adult",
        "ya"
    ],

    "Historical": [
        "historical",
        "history",
        "historical fiction"
    ],

    "Adventure": [
        "adventure",
        "quest",
        "journey"
    ],

    "Comedy": [
        "comedy",
        "humor",
        "humour",
        "funny"
    ],

    "Biography": [
        "biography",
        "memoir",
        "autobiography"
    ],

    "Self Help": [
        "self help",
        "self-help",
        "personal development",
        "motivation"
    ]
}


def detect_genres(text):

    text = normalize_text(text)

    found = []

    for genre, keywords in GENRE_KEYWORDS.items():

        for keyword in keywords:

            if keyword in text:
                found.append(genre)
                break

    return found


def extract_genres(value):

    text = clean_text(value)

    if not text:
        return []

    detected = detect_genres(text)

    if detected:
        return detected

    parts = []

    text = text.replace("|", ",")

    for item in text.split(","):

        item = item.strip()

        if item:
            parts.append(item)

    return parts[:6]


def display_genres(value):

    genres = extract_genres(value)

    if not genres:
        return "Not specified"

    return ", ".join(genres)


# =========================================================
# LANGUAGE SYSTEM
# =========================================================

LANGUAGE_ALIASES = {

    "English": [
        "english",
        "en",
        "eng"
    ],

    "Hindi": [
        "hindi",
        "hin",
        "hi"
    ],

    "Bengali": [
        "bengali",
        "bangla",
        "bn"
    ],

    "French": [
        "french",
        "fr"
    ],

    "Spanish": [
        "spanish",
        "es"
    ],

    "German": [
        "german",
        "de"
    ],

    "Italian": [
        "italian",
        "it"
    ],

    "Portuguese": [
        "portuguese",
        "pt"
    ],

    "Russian": [
        "russian",
        "ru"
    ],

    "Chinese": [
        "chinese",
        "zh"
    ],

    "Japanese": [
        "japanese",
        "ja"
    ],

    "Korean": [
        "korean",
        "ko"
    ],

    "Arabic": [
        "arabic",
        "ar"
    ],

    "Tamil": [
        "tamil",
        "ta"
    ],

    "Telugu": [
        "telugu",
        "te"
    ],

    "Marathi": [
        "marathi",
        "mr"
    ],

    "Malayalam": [
        "malayalam",
        "ml"
    ],

    "Urdu": [
        "urdu",
        "ur"
    ]
}


def normalize_language(value):

    text = normalize_text(value)

    if not text:
        return "Unknown"

    for language, aliases in LANGUAGE_ALIASES.items():

        for alias in aliases:

            if text == alias:
                return language

            parts = re.split(r"[,;|]", text)

            for part in parts:

                if normalize_text(part) == alias:
                    return language

    return clean_text(value)


def language_matches(value, selected_language):

    if selected_language == "All Languages":
        return True

    text = normalize_text(value)

    if not text:
        return selected_language == "Unknown"

    aliases = LANGUAGE_ALIASES.get(
        selected_language,
        [normalize_text(selected_language)]
    )

    parts = re.split(r"[,;|]", text)

    for part in parts:

        part = normalize_text(part)

        if part in aliases:
            return True

    return False


available_languages = set()

for value in df["language"].dropna():

    language = normalize_language(value)

    if language:
        available_languages.add(language)


preferred_language_order = [
    "English",
    "Hindi",
    "Bengali",
    "French",
    "Spanish",
    "German",
    "Italian",
    "Portuguese",
    "Russian",
    "Chinese",
    "Japanese",
    "Korean",
    "Arabic",
    "Tamil",
    "Telugu",
    "Marathi",
    "Malayalam",
    "Urdu"
]

language_options = ["All Languages"]

for language in preferred_language_order:

    if language in available_languages:
        language_options.append(language)

remaining_languages = sorted(
    available_languages.difference(
        set(language_options)
    )
)

language_options.extend(remaining_languages)


# =========================================================
# TF-IDF MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def build_tfidf(data):

    text = (
        data["combined_text"]
        .fillna("")
        .astype(str)
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(text)

    return vectorizer, matrix


vectorizer, tfidf_matrix = build_tfidf(df)


# =========================================================
# BOOK URL
# =========================================================

def get_book_url(title, author):

    query = quote_plus(
        f"{clean_text(title)} {clean_text(author)} book"
    )

    return f"https://www.google.com/search?q={query}"


# =========================================================
# BOOK COVER SYSTEM
# =========================================================

def fetch_image(url):

    try:

        request = Request(
            url,
            headers={
                "User-Agent":
                "ShelfScanner/1.0 "
                "(book discovery ML project)"
            }
        )

        with urlopen(request, timeout=5) as response:

            content_type = (
                response.headers
                .get("Content-Type", "")
                .lower()
            )

            if not content_type.startswith("image"):
                return None

            return response.read()

    except Exception:
        return None


@st.cache_data(
    show_spinner=False,
    ttl=86400
)
def get_book_cover(
    isbn13,
    isbn,
    title,
    author
):

    isbn13 = clean_isbn(isbn13)
    isbn = clean_isbn(isbn)

    title = clean_text(title)
    author = clean_text(author)

    isbn_list = []

    for code in [isbn13, isbn]:

        if code and code not in isbn_list:
            isbn_list.append(code)

    # =====================================================
    # 1. OPEN LIBRARY USING ISBN
    # =====================================================

    for code in isbn_list:

        for size in ["L", "M"]:

            url = (
                "https://covers.openlibrary.org/"
                f"b/isbn/{code}-{size}.jpg"
                "?default=false"
            )

            image = fetch_image(url)

            if image:
                return image

    # =====================================================
    # 2. OPEN LIBRARY SEARCH
    # =====================================================

    search_queries = []

    if title and author:
        search_queries.append(
            f"{title} {author}"
        )

    if title:
        search_queries.append(title)

    for search_query in search_queries:

        try:

            api_url = (
                "https://openlibrary.org/search.json?"
                + urlencode(
                    {
                        "q": search_query,
                        "limit": 10,
                        "fields":
                        "cover_i,title,author_name,isbn,isbn13"
                    }
                )
            )

            request = Request(
                api_url,
                headers={
                    "User-Agent":
                    "ShelfScanner/1.0 "
                    "(book discovery ML project)"
                }
            )

            with urlopen(
                request,
                timeout=7
            ) as response:

                data = json.loads(
                    response
                    .read()
                    .decode("utf-8")
                )

            docs = data.get("docs", [])

            if not docs:
                continue

            requested_title = normalize_text(title)
            requested_author = normalize_text(author)

            # -------------------------------------------------
            # Rank Open Library search results
            # -------------------------------------------------

            ranked_docs = []

            for doc in docs:

                result_title = normalize_text(
                    doc.get("title", "")
                )

                result_authors = [
                    normalize_text(x)
                    for x in doc.get(
                        "author_name",
                        []
                    )
                ]

                score = 0

                if result_title == requested_title:
                    score += 10

                elif (
                    requested_title
                    and requested_title in result_title
                ):
                    score += 5

                if requested_author:

                    for result_author in result_authors:

                        if (
                            requested_author
                            in result_author
                            or
                            result_author
                            in requested_author
                        ):
                            score += 5
                            break

                if doc.get("cover_i"):
                    score += 2

                ranked_docs.append(
                    (score, doc)
                )

            ranked_docs.sort(
                key=lambda x: x[0],
                reverse=True
            )

            for _, doc in ranked_docs:

                cover_id = doc.get("cover_i")

                if not cover_id:
                    continue

                for size in ["L", "M"]:

                    cover_url = (
                        "https://covers.openlibrary.org/"
                        f"b/id/{cover_id}-{size}.jpg"
                        "?default=false"
                    )

                    image = fetch_image(
                        cover_url
                    )

                    if image:
                        return image

        except Exception:
            continue

    # =====================================================
    # 3. GOOGLE BOOKS
    # =====================================================

    google_queries = []

    for code in isbn_list:

        google_queries.append(
            f"isbn:{code}"
        )

    if title:

        google_queries.append(
            f'intitle:{title}'
        )

    if title and author:

        google_queries.append(
            f'intitle:{title} inauthor:{author}'
        )

    for query in google_queries:

        try:

            api_url = (
                "https://www.googleapis.com/books/v1/volumes?"
                + urlencode(
                    {
                        "q": query,
                        "maxResults": 10
                    }
                )
            )

            request = Request(
                api_url,
                headers={
                    "User-Agent":
                    "ShelfScanner/1.0"
                }
            )

            with urlopen(
                request,
                timeout=7
            ) as response:

                data = json.loads(
                    response
                    .read()
                    .decode("utf-8")
                )

            items = data.get(
                "items",
                []
            )

            for item in items:

                volume_info = item.get(
                    "volumeInfo",
                    {}
                )

                image_links = volume_info.get(
                    "imageLinks",
                    {}
                )

                image_url = (
                    image_links.get("extraLarge")
                    or image_links.get("large")
                    or image_links.get("medium")
                    or image_links.get("thumbnail")
                    or image_links.get("smallThumbnail")
                )

                if not image_url:
                    continue

                image_url = image_url.replace(
                    "http://",
                    "https://"
                )

                image = fetch_image(
                    image_url
                )

                if image:
                    return image

        except Exception:
            continue

    return None


# =========================================================
# BOOK HELPERS
# =========================================================

def get_book(title):

    matches = df[
        df["title"].apply(
            normalize_text
        )
        ==
        normalize_text(title)
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def save_book(title):

    if title not in st.session_state.favorites:
        st.session_state.favorites.append(title)


def remove_book(title):

    if title in st.session_state.favorites:
        st.session_state.favorites.remove(title)


def short_description(
    description,
    length=300
):

    text = clean_text(description)

    if not text:
        return ""

    if len(text) <= length:
        return text

    return (
        text[:length]
        .rsplit(" ", 1)[0]
        + "..."
    )


# =========================================================
# SMART SEARCH
# =========================================================

def smart_search(
    query,
    number_of_results=10,
    minimum_rating=0.0,
    selected_genre="All Genres",
    selected_language="All Languages",
    maximum_pages=0
):

    query = clean_text(query)

    if not query:
        return pd.DataFrame()

    query_vector = vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    results = df.copy()

    results["similarity_score"] = similarities

    # -----------------------------------------------------
    # Genre boost
    # -----------------------------------------------------

    detected_query_genres = detect_genres(query)

    genre_boost = []

    for _, row in results.iterrows():

        book_genres = extract_genres(
            row.get("genres", "")
        )

        score = 0

        for genre in detected_query_genres:

            if genre in book_genres:
                score += 0.35

        genre_boost.append(score)

    results["genre_boost"] = genre_boost

    # -----------------------------------------------------
    # Keyword matching
    # -----------------------------------------------------

    query_words = [
        word
        for word in normalize_text(query).split()
        if len(word) > 2
    ]

    keyword_scores = []

    for _, row in results.iterrows():

        searchable = " ".join([
            clean_text(
                row.get("title", "")
            ),
            clean_text(
                row.get("author", "")
            ),
            clean_text(
                row.get("genres", "")
            ),
            clean_text(
                row.get("description", "")
            ),
            clean_text(
                row.get("characters", "")
            )
        ]).lower()

        matches = sum(
            1
            for word in query_words
            if word in searchable
        )

        score = (
            matches
            / max(len(query_words), 1)
        )

        keyword_scores.append(score)

    results["keyword_score"] = keyword_scores

    # -----------------------------------------------------
    # Rating bonus
    # -----------------------------------------------------

    results["rating_bonus"] = (
        results["avg_rating"]
        .clip(0, 5)
        / 5
    ) * 0.08

    # -----------------------------------------------------
    # Final ML score
    # -----------------------------------------------------

    results["final_score"] = (
        results["similarity_score"] * 0.60
        + results["genre_boost"]
        + results["keyword_score"] * 0.25
        + results["rating_bonus"]
    )

    # -----------------------------------------------------
    # Filters
    # -----------------------------------------------------

    results = results[
        results["avg_rating"]
        >= minimum_rating
    ]

    if selected_genre != "All Genres":

        results = results[
            results["genres"].apply(
                lambda x:
                selected_genre
                in extract_genres(x)
            )
        ]

    if selected_language != "All Languages":

        results = results[
            results["language"].apply(
                lambda x:
                language_matches(
                    x,
                    selected_language
                )
            )
        ]

    if maximum_pages > 0:

        results = results[
            (
                results["num_pages"] == 0
            )
            |
            (
                results["num_pages"]
                <= maximum_pages
            )
        ]

    results = results.sort_values(
        "final_score",
        ascending=False
    )

    return results.head(
        number_of_results
    )


# =========================================================
# SIMILAR BOOK RECOMMENDATIONS
# =========================================================

def recommend_similar_books(
    book_title,
    number_of_recommendations=5,
    minimum_rating=0.0,
    selected_genre="All Genres"
):

    matches = df[
        df["title"].apply(
            normalize_text
        )
        ==
        normalize_text(book_title)
    ]

    if matches.empty:
        return pd.DataFrame()

    book_index = matches.index[0]

    similarities = cosine_similarity(
        tfidf_matrix[book_index],
        tfidf_matrix
    ).flatten()

    results = df.copy()

    results["similarity_score"] = similarities

    # Remove selected book
    results = results[
        results.index != book_index
    ]

    # -----------------------------------------------------
    # Genre similarity
    # -----------------------------------------------------

    selected_genres = set(
        extract_genres(
            df.loc[
                book_index,
                "genres"
            ]
        )
    )

    genre_scores = []

    for _, row in results.iterrows():

        other_genres = set(
            extract_genres(
                row.get(
                    "genres",
                    ""
                )
            )
        )

        if (
            selected_genres
            and other_genres
        ):

            overlap = len(
                selected_genres
                .intersection(
                    other_genres
                )
            )

            genre_score = (
                overlap
                /
                max(
                    len(selected_genres),
                    1
                )
            )

        else:

            genre_score = 0

        genre_scores.append(
            genre_score
        )

    results["genre_similarity"] = (
        genre_scores
    )

    # -----------------------------------------------------
    # Rating bonus
    # -----------------------------------------------------

    results["rating_bonus"] = (
        results["avg_rating"]
        .clip(0, 5)
        / 5
    ) * 0.06

    # -----------------------------------------------------
    # Final score
    # -----------------------------------------------------

    results["final_score"] = (
        results["similarity_score"] * 0.75
        + results["genre_similarity"] * 0.20
        + results["rating_bonus"]
    )

    # -----------------------------------------------------
    # Filters
    # -----------------------------------------------------

    results = results[
        results["avg_rating"]
        >= minimum_rating
    ]

    if selected_genre != "All Genres":

        results = results[
            results["genres"].apply(
                lambda x:
                selected_genre
                in extract_genres(x)
            )
        ]

    results = results.sort_values(
        "final_score",
        ascending=False
    )

    return results.head(
        number_of_recommendations
    )


# =========================================================
# BOOK CARD
# =========================================================

def render_book_card(
    row,
    card_key,
    show_similarity=False
):

    title = clean_text(
        row.get("title", "")
    )

    author = clean_text(
        row.get("author", "")
    )

    # -----------------------------------------------------
    # Cover
    # -----------------------------------------------------

    cover = get_book_cover(
        row.get("isbn13", ""),
        row.get("isbn", ""),
        title,
        author
    )

    st.markdown(
        '<div class="book-card">',
        unsafe_allow_html=True
    )

    image_col, info_col = st.columns(
        [1, 3]
    )

    with image_col:

        if cover:

            st.image(
                cover,
                width=180
            )

        else:

            st.info(
                "Cover image is not available "
                "for this book."
            )

    with info_col:

        st.markdown(
            f'<div class="book-title">'
            f'{title}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="book-author">'
            f'by {author or "Unknown Author"}'
            f'</div>',
            unsafe_allow_html=True
        )

        metric1, metric2, metric3 = st.columns(3)

        with metric1:

            st.metric(
                "Rating",
                f'{float(row.get("avg_rating", 0)):.2f}'
            )

        with metric2:

            st.metric(
                "Pages",
                int(
                    float(
                        row.get(
                            "num_pages",
                            0
                        )
                    )
                )
            )

        with metric3:

            if show_similarity:

                st.metric(
                    "Match",
                    f'{float(row.get("final_score", 0)) * 100:.1f}%'
                )

            else:

                st.metric(
                    "Ratings",
                    f'{int(float(row.get("num_ratings", 0))):,}'
                )

        st.write(
            f'**Genres:** '
            f'{display_genres(row.get("genres", ""))}'
        )

        description = short_description(
            row.get("description", ""),
            350
        )

        if description:

            st.write(
                description
            )

        else:

            st.write(
                "No description available."
            )

        if show_similarity:

            st.caption(
                "Matched using TF-IDF text similarity, "
                "genre relevance, keyword matching "
                "and rating signals."
            )

        else:

            st.caption(
                "Recommended using content similarity, "
                "genre similarity and rating signals."
            )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # Buttons
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        if title in st.session_state.favorites:

            if st.button(
                "Remove from My Shelf",
                key=f"remove_{card_key}"
            ):

                remove_book(title)
                st.rerun()

        else:

            if st.button(
                "Save to My Shelf",
                key=f"save_{card_key}"
            ):

                save_book(title)
                st.rerun()

    with col2:

        st.link_button(
            "Find Book Online",
            get_book_url(
                title,
                author
            ),
            use_container_width=True
        )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("ShelfScanner")

st.sidebar.caption(
    "AI-powered book discovery and recommendation"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Discover",
        "Recommendations",
        "Smart Search",
        "My Shelf",
        "About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"Books available: **{len(df):,}**"
)

st.sidebar.write(
    f"My Shelf: **{len(st.session_state.favorites)}**"
)


# =========================================================
# DISCOVER
# =========================================================

if page == "Discover":

    st.title("Discover Books")

    st.markdown(
        '<p class="subtitle">'
        'Search the collection and explore books that match your interests.'
        '</p>',
        unsafe_allow_html=True
    )

    search_text = st.text_input(
        "Search by title",
        placeholder="Type a book title..."
    )

    if search_text:

        title_matches = df[
            df["title"].str.contains(
                search_text,
                case=False,
                na=False,
                regex=False
            )
        ].head(50)

    else:

        title_matches = df.head(50)

    if title_matches.empty:

        st.warning(
            "No books found with that title."
        )

    else:

        selected_title = st.selectbox(
            "Select a book",
            title_matches["title"].tolist()
        )

        book = get_book(
            selected_title
        )

        if book is not None:

            cover = get_book_cover(
                book.get("isbn13", ""),
                book.get("isbn", ""),
                book["title"],
                book["author"]
            )

            st.markdown(
                '<div class="book-card">',
                unsafe_allow_html=True
            )

            image_col, info_col = st.columns(
                [1, 3]
            )

            with image_col:

                if cover:

                    st.image(
                        cover,
                        width=220
                    )

                else:

                    st.info(
                        "Cover image is not available."
                    )

            with info_col:

                st.markdown(
                    f'<div class="book-title">'
                    f'{book["title"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="book-author">'
                    f'by {book["author"] or "Unknown Author"}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Rating",
                        f'{book["avg_rating"]:.2f}'
                    )

                with col2:

                    st.metric(
                        "Ratings",
                        f'{int(book["num_ratings"]):,}'
                    )

                with col3:

                    st.metric(
                        "Pages",
                        int(book["num_pages"])
                    )

                with col4:

                    st.metric(
                        "Language",
                        normalize_language(
                            book["language"]
                        )
                    )

                st.write(
                    f'**Genres:** '
                    f'{display_genres(book["genres"])}'
                )

                if book["series"]:

                    st.write(
                        f'**Series:** '
                        f'{book["series"]}'
                    )

                description = short_description(
                    book["description"],
                    600
                )

                if description:

                    st.write(
                        description
                    )

                else:

                    st.write(
                        "No description available."
                    )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            button1, button2, button3 = st.columns(3)

            with button1:

                if selected_title in st.session_state.favorites:

                    if st.button(
                        "Remove from My Shelf",
                        key="discover_remove"
                    ):

                        remove_book(
                            selected_title
                        )

                        st.rerun()

                else:

                    if st.button(
                        "Save to My Shelf",
                        key="discover_save"
                    ):

                        save_book(
                            selected_title
                        )

                        st.rerun()

            with button2:

                st.link_button(
                    "Find Book Online",
                    get_book_url(
                        book["title"],
                        book["author"]
                    )
                )

            with button3:

                if st.button(
                    "Find Similar Books",
                    key="discover_similar"
                ):

                    st.session_state.recommendations = (
                        recommend_similar_books(
                            selected_title,
                            5
                        )
                    )

                    st.success(
                        "Similar books generated. "
                        "Open Recommendations from the sidebar."
                    )


# =========================================================
# RECOMMENDATIONS
# =========================================================

elif page == "Recommendations":

    st.title("Book Recommendations")

    st.markdown(
        '<p class="subtitle">'
        'Choose a book and let ShelfScanner find books with similar content, themes and genres.'
        '</p>',
        unsafe_allow_html=True
    )

    selected_book = st.selectbox(
        "Choose a book",
        df["title"].tolist()
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        recommendation_count = st.slider(
            "Number of recommendations",
            3,
            15,
            5
        )

    with col2:

        minimum_rating = st.slider(
            "Minimum rating",
            0.0,
            5.0,
            0.0,
            0.1
        )

    with col3:

        genre_options = [
            "All Genres"
        ] + list(
            GENRE_KEYWORDS.keys()
        )

        selected_genre = st.selectbox(
            "Genre",
            genre_options
        )

    if st.button(
        "Generate Recommendations",
        type="primary"
    ):

        st.session_state.recommendations = (
            recommend_similar_books(
                selected_book,
                recommendation_count,
                minimum_rating,
                selected_genre
            )
        )

    results = st.session_state.recommendations

    if not results.empty:

        st.subheader(
            "Recommended for You"
        )

        for index, row in results.iterrows():

            render_book_card(
                row,
                f"rec_{index}",
                show_similarity=False
            )

    else:

        st.info(
            "Select a book and click Generate Recommendations."
        )


# =========================================================
# SMART SEARCH
# =========================================================

elif page == "Smart Search":

    st.title("Smart Search")

    st.markdown(
        '<p class="subtitle">'
        'Describe the kind of book you want in your own words.'
        '</p>',
        unsafe_allow_html=True
    )

    query = st.text_area(
        "What are you looking for?",
        placeholder=(
            "Example: romantic story with emotional characters\n"
            "Example: dark psychological thriller\n"
            "Example: fantasy adventure with magic"
        ),
        height=120
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        number_of_results = st.slider(
            "Results",
            5,
            20,
            10
        )

    with col2:

        minimum_rating = st.slider(
            "Minimum rating",
            0.0,
            5.0,
            0.0,
            0.1,
            key="smart_rating"
        )

    with col3:

        genre_options = [
            "All Genres"
        ] + list(
            GENRE_KEYWORDS.keys()
        )

        selected_genre = st.selectbox(
            "Genre",
            genre_options,
            key="smart_genre"
        )

    col4, col5 = st.columns(2)

    with col4:

        selected_language = st.selectbox(
            "Language",
            language_options,
            key="smart_language"
        )

    with col5:

        maximum_pages = st.number_input(
            "Maximum pages (0 = no limit)",
            min_value=0,
            max_value=5000,
            value=0,
            step=50
        )

    if st.button(
        "Search with AI",
        type="primary"
    ):

        if not query.strip():

            st.warning(
                "Please describe the type of book you are looking for."
            )

        else:

            st.session_state.smart_results = (
                smart_search(
                    query,
                    number_of_results,
                    minimum_rating,
                    selected_genre,
                    selected_language,
                    maximum_pages
                )
            )

    results = st.session_state.smart_results

    if not results.empty:

        st.success(
            f"Found {len(results)} matching books."
        )

        for index, row in results.iterrows():

            render_book_card(
                row,
                f"smart_{index}",
                show_similarity=True
            )

    else:

        st.info(
            "Describe the kind of book you want and click Search with AI."
        )


# =========================================================
# MY SHELF
# =========================================================

elif page == "My Shelf":

    st.title("My Shelf")

    st.markdown(
        '<p class="subtitle">'
        'Books you saved for later.'
        '</p>',
        unsafe_allow_html=True
    )

    if not st.session_state.favorites:

        st.info(
            "Your shelf is empty. Save books from Discover, "
            "Recommendations or Smart Search."
        )

    else:

        st.write(
            f"You have saved "
            f"**{len(st.session_state.favorites)} books**."
        )

        for index, title in enumerate(
            st.session_state.favorites
        ):

            book = get_book(title)

            if book is not None:

                render_book_card(
                    book,
                    f"shelf_{index}",
                    show_similarity=False
                )


# =========================================================
# ABOUT
# =========================================================

elif page == "About":

    st.title("About ShelfScanner")

    st.markdown(
        '<p class="subtitle">'
        'A machine-learning based book discovery and recommendation application.'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="about-box">',
        unsafe_allow_html=True
    )

    st.subheader(
        "What is ShelfScanner?"
    )

    st.write(
        "ShelfScanner is an AI/ML-based book discovery "
        "application designed to help readers find books "
        "according to their interests, genres and reading preferences."
    )

    st.write(
        "The application uses Natural Language Processing "
        "and TF-IDF based text similarity to understand book "
        "information and generate relevant recommendations."
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    st.subheader(
        "Key Features"
    )

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:

        st.write("### Smart Search")

        st.write(
            "Describe the kind of book you want using "
            "natural language and ShelfScanner searches "
            "for relevant books."
        )

        st.write("### Similar Book Recommendations")

        st.write(
            "Select a book and receive recommendations "
            "based on content and genre similarity."
        )

        st.write("### Filters")

        st.write(
            "Filter results using genre, rating, language "
            "and page count."
        )

    with feature_col2:

        st.write("### My Shelf")

        st.write(
            "Save interesting books and keep them in "
            "your personal shelf."
        )

        st.write("### Online Book Search")

        st.write(
            "Each book includes an option to find it online."
        )

        st.write("### Machine Learning")

        st.write(
            "The recommendation engine uses TF-IDF "
            "vectorization and cosine similarity along "
            "with additional relevance signals."
        )

    st.markdown("---")

    st.subheader(
        "About the Creator"
    )

    st.markdown(
        '<div class="section-box">',
        unsafe_allow_html=True
    )

    st.subheader(
        "Komal Kumari"
    )

    st.write(
        "CSE (AI) student with an interest in Artificial "
        "Intelligence, Machine Learning and practical technology projects."
    )

    st.write(
        "ShelfScanner was developed as an ML project to apply "
        "Natural Language Processing, recommendation techniques "
        "and interactive web application development to a "
        "real-world book discovery problem."
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    st.subheader(
        "Contact Us"
    )

    st.write(
        "For questions, suggestions or feedback:"
    )

    st.write(
        "**Email:** komalthakur150197@gmail.com"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        <strong>ShelfScanner</strong><br>
        AI-powered book discovery and recommendation system<br><br>
        Created by <strong>Komal Kumari</strong> · CSE (AI)<br>
        © 2026 ShelfScanner. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)