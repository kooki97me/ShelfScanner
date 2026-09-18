import pandas as pd
from pathlib import Path
import re


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

input_file = Path("data") / "Goodreads Books.csv"
output_file = Path("processed") / "books_cleaned.csv"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(input_file)

print(f"Original dataset: {df.shape}")


# ---------------------------------------------------------
# SELECT USEFUL COLUMNS
# ---------------------------------------------------------

useful_columns = [
    "bookId",
    "title",
    "author",
    "series",
    "description",
    "genres",
    "characters",
    "language",
    "num_pages",
    "num_ratings",
    "avg_rating"
]

df = df[useful_columns].copy()


# ---------------------------------------------------------
# REMOVE INVALID TITLES
# ---------------------------------------------------------

df = df.dropna(subset=["title"])

df["title"] = (
    df["title"]
    .astype(str)
    .str.strip()
)


# ---------------------------------------------------------
# FILL MISSING TEXT VALUES
# ---------------------------------------------------------

text_columns = [
    "author",
    "series",
    "description",
    "genres",
    "characters",
    "language"
]

for column in text_columns:
    df[column] = df[column].fillna("").astype(str)


# ---------------------------------------------------------
# CLEAN GENRES
# ---------------------------------------------------------

def clean_genres(text):

    if not text:
        return ""

    # Remove brackets and quotes
    text = re.sub(r"[\[\]'\"{}()]", "", text)

    # Replace separators with commas
    text = text.replace("|", ",")
    text = text.replace(";", ",")
    
    # Split genres
    genres = text.split(",")

    cleaned = []
    seen = set()

    for genre in genres:

        genre = genre.strip()

        if not genre:
            continue

        # Normalize spacing
        genre = re.sub(r"\s+", " ", genre)

        # Case-insensitive duplicate removal
        key = genre.lower()

        if key not in seen:
            seen.add(key)
            cleaned.append(genre)

    return ", ".join(cleaned)


df["genres"] = df["genres"].apply(clean_genres)


# ---------------------------------------------------------
# NUMERIC COLUMNS
# ---------------------------------------------------------

df["num_pages"] = pd.to_numeric(
    df["num_pages"],
    errors="coerce"
).fillna(0)

df["num_ratings"] = pd.to_numeric(
    df["num_ratings"],
    errors="coerce"
).fillna(0)

df["avg_rating"] = pd.to_numeric(
    df["avg_rating"],
    errors="coerce"
).fillna(0)


# ---------------------------------------------------------
# CREATE COMBINED TEXT
# ---------------------------------------------------------

df["combined_text"] = (
    df["title"] + " "
    + df["author"] + " "
    + df["series"] + " "
    + df["description"] + " "
    + df["genres"] + " "
    + df["characters"]
)


# ---------------------------------------------------------
# CLEAN COMBINED TEXT
# ---------------------------------------------------------

df["combined_text"] = (
    df["combined_text"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# ---------------------------------------------------------
# REMOVE DUPLICATES
# ---------------------------------------------------------

df = df.drop_duplicates(
    subset=["title", "author"]
)


# ---------------------------------------------------------
# SAVE CLEANED DATA
# ---------------------------------------------------------

df.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# OUTPUT INFORMATION
# ---------------------------------------------------------

print("\nPreprocessing completed!")

print(f"Cleaned dataset: {df.shape}")

print(f"Saved to: {output_file}")

print("\nColumns:")
print(df.columns.tolist())

print("\nSample cleaned genres:")

print(
    df[["title", "genres"]]
    .head(10)
    .to_string(index=False)
)