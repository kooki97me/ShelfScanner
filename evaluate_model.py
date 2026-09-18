import pandas as pd
from model.recommender import recommend_books


# ---------------------------------------------------------
# BOOKS USED FOR EVALUATION
# ---------------------------------------------------------

test_books = [
    "Harry Potter and the Half-Blood Prince",
    "Harry Potter and the Order of the Phoenix",
    "The Hobbit",
    "The Hunger Games",
    "Pride and Prejudice"
]


# ---------------------------------------------------------
# GENRE CLEANING
# ---------------------------------------------------------

def get_genres(text):

    if pd.isna(text):
        return set()

    text = str(text)

    for char in ["[", "]", "'", '"', "{", "}"]:
        text = text.replace(char, "")

    text = text.replace("|", ",")
    text = text.replace(";", ",")

    return {
        genre.strip().lower()
        for genre in text.split(",")
        if genre.strip()
    }


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(
    "processed/books_cleaned.csv"
)

df["genres"] = df["genres"].fillna("")


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

results = []

print("\n" + "=" * 70)
print("SHELFSCANNER - RECOMMENDATION MODEL EVALUATION")
print("=" * 70)


for book in test_books:

    matches = df[
        df["title"].astype(str).str.lower()
        == book.lower()
    ]

    if matches.empty:

        print(f"\nBook not found: {book}")
        continue

    source_genres = get_genres(
        matches.iloc[0]["genres"]
    )

    recommendations = recommend_books(
        book,
        number_of_recommendations=5
    )

    if recommendations.empty:
        continue

    overlap_scores = []

    for _, row in recommendations.iterrows():

        recommended_genres = get_genres(
            row["genres"]
        )

        if source_genres and recommended_genres:

            overlap = len(
                source_genres.intersection(
                    recommended_genres
                )
            ) / len(source_genres)

        else:

            overlap = 0.0

        overlap_scores.append(overlap)

    average_overlap = sum(
        overlap_scores
    ) / len(overlap_scores)

    results.append(
        {
            "book": book,
            "average_genre_overlap": average_overlap
        }
    )

    print(f"\nBook: {book}")
    print(
        f"Average Genre Overlap@5: "
        f"{average_overlap:.2f}"
    )

    print("\nTop Recommendations:")

    for i, (_, row) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        print(
            f"{i}. {row['title']} "
            f"({row['similarity_score']:.3f})"
        )


# ---------------------------------------------------------
# FINAL RESULT
# ---------------------------------------------------------

if results:

    evaluation_df = pd.DataFrame(results)

    overall_score = (
        evaluation_df[
            "average_genre_overlap"
        ].mean()
    )

    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    print(
        f"Average Genre Overlap@5: "
        f"{overall_score:.2f}"
    )

    print(
        "\nHigher overlap indicates that the "
        "recommended books share more genre "
        "characteristics with the input books."
    )

else:

    print("\nNo books could be evaluated.")