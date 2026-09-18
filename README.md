# ShelfScanner

ShelfScanner is an AI-powered book discovery and recommendation application built using Python, Streamlit, Pandas, NumPy, and Scikit-learn.

The application helps users discover books based on their interests, search for books, view book details, and receive content-based recommendations.

## Features

- Smart book search
- Content-based book recommendations
- Similar book recommendations
- Book details and ratings
- Genre and language filters
- Rating and page-count filters
- Recommendation explanations
- Book links for further information
- Interactive Streamlit interface

## Machine Learning

ShelfScanner uses a content-based recommendation approach.

The recommendation system uses:

- TF-IDF Vectorization
- Cosine Similarity
- Genre information
- Ratings
- Keyword matching

Book information such as title, author, description, genres, series, and characters is processed to create a combined representation of each book.

## Dataset

The project uses a Goodreads books dataset containing information such as:

- Book title
- Author
- Description
- Genres
- Language
- Number of ratings
- Average rating
- Number of pages

The dataset is cleaned and prepared using the preprocessing pipeline before being used by the recommendation system.

## Project Structure

```text
ShelfScanner/
│
├── app.py
├── preprocess.py
├── check_data.py
├── evaluate_model.py
├── README.md
│
├── data/
│   └── Goodreads Books.csv
│
├── processed/
│   └── books_cleaned.csv
│
└── model/
    ├── recommender.py
    ├── tfidf_matrix.joblib
    └── tfidf_vectorizer.joblib