# Nollywood Movie Recommender

**3MTT NextGen AI/ML Project — AI-13**

## Problem

Viewers struggle to find Nigerian films they will enjoy. There's no simple, local
equivalent of "if you liked this, try this" built specifically for Nollywood.

## Approach

This is a **content-based recommender** using **TF-IDF (Term Frequency–Inverse
Document Frequency)** and **cosine similarity** — a standard, real recommender
technique that doesn't require deep learning for a project of this scope.

Each movie's genre and lead actor(s) are combined into one text feature. TF-IDF
converts that text into numeric vectors, weighting distinctive words (like a rare
actor name) more heavily than common ones (like "Drama," which appears everywhere).
Cosine similarity then measures how close any two movies are in that numeric space —
1.0 means identical, 0 means nothing in common.

Given a movie title, the model finds the most similar titles in the dataset and
returns the top 5, ranked by similarity score.

## Dataset

`movies.csv` — 100 Nollywood titles, hand-compiled from public sources (Wikipedia's
list of Nigerian films, AMVCA nominee/winner records, and Nollywood film databases),
spanning 1993–2026 across genres including Drama, Thriller, Comedy, Action, Horror,
Fantasy, and History.

Columns:
| Column | Description |
|---|---|
| `title` | Movie title |
| `genre` | One or more genres, comma-separated |
| `lead_actor` | One or more lead actors, comma-separated |
| `year` | Release year |
| `tag` | Short descriptive tag (not used in the model, included for context) |

## How to run

1. Open `nollywood_recommender.ipynb` in Google Colab.
2. Upload `movies.csv` to the Colab file browser (left sidebar → Files → upload).
3. Runtime → Run all.
4. In the "Try it out" section, call `recommend("Movie Title")` with any title from
   the dataset to see its top 5 recommendations.

## Evaluation

Since this is an unsupervised recommender (no "correct answer" labels exist to
check against), it's evaluated with two metrics appropriate to that constraint:

- **Coverage — 100%**: every movie in the dataset receives at least one meaningful
  recommendation (similarity score above 0). No title is orphaned by the model.
- **Genre consistency — 75%**: for a sample across the dataset, 75% of top
  recommendations share at least one genre with the original movie — a sanity
  check confirming the model is finding real patterns, not recommending randomly.

## Tools

Python, pandas, scikit-learn (`TfidfVectorizer`, `cosine_similarity`), Google Colab.

## Author

Prince Kalu Odoh — 3MTT NextGen AI & Machine Learning Fellow, Abia Tech Hub.
