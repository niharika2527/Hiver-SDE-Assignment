import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load data
golden = pd.read_csv("DATA/processed/apple_labeled_clean.csv")
history = pd.read_csv("DATA/processed/apple_history.csv")


# Clean text
golden["text_customer"] = (
    golden["text_customer"]
    .fillna("")
    .astype(str)
    .str.strip()
)

history["customer_text"] = (
    history["customer_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# Build TF-IDF index
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2
)

history_vectors = vectorizer.fit_transform(
    history["customer_text"]
)


results = []


for _, row in golden.iterrows():

    query = row["text_customer"]

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        history_vectors
    ).flatten()

    # Remove ALL exact copies of the golden query.
    # This prevents duplicate historical messages
    # from producing artificially perfect scores.
    exact_matches = history["customer_text"] == query
    similarities[exact_matches.values] = -1

    best_index = np.argmax(similarities)
    best_similarity = similarities[best_index]

    results.append({
        "query": query,
        "intent": row["intent"],
        "best_similarity": best_similarity,
        "retrieved_customer": history.iloc[best_index]["customer_text"],
        "retrieved_response": history.iloc[best_index]["apple_response"]
    })


results_df = pd.DataFrame(results)


# Save detailed results
results_df.to_csv(
   "results/retrieval_evaluation_clean.csv",
    index=False
)


# Summary
print("\n=== CLEAN RETRIEVAL EVALUATION ===")

print(f"Golden examples evaluated: {len(results_df)}")

print(
    f"Mean similarity: "
    f"{results_df['best_similarity'].mean():.4f}"
)

print(
    f"Median similarity: "
    f"{results_df['best_similarity'].median():.4f}"
)

print(
    f"Min similarity: "
    f"{results_df['best_similarity'].min():.4f}"
)

print(
    f"Max similarity: "
    f"{results_df['best_similarity'].max():.4f}"
)


print("\nThreshold coverage:")

for threshold in [0.20, 0.30, 0.40, 0.50, 0.60]:

    percentage = (
        results_df["best_similarity"] >= threshold
    ).mean() * 100

    print(
        f">= {threshold:.2f}: "
        f"{percentage:.1f}%"
    )


# Strongest retrieval examples
print("\n=== TOP 5 STRONGEST RETRIEVALS ===")

strongest = results_df.sort_values(
    "best_similarity",
    ascending=False
).head(5)

for _, row in strongest.iterrows():

    print("\nQuery:")
    print(row["query"])

    print(f"Intent: {row['intent']}")
    print(f"Similarity: {row['best_similarity']:.4f}")

    print("Retrieved:")
    print(row["retrieved_customer"])


# Weakest retrieval examples
print("\n=== TOP 5 WEAKEST RETRIEVALS ===")

weakest = results_df.sort_values(
    "best_similarity",
    ascending=True
).head(5)

for _, row in weakest.iterrows():

    print("\nQuery:")
    print(row["query"])

    print(f"Intent: {row['intent']}")
    print(f"Similarity: {row['best_similarity']:.4f}")

    print("Retrieved:")
    print(row["retrieved_customer"])