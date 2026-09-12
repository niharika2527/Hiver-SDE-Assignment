import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load historical Apple conversations
df = pd.read_csv("DATA/processed/apple_history.csv")

print("Loaded historical conversations:", len(df))


# Build TF-IDF representation of customer messages
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

history_vectors = vectorizer.fit_transform(
    df["customer_text"]
)


def retrieve_similar_messages(message, top_k=3):
    """
    Find the most similar historical Apple customer conversations.
    """

    message_vector = vectorizer.transform([message])

    similarities = cosine_similarity(
        message_vector,
        history_vectors
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append({
            "customer_message": df.iloc[index]["customer_text"],
            "apple_response": df.iloc[index]["apple_response"],
            "similarity": float(similarities[index])
        })

    return results


# Test
message = input("\nEnter a customer message: ")

results = retrieve_similar_messages(message)

print("\nSimilar historical conversations:\n")

for i, result in enumerate(results, 1):

    print("=" * 60)

    print(f"RESULT {i}")
    print("Similarity:", round(result["similarity"], 4))

    print("\nCUSTOMER:")
    print(result["customer_message"])

    print("\nAPPLE RESPONSE:")
    print(result["apple_response"])