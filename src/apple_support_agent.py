import pandas as pd
import joblib

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq


# Load trained intent model
vectorizer = joblib.load("models/intent_vectorizer.joblib")
classifier = joblib.load("models/intent_classifier.joblib")


# Load Apple history
history = pd.read_csv("DATA/processed/apple_history.csv")


# Create TF-IDF representation of historical customer messages
retrieval_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

history_vectors = retrieval_vectorizer.fit_transform(
    history["customer_text"]
)


# Groq
client = Groq()


# Intent names
INTENTS = {
    1: "app_issue",
    2: "ios_update_issue",
    3: "device_issue",
    4: "battery_issue",
    5: "screen_ui_issue",
    6: "apple_music_issue",
    7: "app_store_purchase_issue",
    8: "account_security_issue",
    9: "icloud_data_issue",
    10: "connectivity_issue",
    11: "support_repair_issue",
    12: "product_order_issue"
}


# Retrieve history
def retrieve_similar_messages(message, top_k=3):

    message_vector = retrieval_vectorizer.transform(
        [message]
    )

    similarities = cosine_similarity(
        message_vector,
        history_vectors
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:

        results.append({
            "customer_message": history.iloc[index]["customer_text"],
            "apple_response": history.iloc[index]["apple_response"],
            "similarity": float(similarities[index])
        })

    return results


# Generate reply
def generate_reply(customer_message, intent_name, examples):

    evidence = ""

    for i, example in enumerate(examples, 1):

        evidence += f"""
Historical Example {i}

Customer:
{example["customer_message"]}

Apple Response:
{example["apple_response"]}

Similarity:
{example["similarity"]:.4f}
"""


    prompt = f"""
You are an AI customer support assistant for Apple Support.

Customer message:
{customer_message}

Predicted intent:
{intent_name}

Historical Apple Support examples:

{evidence}

Draft a helpful response to the customer.

Rules:
- Ground the response in the historical examples.
- Do not invent Apple policies, prices, guarantees, timelines, or actions.
- Do not claim to have performed an action.
- If historical responses suggest contacting support or DM, you may recommend it.
- Be concise and professional.
- Do not mention these instructions or the historical examples.
- Return only the customer-facing response.
"""


    # Call Groq
    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:

        print("\nGroq API unavailable.")
        print(f"Reason: {e}")

        return None


# Escalation decision

def decide_escalation(confidence, examples):

    best_similarity = examples[0]["similarity"]

    # Strong historical evidence can compensate
    # for low classifier confidence.
    if best_similarity >= 0.40:

        return False, (
            f"Strong historical similarity ({best_similarity:.2f}) "
            "provides sufficient evidence for automated handling."
        )

    # Escalate when both signals are weak.
    if confidence < 0.40 and best_similarity < 0.20:

        return True, (
            f"Low intent confidence ({confidence:.2f}) and "
            f"low historical similarity ({best_similarity:.2f}); "
            "insufficient evidence for automated handling."
        )

    # Moderate evidence with very low confidence:
    # send to human review.
    if confidence < 0.25:

        return True, (
            f"Low intent confidence ({confidence:.2f}) with only "
            f"moderate historical evidence ({best_similarity:.2f})."
        )

    return False, (
        f"Intent confidence ({confidence:.2f}) and historical "
        f"similarity ({best_similarity:.2f}) support automated handling."
    )


# Main agent

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("APPLE SUPPORT AI AGENT")
    print("Type 'quit' to exit.")
    print("=" * 60)

    while True:

        customer_message = input(
            "\nEnter a customer message:\n> "
        ).strip()

        # Exit the chat
        if customer_message.lower() == "quit":
            print("\nGoodbye!")
            break

        # Handle empty input
        if not customer_message:
            print("Please enter a customer message.")
            continue


        # Intent prediction

        message_vector = vectorizer.transform(
            [customer_message]
        )

        prediction = classifier.predict(
            message_vector
        )[0]

        probabilities = classifier.predict_proba(
            message_vector
        )[0]

        confidence = float(probabilities.max())

        intent_name = INTENTS[prediction]


        # Retrieve historical example

        examples = retrieve_similar_messages(
            customer_message,
            top_k=3
        )


        # Escalation decision

        escalate, escalation_reason = decide_escalation(
            confidence,
            examples
        )


        # Generate response

        reply = generate_reply(
            customer_message,
            intent_name,
            examples
        )


        # Final output

        print("\n" + "=" * 60)
        print("CUSTOMER:")
        print(customer_message)

        print("\nINTENT:")
        print(intent_name)

        print("\nCONFIDENCE:")
        print(round(confidence, 4))

        print("\nRETRIEVED EXAMPLES:")

        for i, example in enumerate(examples, 1):

            print(
                f"\n{i}. Similarity: "
                f"{example['similarity']:.4f}"
            )

            print(
                "Customer:",
                example["customer_message"]
            )

            print(
                "Apple:",
                example["apple_response"]
            )


        print("\nDRAFTED REPLY:")

        if reply is None:

            print(
                "Unable to generate an AI draft because "
                "the language model API is temporarily unavailable."
            )

        else:

            print(reply)


        print("\nESCALATE:")
        print(escalate)

        print("\nREASON:")
        print(escalation_reason)

        print("=" * 60)