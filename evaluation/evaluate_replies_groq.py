import json
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq



# LOAD DATA


vectorizer = joblib.load("models/intent_vectorizer.joblib")
classifier = joblib.load("models/intent_classifier.joblib")

history = pd.read_csv("DATA/processed/apple_history.csv")
golden = pd.read_csv("DATA/processed/apple_labeled_clean.csv")

retrieval_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

history_vectors = retrieval_vectorizer.fit_transform(
    history["customer_text"]
)

client = Groq()


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



# RETRIEVAL


def retrieve_similar_messages(message, top_k=3):

    message_vector = retrieval_vectorizer.transform([message])

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



# GENERATE REPLY


def generate_reply(customer_message, intent_name, examples):

    evidence = ""

    for i, example in enumerate(examples, 1):

        evidence += f"""
Historical Example {i}

Customer:
{example["customer_message"]}

Apple Response:
{example["apple_response"]}
"""

    prompt = f"""
You are an AI customer support assistant for Apple Support.

Customer message:
{customer_message}

Predicted intent:
{intent_name}

Historical Apple Support examples:
{evidence}

Draft a helpful customer-facing response.

Rules:
- Ground the response in the historical examples.
- Do not invent Apple policies, prices, guarantees, timelines, or actions.
- Do not claim to have performed an action.
- Do not introduce technical instructions unless they are supported by the historical examples.
- Be concise and professional.
- Return only the customer-facing response.
"""

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=500,
            timeout=30
        )

        content = response.choices[0].message.content

        if not content:
            return None

        return content.strip()

    except Exception as e:

        print("Reply generation failed:")
        print(e)
        return None



# JUDGE REPLY


def judge_reply(customer_message, reply, examples):

    evidence = ""

    for i, example in enumerate(examples, 1):

        evidence += f"""
Historical Example {i}

Customer:
{example["customer_message"]}

Apple Response:
{example["apple_response"]}
"""

    prompt = f"""
You are evaluating an AI customer support reply.

Customer message:
{customer_message}

AI-generated reply:
{reply}

Historical Apple Support evidence:
{evidence}

Evaluate the AI-generated reply using these four criteria.

Groundedness:
1 = mostly unsupported by historical evidence
2 = weakly grounded
3 = partly grounded
4 = well grounded
5 = strongly grounded in the historical evidence

Helpfulness:
1 = not helpful
2 = minimally helpful
3 = reasonably helpful
4 = helpful
5 = highly helpful

Professionalism:
1 = unprofessional
2 = weak
3 = acceptable
4 = professional
5 = highly professional

Overall quality:
1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Return ONLY valid JSON in exactly this format:

{{"groundedness": 1, "helpfulness": 1, "professionalism": 1, "overall_quality": 1}}

Use integer scores from 1 to 5.
Do not include explanations.
"""

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=500,
            timeout=30
        )

        content = response.choices[0].message.content

        if not content:
            print("Judge returned empty response.")
            return None

        print("Judge response:", content)

        # Try direct JSON parsing first
        try:
            result = json.loads(content)
        except json.JSONDecodeError:

            # Try extracting JSON if model added extra text
            start = content.find("{")
            end = content.rfind("}")

            if start == -1 or end == -1:
                print("Could not find JSON in judge response.")
                return None

            result = json.loads(
                content[start:end + 1]
            )

        required = [
            "groundedness",
            "helpfulness",
            "professionalism",
            "overall_quality"
        ]

        for key in required:

            if key not in result:
                print(f"Missing score: {key}")
                return None

            value = int(result[key])

            if value < 1 or value > 5:
                print(f"Invalid score for {key}: {value}")
                return None

            result[key] = value

        return result

    except Exception as e:

        print("Judge failed:")
        print(type(e).__name__)
        print(e)
        return None



# MAIN


if __name__ == "__main__":

    print("=" * 50)
    print("LLM REPLY EVALUATION")
    print("=" * 50)

    # Use only 10 examples
    sample = golden.head(30).copy()

    results = []

    for position, (_, row) in enumerate(
        sample.iterrows(),
        start=1
    ):

        print(
            print(f"\nSuccessfully evaluated: {len(results)}/{len(sample)}")
        )

        customer_message = row["text_customer"]

        # Classify
        message_vector = vectorizer.transform(
            [customer_message]
        )

        prediction = classifier.predict(
            message_vector
        )[0]

        intent_name = INTENTS[prediction]

        # Retrieve evidence
        examples = retrieve_similar_messages(
            customer_message,
            top_k=3
        )

        # Generate reply
        reply = generate_reply(
            customer_message,
            intent_name,
            examples
        )

        if reply is None:

            print("Skipping because reply generation failed.")
            continue

        print("Generated reply:")
        print(reply)

        # Judge reply
        print("Judging reply...")

        scores = judge_reply(
            customer_message,
            reply,
            examples
        )

        if scores is None:

            print("Skipping because judge failed.")
            continue

        results.append({
            "customer_message": customer_message,
            "predicted_intent": intent_name,
            "reply": reply,
            "groundedness": scores["groundedness"],
            "helpfulness": scores["helpfulness"],
            "professionalism": scores["professionalism"],
            "overall_quality": scores["overall_quality"]
        })

        print("Scores:")
        print(scores)


    
    # SAVE RESULTS
    

    if results:

        output = pd.DataFrame(results)

        output.to_csv(
             "results/reply_evaluation_results.csv",
            index=False
        )

        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)

        print(
            f"Successfully evaluated: {len(output)}/30"
        )

        print(
            f"Mean groundedness: "
            f"{output['groundedness'].mean():.2f}/5"
        )

        print(
            f"Mean helpfulness: "
            f"{output['helpfulness'].mean():.2f}/5"
        )

        print(
            f"Mean professionalism: "
            f"{output['professionalism'].mean():.2f}/5"
        )

        print(
            f"Mean overall quality: "
            f"{output['overall_quality'].mean():.2f}/5"
        )

        print("\nSaved:")
        print("reply_evaluation_results.csv")

    else:

        print("\nNo valid evaluations were produced.")