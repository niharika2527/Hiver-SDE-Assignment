import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Load golden dataset
df = pd.read_csv("DATA/processed/apple_labeled_clean.csv")

X = df["text_customer"]
y = df["intent"]


# Train classifier on the complete golden dataset
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1
)

X_vectorized = vectorizer.fit_transform(X)


classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

classifier.fit(X_vectorized, y)


# Save model
joblib.dump(
    vectorizer,
    "models/intent_vectorizer.joblib"
)

joblib.dump(
    classifier,
    "models/intent_classifier.joblib"
)


print("Intent classifier trained successfully.")
print("Saved:")
print("- models/intent_vectorizer.joblib")
print("- models/intent_classifier.joblib")


# Test prediction
message = input("\nEnter a customer message: ")

message_vectorized = vectorizer.transform([message])

prediction = classifier.predict(message_vectorized)[0]

probabilities = classifier.predict_proba(message_vectorized)[0]
confidence = probabilities.max()

print("\nPredicted Intent:", prediction)
print("Confidence:", round(confidence, 4))