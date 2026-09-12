import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# Load labeled dataset
df = pd.read_csv("DATA/processed/apple_labeled_clean.csv")

X = df["text_customer"]
y = df["intent"]


# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Train classifier
classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

classifier.fit(X_train_vec, y_train)


# Predict
y_pred = classifier.predict(X_test_vec)


# Overall metrics
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


print("\n" + "=" * 60)
print("INTENT CLASSIFIER EVALUATION")
print("=" * 60)

print("\nTest examples:", len(y_test))
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))


# Per-intent results
print("\nCLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# Confusion matrix
print("\nCONFUSION MATRIX")
print("=" * 60)

print(confusion_matrix(y_test, y_pred))