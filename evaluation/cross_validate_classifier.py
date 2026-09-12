import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# Load golden labeled dataset
df = pd.read_csv("DATA/processed/apple_labeled_clean.csv")

X = df["text_customer"]
y = df["intent"]


# 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# TF-IDF + Logistic Regression
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


# Evaluate
results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=["accuracy", "f1_macro"]
)


accuracy_scores = results["test_accuracy"]
f1_scores = results["test_f1_macro"]


print("\n" + "=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)

print("\nAccuracy per fold:")
for i, score in enumerate(accuracy_scores, 1):
    print(f"Fold {i}: {score:.4f}")

print("\nMacro F1 per fold:")
for i, score in enumerate(f1_scores, 1):
    print(f"Fold {i}: {score:.4f}")

print("\nMean Accuracy:", round(accuracy_scores.mean(), 4))
print("Std Accuracy:", round(accuracy_scores.std(), 4))

print("\nMean Macro F1:", round(f1_scores.mean(), 4))
print("Std Macro F1:", round(f1_scores.std(), 4))