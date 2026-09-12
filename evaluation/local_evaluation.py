import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# Load the 181 human-labeled golden examples
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


# Evaluate locally — NO API calls
results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=[
        "accuracy",
        "f1_macro",
        "f1_weighted"
    ]
)


accuracy = results["test_accuracy"]
macro_f1 = results["test_f1_macro"]
weighted_f1 = results["test_f1_weighted"]


print("\n" + "=" * 60)
print("GOLDEN SET EVALUATION")
print("=" * 60)

print("\nHuman-labeled examples:", len(df))
print("Number of intents:", y.nunique())

print("\nAccuracy:")
for i, score in enumerate(accuracy, 1):
    print(f"Fold {i}: {score:.4f}")

print("Mean Accuracy:", round(accuracy.mean(), 4))
print("Std Accuracy:", round(accuracy.std(), 4))

print("\nMacro F1:")
for i, score in enumerate(macro_f1, 1):
    print(f"Fold {i}: {score:.4f}")

print("Mean Macro F1:", round(macro_f1.mean(), 4))
print("Std Macro F1:", round(macro_f1.std(), 4))

print("\nWeighted F1:")
for i, score in enumerate(weighted_f1, 1):
    print(f"Fold {i}: {score:.4f}")

print("Mean Weighted F1:", round(weighted_f1.mean(), 4))
print("Std Weighted F1:", round(weighted_f1.std(), 4))

print("\n" + "=" * 60)
print("INTENT DISTRIBUTION")
print("=" * 60)

print(y.value_counts().sort_index())