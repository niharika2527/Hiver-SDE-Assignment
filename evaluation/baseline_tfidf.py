import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


# Load cleaned dataset
df = pd.read_csv("DATA/processed/apple_labeled_clean.csv")

X = df["text_customer"]
y = df["intent"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
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


# Train
model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_test)


# Metrics
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


print("TF-IDF + Logistic Regression")
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))