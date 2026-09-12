import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load cleaned labels
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

# Most common class in training data
majority_class = y_train.mode()[0]

# Predict the same class for every test example
y_pred = [majority_class] * len(y_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

print("Majority Class:", majority_class)
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))