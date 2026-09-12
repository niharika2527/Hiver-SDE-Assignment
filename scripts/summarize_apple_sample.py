import pandas as pd
import re
from collections import Counter

FILE_PATH = "DATA/processed/apple_support_sample.csv"

df = pd.read_csv(FILE_PATH)

print("Total examples:", len(df))

# Combine customer messages into one string
texts = df["text_customer"].fillna("").str.lower()

# Words we want to investigate
keywords = [
    "ios",
    "update",
    "iphone",
    "ipad",
    "app",
    "apps",
    "battery",
    "charging",
    "icloud",
    "backup",
    "apple id",
    "password",
    "music",
    "itunes",
    "purchase",
    "payment",
    "siri",
    "store",
    "download",
    "install",
    "crash",
    "freeze",
    "slow",
    "restart",
    "error",
    "wifi",
    "bluetooth",
    "camera",
    "screen",
]

print("\nKeyword occurrence:")
print("-" * 40)

counts = Counter()

for text in texts:
    for keyword in keywords:
        if keyword in text:
            counts[keyword] += 1

for keyword, count in counts.most_common():
    print(f"{keyword:15} {count}")

print("\nDone.")