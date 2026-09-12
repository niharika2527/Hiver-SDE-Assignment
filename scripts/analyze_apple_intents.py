import pandas as pd

DATA_PATH = "DATA/raw/twcs.csv"

# Load only the columns we need
df = pd.read_csv(
    DATA_PATH,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "in_response_to_tweet_id"
    ]
)

# Keep only customer messages that AppleSupport responded to
customers = df[df["inbound"] == True].copy()

apple_replies = df[
    (df["author_id"] == "AppleSupport") &
    (df["inbound"] == False)
].copy()

# Connect AppleSupport replies to the customer message they answered
pairs = apple_replies.merge(
    customers[["tweet_id", "text"]],
    left_on="in_response_to_tweet_id",
    right_on="tweet_id",
    suffixes=("_apple", "_customer")
)

print("AppleSupport customer -> response pairs:", len(pairs))

# Take a larger sample for intent analysis
sample_size = min(200, len(pairs))

sample = pairs.sample(
    sample_size,
    random_state=42
)

# Save the sample so we can inspect it easily
output = sample[
    ["text_customer", "text_apple"]
].copy()

output.to_csv(
    "DATA/processed/apple_support_sample.csv",
    index=False
)

print(
    "Saved",
    sample_size,
    "examples to DATA/processed/apple_support_sample.csv"
)

print("\nFirst 30 customer messages:\n")

for i, text in enumerate(output["text_customer"].head(30), 1):
    print(f"{i}. {text}")

print("\nDone.")