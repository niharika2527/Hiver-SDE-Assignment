import pandas as pd
import os

input_file = "DATA/raw/twcs.csv"
output_file = "DATA/processed/apple_history.csv"

print("Loading dataset...")

df = pd.read_csv(
    input_file,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ]
)

# AppleSupport's outgoing replies
apple_replies = df[
    (df["author_id"] == "AppleSupport") &
    (df["inbound"] == False)
].copy()

print("AppleSupport replies:", len(apple_replies))

# Create lookup: tweet_id -> tweet text
tweet_lookup = df.set_index("tweet_id")["text"].to_dict()

# Find the customer message that each AppleSupport reply responds to
apple_replies["customer_text"] = (
    apple_replies["in_response_to_tweet_id"]
    .map(tweet_lookup)
)

# Keep only replies where we successfully found the customer message
apple_history = apple_replies.dropna(
    subset=["customer_text"]
).copy()

# Keep useful columns
apple_history = apple_history[
    [
        "tweet_id",
        "customer_text",
        "text"
    ]
]

apple_history = apple_history.rename(
    columns={"text": "apple_response"}
)

# Remove empty messages
apple_history = apple_history[
    (apple_history["customer_text"].str.strip() != "") &
    (apple_history["apple_response"].str.strip() != "")
]

apple_history.to_csv(output_file, index=False)

print("Historical Apple conversations:", len(apple_history))
print("Saved to:", output_file)

print("\nExample:")
print("\nCUSTOMER:")
print(apple_history.iloc[0]["customer_text"])

print("\nAPPLE RESPONSE:")
print(apple_history.iloc[0]["apple_response"])