import pandas as pd

DATA_PATH = "DATA/raw/twcs.csv"

# Brands we are considering
BRANDS = [
    "AppleSupport",
    "AmazonHelp",
    "SpotifyCares",
    "ComcastCares",
    "Delta"
]

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

print("Dataset loaded.")
print("Total tweets:", len(df))

# Customer tweets
customers = df[df["inbound"] == True][
    ["tweet_id", "text"]
].copy()

for brand in BRANDS:

    print("\n" + "=" * 80)
    print("BRAND:", brand)
    print("=" * 80)

    # Tweets sent by this brand
    brand_replies = df[
        (df["author_id"] == brand) &
        (df["inbound"] == False)
    ].copy()

    # Connect each brand reply to the customer tweet it answered
    pairs = brand_replies.merge(
        customers,
        left_on="in_response_to_tweet_id",
        right_on="tweet_id",
        suffixes=("_brand", "_customer")
    )

    print("Customer -> Brand pairs found:", len(pairs))

    # Take up to 20 examples
    sample = pairs.sample(
        min(20, len(pairs)),
        random_state=42
    )

    for i, (_, row) in enumerate(sample.iterrows(), 1):

        print(f"\n--- Example {i} ---")

        print("CUSTOMER:")
        print(row["text_customer"])

        print("\nBRAND RESPONSE:")
        print(row["text_brand"])

print("\nDone.")