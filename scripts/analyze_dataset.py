import pandas as pd

DATA_PATH = "DATA/raw/twcs.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

brands = [
    "AmazonHelp",
    "AppleSupport",
    "SpotifyCares",
    "Delta",
    "comcastcares"
]

for brand in brands:

    print("\n" + "=" * 80)
    print(f"BRAND: {brand}")
    print("=" * 80)

    # Get tweets sent by this brand
    brand_tweets = df[
        (df["author_id"] == brand) &
        (df["inbound"] == False)
    ]

    # Get customer messages that directly respond to brand tweets
    customer_tweets = df[
        (df["inbound"] == True) &
        (df["in_response_to_tweet_id"].isin(
            brand_tweets["tweet_id"]
        ))
    ]

    # Random sample of customer messages
    sample_size = min(30, len(customer_tweets))

    sample = customer_tweets.sample(
        n=sample_size,
        random_state=42
    )

    for i, text in enumerate(sample["text"], start=1):
        print(f"\n{i}. {text}")