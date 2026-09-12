import pandas as pd
import os

sample_file = "DATA/processed/apple_support_sample.csv"
labeled_file = "DATA/processed/apple_labeled_sample.csv"

df = pd.read_csv(sample_file)

intents = {
    1: "app_issue",
    2: "ios_update_issue",
    3: "device_issue",
    4: "battery_issue",
    5: "screen_ui_issue",
    6: "apple_music_issue",
    7: "app_store_purchase_issue",
    8: "account_security_issue",
    9: "icloud_data_issue",
    10: "connectivity_issue",
    11: "support_repair_issue",
    12: "product_order_issue"
}

# Load already-labeled examples
if os.path.exists(labeled_file):
    labeled_df = pd.read_csv(labeled_file)
    labeled_texts = set(labeled_df["text_customer"].astype(str))
else:
    labeled_texts = set()

remaining_df = df[
    ~df["text_customer"].astype(str).isin(labeled_texts)
].copy()

print(f"Already labeled: {len(labeled_texts)}")
print(f"Remaining: {len(remaining_df)}")

# Label examples
for question_number, (_, row) in enumerate(remaining_df.iterrows(), start=1):

    print("\n" + "=" * 70)

    print(f"\nQUESTION {question_number}")

    print("\nCUSTOMER:")
    print(row["text_customer"])

    print("\nAPPLE RESPONSE:")
    print(row["text_apple"])

    while True:

        choice = input("\nIntent (1-12, s=skip, q=quit): ").strip().lower()

        if choice == "q":
            print("\nProgress saved.")
            exit()

        if choice == "s":
            print("Skipped.")
            break

        if choice.isdigit() and int(choice) in intents:

            number = int(choice)

            new_row = pd.DataFrame([{
                "text_customer": row["text_customer"],
                "text_apple": row["text_apple"],
                "intent": number
            }])

            new_row.to_csv(
                labeled_file,
                mode="a",
                header=not os.path.exists(labeled_file),
                index=False
            )

            break

        print("Please enter a number from 1-12, s, or q.")

print("\nFinished labeling!")