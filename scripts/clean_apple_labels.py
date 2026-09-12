import pandas as pd

input_file = "DATA/processed/apple_labeled_sample.csv"
output_file = "DATA/processed/apple_labeled_clean.csv"

df = pd.read_csv(input_file)

# Convert old taxonomy labels to the current 12-intent taxonomy
old_to_new = {
    "app_issue": 1,
    "ios_update_issue": 2,
    "device_issue": 3,
    "battery_issue": 4,
    "screen_ui_issue": 5,
    "apple_music_issue": 6,
    "app_store_purchase_issue": 7,
    "account_data_issue": 9,
    "support_repair_issue": 11,
}

# Convert intent values
clean_intents = []

for value in df["intent"]:

    # Already a numeric label from the new taxonomy
    if str(value).isdigit():
        number = int(value)

        if 1 <= number <= 12:
            clean_intents.append(number)
        else:
            clean_intents.append(None)

    # Old textual label
    else:
        label = str(value).strip()

        if label in old_to_new:
            clean_intents.append(old_to_new[label])
        else:
            # "other" and anything unrecognized
            clean_intents.append(None)

df["intent_clean"] = clean_intents

# Remove examples that cannot be reliably mapped
df = df.dropna(subset=["intent_clean"]).copy()

df["intent"] = df["intent_clean"].astype(int)

df = df.drop(columns=["intent_clean"])

df.to_csv(output_file, index=False)

print("Clean dataset created!")
print("Total usable examples:", len(df))
print("\nIntent distribution:")
print(df["intent"].value_counts().sort_index())

print("\nSaved as:", output_file)