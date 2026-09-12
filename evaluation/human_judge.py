import pandas as pd

# Load the 23 successfully evaluated examples
results = pd.read_csv("results/reply_evaluation_results.csv")

# We already manually scored the first 9.
# This script scores only examples 10 onward.
remaining = results.iloc[9:].copy()

print("=" * 60)
print("HUMAN EVALUATION — REMAINING EXAMPLES")
print("=" * 60)

print(f"\nAlready evaluated by human: 9")
print(f"Remaining to evaluate: {len(remaining)}")
print(f"Final agreement set: {len(results)} examples")

print("\nScoring:")
print("1 = very poor")
print("2 = poor")
print("3 = acceptable")
print("4 = good")
print("5 = excellent")

human_scores = []

for index, (_, row) in enumerate(
    remaining.iterrows(),
    start=10
):

    print("\n" + "=" * 70)
    print(f"EXAMPLE {index}/23")
    print("=" * 70)

    print("\nCUSTOMER:")
    print(row["customer_message"])

    print("\nAI REPLY:")
    print(row["reply"])

    print("\nEnter scores from 1 to 5.")

    while True:
        try:
            groundedness = int(input("Groundedness: "))
            helpfulness = int(input("Helpfulness: "))
            professionalism = int(input("Professionalism: "))
            overall_quality = int(input("Overall quality: "))

            scores = [
                groundedness,
                helpfulness,
                professionalism,
                overall_quality
            ]

            if all(1 <= x <= 5 for x in scores):
                break

            print("All scores must be between 1 and 5.")

        except ValueError:
            print("Please enter numbers only.")

    human_scores.append({
        "groundedness_human": groundedness,
        "helpfulness_human": helpfulness,
        "professionalism_human": professionalism,
        "overall_quality_human": overall_quality
    })


# Load the existing human scores for examples 1–9
old_human = pd.read_csv("results/human_llm_agreement.csv")

# Keep only the existing human-score columns
old_human = old_human[
    [
        "groundedness_human",
        "helpfulness_human",
        "professionalism_human",
        "overall_quality_human"
    ]
]

new_human = pd.DataFrame(human_scores)

# Combine old 9 + new 14
all_human = pd.concat(
    [old_human, new_human],
    ignore_index=True
)

# Make sure the number of human scores matches the LLM results
if len(all_human) != len(results):
    raise ValueError(
        f"Mismatch: {len(all_human)} human scores "
        f"for {len(results)} LLM results."
    )

combined = pd.concat(
    [
        results.reset_index(drop=True),
        all_human.reset_index(drop=True)
    ],
    axis=1
)

combined.to_csv(
     "results/human_llm_agreement.csv",
    index=False
)

print("\n" + "=" * 60)
print("HUMAN EVALUATION COMPLETE")
print("=" * 60)

print(f"\nTotal agreement examples: {len(combined)}")

print("\nHuman mean scores:")

print(
    f"Groundedness: "
    f"{combined['groundedness_human'].mean():.2f}/5"
)

print(
    f"Helpfulness: "
    f"{combined['helpfulness_human'].mean():.2f}/5"
)

print(
    f"Professionalism: "
    f"{combined['professionalism_human'].mean():.2f}/5"
)

print(
    f"Overall quality: "
    f"{combined['overall_quality_human'].mean():.2f}/5"
)

print("\nSaved:")
print("human_llm_agreement.csv")