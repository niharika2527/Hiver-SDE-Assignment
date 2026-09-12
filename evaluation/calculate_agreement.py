import pandas as pd
from scipy.stats import spearmanr


# Load human + LLM scores
df = pd.read_csv("results/human_llm_agreement.csv")


metrics = [
    ("groundedness", "groundedness_human"),
    ("helpfulness", "helpfulness_human"),
    ("professionalism", "professionalism_human"),
    ("overall_quality", "overall_quality_human")
]


print("=" * 60)
print("HUMAN vs LLM JUDGE AGREEMENT")
print("=" * 60)

print(f"\nExamples compared: {len(df)}")


for llm_col, human_col in metrics:

    llm_scores = df[llm_col]
    human_scores = df[human_col]

    correlation, p_value = spearmanr(
        human_scores,
        llm_scores
    )

    exact_agreement = (
        human_scores == llm_scores
    ).mean()

    within_one = (
        (human_scores - llm_scores).abs() <= 1
    ).mean()

    print("\n" + "-" * 60)

    print(
        f"{llm_col.upper()}"
    )

    print(
        f"Spearman correlation: {correlation:.3f}"
    )

    print(
        f"Exact agreement: {exact_agreement * 100:.1f}%"
    )

    print(
        f"Agreement within ±1 point: {within_one * 100:.1f}%"
    )


print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print("""
Spearman correlation measures whether the LLM judge
ranks replies similarly to the human evaluator.

Exact agreement measures how often the two judges
gave exactly the same score.

Within ±1 measures how often their scores differed
by no more than one point.
""")