import pandas as pd
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

# Load CSVs

participants = pd.read_csv("./data/participants.csv")
responses = pd.read_csv("./data/responses.csv")

# Filter to false post IDs

post_ids = []
post_ids += ["1kgcctc", "1fnz3te", "a36e6n"]
post_ids += ["ml5r4y", "ktsyx5", "1ig8qte"]
post_ids += ["knhbhd", "18eha3t", "1cvea44"]
responses = responses[responses["post_id"].isin(post_ids)]

# Normalize ratings

responses["normalized_rating"] = responses.groupby(
    "post_id")["rating"].transform(lambda x: (x - x.mean()) / x.std())

# Create groups

groups = [
    group["normalized_rating"]
    for _, group in responses.groupby("fact_check_source")
]

# Overall

print("\n")
print("Overall")
print(f"Participants: {len(participants)}, responses: {len(responses)}")
print("\n% Credence Reduction:")
none_mean = responses[responses["fact_check_source"] ==
                        "none"]["rating"].mean()
for source in responses["fact_check_source"].unique():
    print(f"{source}: {(100*(responses[responses["fact_check_source"] == source]["rating"].mean() - none_mean) / none_mean):.2f}%")

h_stat, p_val = kruskal(*groups)
print(f"\nKruskal-Wallis Test: p = {p_val:.4f}")

if (p_val < 0.05):
    dunn_results = posthoc_dunn(responses,
                            val_col="normalized_rating",
                            group_col="fact_check_source",
                            p_adjust="hommel")
    print("\nDunn's Post-HOC Test (Hommel Corrected):")
    print(dunn_results)

# Filter by demographic

for category in [
        "political_affiliation", "student_status", "employment_status"
]:
    print("\n")
    print(category)

    labels = participants[category].dropna().unique()

    for label in labels:
        participants_subset = participants[participants[category] == label]
        responses_subset = responses[responses["prolific_pid"].isin(
            participants_subset["prolific_pid"])]

        print("\n")
        print(label)
        print(f"Participants: {len(participants_subset)}, responses: {len(responses_subset)}")
        print("\n% Credence Reduction:")
        none_mean = responses_subset[responses_subset["fact_check_source"] ==
                              "none"]["rating"].mean()
        for source in responses_subset["fact_check_source"].unique():
            print(f"{source}: {(100*(responses_subset[responses_subset["fact_check_source"] == source]["rating"].mean() - none_mean) / none_mean):.2f}%")

        h_stat, p_val = kruskal(*groups)
        print(f"\nKruskal-Wallis Test: p = {p_val:.4f}")

        if (p_val < 0.05):
            dunn_results = posthoc_dunn(responses_subset,
                                   val_col="normalized_rating",
                                   group_col="fact_check_source",
                                   p_adjust="hommel")
            print("\nDunn's Post-HOC Test (Hommel Corrected):")
            print(dunn_results.round(4))

