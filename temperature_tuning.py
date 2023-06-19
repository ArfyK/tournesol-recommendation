import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from recommendation import CRITERIA, random_greedy, random


#### DATA SET UP ####
df = pd.read_csv(sys.argv[1])

#### TESTS ####
if len(sys.argv) < 3:  # no results file provided
    n_tests = 1

    alpha = 0.5  # exponent of the power function used in the objective function

    n_vid = 10

    temp_list = [0.01, 0.1, 1, 10, 100]

    results = []

    for k in range(n_tests):
        print("Test " + str(k + 1) + " out of " + str(n_tests))

        maxs = df[CRITERIA].max()

        for t in temp_list:
            m = (df[CRITERIA] - df[CRITERIA].min()).mean().mean()

            rg = random_greedy(df, n_vid=n_vid, alpha=alpha, l = 1/10*m , T=t)
            maxs_rg = (
                df.loc[df["uid"].isin(rg["uids"]), CRITERIA]
                .max()
                .divide(maxs)
                .to_list()
            )
            results.append(
                [k + 1, "rg_l=1/10*m_" + str(t), rg["uids"], rg["obj"]] + maxs_rg
            )

        r_50 = random(df, n_vid=n_vid, alpha=alpha, pre_selection=True, quantile=0.5)
        maxs_50 = (
            df.loc[df["uid"].isin(r_50["uids"]), CRITERIA].max().divide(maxs).to_list()
        )
        results.append(
            [
                k + 1,
                "r_50",
                alpha,
                r_50["uids"],
                r_50["obj"],
            ]
            + maxs_50
        )

        r_50 = random(
            df,
            n_vid=n_vid,
            alpha=alpha,
            pre_selection=True,
            quantile=0.5,
            key=aggregated_score,
        )
        maxs_50 = (
            df.loc[df["uid"].isin(r_50["uids"]), CRITERIA].max().divide(maxs).to_list()
        )
        results.append(
            [
                k + 1,
                "r_agg_50",
                alpha,
                r_50["uids"],
                r_50["obj"],
            ]
            + maxs_50
        )

    # Set up a dataframe to hold the results
    columns = ["test", "algorithm", "uids", "objective_value"] + CRITERIA
    results = pd.DataFrame(data=results, columns=columns).set_index("test")

    results.to_csv("temp_tuning_" + "n_test=" + str(n_tests) + ".csv")

#### PLOTS ####

# Load result dataset if necessary
if len(sys.argv) == 3:  # results file provided
    results = pd.read_csv(sys.argv[2])
    # hack to get the uids as a python list instead of a string
    results["uids"] = results["uids"].apply(lambda x: x[2:-2].split("', '"))

X = ["objective_value"] + CRITERIA

# Comparison between objective values and the maximum of each criteria
f, axs = plt.subplots(3, 4, figsize=(13, 7), sharey=True)

for i in range(len(X)):
    sns.boxplot(data=results, x=X[i], y="algorithm", ax=axs[i % 3, i % 4], orient="h")
    sns.stripplot(
        data=results,
        x=X[i],
        y="algorithm",
        ax=axs[i % 3, i % 4],
    )

    axs[i % 3, i % 4].xaxis.grid(True)
    axs[i % 3, i % 4].set_ylabel("")


# Number of different channel featured in the selection:
def unique_channel(
    df, uids
):  # used to count how many channel are featured in each selection
    return df.loc[df["uid"].isin(uids), "uploader"].unique().shape[0]


n_vid_per_recommendation = len(results.loc[1, "uids"])
results["n_channel"] = results["uids"].apply(lambda x: unique_channel(df, x))

sns.boxplot(
    data=results,
    x="n_channel",
    y="algorithm",
    orient="h",
    ax=axs[2, 3],
)
sns.stripplot(
    data=results,
    x="n_channel",
    y="algorithm",
    ax=axs[2, 3],
)

axs[2, 3].xaxis.grid(True)
axs[2, 3].set_ylabel("")

plt.subplots_adjust(
    left=0.12, bottom=0.074, right=0.998, top=0.976, wspace=0.062, hspace=0.264
)

plt.savefig(fname="temperature_criteria_comparison.png")

# Coverage of the top K of tournesol scores
K = 200

algo_list = list(results["algorithm"].unique())

coverage = pd.DataFrame(columns=["uid", "rank"] + algo_list)
coverage["uid"] = list(
    df.sort_values(by="largely_recommended", ascending=False)["uid"].iloc[0:K]
)

coverage["rank"] = list(range(1, K + 1))
coverage[algo_list] = np.zeros((K, len(algo_list)))


def compute_coverage(coverage_df, result_series):
    coverage_df.loc[
        coverage_df["uid"].isin(result_series["uids"]), result_series["algorithm"]
    ] = (
        coverage_df.loc[
            coverage_df["uid"].isin(result_series["uids"]), result_series["algorithm"]
        ]
        + 1
    )


results.apply(lambda x: compute_coverage(coverage, x), axis=1)
coverage[algo_list] = coverage[algo_list] * len(algo_list) / results.size

f, axs = plt.subplots(3, 2, figsize=(13, 7), sharex=True, sharey=True)
for i in range(len(algo_list)):
    sns.barplot(data=coverage, x="rank", y=algo_list[i], ax=axs[i % 3, i % 2])
    axs[i % 3, i % 2].axhline(y=results["test"].max() / 200)
    axs[i % 3, i % 2].set_title(algo_list[i])
    axs[i % 3, i % 2].yaxis.set_label_text("count")

f.suptitle("Coverage of the top " + str(K) + " tournesol scores")
plt.subplots_adjust(
    left=0.055, bottom=0.076, right=0.994, top=0.907, wspace=0.072, hspace=0.238
)

plt.savefig(fname="temperature_coverage_tournesolscores.png")

# Coverage of top K ranking the videos with a function resembling the objective function


def aggregated_score(video_scores_series, l=1 / 10, alpha=1 / 2):
    tournesol_score = video_scores_series[CRITERIA[0]]
    criteria_aggregation = (
        l * video_scores_series[CRITERIA[1:]].apply(lambda x: x**alpha).sum()
    )
    return tournesol_score + criteria_aggregation


df["aggregated_score"] = df.apply(aggregated_score, axis="columns")

coverage = pd.DataFrame(columns=["uid", "rank"] + algo_list)
coverage["uid"] = list(
    df.sort_values(by="aggregated_score", ascending=False)["uid"].iloc[0:K]
)

coverage["rank"] = list(range(1, K + 1))
coverage[algo_list] = np.zeros((K, len(algo_list)))

results.apply(lambda x: compute_coverage(coverage, x), axis=1)
coverage[algo_list] = coverage[algo_list] * len(algo_list) / results.size

f, axs = plt.subplots(3, 2, figsize=(13, 7), sharex=True, sharey=True)
for i in range(len(algo_list)):
    sns.barplot(data=coverage, x="rank", y=algo_list[i], ax=axs[i % 3, i % 2])
    axs[i % 3, i % 2].axhline(y=results["test"].max() / 200)
    axs[i % 3, i % 2].set_title(algo_list[i])
    axs[i % 3, i % 2].yaxis.set_label_text("count")

f.suptitle("Coverage of the top " + str(K) + " aggregated scores")
plt.subplots_adjust(
    left=0.055, bottom=0.076, right=0.994, top=0.907, wspace=0.072, hspace=0.238
)

plt.savefig(fname="temperature_coverage_aggregatedscore.png")
