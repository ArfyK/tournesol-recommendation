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
            rg = random_greedy(df, n_vid=n_vid, alpha=alpha, T=t)
            maxs_rg = (
                df.loc[df["uid"].isin(rg["uids"]), CRITERIA]
                .max()
                .divide(maxs)
                .to_list()
            )
            results.append(
                [k + 1, "random_greedy_" + str(t), rg["uids"], rg["obj"]] + maxs_rg
            )

        r_thresh_20 = random(
            df, n_vid=n_vid, alpha=alpha, pre_selection=True, threshold=20
        )
        maxs_thresh_20 = (
            df.loc[df["uid"].isin(r_thresh_20["uids"]), CRITERIA]
            .max()
            .divide(maxs)
            .to_list()
        )
        results.append(
            [
                k + 1,
                "random_threshold_20",
                r_thresh_20["uids"],
                r_thresh_20["obj"],
            ]
            + maxs_thresh_20
        )

    # Set up a dataframe to hold the results
    columns = ["test", "algorithm", "uids", "objective_value"] + CRITERIA
    results = pd.DataFrame(data=results, columns=columns).set_index("test")

    results.to_csv("temp_tuning_" + "n_test=" + str(n_tests) + ".csv")

#### PLOTS ####


def unique_channel(
    df, uids
):  # used to count how many channel are featured in each selection
    return df.loc[df["uid"].isin(uids), "uploader"].unique().shape[0]


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

# Coverage of the top K
K = 200

algo_list = list(results["algorithm"].unique())

coverage = pd.DataFrame(columns=["uid", "rank"] + algo_list)
coverage["uid"] = list(
    df.sort_values(by="largely_recommended", ascending=False)["uid"].iloc[0:200]
)  # we could also rank the videos according to our objective function instead of looking only at the tournesol score

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

plt.subplots_adjust(
    left=0.055, bottom=0.076, right=0.994, top=0.946, wspace=0.072, hspace=0.22
)

plt.savefig(fname="temperature_coverage_comparison.png")
