import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import datetime
from recommendation import (
    CRITERIA,
    deterministic_greedy,
    random,
    api_get_tournesol_scores,
    aggregated_score,
)

#### DATA SET UP ####
if len(sys.argv) == 1:  # no data file provided
    dataFrame = api_get_tournesol_scores()
    dataFrame.to_csv(
        "tournesol_scores_" + str(datetime.datetime.today()).split(" ")[0] + ".csv"
    )
else:
    dataFrame = pd.read_csv(sys.argv[1])  # data file provided

#### TESTS ####
if len(sys.argv) < 3:  # no results file provided
    n_tests = 100

    size = 1000

    alpha = 0.5  # exponent of the power function used in the objective function

    n_vid = 10

    results = []

    for k in range(n_tests):
        print("Test " + str(k + 1) + " out of " + str(n_tests))

        df = dataFrame.loc[
            np.random.choice(a=dataFrame.index, size=size, replace=False)
        ]
        maxs = (df[CRITERIA] - df[CRITERIA].min()).max()

        dg = deterministic_greedy(df, n_vid=n_vid, alpha=alpha, l=1 / 10)
        maxs_dg = (
            df.loc[df["uid"].isin(dg["uids"]), CRITERIA].max().divide(maxs).to_list()
        )
        results.append([k + 1, "dg_l=1/10", alpha, dg["uids"], dg["obj"]] + maxs_dg)

        m = (df[CRITERIA] - df[CRITERIA].min()).mean().mean()
        dg = deterministic_greedy(
            df, n_vid=n_vid, alpha=alpha, l=1 / 10 * m
        )  # multiplying by m ensures the two terms in the objective function are "homogeneous"
        maxs_dg = (
            df.loc[df["uid"].isin(dg["uids"]), CRITERIA].max().divide(maxs).to_list()
        )
        results.append([k + 1, "dg_l=1/10*m", alpha, dg["uids"], dg["obj"]] + maxs_dg)

        r = random(df, n_vid=n_vid, alpha=alpha)
        maxs_r = (
            df.loc[df["uid"].isin(r["uids"]), CRITERIA].max().divide(maxs).to_list()
        )
        results.append([k + 1, "random", alpha, r["uids"], r["obj"]] + maxs_r)

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

        r_75 = random(df, n_vid=n_vid, alpha=alpha, pre_selection=True, quantile=0.75)
        maxs_75 = (
            df.loc[df["uid"].isin(r_75["uids"]), CRITERIA].max().divide(maxs).to_list()
        )
        results.append(
            [
                k + 1,
                "r_75",
                alpha,
                r_75["uids"],
                r_75["obj"],
            ]
            + maxs_75
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

        r_75 = random(
            df,
            n_vid=n_vid,
            alpha=alpha,
            pre_selection=True,
            quantile=0.75,
            key=aggregated_score,
        )
        maxs_75 = (
            df.loc[df["uid"].isin(r_75["uids"]), CRITERIA].max().divide(maxs).to_list()
        )
        results.append(
            [
                k + 1,
                "r_agg_75",
                alpha,
                r_75["uids"],
                r_75["obj"],
            ]
            + maxs_75
        )

    # Set up a dataframe to hold the results
    columns = ["test", "algorithm", "alpha", "uids", "objective_value"] + CRITERIA
    results = pd.DataFrame(data=results, columns=columns).set_index("test")

    results.to_csv("dg_r_" + "n_test=" + str(n_tests) + "_size=" + str(size) + ".csv")

#### PLOTS ####


def unique_channel(
    df, uids
):  # used to count how many channel are featured in each selection
    return df.loc[df["uid"].isin(uids), "uploader"].unique().shape[0]


if len(sys.argv) == 3:  # results file provided
    results = pd.read_csv(sys.argv[2]).set_index("test")
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
results["n_channel"] = results["uids"].apply(lambda x: unique_channel(dataFrame, x))

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
    left=0.08, bottom=0.074, right=0.998, top=0.976, wspace=0.062, hspace=0.264
)

f.savefig(fname="dg_r_comparison.png")
