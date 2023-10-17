import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import datetime
from recommendation import (
    CRITERIA,
    random_greedy,
    random,
    rank_by_tournesol_score,
    get_age_in_days,
)

ref_date = datetime.datetime(2023, 9, 25, 0, 0)

#### DATA SET UP ####
dataFrame = pd.read_csv(sys.argv[1])

#### TESTS ####
if len(sys.argv) < 3:  # no results file provided
    n_tests = 1

    alpha = 0.5  # exponent of the power function used in the objective function

    n_vid = 12

    q = 0

    quantile = 0.75

    temperature = 5.5

    clipping_parameter = 1 / 2 * np.log(1000)

    mu_list = [0.5, 5, 50]

    t_0_list = [0, 15]  # days

    results = []

    for k in range(n_tests):
        print("Test " + str(k + 1) + " out of " + str(n_tests))

        maxs = df[CRITERIA].max()

        print("     Running Random Greedy: ")
        for mu in mu_list:
            print("          Mu = " + str(t) + " from " + str(mu_list))
            for t_0 in t_0_list:
                print("               t_0 = " + str(t_0) + " from " + str(t_0_list))

                rg = random_greedy(
                    df,
                    n_vid=n_vid,
                    alpha=alpha,
                    l=1 / 10,
                    T=temperature,
                    clipping_parameter=clipping_parameter,
                    mu=mu,
                    t_0=t_0,
                )
                maxs_rg = (
                    df.loc[df["uid"].isin(rg["uids"]), CRITERIA]
                    .max()
                    .divide(maxs)
                    .to_list()
                )
                results.append(
                    [
                        k + 1,
                        "random_greedy mu="
                        + str(t)
                        + " t_0="
                        + str(relative_upper_bound),
                        mu,
                        t_0,
                        rg["uids"],
                        rg["obj"],
                    ]
                    + maxs_rg
                )  # We keep the relative upper bound instead of the clipping parameter for interpretability

            print("     Running random")
            r_75 = random(
                df,
                n_vid=n_vid,
                alpha=alpha,
                pre_selection=True,
                quantile=0.75,
                mu=mu,
                t_0=t_0,
            )
            maxs_75 = (
                df.loc[df["uid"].isin(r_75["uids"]), CRITERIA]
                .max()
                .divide(maxs)
                .to_list()
            )
            results.append(
                [
                    k + 1,
                    "r_75",
                    None,
                    None,
                    r_75["uids"],
                    r_75["obj"],
                ]
                + maxs_75
            )

    # Set up a dataframe to hold the results
    columns = [
        "test",
        "algorithm",
        "mu",
        "t_0",
        "uids",
        "objective_value",
    ] + CRITERIA
    results = pd.DataFrame(data=results, columns=columns).set_index("test")

    results.to_csv(
        "mu_tuning_"
        + "n_test="
        + str(n_tests)
        + "_mu="
        + str(mu_list)[1:-1].replace(", ", "_")
        + "_t_0="
        + str(t_0_list)[1:-1].replace(", ", "_")
        + ".csv"
    )

#### PLOTS ####

# Load result dataset if necessary
if len(sys.argv) == 3:  # results file provided
    results = pd.read_csv(sys.argv[2])
    # hack to get the uids as a python list instead of a string
    results["uids"] = results["uids"].apply(lambda x: x[2:-2].split("', '"))

    algo_list = results["algorithm"].unique()
    n_tests = results.shape[0] / len(algo_list)
    mu_list = results["mu"].unique()
    quantile = float(algo_list[0].split("_")[2].split("=").pop())

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

plt.savefig(
    fname="mu_criteria_comparison_q="
    + str(quantile)
    + "_size="
    + str(size_list)[1:-1].replace(", ", "_")
    + "_n_tests="
    + str(n_tests)
    + ".png"
)

# Age in days distribution
# We plot two distributions:
#  - the proportion p1 of videos from the top 4 of the bundle that are more recent than 3 weeks
#  - the proportion p2 of videos from the rest of the bundle that are more recent than 3 weeks
algo_list = list(results["algorithm"].unique())

df["age_in_days"] = df.apply(lambda x: get_age_in_days(x, ref_date), axis="columns")

results["p1"]

f, axs = plt.subplots(3, 2, figsize=(13, 7), sharex=True, sharey=True)
for i in range(len(algo_list)):
    sns.barplot(data=coverage, x="rank", y=algo_list[i], ax=axs[i % 3, i % 2])
    axs[i % 3, i % 2].set_title(algo_list[i])
    axs[i % 3, i % 2].yaxis.set_label_text("count / nbr of tests")

f.suptitle("Coverage of the top " + str(K) + " tournesol scores")
plt.subplots_adjust(
    left=0.055, bottom=0.076, right=0.994, top=0.907, wspace=0.072, hspace=0.238
)

plt.savefig(
    fname="sample_size_coverage_size="
    + str(size_list)[1:-1].replace(", ", "_")
    + "_q="
    + str(quantile)
    + "n_tests="
    + str(n_tests)
    + ".png"
)
