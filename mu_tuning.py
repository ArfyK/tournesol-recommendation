import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import datetime
from recommendation import (
    CRITERIA,
    deterministic_greedy,
    random,
    rank_by_tournesol_score,
    get_age_in_days
)

ref_date = datetime.datetime(2023, 5, 10, 0, 0)

#### DATA SET UP ####
dataFrame = pd.read_csv(sys.argv[1])

#### TESTS ####
if len(sys.argv) < 3:  # no results file provided
    n_tests = 1

    sample_size = 90

    alpha = 0.5  # exponent of the power function used in the objective function

    n_vid = 12

    q = 0.15

    quantile = 0.75

    mu_list = [0.5, 5, 50]

    results = []

    for k in range(n_tests):
        print("Test " + str(k + 1) + " out of " + str(n_tests))

        df = dataFrame.loc[
            np.random.choice(a=dataFrame.index, size=sample_size, replace=False)
        ]
        maxs = (df[CRITERIA] - df[CRITERIA].min()).max()

        for mu in mu_list:
            dg = deterministic_greedy(
                df, ref_date, n_vid=n_vid, alpha=alpha, l=1 / 10, mu=mu
            )
            maxs_dg = (
                df.loc[df["uid"].isin(dg["uids"]), CRITERIA]
                .max()
                .divide(maxs)
                .to_list()
            )
            results.append(
                [k + 1, "dg_mu="+str(mu), mu, dg["uids"], dg["obj"]] + maxs_dg
            )

            r_75 = random(df, ref_date, n_vid=n_vid, alpha=alpha, mu=mu, pre_selection=True, quantile=0.75)
            maxs_75 = (
                df.loc[df["uid"].isin(r_75["uids"]), CRITERIA].max().divide(maxs).to_list()
            )
            results.append(
                [
                    k + 1,
                    "r_75",
                    mu,
                    r_75["uids"],
                    r_75["obj"],
                ]
                + maxs_75
            )

    # Set up a dataframe to hold the results
    columns = ["test", "algorithm", "mu", "uids", "objective_value"] + CRITERIA
    results = pd.DataFrame(data=results, columns=columns).set_index("test")

    results.to_csv(
        "mu_tuning_"
        + "n_test="
        + str(n_tests)
        + "_sample_size="
        + str(sample_size)
        + "_mus="
        + str(mu_list)[1:-1].replace(", ", "_")
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

df['age_in_days'] = df.apply(lambda x: get_age_in_days(x, ref_date), axis="columns")

results['p1']

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
