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


ref_date = datetime.datetime(2023, 9, 19, 0, 0)  # one day older than the video database

#### DATA SET UP ####
df = pd.read_csv(sys.argv[1])

#### TESTS ####
if len(sys.argv) < 3:  # no results file provided
    n_tests = 1

    alpha = 0.5  # exponent of the power function used in the objective function

    n_vid = 12

    q = 0

    quantile = 0.75

    temperature = 5.5

    clipping_parameter = 1 / 2 * np.log(1000)


    mu_list = [0.5, 5] #[0.5, 5, 50]

    t_0_list = [0, 15]  # days

    results = []

    for k in range(n_tests):
        print("Test " + str(k + 1) + " out of " + str(n_tests))

        maxs = df[CRITERIA].max()

        print("     Running Random Greedy: ")
        for mu in mu_list:
            print("          Mu = " + str(mu) + " from " + str(mu_list))
            for t_0 in t_0_list:
                print("               t_0 = " + str(t_0) + " from " + str(t_0_list))

                rg = random_greedy(
                    data=df,
                    ref_date=ref_date,
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
                        "random_greedy mu=" + str(mu) + " t_0=" + str(t_0),
                        mu,
                        t_0,
                        rg["uids"],
                        rg["obj"],
                    ]
                    + maxs_rg
                )

            print("     Running random")
            r_75 = random(
                data=df,
                ref_date=ref_date,
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

    n_tests = results["test"].max()
    mu_list = results.loc[results["mu"].notna(), "mu"].unique()
    t_0_list = results.loc[results["t_0"].notna(), "t_0"].unique()

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
    fname="mu_criteria_comparison"
    + "_n_tests="
    + str(n_tests)
    + "_mu="
    + str(mu_list)[1:-1].replace(", ", "_")
    + "_t_0="
    + str(t_0_list)[1:-1].replace(", ", "_")
    + ".png"
)

# Selection frequencies
results = results.dropna()  # removes the results from the uniformly random algorithm
algo_list = list(results["algorithm"].unique())

selection_frequencies = pd.DataFrame(columns=["uid", "rank"] + algo_list)
selection_frequencies["uid"] = list(
    df.sort_values(by="largely_recommended", ascending=False)["uid"]
)

selection_frequencies["rank"] = list(range(1, df.shape[0] + 1))
selection_frequencies[algo_list] = np.zeros((df.shape[0], len(algo_list)))


def compute_frequencies(selection_frequencies_dataFrame, result_series):
    selection_frequencies_dataFrame.loc[
        selection_frequencies_dataFrame["uid"].isin(result_series["uids"]),
        result_series["algorithm"],
    ] = (
        selection_frequencies_dataFrame.loc[
            selection_frequencies_dataFrame["uid"].isin(result_series["uids"]),
            result_series["algorithm"],
        ]
        + 1
    )


results.apply(lambda x: compute_frequencies(selection_frequencies, x), axis=1)
selection_frequencies[algo_list] = selection_frequencies[algo_list] / n_tests

selection_frequencies.to_csv(
    "selection_frequencies"
    + "_mu="
    + str(mu_list)[1:-1].replace(" ", "_")
    + "t_0="
    + str(t_0_list)[1:-1].replace(" ", "_")
    + "n_tests="
    + str(n_tests)
    + ".csv"
)

f, axs = plt.subplots(
    len(mu_list),
    len(t_0_list),
    figsize=(13, 7),
    sharex=True,
    sharey=True,
)
for i in range(len(mu_list)):
    for j in range(len(t_0_list)):
        sns.scatterplot(
            data=selection_frequencies,
            x="rank",
            y=algo_list[i * len(mu_list) + j],
            ax=axs[i, j],
        )
        axs[i, j].set_title("mu = " + str(mu_list[i]) + " t_0 = " + str(t_0_list[j]))
        if (i == int(len(mu_list) / 2)) and (j == 0):
            axs[i, j].yaxis.set_label_text("frequency")
        else:
            axs[i, j].yaxis.set_label_text("")
        if (i == len(mu_list) - 1) and (j == int(len(t_0_list) / 2)):
            axs[i, j].xaxis.set_label_text("rank")
        else:
            axs[i, j].xaxis.set_label_text("")
        axs[i, j].set_xticks([], minor=True)
        axs[i, j].set_xticks(list(range(0, 2000, 500)))
f.suptitle("Selection frequencies of random greedy")
plt.subplots_adjust(
    left=0.04, bottom=0.043, right=0.998, top=0.907, wspace=0.055, hspace=0.34
)

plt.savefig(
    fname="mu_selection_frequencies"
    + "_mu="
    + str(mu_list)[1:-1].replace(" ", "_")
    + "t_0="
    + str(t_0_list)[1:-1].replace(" ", "_")
    + "n_tests="
    + str(n_tests)
    + ".png"
)

# Distributions of the number of videos from the top 5% in the bundle
quantile_95 = df["tournesol_score"].quantile(0.95)


def count_videos_within_threshold(uids_list, dataFrame, quantile, above=True):
    if above:
        # returns how many videos from the uids_list have a tournesol_score above the quantile
        return dataFrame.loc[
            (dataFrame["uid"].isin(uids_list))
            & (dataFrame["tournesol_score"] >= quantile)
        ].shape[0]
    else:
        # returns how many videos from the uids_list have a tournesol_score below the quantile
        return dataFrame.loc[
            (dataFrame["uid"].isin(uids_list))
            & (dataFrame["tournesol_score"] <= quantile)
        ].shape[0]


results["top_5%"] = results["uids"].apply(
    lambda x: count_videos_within_threshold(x, df, quantile_95, above=True)
)

g = sns.FacetGrid(
    results[["top_5%", "mu", "t_0"]],
    row="mu",
    col="t_0",
)
g.map_dataframe(sns.boxplot, x="top_5%")
g.set_titles(col_template="mu = {col_name}", row_template="t_0 = {row_name}")
g.fig.suptitle("Distribution of number of videos from the top 5%")

# Display the total number of videos from top 5% for each algorithm in the subplot title
for i_t0 in range(len(t_0_list)):
    for i_mu in range(len(mu_list)):
        g.axes[i_t0][i_mu].set_title(
            g.axes[i_t0][i_mu].get_title()
            + " | Total: "
            + str(
                int(
                    results.loc[
                        (results["t_0"] == t_0_list[i_t0])
                        & (results["mu"] == mu_list[i_mu]),
                        "top_5%",
                    ].sum()
                )
            )
        )

g.fig.subplots_adjust(
    left=0.013, bottom=0.038, right=0.99, top=0.905, wspace=0.072, hspace=0.536
)

g.savefig(
    fname="video_from_top_5_distribution"
    + "_mu="
    + str(mu_list)[1:-1].replace(" ", "_")
    + "t_0="
    + str(t_0_list)[1:-1].replace(" ", "_")
    + "n_tests="
    + str(n_tests)
    + ".png"
)

# Distributions of the number of videos from the bottom 50% in the bundle
quantile_50 = df["tournesol_score"].quantile(0.5)

results["bottom_50%"] = results["uids"].apply(
    lambda x: count_videos_within_threshold(x, df, quantile_50, above=False)
)

g = sns.FacetGrid(
    results[["bottom_50%", "mu", "t_0"]],
    row="t_0",
    col="mu",
)
g.map_dataframe(sns.boxplot, x="bottom_50%")
g.set_titles(col_template="T = {col_name}", row_template="c = {row_name}")
g.fig.suptitle("Distribution of number of videos from the bottom 50%")

# Display the total number of videos from bottom 50% for each algorithm in the subplot title
for i_t0 in range(len(t_0_list)):
    for i_mu in range(len(mu_list)):
        g.axes[i_t0][i_mu].set_title(
            g.axes[i_t0][i_mu].get_title()
            + " | Total: "
            + str(
                int(
                    results.loc[
                        (results["t_0"] == t_0_list[i_t0])
                        & (results["mu"] == mu_list[i_mu]),
                        "bottom_50%",
                    ].sum()
                )
            )
        )


g.fig.subplots_adjust(
    left=0.013, bottom=0.038, right=0.99, top=0.905, wspace=0.072, hspace=0.536
)

g.savefig(
    fname="video_from_bottom_50_distribution"
    + "_mu="
    + str(mu_list)[1:-1].replace(" ", "_")
    + "t_0="
    + str(t_0_list)[1:-1].replace(" ", "_")
    + "n_tests="
    + str(n_tests)
    + ".png"
)

# Distributions of the number of videos from the top 20 of the month
df["age_in_days"] = df.apply(lambda x: get_age_in_days(x, ref_date), axis="columns")
top_20_of_last_month = (
    df.loc[df["age_in_days"] <= 30]
    .sort_values(by="tournesol_score", ascending=False)
    .iloc[0:20]["uid"]
)


def count_videos_in_subset(uids_list, subset):
    return subset.loc[(subset.isin(uids_list))].shape[0]


results["top_20_of_last_month"] = results["uids"].apply(
    lambda x: count_videos_in_subset(x, top_20_of_last_month)
)

g = sns.FacetGrid(
    results[["top_20_of_last_month", "mu", "t_0"]],
    row="mu",
    col="t_0",
)
g.map_dataframe(sns.boxplot, x="top_20_of_last_month")
g.set_titles(col_template="mu = {col_name}", row_template="t_0 = {row_name}")
g.fig.suptitle("Distribution of number of videos from the top 20 of last month")

# Display the total number of videos from top 20 of last month for each algorithm in the subplot title
for i_t0 in range(len(t_0_list)):
    for i_mu in range(len(mu_list)):
        g.axes[i_t0][i_mu].set_title(
            g.axes[i_t0][i_mu].get_title()
            + " | Total: "
            + str(
                int(
                    results.loc[
                        (results["t_0"] == t_0_list[i_t0])
                        & (results["mu"] == mu_list[i_mu]),
                        "top_20_of_last_month",
                    ].sum()
                )
            )
        )

g.fig.subplots_adjust(
    left=0.013, bottom=0.038, right=0.99, top=0.905, wspace=0.072, hspace=0.536
)

g.savefig(
    fname="video_from_top_20_distribution"
    + "_mu="
    + str(mu_list)[1:-1].replace(" ", "_")
    + "t_0="
    + str(t_0_list)[1:-1].replace(" ", "_")
    + "n_tests="
    + str(n_tests)
    + ".png"
)
