import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from recommendation import CRITERIA, random_greedy, random, aggregated_score


#### DATA SET UP ####
df = pd.read_csv(sys.argv[1])

#### TESTS ####
if len(sys.argv) < 3:  # no results file provided
    n_tests = 100

    alpha = 0.5  # exponent of the power function used in the objective function

    n_vid = 12

    temperature_list = [0.001, 0.01, 0.1, 1, 10]

    relative_upper_bound_list = [1000, 3250, 5500, 7750, 10000]

    results = []

    for k in range(n_tests):
        print("Test " + str(k + 1) + " out of " + str(n_tests))

        maxs = df[CRITERIA].max()

        print("     Running Random Greedy: ")
        for t in temperature_list:
            print("          Temperature " + str(t) + " from " + str(temperature_list))
            for relative_upper_bound in relative_upper_bound_list:
                print(
                    "               Relative upper bound from "
                    + str(relative_upper_bound)
                    + " from "
                    + str(relative_upper_bound_list)
                )

                clipping_parameter = 1 / 2 * np.log(relative_upper_bound)

                rg = random_greedy(
                    df,
                    n_vid=n_vid,
                    alpha=alpha,
                    l=1 / 10,
                    T=t,
                    clipping_parameter=clipping_parameter,
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
                        "random_greedy T=" + str(t) + " c=" + str(relative_upper_bound),
                        t,
                        relative_upper_bound,
                        rg["uids"],
                        rg["obj"],
                    ]
                    + maxs_rg
                )  # We keep the relative upper bound instead of the clipping parameter for interpretability

            print("     Running random")
            r_75 = random(
                df, n_vid=n_vid, alpha=alpha, pre_selection=True, quantile=0.75
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
        "temperature",
        "relative_upper_bound",
        "uids",
        "objective_value",
    ] + CRITERIA
    results = pd.DataFrame(data=results, columns=columns).set_index("test")

    results.to_csv(
        "temp_tuning_"
        + "n_test="
        + str(n_tests)
        + "_t="
        + str(temperature_list)[1:-1].replace(", ", "_")
        + "_c="
        + str(relative_upper_bound_list)[1:-1].replace(", ", "_")
        + ".csv"
    )

#### PLOTS ####

# Load result dataset if necessary
if len(sys.argv) == 3:  # results file provided
    results = pd.read_csv(sys.argv[2])
    # hack to get the uids as a python list instead of a string
    results["uids"] = results["uids"].apply(lambda x: x[2:-2].split("', '"))

    n_vid_per_bundle = len(results.loc[1, "uids"])

    n_tests = results["test"].max()

    temperature_list = results.loc[
        ~results["temperature"].isna(), "temperature"
    ].unique()
    relative_upper_bound_list = results.loc[
        ~results["relative_upper_bound"].isna(), "relative_upper_bound"
    ].unique()

X = ["objective_value"] + CRITERIA

# Comparison between objective values and the maximum of each criteria


# Number of different channel featured in the selection:
def unique_channel(
    df, uids
):  # used to count how many channel are featured in each selection
    return df.loc[df["uid"].isin(uids), "uploader"].unique().shape[0]


# 1 plot per temperature value for visibility
for t in temperature_list:
    f, axs = plt.subplots(3, 4, figsize=(13, 7), sharey=True)
    results_temperature_t = results.loc[
        (results["temperature"] == t) | (results["temperature"].isna())
    ]

    # Plot the distributions of the objective value and the maximum of each criteria
    for i in range(len(X)):
        sns.boxplot(
            data=results_temperature_t,
            x=X[i],
            y="algorithm",
            ax=axs[i % 3, i % 4],
            orient="h",
        )
        sns.stripplot(
            data=results_temperature_t,
            x=X[i],
            y="algorithm",
            ax=axs[i % 3, i % 4],
        )

        axs[i % 3, i % 4].set_ylabel("")
        if i == 4:
            axs[i % 3, i % 4].yaxis.set_label_text("relative upper bound")

        axs[i % 3, i % 4].xaxis.grid(True)

    # Plot the number of different channel in each bundle
    results_temperature_t.insert(
        0,
        "n_channel",
        results_temperature_t["uids"].apply(lambda x: unique_channel(df, x)),
    )

    sns.boxplot(
        data=results_temperature_t,
        x="n_channel",
        y="algorithm",
        orient="h",
        ax=axs[2, 3],
    )
    sns.stripplot(
        data=results_temperature_t,
        x="n_channel",
        y="algorithm",
        ax=axs[2, 3],
    )

    axs[2, 3].xaxis.grid(True)
    axs[2, 3].set_ylabel("")

    f.suptitle(
        "Objective value, Maximum of each criteria and number of channel per bundle for T ="
        + str(t)
    )
    plt.subplots_adjust(
        left=0.043, bottom=0.074, right=0.995, top=0.94, wspace=0.062, hspace=0.264
    )

    plt.savefig(
        fname="temperature_criteria_comparison_t="
        + str(t)
        + "_c="
        + str(relative_upper_bound_list)[1:-1].replace(" ", "_")
        + "_n_tests="
        + str(n_tests)
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
    + "_t="
    + str(temperature_list)[1:-1].replace(" ", "_")
    + "_c="
    + str(relative_upper_bound_list)[1:-1].replace(" ", "_")
    + "n_tests="
    + str(n_tests)
    + ".csv"
)

f, axs = plt.subplots(
    len(temperature_list),
    len(relative_upper_bound_list),
    figsize=(13, 7),
    sharex=True,
    sharey=True,
)
for i in range(len(temperature_list)):
    for j in range(len(relative_upper_bound_list)):
        sns.barplot(
            data=selection_frequencies,
            x="rank",
            y=algo_list[i * len(temperature_list) + j],
            ax=axs[i, j],
        )
        axs[i, j].set_title(
            "T = "
            + str(temperature_list[i])
            + " C = "
            + str(relative_upper_bound_list[j])
        )
        if (i == int(len(temperature_list) / 2)) and (j == 0):
            axs[i, j].yaxis.set_label_text("frequency")
        else:
            axs[i, j].yaxis.set_label_text("")
        if (i == len(temperature_list) - 1) and (
            j == int(len(relative_upper_bound_list) / 2)
        ):
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
    fname="temperature_selection_frequencies_t="
    + str(temperature_list)[1:-1].replace(" ", "_")
    + "_c="
    + str(relative_upper_bound_list)[1:-1].replace(" ", "_")
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
    results[["top_5%", "temperature", "relative_upper_bound"]],
    row="relative_upper_bound",
    col="temperature",
)
g.map_dataframe(sns.boxplot, x="top_5%")
g.set_titles(col_template="T = {col_name}", row_template="c = {row_name}")
g.fig.suptitle("Distribution of number of videos from the top 5%")
g.fig.subplots_adjust(
    left=0.013, bottom=0.038, right=0.99, top=0.905, wspace=0.072, hspace=0.536
)

g.savefig(
    fname="video_from_top_5_distribution"
    + "_t="
    + str(temperature_list)[1:-1].replace(" ", "_")
    + "_c="
    + str(relative_upper_bound_list)[1:-1].replace(" ", "_")
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
    results[["bottom_50%", "temperature", "relative_upper_bound"]],
    row="relative_upper_bound",
    col="temperature",
)
g.map_dataframe(sns.boxplot, x="bottom_50%")
g.set_titles(col_template="T = {col_name}", row_template="c = {row_name}")
g.fig.suptitle("Distribution of number of videos from the bottom 50%")
g.fig.subplots_adjust(
    left=0.013, bottom=0.038, right=0.99, top=0.905, wspace=0.072, hspace=0.536
)

g.savefig(
    fname="video_from_bottom_50_distribution"
    + "_t="
    + str(temperature_list)[1:-1].replace(" ", "_")
    + "_c="
    + str(relative_upper_bound_list)[1:-1].replace(" ", "_")
    + "n_tests="
    + str(n_tests)
    + ".png"
)

for t in temperature_list:
    for c in relative_upper_bound_list:
        print(" T = " + str(t) + ", c = " + str(c) + ": ")
        print("Total number of videos from top 5% : ")
        print(
            str(
                results.loc[
                    (results["temperature"] == t)
                    & (results["relative_upper_bound"] == c),
                    ["top_5%"],
                ].sum()
            )
        )

        print("Total number of videos from bottom 50% : ")
        print(
            str(
                results.loc[
                    (results["temperature"] == t)
                    & (results["relative_upper_bound"] == c),
                    ["bottom_50%"],
                ].sum()
            )
        )
        print("\n\n")
