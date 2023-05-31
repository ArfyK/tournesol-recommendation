import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import requests
import sys
import datetime

CRITERIA = [
    "largely_recommended",
    "reliability",
    "importance",
    "engaging",
    "pedagogy",
    "layman_friendly",
    "entertaining_relaxing",
    "better_habits",
    "diversity_inclusion",
    "backfire_risk",
]


# Complete Objective function
def F(partial_sums, new):
    l = 1 / 10
    alpha = 0.5
    # Relevance term
    R = partial_sums["largely_recommended"] + new["largely_recommended"]
    # Diversity term
    C = (
        (partial_sums[CRITERIA[1:]] + new[CRITERIA[1:]])
        .apply(lambda x: x ** (alpha))
        .sum()
    )
    return R + l * C


# Incomplete objective function, only based on the tournesol score
def R(partial_sum, new):
    return partial_sum + new["largely_recommended"]


def deterministic_greedy(data, n_vid=10, q=0.15, l=1 / 10, alpha=0.5):
    df = data.copy()  # copy the dataframe to avoid modifying the original

    # Normalizes the dataframe
    contains_na = df[CRITERIA].isna().apply(any, axis="columns")
    df[CRITERIA] = df[CRITERIA] - df[CRITERIA].min()
    df = df.fillna(0)

    # Determine how many videos with and without na will be selected
    n_incomplete = int(q * n_vid)
    incomplete_available = contains_na.sum()

    if incomplete_available < n_incomplete:
        n_incomplete = incomplete_available

    n_complete = n_vid - n_incomplete

    # Selection of videos scored according to all criteria
    S1 = []  # uids of the selected videos
    partial_sums = pd.Series(data=[0] * len(CRITERIA), index=CRITERIA)

    for i in range(n_complete):
        # Compute the objective function
        obj = df.loc[~df["uid"].isin(S1)].apply(
            lambda x: F(partial_sums, x), axis="columns"
        )

        # Update S1 and partial sums
        new = df.loc[obj.idxmax(), "uid"]
        S1.append(new)
        partial_sums = (partial_sums + df.loc[df["uid"] == new, CRITERIA]).iloc[
            0
        ]  # hack to keep a series

    objective1 = obj.max()

    # Selection of videos only using the tournesol score
    S2 = []  # indexes of the selected videos
    partial_sum = 0
    df_incomplete = df.loc[
        contains_na, ["uid", "largely_recommended"]
    ]  # we remove videos with no missing score
    df_incomplete = df_incomplete[
        ~df_incomplete["uid"].isin(S1)
    ]  # we remove videos that were already selected

    for i in range(n_incomplete):
        # Compute the objective function
        obj = df_incomplete.loc[~df_incomplete["uid"].isin(S2)].apply(
            lambda x: R(partial_sum, x), axis="columns"
        )

        # Update S2 and partial sums
        new = df_incomplete.loc[obj.idxmax(), "uid"]
        S2.append(new)
        partial_sum = (
            partial_sum
            + df_incomplete.loc[df_incomplete["uid"] == new, "largely_recommended"]
        ).iloc[
            0
        ]  # hack to get a series

    objective2 = obj.max()

    return {"uids": S1 + S2, "obj": objective1 + objective2}


def random_greedy(data, n_vid=10, q=0.15, l=1 / 10, alpha=0.5):
    df = data.copy()  # copy the dataframe to avoid modifying the original

    # Normalizes the dataframe
    contains_na = df[CRITERIA].isna().apply(any, axis="columns")

    df[CRITERIA] = df[CRITERIA] - df[CRITERIA].min()
    df = df.fillna(0)

    # Determine how many videos with and without na will be selected
    n_incomplete = int(q * n_vid)
    incomplete_available = contains_na.sum()

    if incomplete_available < n_incomplete:
        n_incomplete = incomplete_available

    n_complete = n_vid - n_incomplete

    # Selection of videos scored according to all criteria
    S1 = []  # uids of the selected videos
    partial_sums = pd.Series(data=[0] * len(CRITERIA), index=CRITERIA)

    for i in range(n_complete):
        # Compute the objective function
        obj = df.loc[~df["uid"].isin(S1)].apply(
            lambda x: F(partial_sums, x), axis="columns"
        )
        # Compute the probability distribution
        p = obj.apply(lambda x: x**5)
        norm = p.sum()
        p = p.apply(lambda x: x / norm)

        # sample a new element from p
        new_idx = np.random.choice(a=obj.index, size=1, replace=False, p=p)[0]
        new = df.loc[new_idx, "uid"]

        # Update S1 and partial sums
        S1.append(new)
        partial_sums = (partial_sums + df.loc[df["uid"] == new, CRITERIA]).iloc[
            0
        ]  # hack to keep a series

    objective1 = obj.max()

    # Selection of videos only using the tournesol score
    S2 = []  # indexes of the selected videos
    partial_sum = 0
    df_incomplete = df.loc[
        contains_na, ["uid", "largely_recommended"]
    ]  # we remove videos with no missing score
    df_incomplete = df_incomplete[
        ~df_incomplete["uid"].isin(S1)
    ]  # we remove videos that were already selected

    for i in range(n_incomplete):
        # Compute the objective function
        obj = df_incomplete.loc[~df_incomplete["uid"].isin(S2)].apply(
            lambda x: R(partial_sum, x), axis="columns"
        )

        # Compute the probability distribution
        p = obj.apply(lambda x: np.exp(x / 100))  # 100 is a normalization
        # ensuring numbers don't get too high
        norm = p.sum()
        p = p.apply(lambda x: x / norm)

        # sample a new element from p
        new_idx = np.random.choice(a=obj.index, size=1, replace=False, p=p)[0]
        new = df_incomplete.loc[new_idx, "uid"]

        # Update S2 and partial sums
        S2.append(new)
        partial_sum = (
            partial_sum
            + df_incomplete.loc[df_incomplete["uid"] == new, "largely_recommended"]
        ).iloc[
            0
        ]  # hack to get a series

    objective2 = obj.max()

    return {"uids": S1 + S2, "obj": objective1 + objective2}


def random(data, n_vid=10, pre_selection=False, threshold=0, alpha=0.5, l=1 / 10):
    df = data.copy()  # copy the dataframe to avoid modifying the original

    if pre_selection:
        df = df.loc[df["largely_recommended"] >= threshold]

    # Normalizes the dataframe
    df[CRITERIA] = df[CRITERIA] - df[CRITERIA].min()
    df = df.fillna(0)

    # uniformly sample a selection
    selection = df.loc[np.random.choice(df.index, n_vid)]
    S = selection["uid"].to_list()

    # Compute the objective value
    sums = selection[CRITERIA].apply(sum, axis=0)
    obj = (
        sums["largely_recommended"]
        + l * sums[CRITERIA[1:]].apply(lambda x: x**alpha).sum()
    )

    return {"uids": S, "obj": obj}


# We re-use the code from tournesol/data-visualization to set up the data
def get_score(row, crit):
    for item in row["criteria_scores"]:
        if item["criteria"] == crit:
            return item["score"]


def api_get_tournesol_scores():
    """Get a dataframe with all videos from tournesol.."""

    response = requests.get(
        f"https://api.tournesol.app/video/?limit=99999&unsafe=true"
    ).json()

    df = pd.DataFrame.from_dict(response["results"])

    for crit in CRITERIA:
        df[crit] = df.apply(lambda x: get_score(x, crit), axis=1)

    df.drop(columns=["criteria_scores"], inplace=True)

    return df


#### TESTS ####
if __name__ == "__main__":
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

            dg = deterministic_greedy(df, n_vid=n_vid, alpha=alpha)
            maxs = df.loc[df["uid"].isin(dg["uids"]), CRITERIA].max().to_list()
            results.append(
                [k + 1, "deterministic_greedy", alpha, dg["uids"], dg["obj"]] + maxs
            )

            rg = random_greedy(df, n_vid=n_vid, alpha=alpha)
            maxs = df.loc[df["uid"].isin(rg["uids"]), CRITERIA].max().to_list()
            results.append(
                [k + 1, "random_greedy", alpha, rg["uids"], rg["obj"]] + maxs
            )

            r = random(df, n_vid=n_vid, alpha=alpha)
            maxs = df.loc[df["uid"].isin(r["uids"]), CRITERIA].max().to_list()
            results.append([k + 1, "random", alpha, r["uids"], r["obj"]] + maxs)

            r_thresh_0 = random(
                df, n_vid=n_vid, alpha=alpha, pre_selection=True, threshold=0
            )
            maxs = df.loc[df["uid"].isin(r_thresh_0["uids"]), CRITERIA].max().to_list()
            results.append(
                [
                    k + 1,
                    "random_threshold_0",
                    alpha,
                    r_thresh_0["uids"],
                    r_thresh_0["obj"],
                ]
                + maxs
            )

            r_thresh_20 = random(
                df, n_vid=n_vid, alpha=alpha, pre_selection=True, threshold=20
            )
            maxs = df.loc[df["uid"].isin(r_thresh_20["uids"]), CRITERIA].max().to_list()
            results.append(
                [
                    k + 1,
                    "random_threshold_20",
                    alpha,
                    r_thresh_20["uids"],
                    r_thresh_20["obj"],
                ]
                + maxs
            )

        # Set up a dataframe to hold the results
        columns = ["test", "algorithm", "alpha", "uids", "objective_value"] + CRITERIA
        results = pd.DataFrame(data=results, columns=columns).set_index("test")

        results.to_csv(
            "tests_" + "n_test=" + str(n_tests) + "_size=" + str(size) + ".csv"
        )

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

    compare_with = ["random", "random_threshold_0", "random_threshold_20"]

    ### Deterministic greedy versus random ###
    # Comparison between objective values and the maximum of each criteria
    for alg in compare_with:
        f, axs = plt.subplots(3, 4)
        for i in range(len(X)):
            sns.histplot(
                data=results.loc[results["algorithm"] == "deterministic_greedy", X]
                - results.loc[results["algorithm"] == alg, X],
                x=X[i],
                element="step",
                ax=axs[i % 3, i % 4],
                legend=(i == 0),
            )

        # Number of different channel featured in the selection:
        n_vid_per_recommendation = len(results.loc[1, "uids"])
        results["n_channel"] = results["uids"].apply(
            lambda x: unique_channel(dataFrame, x)
        )
        sns.histplot(
            data=results[results["algorithm"].isin(["deterministic_greedy", alg])],
            x="n_channel",
            hue="algorithm",
            element="step",
            ax=axs[2, 3],
        )

        plt.show()

    ### Random greedy versus random ###
    # Comparison between objective values and the maximum of each criteria
    for alg in compare_with:
        f, axs = plt.subplots(3, 4)
        for i in range(len(X)):
            sns.histplot(
                data=results.loc[results["algorithm"] == "random_greedy", X]
                - results.loc[results["algorithm"] == alg, X],
                x=X[i],
                element="step",
                ax=axs[i % 3, i % 4],
                legend=(i == 0),
            )

        # Number of different channel featured in the selection:
        n_vid_per_recommendation = len(results.loc[1, "uids"])
        results["n_channel"] = results["uids"].apply(
            lambda x: unique_channel(dataFrame, x)
        )
        sns.histplot(
            data=results[results["algorithm"].isin(["random_greedy", alg])],
            x="n_channel",
            hue="algorithm",
            element="step",
            ax=axs[2, 3],
        )

        plt.show()
