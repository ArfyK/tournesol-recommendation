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


def get_age_in_days(video_series, ref_date):
    # return 1 if the video is less than a day old
    return min(
        (
            ref_date
            - datetime.datetime.strptime(
                video_series["publication_date"].split("T")[0], "%Y-%m-%d"
            )
        ).days,
        1,
    )  # remove the time part of the datetime with the split because some entries only have the date part.


### Greedy objective functions


# Complete Objective function
def F(partial_sums, new, ref_date, l=1 / 10, alpha=0.5, mu=0.1, t_0=0):
    # Relevance term
    R = partial_sums["largely_recommended"] + new["largely_recommended"]
    # Diversity term
    C = (
        (partial_sums[CRITERIA[1:]] + new[CRITERIA[1:]])
        .apply(lambda x: x ** (alpha))
        .sum()
    )

    # Recency term
    D = (
        partial_sums["age_in_days_inverses"]
        + 1 / (t_0 + get_age_in_days(new, ref_date))
    ) ** alpha

    return R + l * C + mu * D


# Incomplete objective function
def F_incomplete(partial_sums, new, ref_date, alpha=0.5, mu=0.1, t_0=0):
    # Relevance term
    R = partial_sums["largely_recommended"] + new["largely_recommended"]
    # Recency term
    D = (
        partial_sums["age_in_days_inverses"]
        + 1 / (t_0 + get_age_in_days(new, ref_date))
    ) ** alpha

    return R + mu * D


def deterministic_greedy(
    data, ref_date, n_vid=10, q=0.15, l=1 / 10, alpha=0.5, mu=0.1, t_0=0
):
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
    index = CRITERIA + ["age_in_days_inverses"]
    partial_sums = pd.Series(data=[0] * len(index), index=index)

    for i in range(n_complete):
        # Compute the objective function
        obj = df.loc[~df["uid"].isin(S1)].apply(
            lambda x: F(partial_sums, x, ref_date, l, alpha, mu, t_0), axis="columns"
        )

        # Update S1 and partial sums
        new = df.at[obj.idxmax(), "uid"]
        S1.append(new)
        partial_sums[CRITERIA] = (
            partial_sums[CRITERIA] + df.loc[df["uid"] == new, CRITERIA]
        ).iloc[
            0
        ]  # hack to keep a series
        partial_sums["age_in_days_inverses"] = partial_sums[
            "age_in_days_inverses"
        ] + 1 / (t_0 + get_age_in_days(df.loc[df["uid"] == new].iloc[0], ref_date))
        objective1 = obj.max()

    # Selection of videos only using the tournesol score
    S2 = []  # indexes of the selected videos
    index = ["largely_recommended", "age_in_days_inverses"]
    partial_sums = pd.Series(data=[0] * len(index), index=index)
    df_incomplete = df.loc[contains_na]  # Remove videos with no missing score
    df_incomplete = df_incomplete[
        ~df_incomplete["uid"].isin(S1)
    ]  # Remove videos that were already selected

    # if the there's not enough videos remaining in df_incomplete we also use "complete" videos
    if df_incomplete.shape[0] < n_incomplete:
        df_incomplete = df.loc[~df["uid"].isin(S1), ["uid", "largely_recommended"]]

    for i in range(n_incomplete):
        # Compute the objective function
        obj = df_incomplete.loc[~df_incomplete["uid"].isin(S2)].apply(
            lambda x: F_incomplete(partial_sums, x, ref_date, alpha, mu, t_0),
            axis="columns",
        )

        # Update S2 and partial sums
        new = df_incomplete.loc[obj.idxmax(), "uid"]
        S2.append(new)
        partial_sums["largely_recommended"] = (
            partial_sums["largely_recommended"]
            + df_incomplete.loc[df_incomplete["uid"] == new, "largely_recommended"]
        ).iloc[
            0
        ]  # hack to get a series
        partial_sums["age_in_days_inverses"] = partial_sums[
            "age_in_days_inverses"
        ] + 1 / (t_0 + get_age_in_days(df.loc[df["uid"] == new].iloc[0], ref_date))
    objective2 = obj.max()

    return {"uids": S1 + S2, "obj": objective1 + objective2}


# Functions used to pre-select a subset of the dataset prior to sampling
def rank_by_tournesol_score(series, l, alpha):
    return series["largely_recommended"]


def aggregated_score(series, l, alpha):
    # Resembles the F objective function
    return series["largely_recommended"] + l * series[CRITERIA[1:]].sum()


def random_greedy(
    data, ref_date, n_vid=10, l=1 / 10, alpha=0.5, T=1, clipping_parameter=1, mu=0.1, t_0=0
):
    df = data.copy()  # copy the dataframe to avoid modifying the original

    # Normalizes the dataframe
    contains_na = df[CRITERIA].isna().apply(any, axis="columns")

    df[CRITERIA] = df[CRITERIA] - df[CRITERIA].min()
    df = df.fillna(0)

    # Selection of videos scored according to all criteria
    S = []  # uids of the selected videos
    index = CRITERIA + ["age_in_days_inverses"]
    partial_sums = pd.Series(data=[0] * len(index), index=index)

    for i in range(n_vid):
        # Compute the objective function
        objective_function_scores = df.loc[~df["uid"].isin(S)].apply(
            lambda x: F(partial_sums, x, ref_date, l, alpha, mu, t_0), axis="columns"
        )

        # Compute the probability distribution
        objective_function_scores_mean = objective_function_scores.mean()
        p = objective_function_scores.apply(
            lambda x: np.exp(
                np.clip(
                    (x - objective_function_scores_mean) / T,
                    -clipping_parameter,
                    clipping_parameter,
                )
            )
        )
        norm = p.sum()
        p = p.apply(lambda x: x / norm)

        # sample a new element from p
        new_idx = np.random.choice(
            a=objective_function_scores.index, size=1, replace=False, p=p
        )[0]
        new = df.loc[new_idx, "uid"]

        # Update S and partial sums
        S.append(new)
        partial_sums[CRITERIA] = (partial_sums[CRITERIA] + df.loc[df["uid"] == new, CRITERIA]).iloc[
            0
        ]  # hack to keep a series
        partial_sums["age_in_days_inverses"] = partial_sums[
            "age_in_days_inverses"
        ] + 1 / (t_0 + get_age_in_days(df.loc[df["uid"] == new].iloc[0], ref_date))

    objective = objective_function_scores[new_idx]

    return {"uids": S, "obj": objective}


def deterministic_random_sample(
    data,
    ref_date,
    sample_size,
    n_vid=10,
    q=0.15,
    l=1 / 10,
    alpha=0.5,
    mu=0.1,
    quantile=0,
    key=rank_by_tournesol_score,
):
    df = data.copy()

    # Sample a subset
    df["key"] = df.apply(lambda x: key(x, l, alpha), axis="columns")
    df = df.loc[df["key"] >= df["key"].quantile(quantile)]
    sample = df.loc[np.random.choice(df.index, sample_size)]

    sample.reset_index(drop=True, inplace=True)

    return deterministic_greedy(
        sample, ref_date, n_vid=n_vid, alpha=alpha, l=l, mu=mu, q=q
    )


### Uniformly Random sampling


def random(
    data,
    ref_date,
    n_vid=10,
    alpha=0.5,
    l=1 / 10,
    mu=0.1,
    t_0=0,
    pre_selection=False,
    quantile=0,
    key=rank_by_tournesol_score,
):
    df = data.copy()  # copy the dataframe to avoid modifying the original

    if pre_selection:
        df["key"] = df.apply(lambda x: key(x, l, alpha), axis="columns")
        df = df.loc[df["key"] >= df["key"].quantile(quantile)]

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
        + mu
        * selection.apply(
            lambda x: 1 / (t_0 + get_age_in_days(x, ref_date)), axis="columns"
        ).sum()
        ** alpha
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
        n_tests = 1

        size = 100

        alpha = 0.5  # exponent of the power function used in the objective function

        n_vid = 12

        results = []

        for k in range(n_tests):
            print("Test " + str(k + 1) + " out of " + str(n_tests))

            df = dataFrame.loc[
                np.random.choice(a=dataFrame.index, size=size, replace=False)
            ]
            maxs = (df[CRITERIA] - df[CRITERIA].min()).max()

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
                [k + 1, "dg_l=1/10", alpha, mu, dg["uids"], dg["obj"]] + maxs_dg
            )

            dg_random_sample = deterministic_random_sample(
                df,
                ref_date,
                sample_size=90,
                n_vid=n_vid,
                q=0.15,
                l=1 / 10,
                alpha=alpha,
                mu=mu,
                quantile=0.75,
                key=rank_by_tournesol_score,
            )
            maxs_dg_random_sample = (
                df.loc[df["uid"].isin(dg_random_sample["uids"]), CRITERIA]
                .max()
                .divide(maxs)
                .to_list()
            )
            results.append(
                [
                    k + 1,
                    "dg_random_sample_l=1/10_size=50",
                    alpha,
                    mu,
                    dg_random_sample["uids"],
                    dg_random_sample["obj"],
                ]
                + maxs_dg_random_sample
            )

            r_75 = random(
                df,
                ref_date,
                n_vid=n_vid,
                alpha=alpha,
                mu=mu,
                pre_selection=True,
                quantile=0.75,
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
                    alpha,
                    mu,
                    r_75["uids"],
                    r_75["obj"],
                ]
                + maxs_75
            )

        # Set up a dataframe to hold the results
        columns = [
            "test",
            "algorithm",
            "alpha",
            "mu",
            "uids",
            "objective_value",
        ] + CRITERIA
        results = pd.DataFrame(data=results, columns=columns).set_index("test")

        results.to_csv(
            "algo_comparison_"
            + "n_test="
            + str(n_tests)
            + "_size="
            + str(size)
            + ".csv"
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

    # Comparison between objective values and the maximum of each criteria
    f, axs = plt.subplots(3, 4, figsize=(13, 7), sharey=True)

    for i in range(len(X)):
        sns.boxplot(
            data=results, x=X[i], y="algorithm", ax=axs[i % 3, i % 4], orient="h"
        )
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
        left=0.12, bottom=0.074, right=0.998, top=0.976, wspace=0.062, hspace=0.264
    )

    plt.show()
    # f.savefig(fname="algorithms_comparison.png")
