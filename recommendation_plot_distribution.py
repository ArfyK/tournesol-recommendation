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
    return max(
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


# Functions used to pre-select a subset of the dataset prior to sampling
def rank_by_tournesol_score(series, l, alpha):
    return series["largely_recommended"]


def aggregated_score(series, l, alpha):
    # Resembles the F objective function
    return series["largely_recommended"] + l * series[CRITERIA[1:]].sum()


def random_greedy(
    df,
    ref_date,
    n_vid=10,
    l=1 / 10,
    alpha=0.5,
    T=1,
    clipping_parameter=1,
    mu=0.1,
    t_0=0,
):

    # Normalizes the dataframe
    contains_na = df[CRITERIA].isna().apply(any, axis="columns")

    df[CRITERIA] = df[CRITERIA] - df[CRITERIA].min()
    df = df.fillna(0)

    df["age_in_days"] = df.apply(lambda x: get_age_in_days(x, ref_date), axis="columns")

    top_20_of_last_month = (
        df.loc[df["age_in_days"] <= 30]
        .sort_values(by="tournesol_score", ascending=False)
        .iloc[0:20]
    )
    df["top_20_of_last_month"] = df["uid"].isin(top_20_of_last_month["uid"])

    q50 = df["largely_recommended"].quantile(0.5)
    q95 = df["largely_recommended"].quantile(0.95)
    df["top_5%"] = df['largely_recommended'] > q95
    df["bottom_50%"] = df['largely_recommended'] < q50

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

        df["p"] = p
        print("Iteration " + str(i+1))
        print("Probability of top 5%: ")
        print(df.loc[~df['uid'].isin(S) & df['top_5%'], ["p"]].sum())
        print("Available top 5%: ")
        print(df.loc[~df['uid'].isin(S), ["top_5%"]].sum())

        print("Probability of bottom 50%: ")
        print(df.loc[~df['uid'].isin(S) & df['bottom_50%'], ["p"]].sum())
        print("Available bottom 50%: ")
        print(df.loc[~df['uid'].isin(S), ["bottom_50%"]].sum())

        print("Probability of top 20 of last month: ")
        print(df.loc[~df['uid'].isin(S) & df['top_20_of_last_month'], ["p"]].sum())
        print("Available top 20 of last month: ")
        print(df.loc[~df['uid'].isin(S), ["top_20_of_last_month"]].sum())
        print("\n\n")

        # sample a new element from p
        new_idx = np.random.choice(
            a=objective_function_scores.index, size=1, replace=False, p=p
        )[0]
        new = df.loc[new_idx, "uid"]

        # Update S and partial sums
        S.append(new)
        partial_sums[CRITERIA] = (
            partial_sums[CRITERIA] + df.loc[df["uid"] == new, CRITERIA]
        ).iloc[
            0
        ]  # hack to keep a series
        partial_sums["age_in_days_inverses"] = partial_sums[
            "age_in_days_inverses"
        ] + 1 / (t_0 + get_age_in_days(df.loc[df["uid"] == new].iloc[0], ref_date))

    objective = objective_function_scores[new_idx]

    print("Top 5% selected: ")
    print(df.loc[df['uid'].isin(S), ["top_5%"]].sum())
    print("Bottom 50% selected: ")
    print(df.loc[df['uid'].isin(S), ["bottom_50%"]].sum())
    print("Top 20 of last month selected: ")
    print(df.loc[df['uid'].isin(S), ["top_20_of_last_month"]].sum())

    return {"uids": S, "obj": objective}


# TEST

# Set up dataframe
df = pd.read_csv("tournesol_scores_above_20_2023-09-18.csv")

ref_date = datetime.datetime(2023, 9, 19, 0, 0)  # one day older than the video database

# random greedy parameters
alpha = 0.5
n_vid = 12
q = 0
temperature = 5.5
clipping_parameter = 1000 #1 / 2 * np.log(1000)
mu = 500
t_0 = 0

random_greedy(
    df=df,
    ref_date=ref_date,
    n_vid=n_vid,
    alpha=alpha,
    l=1 / 10,
    T=temperature,
    clipping_parameter=clipping_parameter,
    mu=mu,
    t_0=t_0,
)
