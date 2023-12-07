import requests
import sys
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


def greedy(
    data,
    score,
    update_state,
    selection,
    normalization=None,
    preselection=None,
    bundle_size=12,
    **kwargs,
):
    if normalization:
        data = normalization(data, **kwargs)
    if preselection:
        data = preselection(data, **kwargs)

    bundle = []
    state = None

    for i in range(bundle_size):
        scores = data.apply(lambda x: score(x, state, **kwargs), axis="columns")

        new_index = selection(scores, **kwargs)
        new_item = data.loc[new_index]
        bundle.append(new_item["uid"])

        state = update_state(new_item, state, **kwargs)

    return {"bundle": bundle, "score": scores.iat[new_index]}


### Normalizations
def normalization(data, **kwargs):
    return (data[CRITERIA] - data[CRITERIA].min()).fillna(0)


### Preselections
def above_quantile(data, key, quantile, **kwargs):
    if type(key) == str:
        return data.loc[data[key] >= data[key].quantile(quantile)]

    data["key"] = data.apply(lambda x: key(x, **kwargs), axis="columns")
    return data.loc[data["key"] >= data["key"].quantile(quantile)]


def uniform_sampling(data, sample_size, **kwargs):
    return data.loc[np.random.choice(a=data.index, size=sample_size, replace=False)]


### Score functions and their associated update_state functions


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


def F(new, partial_sums, ref_date, l=1 / 10, alpha=0.5, mu=0.1, t_0=0, **kwargs):
    try:
        if partial_sums == None:
            index = CRITERIA + ["age_in_days_inverses"]
            partial_sums = pd.Series(data=[0] * len(index), index=index)
    except ValueError:
        pass

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


def F_update_state(new, partial_sums, ref_date, t_0=0, **kwargs):
    try:
        if partial_sums == None:
            index = CRITERIA + ["age_in_days_inverses"]
            partial_sums = pd.Series(data=[0] * len(index), index=index)
    except ValueError:
        pass

    partial_sums[CRITERIA] = (partial_sums[CRITERIA] + new[CRITERIA]).iloc[
        0
    ]  # hack to keep a series
    partial_sums["age_in_days_inverses"] = partial_sums["age_in_days_inverses"] + 1 / (
        t_0 + get_age_in_days(new, ref_date)
    )

    return partial_sums


def F_incomplete(new, partial_sums, ref_date, alpha=0.5, mu=0.1, t_0=0):
    try:
        if partial_sums == None:
            index = ["largely_recommended", "age_in_days_inverses"]
            partial_sums = pd.Series(data=[0] * len(index), index=index)
    except ValueError:
        pass

    # Relevance term
    R = partial_sums["largely_recommended"] + new["largely_recommended"]
    # Recency term
    D = (
        partial_sums["age_in_days_inverses"]
        + 1 / (t_0 + get_age_in_days(new, ref_date))
    ) ** alpha

    return R + mu * D


def F_incomplete_state_update(new, partial_sums, ref_date, t_0=0, **kwargs):
    try:
        if partial_sums == None:
            index = ["largely_recommended", "age_in_days_inverses"]
            partial_sums = pd.Series(data=[0] * len(index), index=index)
    except ValueError:
        pass

    partial_sums["largely_recommended"] = (
        partial_sums["largely_recommended"] + new["largely_recommended"]
    ).iloc[
        0
    ]  # hack to get a series
    partial_sums["age_in_days_inverses"] = partial_sums["age_in_days_inverses"] + 1 / (
        t_0 + get_age_in_days(new, ref_date)
    )

    return partial_sums


### Selection functions


def proxied_idxmax(scores, **kwargs):
    return scores.idxmax()


def random_selection(scores, score_transform=None, center=True, **kwargs):
    if score_transform == None:
        # by default the distribution is uniform
        score_transform = lambda x: 1

    if center:
        scores = scores - scores.mean()

    distribution = scores.apply(lambda x: score_transform(x, **kwargs))
    distribution = distribution / distribution.sum()
    return np.random.choice(a=scores.index, size=1, p=distribution)[0]


def exponential(x, temperature, clipping_parameter, **kwargs):
    return np.exp(np.clip(x / temperature, clipping_parameter, -clipping_parameter))


if __name__ == "__main__":
    df = pd.read_csv("tournesol_scores_above_20_2023-09-18.csv")

    ref_date = datetime.datetime(
        2023, 9, 19, 0, 0
    )  # one day older than the video database

    results = greedy(
        data=df,
        score=F,
        update_state=F_update_state,
        selection=random_selection,
        ref_date=ref_date,
        mu=500,
        temperature=5.5,
        clipping_parameter=1 / 1 * np.log(1000),
        score_transform=exponential,
    )
    print(results)
