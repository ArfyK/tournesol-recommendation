import sys
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from recommendation import (
    CRITERIA,
    random_greedy,
    random,
    rank_by_tournesol_score,
    get_age_in_days,
)

# Set up dataframe
df = pd.read_csv("tournesol_scores_above_20_2023-09-18.csv")

ref_date = datetime.datetime(2023, 9, 19, 0, 0)  # one day older than the video database

df["age_in_days"] = df.apply(lambda x: get_age_in_days(x, ref_date), axis="columns")

df[CRITERIA] = df[CRITERIA] - df[CRITERIA].min()

# Set up results
results = pd.read_csv("mu_tuning_n_test=100_mu=0.5_5_50_500_t_0=0_15.csv")
results["uids"] = results["uids"].apply(lambda x: x[2:-2].split("', '"))


top_20_of_last_month = (
    df.loc[df["age_in_days"] <= 30]
    .sort_values(by="tournesol_score", ascending=False)
    .iloc[0:20]
)

# sns.scatterplot(data=df, x="age_in_days", y="largely_recommended")
# plt.show()


def count_videos_in_subset(uids_list, subset):
    return subset.loc[(subset.isin(uids_list))].shape[0]


median = df["largely_recommended"].quantile(0.5)

# Comparison between the terms of the objective function


# Relevance term
def compute_bundle_relevance(df, uids):
    return df.loc[df["uid"].isin(uids), "largely_recommended"].sum()


results["relevance"] = results.loc[results["mu"] == 500, "uids"].apply(
    lambda x: compute_bundle_relevance(df, x)
)


# Diversity term
def compute_bundle_diversity(df, uids, l=1 / 10):
    diversity = 0
    for criteria in CRITERIA:
        diversity += np.sqrt(df.loc[df["uid"].isin(uids), criteria].sum())
    return l * diversity


results["diversity"] = results.loc[results["mu"] == 500, "uids"].apply(
    lambda x: compute_bundle_diversity(df, x)
)

# Recency term
mu = 500
t0 = 0


def compute_bundle_recency(df, uids, t0=0, mu=500):
    return mu * np.sqrt((1 / (t0 + df.loc[df["uid"].isin(uids), "age_in_days"])).sum())


results["recency_0"] = results.loc[results["mu"] == mu, "uids"].apply(
    lambda x: compute_bundle_recency(df, x, 0, mu)
)
results["recency_15"] = results.loc[results["mu"] == mu, "uids"].apply(
    lambda x: compute_bundle_recency(df, x, 15, mu)
)


# f, axs = plt.subplots(2, 2, figsize=(13, 7))
# sns.boxplot(data=results, x="relevance", ax=axs[0, 0])
# sns.boxplot(data=results, x="diversity", ax=axs[0, 1])
# sns.boxplot(data=results, x="recency_0", ax=axs[1, 0])
# sns.boxplot(data=results, x="recency_15", ax=axs[1, 1])
# f.suptitle("mu = " + str(mu))
# plt.show()

# Comparison of the "cumulative distribution" of each term
bundle_size = 12
mu = 500
t0 = 0
subresults = results.loc[(results["mu"] == mu) & (results["t_0"] == t0)]

successive_increases = pd.DataFrame(
    columns=["test", "iteration", "relevance", "diversity", "recency"]
)


def relevance_successive_increases(df, test_uids):
    global successive_increases

    # Set up te dataframe holding the results
    increases = pd.DataFrame(
        data=np.zeros((bundle_size, 5)),
        columns=["test", "iteration", "relevance", "diversity", "recency"],
    )
    increases["test"] = test_uids["test"]
    increases["iteration"] = range(1, bundle_size + 1)

    # fill the dataframe
    # relevance
    increases.loc[increases["iteration"] == 1, "relevance"] = compute_bundle_relevance(
        df, test_uids["uids"][:1]
    )
    for i in range(2, bundle_size + 1):
        increases.loc[
            increases["iteration"] == i, "relevance"
        ] = compute_bundle_relevance(df, test_uids["uids"][:i])
    # diversity
    increases.loc[increases["iteration"] == 1, "diversity"] = compute_bundle_diversity(
        df, test_uids["uids"][:1]
    )
    for i in range(2, bundle_size + 1):
        increases.loc[
            increases["iteration"] == i, "diversity"
        ] = compute_bundle_diversity(df, test_uids["uids"][:i])
    # recency
    increases.loc[increases["iteration"] == 1, "recency"] = compute_bundle_recency(
        df, test_uids["uids"][:1]
    )
    for i in range(2, bundle_size + 1):
        increases.loc[increases["iteration"] == i, "recency"] = compute_bundle_recency(
            df, test_uids["uids"][:i]
        )

    # accumulate the results
    successive_increases = pd.concat([successive_increases, increases])
    return


subresults[["test", "uids"]].apply(
    lambda x: relevance_successive_increases(df, x),
    axis="columns",
)

f, axs = plt.subplots(1, 3)
sns.boxplot(data=successive_increases, x="iteration", y="relevance", ax=axs[0])
sns.boxplot(data=successive_increases, x="iteration", y="diversity", ax=axs[1])
sns.boxplot(data=successive_increases, x="iteration", y="recency", ax=axs[2])
f.suptitle('mu = ' + str(mu) + ', t0 = ' + str(t0))
plt.show()
