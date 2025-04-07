import pandas
import requests


def build_dataset(collective_criteria_scores, youtube_api_key, tournesol_score_threshold=None):
    dataset = collective_criteria_scores.pivot(
        index="video", columns="criteria", values="score"
    )
    if tournesol_score_threshold:
        dataset = dataset.loc[dataset["largely_recommended"] >= tournesol_score_threshold]
    dataset["publication_date"] = None
    dataset["title"] = None
    dataset["channel"] = None
    dataset["view_count"] = None
    dataset["duration"] = None
    dataset.reset_index(inplace=True)

    # request videos information to the youtube API
    url_prefix = (
        "https://youtube.googleapis.com/youtube/v3/videos?"
        + "part=snippet"
        + "&part=contentDetails"
        + "&part=statistics"
        + "&key="
        + youtube_api_key
    )

    # Split the request in blocks of 50 ids because of the API limitations
    n_videos = dataset.shape[0]
    n_blocks = n_videos // 50
    for i in range(n_blocks + 1):
        url = url_prefix
        if i < n_blocks:  # blocks of 50 ids
            for j in range(0, 50):
                url += (
                    "&id="
                    + dataset.loc[dataset.index == (i * 50 + j), "video"]
                    .to_string(index=False, header=False)
                    .lstrip()
                )
        else:  # last partially filled block
            for j in range(0, n_videos % 50 + 1):  # last partially filled block
                url += (
                    "&id="
                    + dataset.loc[dataset.index == (n_blocks * 50 + j), "video"]
                    .to_string(index=False, header=False)
                    .lstrip()
                )
        r = requests.get(url)
        for item in r.json()["items"]:
            dataset.loc[
                dataset["video"] == item["id"], ["title", "channel", "publication_date", "view_count", "duration"]
            ] = (
                item["snippet"]["title"],
                item["snippet"]["channelTitle"],
                item["snippet"]["publishedAt"],
                item["statistics"]["viewCount"].split('.')[0],
                item["contentDetails"]["duration"][2:]
            )

    return dataset
