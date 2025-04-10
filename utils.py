import io
import zipfile

import pandas as pd
import requests


def build_dataset(youtube_api_key, tournesol_score_threshold=None):
    r"""
    Use the collective_criteria_scores returned by the Tournesol API and the youtube API to build a detailed dataset of youtube videos.

    The returned dataFrame consists in the pivoted collective_criteria_scores and additional metadata sent by the youtube API: publication_date, title, channel, view_count, duration.

    Parameters
    ----------
    youtube_api_key : str
        API key to request "https://youtube.googleapis.com/youtube/v3/videos?part=snippet&part=contentDetails&part=statistics&key=<youtube_api_key>&id=<ids>"
    tournesol_score_threshold : float
        Score threshold to filter out videos having a low tournesol_score.

    Returns
    -------
    pandas.DataFrame
    """
    print("Downloading the Tournesol scores from the Tournesol API")
    response = requests.get("https://api.tournesol.app/exports/all")
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    collective_criteria_scores = pd.read_csv(zip_file.open("collective_criteria_scores.csv"))

    dataset = collective_criteria_scores.pivot(
        index="video", columns="criteria", values="score"
    )
    if tournesol_score_threshold:
        dataset = dataset.loc[
            dataset["largely_recommended"] >= tournesol_score_threshold
        ]
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
    print("Downloading videos metadata from Youtube API")
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
                dataset["video"] == item["id"],
                ["title", "channel", "publication_date", "view_count", "duration"],
            ] = (
                item["snippet"].get("title",""),
                item["snippet"].get("channelTitle",""),
                item["snippet"].get("publishedAt",""),
                item["statistics"].get("viewCount","").split(".")[0],
                item["contentDetails"].get("duration","PT")[2:],
            )
        print("    " + str(i*50 + j + 1) + " videos out of " + str(n_videos))

    return dataset
