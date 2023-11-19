""" Utilizing Method 1 (Verse Outliers) To Segment And Analyze Authorship
"""

from cProfile import label
import enum
from typing import TypedDict, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast


class Statistics(TypedDict):
    data: pd.Series
    filtered_data: pd.Series
    lower_outliers: pd.Series
    upper_outliers: pd.Series
    outliers: pd.Series
    mean: Union[int, float]
    median: Union[int, float]
    std_dev: Union[int, float]
    q1: Union[int, float]
    q3: Union[int, float]
    iqr: Union[int, float]
    min_value: Union[int, float]
    max_value: Union[int, float]
    lower_bound: Union[int, float]
    upper_bound: Union[int, float]


def get_statistics(data: pd.Series, outlier_threshold: float = 1.5) -> Statistics:
    stats = {}
    stats["data"] = data

    stats["mean"] = data.mean()
    stats["median"] = data.median()
    stats["std_dev"] = data.std()
    stats["q1"] = data.quantile(0.25)
    stats["q3"] = data.quantile(0.75)
    stats["iqr"] = stats["q3"] - stats["q1"]

    stats["lower_bound"] = stats["q1"] - outlier_threshold * stats["iqr"]
    stats["upper_bound"] = stats["q3"] + outlier_threshold * stats["iqr"]

    lower_outliers = data[data < stats["lower_bound"]]
    upper_outliers = data[data > stats["upper_bound"]]
    stats["lower_outliers"] = lower_outliers
    stats["upper_outliers"] = data[data > stats["upper_bound"]]
    stats["outliers"] = pd.concat([stats["lower_outliers"], stats["upper_outliers"]])

    stats["filtered_data"] = data[~data.isin(stats["outliers"])]
    stats["min_value"] = stats["filtered_data"].min()
    stats["max_value"] = stats["filtered_data"].max()

    return {key: stats[key] for key in Statistics.__annotations__}  # type: ignore # return in correct order

def segment(stats: Statistics, df: pd.DataFrame):
    authors_df = pd.DataFrame(columns=["start_verse_num", "end_verse_num", "start_verse", "end_verse", "avg_embedding"])
    split_indices = stats["lower_outliers"].index
    prev_indice = 0
    for index, i in enumerate(split_indices):
        author = df.loc[prev_indice:i]
        embeddings = np.array([list(map(float, x.lstrip("[").rstrip("]").split())) for x in author["embedding"].to_list()])
        author_embedding = embeddings.mean(axis=1)
        author_start = author.to_numpy()[0]
        print(author_start)
        author_end = author.to_numpy()[-1]
        authors_df.loc[index] = [prev_indice, i, f"{author_start[1]} {author_start[2]} {author_start[3]}", f"{author_end[1]} {author_end[2]}.{author_end[3]}", author_embedding]
        prev_indice = i
    return authors_df

def box_and_whisker(stats: Statistics) -> None:
    outliers = stats["outliers"]
    min_value = stats["min_value"]
    q1 = stats["q1"]
    median = stats["median"]
    q3 = stats["q3"]
    max_value = stats["max_value"]

    plt.figure(figsize=(6, 4))
    plt.ylim(0, 2)
    plt.yticks([1], ["1"])

    plt.plot(
        outliers,
        [1] * len(outliers),
        "o",
        markersize=1.5,
        markerfacecolor="red",
        markeredgecolor="none",
        label="Outliers",
    )

    plt.vlines([min_value, max_value], 0.9, 1.1, "black", linewidth=.75)
    plt.vlines([q1, median, q3], 0.75, 1.25, "black", linewidth=1.2)
    plt.hlines([1, 1], [min_value, q3], [q1, max_value], "black", linewidth=.75)
    plt.hlines([0.75, 0.75, 1.25, 1.25], [q1, median, q1, median], [median, q3, median, q3], "black", linewidth=1.2)

    plt.xlabel("Values")
    plt.title("Box and Whisker Plot")

    plt.legend()
    plt.show()

def line_graph(stats: Statistics):
    data = stats["data"]
    split_indices = stats["lower_outliers"].index
    plt.figure(figsize=(8, 2))
    colors = plt.cm.get_cmap("hsv", len(split_indices))
    prev_indice = 0
    for index, i in enumerate(split_indices):
        plt.plot(data[prev_indice:i], color=colors(index), linewidth=0.3)
        prev_indice = i
    plt.legend()
    plt.show()

df = pd.read_csv("embeddings.csv")
cos_sim = df["cos_sim"].dropna()
stats = get_statistics(cos_sim, 2)
segmented = segment(stats, df)
segmented.to_csv("method1_segments.csv")
