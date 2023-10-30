import requests
import re


def download(name: str, url: str, start: re.Pattern, end: re.Pattern):
    response = requests.get(url, timeout=1)
    if response.status_code != 200:
        raise requests.RequestException(
            f"Failed to download file from {url} with status code {response.status_code}"
        )
    text = response.content.decode("utf-8")
    text = re.split(start, text, 1)[1]
    text = re.split(end, text, 1)[0]
    text = text.strip()
    with open(f"{name}.txt", "w", encoding="utf-8") as f:
        f.write(text)


downloads = {
    "shakespeare": (
        "https://gutenberg.org/cache/epub/100/pg100.txt",
        re.compile(r"VENUS AND ADONIS"),
        re.compile(r"\*\*\* END OF THE PROJECT GUTENBERG"),
    ),
    "edgar_allen_poe": (
        "https://www.gutenberg.org/cache/epub/10031/pg10031.txt",
        re.compile(r"THE RAVEN."),
        re.compile(r"\*\*\* END OF THE PROJECT GUTENBERG"),
    ),
    "thomas_paine": (
        "https://www.gutenberg.org/cache/epub/31270/pg31270.txt",
        re.compile(r"THE CRISIS"),
        re.compile(r"\*\*\* END OF THE PROJECT GUTENBERG"),
    ),
    "winston_churchill": (
        "https://www.gutenberg.org/cache/epub/5400/pg5400.txt",
        re.compile(r"CHAPTER I"),
        re.compile(r"\*\*\* END OF THE PROJECT GUTENBERG"),
    )
}

for name, data in downloads.items():
    download(name, data[0], data[1], data[2])