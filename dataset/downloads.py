""" Contains downloads dict with information for database downloads """
import re

downloads: dict[str, tuple[str, re.Pattern, re.Pattern]] = {
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
    ),
}
