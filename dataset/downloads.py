""" Contains downloads dict with information for database downloads """

END_OF_GUTENBERG = "*** END OF THE PROJECT GUTENBERG"

downloads: dict[str, tuple[str, str, str]] = {
    "shakespeare": (
        "https://gutenberg.org/cache/epub/100/pg100.txt",
        "VENUS AND ADONIS",
        END_OF_GUTENBERG,
    ),
    "edgar_allen_poe": (
        "https://www.gutenberg.org/cache/epub/10031/pg10031.txt",
        "THE RAVEN.",
        END_OF_GUTENBERG,
    ),
    "thomas_paine": (
        "https://www.gutenberg.org/cache/epub/31270/pg31270.txt",
        "THE CRISIS",
        END_OF_GUTENBERG,
    ),
    "winston_churchill": (
        "https://www.gutenberg.org/cache/epub/5400/pg5400.txt",
        "CHAPTER I",
        END_OF_GUTENBERG,
    ),
    "the_odyssey": (
        "https://www.gutenberg.org/cache/epub/1727/pg1727.txt",
        "TELEMACHUS TO THE SUITORS.",
        "[1] [ Black races are evidently known to the",
    ),
}
