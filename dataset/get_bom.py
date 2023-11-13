import requests
import re


def fetch_bom():
    url = "https://www.gutenberg.org/cache/epub/17/pg17.txt"
    response = requests.get(url, timeout=1)
    if response.status_code != 200:
        raise requests.RequestException(
            f"Failed to download file from {url} with status code {response.status_code}"
        )
    text = response.content.decode("utf-8")
    text = text.split(
        "*** START OF THE PROJECT GUTENBERG EBOOK THE BOOK OF MORMON ***"
    )[-1]
    text = text.split("*** END OF THE PROJECT GUTENBERG EBOOK THE BOOK OF MORMON ***")[
        0
    ]
    return text

def parse_book(book_text: str, book_name: str):
    chapters = re.split(f"{book_name} Chapter \d+", book_text)
    if chapters[0].strip() != "":
        print(chapters[0])
        chapters = chapters[1:]
    return chapters
def parse_books(bom_text: str):
    bom_text = bom_text.split(
        "THE FIRST BOOK OF NEPHI HIS REIGN AND MINISTRY (1 Nephi)"
    )[-1]
    nephi1, bom_text = bom_text.split("THE SECOND BOOK OF NEPHI", maxsplit=1)
    nephi2, bom_text = re.split(
        r"THE BOOK OF JACOB\s+THE BROTHER OF NEPHI", bom_text, maxsplit=1
    )
    jacob, bom_text = bom_text.split("THE BOOK OF ENOS", maxsplit=1)
    enos, bom_text = bom_text.split("THE BOOK OF JAROM", maxsplit=1)
    jarom, bom_text = bom_text.split("THE BOOK OF OMNI", maxsplit=1)
    omni, bom_text = bom_text.split("THE WORDS OF MORMON", maxsplit=1)
    words_of_mormon, bom_text = bom_text.split("THE BOOK OF MOSIAH", maxsplit=1)
    mosiah, bom_text = re.split(
        r"THE BOOK OF ALMA\s+THE SON OF ALMA", bom_text, maxsplit=1
    )
    alma, bom_text = bom_text.split("THE BOOK OF HELAMAN")
    heleman, bom_text = re.split(
        r"THIRD BOOK OF NEPHI\s+THE SON OF NEPHI, WHO WAS THE SON OF HELAMAN",
        bom_text,
        maxsplit=1,
    )
    nephi3, bom_text = re.split(
        r"FOURTH NEPHI\s+WHO IS THE SON OF NEPHI—ONE OF THE DISCIPLES OF JESUS CHRIST",
        bom_text,
        maxsplit=1,
    )
    nephi4, bom_text = bom_text.split("THE BOOK OF MORMON", maxsplit=1)
    mormon, bom_text = bom_text.split("THE BOOK OF ETHER", maxsplit=1)
    ether, moroni = bom_text.split("THE BOOK OF MORONI", maxsplit=1)

    del bom_text

    parse_book(nephi1., "1 Nephi")
    verse_numbers = re.compile(r"\n\d+:\d+ ")
    books = r"([1-4] Nephi|Jacob|Enos|Jarom|Omni|Mormon|Mosiah|Alma|Heleman|Mormon|Ether|Moroni)"


parse_books(fetch_bom())


# # Perform the split
# segments = split_regex.split(text)

# # Create a dictionary from the segments
# result_dict = {}
# for i in range(1, len(segments), 2):
#     key = segments[i - 1].strip()
#     value = segments[i].strip()
#     result_dict[key] = value
