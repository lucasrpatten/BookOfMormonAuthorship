import requests
import re
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
database_path = os.path.join(script_dir, "bom")
if not os.path.exists(database_path):
    os.makedirs(database_path)


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


def parse_chapter(chapter_text: str):
    verse_numbers = re.compile(r"\d+:\d+ ")
    return [
        re.sub(r"\n+", "", i.strip())
        for i in re.split(verse_numbers, chapter_text)
        if i.strip() != ""
    ]


def parse_book(book_text: str, book_name: str, single: bool = False, include_intro: bool = True):
    book_dir = os.path.join(database_path, book_name)
    if not os.path.exists(book_dir):
        os.makedirs(book_dir)

    if book_name == "Heleman":
        book_name = ""
    if single:
        chapter_dir = os.path.join(book_dir, "1")
        if not os.path.exists(chapter_dir):
            os.makedirs(chapter_dir)
        verses = parse_chapter(book_text.strip())
        for j, verse in enumerate(verses):
            verse_path = os.path.join(chapter_dir, f"{j+1}.txt")
            with open(verse_path, "w", encoding="utf-8") as f:
                f.write(verse.strip().replace("\r", " "))
        return

    chapters = re.split(f"{book_name} Chapter \d+", book_text)
    if chapters[0].strip() != "":
        if include_intro:
            intro = chapters[0]
            intro_path = os.path.join(book_dir, "intro.txt")
            with open(intro_path, "w", encoding="utf-8") as f:
                f.write(re.sub(r"\r\n", " ", intro.strip()))

    chapters = chapters[1:]
    for i, chapter in enumerate(chapters):
        chapter_dir = os.path.join(book_dir, str(i + 1))
        if not os.path.exists(chapter_dir):
            os.makedirs(chapter_dir)
        verses = parse_chapter(chapter)
        for j, verse in enumerate(verses):
            verse_path = os.path.join(chapter_dir, f"{j+1}.txt")
            with open(verse_path, "w", encoding="utf-8") as f:
                f.write(verse.strip().replace("\r", " "))
    return


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
        r"FOURTH NEPHI\s+WHO IS THE SON OF NEPHI—ONE OF THE DISCIPLES OF JESUS CHRIST\s+An account of the people of Nephi, according to his record.",
        bom_text,
        maxsplit=1,
    )
    nephi4, bom_text = bom_text.split("THE BOOK OF MORMON", maxsplit=1)
    mormon, bom_text = bom_text.split("THE BOOK OF ETHER", maxsplit=1)
    ether, moroni = bom_text.split("THE BOOK OF MORONI", maxsplit=1)

    del bom_text

    intro_include = False

    parse_book(nephi1, "1 Nephi", include_intro=intro_include)
    parse_book(nephi2, "2 Nephi", include_intro=intro_include)
    parse_book(jacob, "Jacob", include_intro=intro_include)
    parse_book(enos, "Enos", single=True, include_intro=intro_include)
    parse_book(jarom, "Jarom", single=True, include_intro=intro_include)
    parse_book(omni, "Omni", single=True, include_intro=intro_include)
    parse_book(words_of_mormon, "Words of Mormon", single=True, include_intro=intro_include)
    parse_book(mosiah, "Mosiah", include_intro=intro_include)
    parse_book(alma, "Alma", include_intro=intro_include)
    parse_book(heleman, "Heleman", include_intro=intro_include)
    parse_book(nephi3, "3 Nephi", include_intro=intro_include)
    parse_book(nephi4, "4 Nephi", single=True, include_intro=intro_include)
    parse_book(mormon, "Mormon", include_intro=intro_include)
    parse_book(ether, "Ether", include_intro=intro_include)
    parse_book(moroni, "Moroni", include_intro=intro_include)


parse_books(fetch_bom())