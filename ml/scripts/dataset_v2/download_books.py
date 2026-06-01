"""Download external book corpora for the dataset_v2 pipeline.

Fetches individual .txt and .epub files from GitHub. Re-running is safe
(skips existing files). Called automatically by run_pipeline.py step 1.
"""
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlretrieve

_TXT_DIR = Path(__file__).parent.parent.parent / "data" / "txt-books"
_EPUB_DIR = Path(__file__).parent.parent.parent / "data" / "epub-books"

_TENNESSINE = "https://raw.githubusercontent.com/tennessine/corpus/master"
_HANKINGHU = "https://raw.githubusercontent.com/hankinghu/literature-books/master"
_FANCY88 = "https://raw.githubusercontent.com/fancy88/ibook/master"

TXT_BOOKS: list[tuple[str, str]] = [
    # (base_url, filename)

    # --- Classical (pre-1900) ---
    (_TENNESSINE, "红楼梦.txt"),              # 18th c, classical vernacular

    # --- Early modern / 五四 era (1920s-40s) ---
    (_HANKINGHU, "周作人文集.txt"),            # 1920s-30s essays
    (_HANKINGHU, "四世同堂.txt"),              # Lao She, 1940s Beijing vernacular

    # --- Mid-century (1950s-80s) ---
    (_HANKINGHU, "冬天里的春天.txt"),          # 1981, scar literature

    # --- Contemporary literary (1990s-2010s) ---
    (_HANKINGHU, "《白鹿原》全集.txt"),        # 1993, rural realist, Shaanxi
    (_HANKINGHU, "务虚笔记.txt"),              # Shi Tiesheng, 1996, philosophical
    (_HANKINGHU, "繁花.txt"),                  # 2012, Shanghai contemporary

    # --- Essay / 散文 ---
    (_HANKINGHU, "俗世奇人.txt"),              # Feng Jicai, Tianjin dialect vignettes

    # --- Nonfiction ---
    (_HANKINGHU, "万历十五年.txt"),            # Huang Renyu, 1982, historical prose
    (_HANKINGHU, "《常识》梁文道.txt"),        # Leung Man-tao, modern social commentary
]

EPUB_BOOKS: list[tuple[str, str]] = [
    # (base_url, filename)
    (_FANCY88, "看见.epub"),                   # Chai Jing, 2013, investigative journalism
    (_FANCY88, "沉默的大多数(精排版).epub"),    # Wang Xiaobo, essays, intellectual/irreverent
    (_FANCY88, "【精】三体（全集）.epub"),      # Liu Cixin, 2006-2010, sci-fi
]


def _download(dest_dir: Path, books: list[tuple[str, str]]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for base_url, filename in books:
        dest = dest_dir / filename
        if dest.exists():
            continue
        url = f"{base_url}/{quote(filename)}"
        print(f"  downloading: {filename}")
        try:
            urlretrieve(url, dest)
        except Exception as e:
            print(f"  FAILED: {filename} ({e})")
            dest.unlink(missing_ok=True)


def download_books() -> None:
    _download(_TXT_DIR, TXT_BOOKS)
    _download(_EPUB_DIR, EPUB_BOOKS)
    n_txt = len(list(_TXT_DIR.glob("*.[tT][xX][tT]")))
    n_epub = len(list(_EPUB_DIR.glob("*.epub")))
    print(f"  {n_txt} txt + {n_epub} epub in {_TXT_DIR.parent}")


if __name__ == "__main__":
    download_books()
