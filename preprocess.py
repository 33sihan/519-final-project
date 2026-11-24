"""
preprocess.py  —  CIS 5190 Project B (News Source Classification)

Backend contract:
    prepare_data(csv_path) -> (X, y)
    * X: one input per CSV row, same order (no filtering / dropping / reordering)
    * y: labels if they can be read from the URL, otherwise an empty list
"""
import re
from urllib.parse import urlparse

import pandas as pd

# Label names used when we can read the outlet from a full training URL.
# model.py uses the SAME order (index 0 = Fox, index 1 = NBC).
LABELS = ("FoxNews", "NBC")

_URL_COL_CANDIDATES = ("url", "URL", "link", "Link", "urls")


def headline_from_url(url) -> str:
    """Reduce a link to its headline words, the same way the test set is reduced.

    Works on both full training links and already-reduced test links:
      https://www.foxnews.com/politics/some-headline-here     -> "some headline here"
      https://www.nbcnews.com/news/us-news/some-headline-rcna12345 -> "some headline"
      /some-headline                                           -> "some headline"
    Website name, section, and any word containing a digit are removed.
    """
    if not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""

    path = urlparse(url).path if "://" in url else url.split("?")[0].split("#")[0]
    segments = [s for s in path.split("/") if s]
    if not segments:
        return ""
    last = segments[-1]                                  # headline slug
    last = re.sub(r"\.(html?|print|amp)$", "", last, flags=re.I)

    words = re.split(r"[-_+\s]+", last.lower())
    words = [w for w in words if w and not any(c.isdigit() for c in w)]
    words = [re.sub(r"[^a-z']", "", w) for w in words]  # strip leftover punctuation
    return " ".join(w for w in words if w)


def label_from_url(url):
    """Return 0 (Fox) / 1 (NBC) from the website name, or None if unknown."""
    if not isinstance(url, str):
        return None
    u = url.lower()
    if "foxnews" in u:
        return 0
    if "nbcnews" in u:
        return 1
    return None


def _find_url_col(df: pd.DataFrame) -> str:
    for c in _URL_COL_CANDIDATES:
        if c in df.columns:
            return c
    return df.columns[0]


def prepare_data(csv_path: str):
    df = pd.read_csv(csv_path)
    urls = df[_find_url_col(df)].tolist()

    X = [headline_from_url(u) for u in urls]            # exactly one input per row

    labels = [label_from_url(u) for u in urls]
    if all(l is not None for l in labels) and len(labels) > 0:
        y = [LABELS[l] for l in labels]                  # training CSV: full links
    else:
        y = []                                           # test CSV: no labels
    return X, y