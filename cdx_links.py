"""
cdx_links.py  —  extra article links from the Internet Archive's URL index (CDX API),
for pages the sitemaps no longer list (e.g. Fox live-news blogs, old NBC pages).

    py cdx_links.py --out cdx.csv foxnews.com/live-news/ nbcnews.com/politics/ ...

Each argument is a URL prefix. Writes url,label (deduplicated, appends + resumes).
"""
import argparse
import csv
import os
import time
import urllib.parse

from scrape_links import fetch
from preprocess import headline_from_url, label_from_url

API = "https://web.archive.org/cdx/search/cdx"


def prefix_urls(prefix, per_page=150000, start=None):
    key = None
    while True:
        q = {"url": prefix, "matchType": "prefix", "fl": "original", "collapse": "urlkey",
             "limit": per_page, "showResumeKey": "true",
             "filter": ["statuscode:200", "mimetype:text/html"]}
        if start:
            q["from"] = start
        if key:
            q["resumeKey"] = key
        txt = fetch(API + "?" + urllib.parse.urlencode(q, doseq=True), timeout=300, retries=4)
        lines = txt.split("\n")
        key = None
        # resume key = last non-empty line after a blank line
        if len(lines) >= 2 and "" in lines[:-1]:
            blank = len(lines) - 1 - lines[::-1].index("", 1)
            tail = [l for l in lines[blank + 1:] if l.strip()]
            if tail and "://" not in tail[0]:
                key = tail[0].strip()
                lines = lines[:blank]
        yield [l.strip() for l in lines if "://" in l]
        if not key:
            return
        time.sleep(1)


def clean(u):
    u = urllib.parse.unquote(u).split("?")[0].split("#")[0]
    if u.startswith("http://"):
        u = "https://" + u[7:]
    return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--out", default="cdx.csv")
    ap.add_argument("--from", dest="start", default="", help="e.g. 2019 (CDX 'from')")
    args = ap.parse_args()

    seen = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            seen = {r["url"] for r in csv.DictReader(f)}
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["url", "label"])
        for p in args.prefixes:
            n = 0
            for chunk in prefix_urls(p, start=args.start or None):
                for u in chunk:
                    u = clean(u)
                    lab = label_from_url(u)
                    if lab is None or u in seen or " " in u or "$" in u:
                        continue
                    if len(headline_from_url(u).split()) < 2:
                        continue
                    seen.add(u)
                    w.writerow([u, lab])
                    n += 1
                f.flush()
                print(f"  {p}: +{n}", flush=True)
            print(f"{p}: {n} new links", flush=True)


if __name__ == "__main__":
    main()
