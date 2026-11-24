"""
harvest.py  —  download EVERY article/video link from the Fox News and NBC News sitemaps.

    py harvest.py --out harvest.csv

Fox:  https://www.foxnews.com/sitemap.xml lists ~334 pages of 5000 articles each (2014..now).
NBC:  sitemap-index lists one article + one video sitemap per month, plus select/news/etc.
Writes url,date,label. Resumable: sitemaps already done are listed in <out>.done.
"""
import argparse
import csv
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from scrape_links import fetch, parse
from preprocess import headline_from_url, label_from_url

FOX_INDEX = "https://www.foxnews.com/sitemap.xml"
NBC_INDEXES = ["https://www.nbcnews.com/sitemap/nbcnews/sitemap-index",
               "https://www.nbcnews.com/sitemap/nbcnews/sitemap-news",
               "https://www.nbcnews.com/sitemap/nbcnews/sitemap-select.xml",
               "https://www.nbcnews.com/sitemap/nbcnews/sitemap-curations"]
SKIP_TYPES = ("type=categories", "type=sections", "type=shows", "type=personalities",
              "type=pages", "type=videoshows", "games-sitemap", "sitemap-authors")


def list_sitemaps():
    maps = []
    _, e = parse(fetch(FOX_INDEX))
    maps += [l for l, _ in e if not any(s in l for s in SKIP_TYPES)]
    for idx in NBC_INDEXES:
        is_index, e = parse(fetch(idx))
        if is_index:
            maps += [l for l, _ in e if not any(s in l for s in SKIP_TYPES)]
        else:
            maps.append(idx)
    # NBC monthly sitemaps follow a fixed pattern; list them even if the index is blocked
    for y in range(2026, 2002, -1):
        for m in range(12, 0, -1):
            for kind in ("article", "video", "slideshow"):
                maps.append(f"https://www.nbcnews.com/sitemap/nbcnews/sitemap-{y}-{m:02d}-{kind}.xml")
    return list(dict.fromkeys(maps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="harvest.csv")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds before each request")
    ap.add_argument("--only", default="", help="only sitemaps containing this text")
    args = ap.parse_args()

    done_path = args.out + ".done"
    done = set()
    if os.path.exists(done_path):
        done = set(open(done_path, encoding="utf-8").read().split())
    maps = [m for m in list_sitemaps() if m not in done and args.only in m]
    print(f"{len(maps)} sitemaps to fetch ({len(done)} already done)", flush=True)

    new_file = not os.path.exists(args.out)
    out = open(args.out, "a", newline="", encoding="utf-8")
    w = csv.writer(out)
    if new_file:
        w.writerow(["url", "date", "label"])
    dlog = open(done_path, "a", encoding="utf-8")
    lock = threading.Lock()
    counter = [0, 0]

    def work(sm):
        time.sleep(args.delay)
        xml = fetch(sm, timeout=60, retries=3)
        if not xml:
            return
        _, entries = parse(xml)
        rows = []
        for loc, date in entries:
            url = loc.split("?")[0].split("#")[0]
            lab = label_from_url(url)
            if lab is None or url.endswith((".xml", ".gz")):
                continue
            if not headline_from_url(url):
                continue
            rows.append((url, date, lab))
        with lock:
            w.writerows(rows)
            dlog.write(sm + "\n")
            counter[0] += 1
            counter[1] += len(rows)
            if counter[0] % 25 == 0:
                out.flush(); dlog.flush()
                print(f"  {counter[0]}/{len(maps)} sitemaps, {counter[1]} links", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, maps))
    out.close(); dlog.close()
    print(f"done: {counter[1]} links from {counter[0]} sitemaps -> {args.out}")


if __name__ == "__main__":
    main()
