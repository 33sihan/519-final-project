"""
scrape_links.py  (v3)  —  collect extra Fox News / NBC News article links from each
site's sitemaps (the project requires scraping additional links; Section 4.3).

Standard library only. Writes a CSV with columns `url,date` (date = the article's
publication / last-modified date from the sitemap, YYYY-MM-DD, may be empty).

    py scrape_links.py --out scraped_dated.csv                     # broad crawl
    py scrape_links.py --since 2025-01-01 --until 2025-12-31 \
                       --out scraped_window.csv --max-sitemaps 3000 # one time window

With --since/--until, only articles dated inside the window are kept, and child
sitemaps that can't contain such articles are skipped without downloading:
  * the sitemap index says it was last modified before --since, or
  * its URL names a year-month (e.g. sitemap-2019-04-article) outside the window, or
  * 5 sitemaps in a row from the same index held only articles older than --since.
Progress is saved every 20 sitemaps; Ctrl+C stops and saves.
"""
import argparse
import csv
import gzip
import os
import random
import re
import time
import urllib.request
from collections import defaultdict, deque

from preprocess import headline_from_url, label_from_url

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

SITES = {
    "fox": {
        "label": 0,
        "domain": "foxnews.com",
        "robots": "https://www.foxnews.com/robots.txt",
        "fallback": ["https://www.foxnews.com/sitemap.xml",
                     "https://www.foxnews.com/sitemap.xml?type=articles",
                     "https://www.foxnews.com/sitemap.xml?type=news"],
    },
    "nbc": {
        "label": 1,
        "domain": "nbcnews.com",
        "robots": "https://www.nbcnews.com/robots.txt",
        "fallback": ["https://www.nbcnews.com/sitemap/nbcnews/sitemap-index",
                     "https://www.nbcnews.com/sitemap.xml"],
    },
}

SKIP_PARTS = ("/video", "/videos/", "/category/", "/person/", "/tag/", "/tags/",
              "/author/", "/authors/", "/photos/", "/slideshow", "/podcasts/",
              "/shows/", "/search", "/weather/", "/newsletters", "/about")

URL_BLOCK_RE = re.compile(r"<url>(.*?)</url>", re.I | re.S)
SITEMAP_BLOCK_RE = re.compile(r"<sitemap>(.*?)</sitemap>", re.I | re.S)
LOC_RE = re.compile(r"<loc>\s*(?:<!\[CDATA\[)?\s*(.*?)\s*(?:\]\]>)?\s*</loc>", re.I | re.S)
DATE_RE = re.compile(r"<(?:[\w]+:)?(?:publication_date|lastmod)>\s*(?:<!\[CDATA\[)?\s*"
                     r"(\d{4}-\d{2}-\d{2})", re.I)
YM_IN_URL_RE = re.compile(r"(20\d\d|19\d\d)[-/_](0[1-9]|1[0-2])(?!\d)")


def fetch(url, timeout=30, retries=2):
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                       "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
            return data.decode("utf-8", errors="replace")
        except Exception as e:
            if attempt == retries:
                print(f"   ! failed {url}: {e}")
                return ""
            time.sleep(2 * (attempt + 1))


def is_article(url, domain):
    u = url.lower()
    if "://" not in u or domain not in u.split("/")[2]:
        return False
    if any(p in u for p in SKIP_PARTS):
        return False
    if u.endswith((".xml", ".gz", ".jpg", ".png", ".pdf")):
        return False
    return len(headline_from_url(url).split()) >= 3


def looks_like_sitemap(url):
    u = url.lower()
    return "sitemap" in u or u.endswith((".xml", ".xml.gz"))


def _first_date(block):
    m = DATE_RE.search(block)
    return m.group(1) if m else ""


def parse(xml):
    """-> (is_index, [(loc, date)])"""
    is_index = "<sitemapindex" in xml.lower()
    blocks = SITEMAP_BLOCK_RE.findall(xml) if is_index else URL_BLOCK_RE.findall(xml)
    out = []
    for b in blocks:
        m = LOC_RE.search(b)
        if m:
            out.append((m.group(1).replace("&amp;", "&").strip(), _first_date(b)))
    if not blocks:  # unusual format: fall back to bare <loc> tags
        out = [(l.replace("&amp;", "&").strip(), "") for l in LOC_RE.findall(xml)]
    return is_index, out


def child_can_match(loc, lastmod, since, until):
    """Can this child sitemap contain articles inside [since, until]?"""
    if since and lastmod and lastmod < since:
        return False
    m = YM_IN_URL_RE.search(loc)
    if m and (since or until):
        ym = f"{m.group(1)}-{m.group(2)}"
        if since and ym < since[:7]:
            return False
        if until and ym > until[:7]:
            return False
    return True


def crawl_site(name, cfg, args, articles):
    """articles: dict url -> date (updated in place). Yields periodically for saving."""
    since, until = args.since, args.until
    seeds = []
    for line in fetch(cfg["robots"]).splitlines():
        if line.lower().startswith("sitemap:"):
            seeds.append(line.split(":", 1)[1].strip())
    seeds += [u for u in cfg["fallback"] if u not in seeds]
    print(f"[{name}] {len(seeds)} seed sitemaps")

    queue = deque((s, None) for s in seeds)        # (sitemap url, parent index url)
    seen, too_old_streak, dead_parents = set(), defaultdict(int), set()
    n_maps = skipped = 0
    while queue and n_maps < args.max_sitemaps and len(articles) < args.max_urls:
        sm, parent = queue.popleft()
        if sm in seen or parent in dead_parents:
            continue
        seen.add(sm)
        xml = fetch(sm)
        n_maps += 1
        is_index, entries = parse(xml)
        kids = added = old = 0
        if is_index and not since:
            random.shuffle(entries)                   # broad crawl: spread across years
        for loc, date in entries:
            if is_index or (looks_like_sitemap(loc) and not is_article(loc, cfg["domain"])):
                if loc in seen:
                    continue
                if child_can_match(loc, date, since, until):
                    queue.append((loc, sm))
                    kids += 1
                else:
                    skipped += 1
            elif is_article(loc, cfg["domain"]):
                if since and date and date < since:
                    old += 1
                    continue
                if until and date and date > until:
                    continue
                if (since or until) and not date and not args.keep_undated:
                    continue
                url = loc.split("?")[0].split("#")[0]
                if url not in articles:
                    articles[url] = date
                    added += 1
        # paged sitemaps (newest first): stop a parent after 5 all-too-old children
        if since and parent and not is_index:
            if added == 0 and old > 0:
                too_old_streak[parent] += 1
                if too_old_streak[parent] >= 5:
                    dead_parents.add(parent)
                    print(f"[{name}]   older than {since} from here on, skipping rest of {parent}")
            elif added:
                too_old_streak[parent] = 0
        print(f"[{name}] sitemap {n_maps}: {len(entries)} entries, +{added} articles, "
              f"{kids} child maps (skipped {skipped}), total {len(articles)}", flush=True)
        if n_maps % 20 == 0:
            yield
        time.sleep(args.delay + random.random() * args.delay)
    yield


def save(path, fox, nbc):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url", "date"])
        for d in (fox, nbc):
            for u in sorted(d):
                w.writerow([u, d[u]])
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scraped_dated.csv")
    ap.add_argument("--since", default="", help="YYYY-MM-DD, keep articles on/after this date")
    ap.add_argument("--until", default="", help="YYYY-MM-DD, keep articles on/before this date")
    ap.add_argument("--keep-undated", action="store_true",
                    help="with --since/--until, also keep articles that have no date")
    ap.add_argument("--max-sitemaps", type=int, default=400, help="per outlet")
    ap.add_argument("--max-urls", type=int, default=60000, help="per outlet")
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between requests")
    args = ap.parse_args()

    fox, nbc = {}, {}
    if os.path.exists(args.out):           # resume
        with open(args.out, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                (fox if label_from_url(r["url"]) == 0 else nbc)[r["url"]] = r.get("date", "") or ""
        print(f"resuming with fox={len(fox)} nbc={len(nbc)} from {args.out}")

    try:
        for _ in crawl_site("fox", SITES["fox"], args, fox):
            save(args.out, fox, nbc)
        for _ in crawl_site("nbc", SITES["nbc"], args, nbc):
            save(args.out, fox, nbc)
    except KeyboardInterrupt:
        print("\nstopped by user")
    save(args.out, fox, nbc)
    dated = sum(1 for d in (fox, nbc) for v in d.values() if v)
    print(f"saved {args.out}: fox={len(fox)} nbc={len(nbc)} ({dated} with dates)")


if __name__ == "__main__":
    main()