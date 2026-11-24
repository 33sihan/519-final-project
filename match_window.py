"""
match_window.py  —  make the scraped links look like the course data.

    py match_window.py url_train.csv scraped_dated.csv
    py match_window.py url_train.csv scraped_dated.csv scraped_window.csv --out scraped_matched.csv

1. Finds url_train.csv headlines that also appear in the scraped links, and uses their
   sitemap dates to estimate the time period the course data comes from.
2. Prints how the scraped links are spread over the years for each outlet (if Fox and
   NBC come from different years, the model learns "which year" instead of "which outlet").
3. Writes --out with only scraped links inside that period, balanced month by month
   (same number of Fox and NBC links in every month), so time can't leak the label.
"""
import argparse

import pandas as pd

from preprocess import headline_from_url, label_from_url

NAMES = {0: "fox", 1: "nbc"}


def year_table(df, title):
    if df.empty:
        print(f"\n{title}: (none)")
        return
    t = (df.assign(year=df["date"].dt.year, outlet=df["label"].map(NAMES))
           .pivot_table(index="year", columns="outlet", values="url", aggfunc="count", fill_value=0))
    print(f"\n{title}")
    print(t.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_csv")
    ap.add_argument("scraped_csvs", nargs="+")
    ap.add_argument("--out", default="scraped_matched.csv")
    ap.add_argument("--margin-days", type=int, default=30)
    ap.add_argument("--since", default="", help="override the estimated start (YYYY-MM-DD)")
    ap.add_argument("--until", default="", help="override the estimated end (YYYY-MM-DD)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tr = pd.read_csv(args.train_csv)
    col = "url" if "url" in tr.columns else tr.columns[0]
    tr = pd.DataFrame({"url": tr[col]})
    tr["headline"] = tr["url"].map(headline_from_url)
    tr["label"] = tr["url"].map(label_from_url)

    sc = pd.concat([pd.read_csv(p, dtype=str).fillna("") for p in args.scraped_csvs],
                   ignore_index=True)
    if "date" not in sc.columns:
        raise SystemExit("These scraped files have no `date` column. Re-run the v3 "
                         "scrape_links.py (it records dates), then run this again.")
    sc = sc.drop_duplicates("url")
    sc["label"] = sc["url"].map(label_from_url)
    sc["headline"] = sc["url"].map(headline_from_url)
    sc["date"] = pd.to_datetime(sc["date"], errors="coerce")
    print(f"scraped links: {len(sc)}  (with a date: {sc['date'].notna().sum()})")
    print(sc["label"].map(NAMES).value_counts().to_string())

    dated = sc.dropna(subset=["date"])
    year_table(dated, "Scraped links per year (all):")

    # --- course-data time period from exact headline matches ---------------------
    first = dated.drop_duplicates(["headline", "label"])[["headline", "label", "date"]]
    hits = tr.merge(first, on=["headline", "label"], how="inner")
    print(f"\nurl_train.csv headlines found in the scrape (with a date): "
          f"{len(hits)} / {len(tr)}")
    if len(hits):
        print(hits["label"].map(NAMES).value_counts().to_string())
        year_table(hits.assign(url=hits["headline"]), "Matched url_train.csv headlines per year:")

    if args.since and args.until:
        lo, hi = pd.Timestamp(args.since), pd.Timestamp(args.until)
    elif len(hits) >= 20:
        margin = pd.Timedelta(days=args.margin_days)
        lo = hits["date"].quantile(0.02) - margin
        hi = hits["date"].quantile(0.98) + margin
        if args.since:
            lo = pd.Timestamp(args.since)
        if args.until:
            hi = pd.Timestamp(args.until)
    else:
        print("\nToo few matches to estimate a time period. Using every month where both "
              "outlets have links (still balanced month by month).")
        lo, hi = dated["date"].min(), dated["date"].max()

    print(f"\nTime period used: {lo.date()} .. {hi.date()}")

    win = dated[(dated["date"] >= lo) & (dated["date"] <= hi)].copy()
    win["month"] = win["date"].dt.to_period("M")
    parts = []
    for _, g in win.groupby("month"):
        f, n = g[g["label"] == 0], g[g["label"] == 1]
        k = min(len(f), len(n))
        if k:
            parts += [f.sample(k, random_state=args.seed), n.sample(k, random_state=args.seed)]
    out = pd.concat(parts) if parts else win.iloc[:0]
    out = out[~out["headline"].isin(set(tr["headline"]))]   # don't duplicate course data

    out[["url"]].assign(date=out["date"].dt.strftime("%Y-%m-%d")).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}: {len(out)} links "
          f"(fox={int((out['label'] == 0).sum())}, nbc={int((out['label'] == 1).sum())})")
    if len(out) < 20000:
        print("\nTip: to collect more links from this period, run:\n"
              f"  py scrape_links.py --since {lo.date()} --until {hi.date()} "
              f"--out scraped_window.csv --max-sitemaps 3000\n"
              f"then:\n"
              f"  py match_window.py {args.train_csv} {' '.join(args.scraped_csvs)} "
              f"scraped_window.csv --out {args.out}")


if __name__ == "__main__":
    main()