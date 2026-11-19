"""Step A4 of the payment workload plan: profile TabFormer before any mapping.

Applies only the plan's step A3 filters (drop refunds/reversals with amount <= 0, fraud, and rows with errors), and
reports on the whole dataset and on one window (by default the last full calendar year).

Usage (from workload/):
    python3 profile_tabformer.py [--window-year YEAR]

Input:  data/raw/tabformer/transactions.parquet (from tabformer_to_parquet.py)
Output: data/processed/tabformer_profile.md
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".pylib"))

import duckdb  # noqa: E402

SRC = os.path.join(HERE, "data", "raw", "tabformer", "transactions.parquet")
OUT_DIR = os.path.join(HERE, "data", "processed")

ap = argparse.ArgumentParser()
ap.add_argument("--window-year", type=int, help="calendar year to profile (default: last full year in the data)")
args = ap.parse_args()

con = duckdb.connect()
con.sql(f"CREATE VIEW raw AS SELECT * FROM '{SRC}'")
con.sql("CREATE VIEW clean AS SELECT * FROM raw WHERE amount_usd > 0 AND NOT is_fraud AND errors IS NULL")


def one(q):
    return con.sql(q).fetchone()


def table(q, headers=None):
    rel = con.sql(q)
    cols = headers or rel.columns
    rows = rel.fetchall()
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt(c, v) for c, v in zip(cols, r)) + " |")
    return "\n".join(out)


# Columns that are identifiers or labels, written without thousands separators.
PLAIN = {"year", "mcc", "hour", "day"}


def fmt(col, v):
    if col in PLAIN or v is None:
        return str(v)
    if col.startswith("pct"):
        return f"{float(v):.1f}%"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float) or hasattr(v, "as_tuple"):
        return f"{float(v):,.2f}"
    return str(v)


lines = ["# TabFormer profile", ""]

# Whole dataset, before and after the A3 filters.
n_raw, first, last = one("SELECT count(*), min(ts), max(ts) FROM raw")
n_refund = one("SELECT count(*) FROM raw WHERE amount_usd <= 0")[0]
n_fraud = one("SELECT count(*) FROM raw WHERE is_fraud")[0]
n_err = one("SELECT count(*) FROM raw WHERE errors IS NOT NULL")[0]
n_clean = one("SELECT count(*) FROM clean")[0]
lines += [
    "## Whole dataset",
    "",
    f"- Rows: {n_raw:,}, from {first:%Y-%m-%d} to {last:%Y-%m-%d}.",
    f"- Dropped by the A3 filters (rows can match more than one): amount <= 0: {n_refund:,}; fraud: {n_fraud:,}; "
    f"errors: {n_err:,}. Remaining: {n_clean:,} ({n_clean / n_raw:.1%}).",
    "",
    "Rows per year after filtering:",
    "",
    table(
        "SELECT year(ts) AS year, count(*) AS payments, count(DISTINCT user_id) AS users, "
        "count(DISTINCT merchant_id) AS merchants FROM clean GROUP BY 1 ORDER BY 1"
    ),
    "",
]

# Window: last full calendar year by default.
last_full_year = last.year - 1 if (last.month, last.day) != (12, 31) else last.year
year = args.window_year or last_full_year
con.sql(f"CREATE VIEW w AS SELECT * FROM clean WHERE year(ts) = {year}")
n, users, merchants, cards = one(
    "SELECT count(*), count(DISTINCT user_id), count(DISTINCT merchant_id), count(DISTINCT (user_id, card_id)) FROM w"
)
lines += [
    f"## Window: {year}",
    "",
    f"- Payments: {n:,}; users: {users:,}; cards: {cards:,}; merchants: {merchants:,}.",
    "",
    "### Channels",
    "",
    table(
        "SELECT channel, count(*) AS payments, round(100 * count(*) / sum(count(*)) OVER (), 1) AS pct_payments, "
        "round(100 * sum(amount_usd) / sum(sum(amount_usd)) OVER (), 1) AS pct_volume FROM w GROUP BY 1 ORDER BY 2 DESC"
    ),
    "",
    "### Amounts (USD)",
    "",
    table(
        "SELECT round(avg(amount_usd), 2) AS mean, quantile_cont(amount_usd, 0.5) AS median, "
        "quantile_cont(amount_usd, 0.9) AS p90, quantile_cont(amount_usd, 0.99) AS p99, max(amount_usd) AS max, "
        "round(sum(amount_usd)) AS total FROM w"
    ),
    "",
    "Log-scale histogram:",
    "",
    table(
        """
        SELECT CASE WHEN amount_usd < 1 THEN 'under $1' WHEN amount_usd < 10 THEN '$1-10'
                    WHEN amount_usd < 100 THEN '$10-100' WHEN amount_usd < 1000 THEN '$100-1,000'
                    ELSE '$1,000+' END AS amount,
               count(*) AS payments, round(100 * count(*) / sum(count(*)) OVER (), 1) AS pct_payments,
               round(100 * sum(amount_usd) / sum(sum(amount_usd)) OVER (), 1) AS pct_volume
        FROM w GROUP BY 1 ORDER BY min(amount_usd)
        """
    ),
    "",
    "Amounts by channel:",
    "",
    table(
        "SELECT channel, quantile_cont(amount_usd, 0.5) AS median, round(avg(amount_usd), 2) AS mean, "
        "quantile_cont(amount_usd, 0.99) AS p99 FROM w GROUP BY 1 ORDER BY 1"
    ),
    "",
    "### Users",
    "",
    table(
        """
        WITH per_user AS (
            SELECT user_id, count(*) AS n, count(DISTINCT merchant_id) AS merchants, sum(amount_usd) AS spend
            FROM w GROUP BY 1)
        SELECT round(avg(n) / 365, 2) AS payments_per_day_mean, quantile_cont(n, 0.5) / 365 AS payments_per_day_median,
               round(avg(merchants), 1) AS distinct_merchants_mean, quantile_cont(merchants, 0.5) AS distinct_merchants_median,
               round(avg(spend)) AS spend_mean_usd, quantile_cont(spend, 0.5) AS spend_median_usd,
               quantile_cont(spend, 0.99) AS spend_p99_usd
        FROM per_user
        """
    ),
    "",
    "### Repeat pairs",
    "",
]

repeat = one(
    """
    SELECT avg(CASE WHEN rn > 1 THEN 1 ELSE 0 END)
    FROM (SELECT row_number() OVER (PARTITION BY user_id, merchant_id ORDER BY ts) AS rn FROM w)
    """
)[0]
lines += [
    f"Share of payments where the user had already paid the same merchant earlier in the window: {repeat:.1%}.",
    "",
    "### Merchant concentration",
    "",
]

conc = con.sql(
    """
    WITH m AS (SELECT merchant_id, sum(amount_usd) AS vol, count(*) AS n FROM w GROUP BY 1),
         r AS (SELECT *, row_number() OVER (ORDER BY vol DESC) AS rk, count(*) OVER () AS total,
                      sum(vol) OVER () AS all_vol FROM m)
    SELECT sum(vol) FILTER (WHERE rk <= greatest(1, total * 0.01)) / any_value(all_vol),
           sum(vol) FILTER (WHERE rk <= greatest(1, total * 0.10)) / any_value(all_vol),
           quantile_cont(n, 0.5), max(n)
    FROM r
    """
).fetchone()
lines += [
    f"- Top 1% of merchants by volume receive {conc[0]:.1%} of volume; top 10% receive {conc[1]:.1%}.",
    f"- Payments per merchant: median {conc[2]:,.0f}, max {conc[3]:,}.",
    "",
    "Top merchant categories (MCC) by volume:",
    "",
    table(
        "SELECT mcc, count(DISTINCT merchant_id) AS merchants, count(*) AS payments, "
        "round(100 * sum(amount_usd) / sum(sum(amount_usd)) OVER (), 1) AS pct_volume "
        "FROM w GROUP BY 1 ORDER BY sum(amount_usd) DESC LIMIT 10"
    ),
    "",
    "### Timing",
    "",
    "Hour of day:",
    "",
    table(
        "SELECT hour(ts) AS hour, round(100 * count(*) / sum(count(*)) OVER (), 1) AS pct_payments FROM w "
        "GROUP BY 1 ORDER BY 1"
    ),
    "",
    "Day of week (0 = Sunday):",
    "",
    table(
        "SELECT dayofweek(ts) AS day, round(100 * count(*) / sum(count(*)) OVER (), 1) AS pct_payments FROM w "
        "GROUP BY 1 ORDER BY 1"
    ),
    "",
]

os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, "tabformer_profile.md")
with open(out, "w") as f:
    f.write("\n".join(lines))
print(f"wrote {out}")
