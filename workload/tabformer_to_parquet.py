"""Step A2 of the payment workload plan: convert IBM TabFormer's credit card CSV to Parquet, once.

Nothing is filtered or mapped here: every row is kept, only parsed into typed columns.

Usage (from workload/):
    PYTHONPATH=.pylib python3 tabformer_to_parquet.py

Input:  data/raw/tabformer/card_transaction.v1.csv (from IBM/TabFormer data/credit_card/transactions.tgz)
Output: data/raw/tabformer/transactions.parquet
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".pylib"))

import duckdb  # noqa: E402

SRC = os.path.join(HERE, "data", "raw", "tabformer", "card_transaction.v1.csv")
DST = os.path.join(HERE, "data", "raw", "tabformer", "transactions.parquet")

duckdb.sql(
    f"""
    COPY (
        SELECT
            "User"::INTEGER                                           AS user_id,
            "Card"::INTEGER                                           AS card_id,
            make_timestamp("Year"::INTEGER, "Month"::INTEGER, "Day"::INTEGER,
                           split_part("Time", ':', 1)::INTEGER,
                           split_part("Time", ':', 2)::INTEGER, 0)    AS ts,
            replace("Amount", '$', '')::DECIMAL(12, 2)                AS amount_usd,
            "Use Chip"                                                AS channel,
            "Merchant Name"::BIGINT                                   AS merchant_id,
            "Merchant City"                                           AS merchant_city,
            "Merchant State"                                          AS merchant_state,
            "MCC"::INTEGER                                            AS mcc,
            nullif("Errors?", '')                                     AS errors,
            "Is Fraud?" = 'Yes'                                       AS is_fraud
        FROM read_csv('{SRC}', all_varchar = true)
        ORDER BY ts, user_id
    ) TO '{DST}' (FORMAT parquet, COMPRESSION zstd)
    """
)

rows = duckdb.sql(f"SELECT count(*) FROM '{DST}'").fetchone()[0]
print(f"wrote {DST}: {rows:,} rows, {os.path.getsize(DST) / 1e6:,.0f} MB")
