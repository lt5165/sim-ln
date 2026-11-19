# Payment workload from card transactions

Builds a trace for sim-ln's trace replay (`time_secs,source,destination,amount_msat`) by mapping IBM's TabFormer
synthetic credit card transactions onto a Lightning graph snapshot, following the payment workload plan (parts A and
B1-B4; card payments only). Raw and generated data live in `data/`, which is git-ignored.

## Setup

DuckDB is the only dependency (`requirements.txt`). Without pip or venv, unpack its wheel into `.pylib/` (the scripts
add `.pylib/` to the import path):

```
pip install --target .pylib -r requirements.txt     # or unpack the wheel from PyPI with python3 -m zipfile -e
```

## Steps

```
# A1: TabFormer (git-lfs file, fetched directly; SHA-256 must match the LFS pointer's oid)
mkdir -p data/raw/tabformer && cd data/raw/tabformer
curl -LO https://media.githubusercontent.com/media/IBM/TabFormer/main/data/credit_card/transactions.tgz
tar -xzf transactions.tgz && cd -

python3 tabformer_to_parquet.py          # A2: CSV -> data/raw/tabformer/transactions.parquet (no filtering)
python3 profile_tabformer.py             # A4: data/processed/tabformer_profile.md
python3 profile_topology.py GRAPH.json --drop-disabled   # role counts for a snapshot (report only)

python3 build_workload.py --graph GRAPH.json --btc-usd 63114.76   # B1-B4: the trace
```

`GRAPH.json` is an LND `describegraph` dump with capacities filled in.

## Choices for the current workload

| Parameter | Value | Why |
|---|---|---|
| `--graph` | `graph_2026-11-19.json` (own LND node, capacities from mempool.space) | current topology |
| `--disabled` | `block` | disabled directions get max HTLC 0; channels disabled both ways are dropped |
| `--btc-usd` | 63114.76 | fixed price, no rescaling of card amounts |
| timing | TabFormer's own pace, no compression | keeps each user's rhythm |
| `--window-year` | 2019 | last full year in TabFormer |
| roles | plan defaults: hubs = top 2% by capacity, exchanges = top 10, leaves = degree <= 3 | |
| `--merchant-share-hubs` | 0.70 | plan default |
| `--user-assignment`, `--rank-window` | rank, 0.05 | leaves draw users within +-5 percentile points of their own rank |
| clones | time jitter +-7 days, amount scale lognormal sigma 0.2, no per-payment noise | plan defaults; per-payment noise left off |
| payment types | card payments only | no p2p, exchange or sweep traffic yet |
| `--seed` | 42 | outputs are byte-identical across runs |

Senders are leaves with at least one usable outgoing direction, and merchants are only mapped to hubs and mid nodes with
at least one usable incoming direction.

## Outputs

`data/processed/workload/<graph>_<config hash>/`: `sim.json` (network + `"trace": "trace.csv"`, ready to run),
`trace.csv`, `payments.parquet` (the trace with source user, card, merchant, MCC, original timestamp and amount, and an
`oversize_flag` for payments larger than the sender could send over any single channel at the start),
`node_roles.csv`, `merchant_map.csv`, `user_map.csv` and `manifest.json` (parameters, input hashes, counts).
