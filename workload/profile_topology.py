"""Profiles a Lightning graph snapshot for the payment workload plan: capacity and degree distributions, and how the
nodes would split into the plan's roles (step B1) under given thresholds. Nothing is assigned or written for use by
the mapping; this is a report to choose the thresholds from.

Only channels that sim-ln can run are counted: both directions have a policy and the capacity is known and non-zero
(as in scripts/describegraph_to_simln.py). Disabled channels are counted unless --drop-disabled is passed.

Usage (from workload/):
    python3 profile_topology.py GRAPH.json [--hub-top-frac 0.02] [--n-exchanges 10] [--leaf-max-degree 3]

Output: data/processed/topology_profile_<graph name>.md
"""

import argparse
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("graph")
ap.add_argument("--hub-top-frac", type=float, default=0.02)
ap.add_argument("--n-exchanges", type=int, default=10)
ap.add_argument("--leaf-max-degree", type=int, default=3)
ap.add_argument("--drop-disabled", action="store_true")
args = ap.parse_args()

graph = json.load(open(args.graph))
stats = collections.Counter()
capacity = collections.Counter()  # node -> total capacity (sat)
degree = collections.Counter()
for e in graph["edges"]:
    p1, p2 = e.get("node1_policy"), e.get("node2_policy")
    cap = int(e.get("capacity") or 0)
    if not p1 or not p2:
        stats["skipped: missing a policy"] += 1
        continue
    if cap == 0:
        stats["skipped: no capacity"] += 1
        continue
    if args.drop_disabled and (p1.get("disabled") or p2.get("disabled")):
        stats["skipped: disabled"] += 1
        continue
    stats["channels"] += 1
    for pk in (e["node1_pub"], e["node2_pub"]):
        capacity[pk] += cap
        degree[pk] += 1

nodes = sorted(capacity, key=lambda pk: (-capacity[pk], pk))
n = len(nodes)
total_cap = sum(capacity.values()) / 2  # each channel counted at both ends


def pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


n_hubs = max(1, round(args.hub_top_frac * n))
hubs = set(nodes[:n_hubs])
exchanges = set(nodes[: min(args.n_exchanges, n_hubs)])
roles = {}
for pk in nodes:
    if pk in exchanges:
        roles[pk] = "exchange (hub)"
    elif pk in hubs:
        roles[pk] = "hub"
    elif degree[pk] <= args.leaf_max_degree:
        roles[pk] = "leaf"
    else:
        roles[pk] = "mid"

lines = [
    f"# Topology profile: {os.path.basename(args.graph)}",
    "",
    f"- Channels counted: {stats['channels']:,} ({', '.join(f'{k}: {v:,}' for k, v in stats.items() if k != 'channels')}).",
    f"- Nodes with at least one counted channel: {n:,}. Total capacity: {total_cap / 1e8:,.1f} BTC.",
    "",
    "| | min | median | p90 | p99 | max |",
    "|---|---|---|---|---|---|",
]
caps = [capacity[pk] for pk in nodes]
degs = [degree[pk] for pk in nodes]
lines.append(
    "| node capacity (sat) | " + " | ".join(f"{v:,}" for v in (min(caps), pct(caps, .5), pct(caps, .9), pct(caps, .99), max(caps))) + " |"
)
lines.append(
    "| node degree | " + " | ".join(f"{v:,}" for v in (min(degs), pct(degs, .5), pct(degs, .9), pct(degs, .99), max(degs))) + " |"
)

lines += [
    "",
    f"## Roles under: hub = top {args.hub_top_frac:.0%} by capacity, exchanges = top {args.n_exchanges} hubs, "
    f"leaf = degree <= {args.leaf_max_degree} and not a hub",
    "",
    "| role | nodes | share of nodes | share of node capacity | median capacity (sat) | median degree |",
    "|---|---|---|---|---|---|",
]
by_role = collections.defaultdict(list)
for pk, r in roles.items():
    by_role[r].append(pk)
all_node_cap = sum(caps)
for r in ("exchange (hub)", "hub", "mid", "leaf"):
    members = by_role[r]
    if not members:
        continue
    lines.append(
        f"| {r} | {len(members):,} | {len(members) / n:.1%} | {sum(capacity[p] for p in members) / all_node_cap:.1%} | "
        f"{pct([capacity[p] for p in members], .5):,} | {pct([degree[p] for p in members], .5):,} |"
    )

lines += ["", "Degree distribution:", "", "| degree | nodes |", "|---|---|"]
for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 10), (11, 100), (101, 10**9)):
    label = str(lo) if lo == hi else (f"{lo}+" if hi > 10**6 else f"{lo}-{hi}")
    lines.append(f"| {label} | {sum(lo <= d <= hi for d in degs):,} |")

os.makedirs(os.path.join(HERE, "data", "processed"), exist_ok=True)
out = os.path.join(HERE, "data", "processed", f"topology_profile_{os.path.splitext(os.path.basename(args.graph))[0]}.md")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {out}")
