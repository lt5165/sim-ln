#!/usr/bin/env python3
"""Convert an LND `describegraph` JSON dump into a sim-ln simulated network.

Usage:
    lncli describegraph > graph.json
    python3 scripts/describegraph_to_simln.py graph.json sim.json

The output is a simulation file with a `sim_network` section, which can be run as is
(random activity) or extended with `activity`, `exclude` or `probing` sections.

Gossip doesn't carry everything that a simulated channel needs, so some fields are
filled with defaults:
- max_htlc_count: 483, the maximum allowed by BOLT 2.
- max_in_flight_msat: the channel capacity.
- max_htlc_size_msat: the policy's max_htlc_msat if present (older snapshots don't
  have it), otherwise the channel capacity, and never more than the capacity.
- Balances are never gossiped: sim-ln splits each channel's capacity 50/50.

Channels are skipped if either direction has no policy, if their capacity is zero,
if both ends are the same node, or if their channel id was already seen. sim-ln
can't represent disabled channels, so they are kept as usable unless
--drop-disabled is passed (which drops a channel if either direction is disabled).
Node aliases that are not unique are cleared, because sim-ln rejects duplicates.

Only the Python standard library is needed.
"""

import argparse
import collections
import json
import sys

# The maximum number of HTLCs a channel can have in flight in each direction (BOLT 2).
MAX_HTLC_COUNT = 483


def convert_policy(pubkey, alias, policy, capacity_msat):
    """Converts an LND routing policy into a sim-ln channel policy for the node that set it."""
    max_htlc_msat = int(policy.get("max_htlc_msat") or 0) or capacity_msat
    return {
        "pubkey": pubkey,
        "alias": alias,
        "max_htlc_count": MAX_HTLC_COUNT,
        "max_in_flight_msat": capacity_msat,
        "min_htlc_size_msat": int(policy.get("min_htlc") or 0),
        "max_htlc_size_msat": min(max_htlc_msat, capacity_msat),
        "cltv_expiry_delta": int(policy["time_lock_delta"]),
        "base_fee": int(policy["fee_base_msat"]),
        "fee_rate_prop": int(policy["fee_rate_milli_msat"]),
    }


def convert_graph(graph, drop_disabled=False):
    """Converts a parsed describegraph dump, returning the sim-ln simulation file and a counter of what was
    kept and skipped."""
    stats = collections.Counter()

    # sim-ln rejects duplicate aliases, but allows empty ones, so only unique aliases are kept.
    alias_counts = collections.Counter(n.get("alias", "") for n in graph["nodes"])
    aliases = {}
    for node in graph["nodes"]:
        alias = node.get("alias", "")
        if alias and alias_counts[alias] > 1:
            stats["aliases cleared (not unique)"] += 1
            alias = ""
        aliases[node["pub_key"]] = alias

    channels = []
    seen_ids = set()
    for edge in graph["edges"]:
        policy_1, policy_2 = edge.get("node1_policy"), edge.get("node2_policy")
        capacity_msat = int(edge["capacity"]) * 1000
        scid = int(edge["channel_id"])

        if not policy_1 or not policy_2:
            stats["skipped: missing a policy"] += 1
            continue
        if capacity_msat == 0:
            stats["skipped: zero capacity"] += 1
            continue
        if edge["node1_pub"] == edge["node2_pub"]:
            stats["skipped: both ends are the same node"] += 1
            continue
        if scid in seen_ids:
            stats["skipped: duplicate channel id"] += 1
            continue

        disabled = policy_1.get("disabled") or policy_2.get("disabled")
        if disabled and drop_disabled:
            stats["skipped: disabled"] += 1
            continue
        if disabled:
            stats["kept, with a disabled direction treated as usable"] += 1

        seen_ids.add(scid)
        channels.append(
            {
                "scid": scid,
                "capacity_msat": capacity_msat,
                "node_1": convert_policy(
                    edge["node1_pub"], aliases.get(edge["node1_pub"], ""), policy_1, capacity_msat
                ),
                "node_2": convert_policy(
                    edge["node2_pub"], aliases.get(edge["node2_pub"], ""), policy_2, capacity_msat
                ),
            }
        )
        stats["channels"] += 1

    nodes = {c["node_1"]["pubkey"] for c in channels} | {c["node_2"]["pubkey"] for c in channels}
    stats["nodes"] = len(nodes)
    return {"sim_network": channels}, stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert an LND describegraph JSON dump into a sim-ln simulation file."
    )
    parser.add_argument("graph", help="path to the describegraph JSON dump")
    parser.add_argument("output", help="path to write the sim-ln simulation file to")
    parser.add_argument(
        "--drop-disabled",
        action="store_true",
        help="drop channels with either direction disabled (by default they are kept as usable)",
    )
    args = parser.parse_args()

    with open(args.graph) as f:
        graph = json.load(f)

    sim, stats = convert_graph(graph, drop_disabled=args.drop_disabled)

    with open(args.output, "w") as f:
        json.dump(sim, f)

    for key, count in sorted(stats.items()):
        print(f"{key}: {count}", file=sys.stderr)


if __name__ == "__main__":
    main()
