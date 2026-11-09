"""Tests for describegraph_to_simln.py. Run with: python3 -m unittest discover scripts"""

import unittest

from describegraph_to_simln import convert_graph

A = "02" + "aa" * 32
B = "03" + "bb" * 32
C = "02" + "cc" * 32


def policy(**overrides):
    p = {
        "time_lock_delta": 40,
        "min_htlc": "1000",
        "fee_base_msat": "1000",
        "fee_rate_milli_msat": "100",
        "disabled": False,
    }
    p.update(overrides)
    return p


def edge(channel_id, node1, node2, capacity="1000000", p1=None, p2=None):
    return {
        "channel_id": channel_id,
        "node1_pub": node1,
        "node2_pub": node2,
        "capacity": capacity,
        "node1_policy": p1 if p1 is not None else policy(),
        "node2_policy": p2 if p2 is not None else policy(),
    }


def graph(edges, aliases=None):
    aliases = aliases or {}
    return {
        "nodes": [{"pub_key": pk, "alias": aliases.get(pk, "")} for pk in (A, B, C)],
        "edges": edges,
    }


class ConvertGraphTest(unittest.TestCase):
    def test_maps_fields_and_policy_directions(self):
        sim, stats = convert_graph(
            graph(
                [
                    edge(
                        "557504072480260097",
                        A,
                        B,
                        p1=policy(time_lock_delta=14, fee_base_msat="500", fee_rate_milli_msat="10"),
                        p2=policy(min_htlc="1", max_htlc_msat="400000000"),
                    )
                ],
                aliases={A: "alice", B: "bob"},
            )
        )

        self.assertEqual(stats["channels"], 1)
        self.assertEqual(stats["nodes"], 2)
        channel = sim["sim_network"][0]
        self.assertEqual(channel["scid"], 557504072480260097)
        self.assertEqual(channel["capacity_msat"], 1_000_000_000)

        # node1_policy is the policy node 1 sets for forwarding out over the channel, which is sim-ln's node_1.
        node_1, node_2 = channel["node_1"], channel["node_2"]
        self.assertEqual((node_1["pubkey"], node_1["alias"]), (A, "alice"))
        self.assertEqual(node_1["cltv_expiry_delta"], 14)
        self.assertEqual(node_1["base_fee"], 500)
        self.assertEqual(node_1["fee_rate_prop"], 10)
        self.assertEqual(node_2["pubkey"], B)
        self.assertEqual(node_2["min_htlc_size_msat"], 1)

        # Defaults for fields that gossip doesn't carry.
        self.assertEqual(node_1["max_htlc_count"], 483)
        self.assertEqual(node_1["max_in_flight_msat"], 1_000_000_000)
        self.assertEqual(node_1["max_htlc_size_msat"], 1_000_000_000)
        self.assertEqual(node_2["max_htlc_size_msat"], 400_000_000)

    def test_max_htlc_never_exceeds_capacity(self):
        sim, _ = convert_graph(graph([edge("1", A, B, p1=policy(max_htlc_msat="5000000000"))]))
        self.assertEqual(sim["sim_network"][0]["node_1"]["max_htlc_size_msat"], 1_000_000_000)

    def test_skips_channels_sim_ln_would_reject(self):
        _, stats = convert_graph(
            graph(
                [
                    edge("1", A, B),
                    edge("2", A, C, p2={}),
                    edge("3", B, C, capacity="0"),
                    edge("4", C, C),
                    edge("1", B, C),
                ]
            )
        )
        self.assertEqual(stats["channels"], 1)
        self.assertEqual(stats["skipped: missing a policy"], 1)
        self.assertEqual(stats["skipped: zero capacity"], 1)
        self.assertEqual(stats["skipped: both ends are the same node"], 1)
        self.assertEqual(stats["skipped: duplicate channel id"], 1)

    def test_disabled_channels(self):
        edges = [edge("1", A, B, p1=policy(disabled=True)), edge("2", B, C)]

        sim, stats = convert_graph(graph(edges))
        self.assertEqual(len(sim["sim_network"]), 2)
        self.assertEqual(stats["kept, with a disabled direction treated as usable"], 1)

        sim, stats = convert_graph(graph(edges), drop_disabled=True)
        self.assertEqual([c["scid"] for c in sim["sim_network"]], [2])
        self.assertEqual(stats["skipped: disabled"], 1)

    def test_clears_duplicate_aliases(self):
        sim, stats = convert_graph(
            graph([edge("1", A, B), edge("2", B, C)], aliases={A: "Satoshi", B: "Satoshi", C: "carol"})
        )
        aliases = {}
        for c in sim["sim_network"]:
            for side in ("node_1", "node_2"):
                aliases[c[side]["pubkey"]] = c[side]["alias"]
        self.assertEqual(aliases, {A: "", B: "", C: "carol"})
        self.assertEqual(stats["aliases cleared (not unique)"], 2)


if __name__ == "__main__":
    unittest.main()
