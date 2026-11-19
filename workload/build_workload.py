"""Builds a payment trace for sim-ln's trace replay from TabFormer card transactions and a Lightning graph snapshot,
following the payment workload plan (parts B1-B5): card payments, plus top-ups from exchanges to leaves and sweeps
from merchant nodes to exchanges.

Steps:
1. Convert the graph with sim-ln's converter (scripts/describegraph_to_simln.py), so that the trace and the simulated
   network contain exactly the same nodes and channels. By default, directions that were disabled in the snapshot are
   blocked (max HTLC of 0), and channels disabled in both directions are dropped. With --unblock-leaf-inbound (on by
   default), a disabled direction into a leaf is kept usable instead: a peer usually disables its side when the leaf
   goes offline, and a wallet is taken to be online whenever it pays or is paid.
2. B1: classify nodes into roles by total capacity and degree.
3. B2: map TabFormer merchants onto hub and mid nodes, many merchants per node, in proportion to node capacity.
4. B3: give every leaf node a TabFormer user (users matched to leaves by rank), and replay that user's payments from the
   leaf with one time offset and one amount scale per clone.
5. B4: convert USD to msat at a fixed BTC price, keeping TabFormer's own timing.
6. B5: inflows, precomputed from running totals of the card payments (assuming they all succeed):
   - top-ups: once a leaf's card spend since its last top-up reaches --topup-threshold of its starting outbound
     balance, its home exchange (one per leaf, drawn in proportion to exchange capacity) sends it that amount, at most
     what the leaf's largest channel could receive at the start; anything left over counts towards the next top-up.
   - sweeps: once a hub or mid node's card receipts since its last sweep reach --sweep-threshold of its starting
     inbound balance, it sends them to an exchange drawn in proportion to capacity, split into payments no larger
     than it could send over its largest channel at the start.
   Each goes out --inflow-delay-s after the card payment that triggered it, and sweep chunks follow at the same
   spacing. Inflows use their own random generator, so the card payments are the same as without them.

Usage (from workload/):
    python3 build_workload.py --graph data/raw/topology/graph_2026-11-19.json --btc-usd 63114.76

Outputs, in data/processed/workload/<graph name>_<config hash>/:
    sim.json          sim-ln simulation file: the converted network plus "trace": "trace.csv"
    trace.csv         time_secs,source,destination,amount_msat (sorted by time)
    payments.parquet  the trace with its type (card, topup, sweep) and provenance: user, merchant, clone, oversize flag
    node_roles.csv    pubkey, role, total_capacity_sat, degree
    merchant_map.csv  merchant_id, mcc (several separated by ';' if a merchant has more than one), node_pubkey,
                      window_volume_usd
    user_map.csv      leaf_pubkey, tabformer_user, time_offset_s, amount_scale, home_exchange
    manifest.json     parameters, input hashes and row counts
"""

import argparse
import bisect
import hashlib
import heapq
import json
import math
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, ".pylib"))

import duckdb  # noqa: E402

MSAT_PER_BTC = 100_000_000_000


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True, help="LND describegraph JSON with capacities filled in")
    ap.add_argument("--tabformer", default=os.path.join(HERE, "data", "raw", "tabformer", "transactions.parquet"))
    ap.add_argument("--out-root", default=os.path.join(HERE, "data", "processed", "workload"))
    ap.add_argument("--btc-usd", type=float, required=True, help="fixed USD price of 1 BTC for converting amounts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window-year", type=int, default=2019, help="TabFormer calendar year to replay")
    ap.add_argument("--online-only", action="store_true", help="use only online card transactions")
    ap.add_argument("--disabled", choices=["keep", "drop", "block"], default="block",
                    help="how the converter treats directions disabled in the snapshot (see the converter)")
    ap.add_argument("--unblock-leaf-inbound", action=argparse.BooleanOptionalAction, default=True,
                    help="with --disabled block, keep a disabled direction usable when it points into a leaf")
    ap.add_argument("--hub-top-frac", type=float, default=0.02)
    ap.add_argument("--n-exchanges", type=int, default=10)
    ap.add_argument("--leaf-max-degree", type=int, default=3)
    ap.add_argument("--merchant-share-hubs", type=float, default=0.70)
    ap.add_argument("--user-assignment", choices=["rank", "random"], default="rank")
    ap.add_argument("--rank-window", type=float, default=0.05,
                    help="with rank assignment, users within this many percentile points (as a fraction) of the "
                         "leaf's rank are candidates")
    ap.add_argument("--clone-time-jitter-days", type=float, default=7.0)
    ap.add_argument("--clone-amount-sigma", type=float, default=0.2)
    ap.add_argument("--payment-amount-sigma", type=float, default=0.0,
                    help="per-payment lognormal amount noise (0 = off)")
    ap.add_argument("--sender-fraction", type=float, default=1.0)
    ap.add_argument("--topup-threshold", type=float, default=0.5,
                    help="share of a leaf's starting outbound balance spent before its exchange tops it up (0 = no "
                         "top-ups)")
    ap.add_argument("--sweep-threshold", type=float, default=0.10,
                    help="share of a merchant node's starting inbound balance received before it sweeps to an "
                         "exchange (0 = no sweeps)")
    ap.add_argument("--inflow-delay-s", type=float, default=60.0,
                    help="delay between the card payment that triggers a top-up or sweep and its first payment, and "
                         "between a sweep's chunks")
    return ap.parse_args(argv)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_graph(graph_path, out_path, disabled):
    """Runs sim-ln's converter, returning its summary output."""
    cmd = [sys.executable, os.path.join(REPO, "scripts", "describegraph_to_simln.py"), "--disabled", disabled,
           graph_path, out_path]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stderr


def node_stats(sim_network):
    """Returns per-node dicts, all at the start of a simulation (every channel split 50/50):
    - capacity: total capacity (sat); degree: channel count;
    - max_send / max_recv: the largest amount (msat) the node could send / receive over a single channel (half the
      capacity, at most the sending side's max HTLC);
    - out_balance / in_balance: total balance (msat) it could send / receive, over directions that aren't blocked.
    A node can receive if max_recv > 0."""
    stats = {k: {} for k in ("capacity", "degree", "max_send", "max_recv", "out_balance", "in_balance")}
    for c in sim_network:
        half = c["capacity_msat"] // 2
        for side, other in (("node_1", "node_2"), ("node_2", "node_1")):
            pk = c[side]["pubkey"]
            out_htlc, in_htlc = c[side]["max_htlc_size_msat"], c[other]["max_htlc_size_msat"]
            for key, value in (
                ("capacity", c["capacity_msat"] // 1000),
                ("degree", 1),
                ("out_balance", half if out_htlc > 0 else 0),
                ("in_balance", half if in_htlc > 0 else 0),
            ):
                stats[key][pk] = stats[key].get(pk, 0) + value
            stats["max_send"][pk] = max(stats["max_send"].get(pk, 0), min(half, out_htlc))
            stats["max_recv"][pk] = max(stats["max_recv"].get(pk, 0), min(half, in_htlc))
    return stats


def classify_roles(capacity, degree, hub_top_frac, n_exchanges, leaf_max_degree):
    """B1: hubs are the top share of nodes by capacity, exchanges the largest hubs, leaves small-degree non-hubs."""
    ranked = sorted(capacity, key=lambda pk: (-capacity[pk], pk))
    n_hubs = max(1, round(hub_top_frac * len(ranked)))
    roles = {}
    for i, pk in enumerate(ranked):
        if i < min(n_exchanges, n_hubs):
            roles[pk] = "exchange"
        elif i < n_hubs:
            roles[pk] = "hub"
        elif degree[pk] <= leaf_max_degree:
            roles[pk] = "leaf"
        else:
            roles[pk] = "mid"
    return roles


def assign_merchants(merchants, hub_nodes, mid_nodes, capacity, share_hubs):
    """B2: splits merchants (sorted by volume, largest first) into a hub pool covering share_hubs of volume and a mid
    pool, then fills each pool's nodes greedily so that node volume is proportional to node capacity.

    merchants: list of (merchant_id, volume). Returns {merchant_id: pubkey}.
    """
    merchants = sorted(merchants, key=lambda m: (-m[1], m[0]))
    total = sum(v for _, v in merchants)
    hub_pool, mid_pool, running = [], [], 0.0
    for m in merchants:
        (hub_pool if running < share_hubs * total else mid_pool).append(m)
        running += m[1]

    mapping = {}
    for pool, nodes in ((hub_pool, hub_nodes), (mid_pool, mid_nodes)):
        if not pool:
            continue
        if not nodes:
            raise SystemExit("a merchant pool has no nodes to map to")
        pool_volume = sum(v for _, v in pool)
        pool_capacity = sum(capacity[pk] for pk in nodes)
        # Max-heap of remaining budget; ties broken by pubkey so the result is deterministic.
        heap = [(-pool_volume * capacity[pk] / pool_capacity, pk) for pk in sorted(nodes)]
        heapq.heapify(heap)
        for merchant_id, volume in pool:
            remaining, pk = heapq.heappop(heap)
            mapping[merchant_id] = pk
            heapq.heappush(heap, (remaining + volume, pk))
    return mapping


def assign_users(leaves, users, capacity, rng, method, rank_window):
    """B3: gives each leaf a user. With "rank", leaves and users are sorted by capacity and spend, and a leaf at
    percentile q draws uniformly from users whose percentile is within rank_window of q.

    leaves: list of pubkeys. users: list of (user_id, spend). Returns {leaf: user_id}.
    """
    if method == "random":
        ids = sorted(u for u, _ in users)
        return {leaf: rng.choice(ids) for leaf in sorted(leaves)}

    users = sorted(users, key=lambda u: (u[1], u[0]))
    leaves = sorted(leaves, key=lambda pk: (capacity[pk], pk))
    user_pct = [i / (len(users) - 1) if len(users) > 1 else 0.5 for i in range(len(users))]
    mapping = {}
    for i, leaf in enumerate(leaves):
        q = i / (len(leaves) - 1) if len(leaves) > 1 else 0.5
        lo = bisect.bisect_left(user_pct, q - rank_window)
        hi = bisect.bisect_right(user_pct, q + rank_window)
        mapping[leaf] = users[rng.randrange(lo, hi)][0]
    return mapping


def build_inflows(cards, roles, stats, rng, topup_threshold, sweep_threshold, delay_s):
    """B5: top-ups and sweeps from running totals of the card payments.

    cards: (time_secs, source, destination, amount_msat) in time order. Returns a list of
    (time_secs, source, destination, amount_msat, type) and each leaf's home exchange."""
    exchanges = sorted(pk for pk, r in roles.items() if r == "exchange" and stats["max_send"][pk] > 0)
    weights = [stats["capacity"][pk] for pk in exchanges]
    senders = sorted({source for _, source, _, _ in cards})
    home = {leaf: rng.choices(exchanges, weights)[0] for leaf in senders} if topup_threshold > 0 else {}

    spent, received, inflows = {}, {}, []
    for time_secs, source, destination, amount in cards:
        if topup_threshold > 0 and stats["max_recv"][source] > 0:
            spent[source] = spent.get(source, 0) + amount
            if spent[source] >= topup_threshold * stats["out_balance"][source]:
                topup = min(spent[source], stats["max_recv"][source])
                spent[source] -= topup
                inflows.append((time_secs + delay_s, home[source], source, topup, "topup"))
        if sweep_threshold > 0 and roles[destination] in ("hub", "mid"):
            received[destination] = received.get(destination, 0) + amount
            if received[destination] >= sweep_threshold * stats["in_balance"][destination]:
                to = rng.choices(exchanges, weights)[0]
                remaining, chunk = received.pop(destination), stats["max_send"][destination]
                if chunk == 0:
                    continue  # can't send at all
                n = 0
                while remaining > 0:
                    n += 1
                    amount_now = min(remaining, chunk)
                    inflows.append((time_secs + n * delay_s, destination, to, amount_now, "sweep"))
                    remaining -= amount_now
    return inflows, home


def config_hash(args):
    keys = {k: v for k, v in vars(args).items() if k not in ("out_root", "tabformer", "graph")}
    keys["graph"] = os.path.basename(args.graph)
    return hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()[:8]


def main(argv=None):
    args = parse_args(argv)
    rng = random.Random(args.seed)
    graph_name = os.path.splitext(os.path.basename(args.graph))[0]
    out = os.path.join(args.out_root, f"{graph_name}_{config_hash(args)}")
    os.makedirs(out, exist_ok=True)

    # 1. Convert the graph with sim-ln's converter.
    network_path = os.path.join(out, "sim_network.json")
    conversion = convert_graph(args.graph, network_path, args.disabled)
    with open(network_path) as f:
        sim_network = json.load(f)["sim_network"]
    os.remove(network_path)
    stats = node_stats(sim_network)

    # 2. Roles. They depend only on capacity and degree, which unblocking directions below doesn't change.
    roles = classify_roles(stats["capacity"], stats["degree"], args.hub_top_frac, args.n_exchanges,
                           args.leaf_max_degree)

    unblocked = 0
    if args.unblock_leaf_inbound and args.disabled == "block":
        # The keep conversion has the same channels plus those disabled both ways, with every direction's own max HTLC.
        convert_graph(args.graph, network_path, "keep")
        with open(network_path) as f:
            original = {c["scid"]: c for c in json.load(f)["sim_network"]}
        os.remove(network_path)
        for c in sim_network:
            for side, other in (("node_1", "node_2"), ("node_2", "node_1")):
                kept = original[c["scid"]][side]["max_htlc_size_msat"]
                if c[side]["max_htlc_size_msat"] == 0 and kept > 0 and roles[c[other]["pubkey"]] == "leaf":
                    c[side]["max_htlc_size_msat"] = kept
                    unblocked += 1
        stats = node_stats(sim_network)
    capacity, degree, max_send = stats["capacity"], stats["degree"], stats["max_send"]
    can_receive = {pk: v > 0 for pk, v in stats["max_recv"].items()}
    by_role = {r: sorted(pk for pk, x in roles.items() if x == r) for r in ("exchange", "hub", "mid", "leaf")}

    # TabFormer window, with the plan's A3 filters.
    con = duckdb.connect()
    con.sql(
        f"""
        CREATE TABLE tx AS
        SELECT user_id, card_id, ts, amount_usd, merchant_id, mcc, channel
        FROM '{args.tabformer}'
        WHERE amount_usd > 0 AND NOT is_fraud AND errors IS NULL AND year(ts) = {args.window_year}
              {"AND channel = 'Online Transaction'" if args.online_only else ""}
        """
    )

    # 3. Merchants -> hub (incl. exchanges) and mid nodes that can receive. Volumes are summed as exact decimals, so
    # that ties and near-ties sort the same way on every run.
    merchants = [(m, float(v)) for m, v in con.sql("SELECT merchant_id, sum(amount_usd) FROM tx GROUP BY 1").fetchall()]
    receiving = {r: [pk for pk in by_role[r] if can_receive[pk]] for r in ("exchange", "hub", "mid")}
    merchant_map = assign_merchants(
        merchants, receiving["exchange"] + receiving["hub"], receiving["mid"], capacity, args.merchant_share_hubs
    )

    # 4. Users -> leaves, with one time offset and amount scale per clone.
    users = [(u, float(v)) for u, v in con.sql("SELECT user_id, sum(amount_usd) FROM tx GROUP BY 1").fetchall()]
    senders = [pk for pk in by_role["leaf"] if max_send[pk] > 0]
    if args.sender_fraction < 1.0:
        senders = sorted(rng.sample(senders, round(args.sender_fraction * len(senders))))
    user_map = assign_users(senders, users, capacity, rng, args.user_assignment, args.rank_window)
    jitter_s = args.clone_time_jitter_days * 86400
    clones = []
    for leaf in sorted(user_map):
        offset = rng.uniform(-jitter_s, jitter_s)
        scale = math.exp(rng.gauss(0.0, args.clone_amount_sigma)) if args.clone_amount_sigma > 0 else 1.0
        clones.append((leaf, user_map[leaf], offset, scale))

    # 5. Expand clones into payments: shift and wrap times within the window, scale and convert amounts.
    con.sql("CREATE TABLE merchant_map (merchant_id BIGINT, node_pubkey VARCHAR)")
    con.executemany("INSERT INTO merchant_map VALUES (?, ?)", list(merchant_map.items()))
    con.sql("CREATE TABLE user_map (leaf_pubkey VARCHAR, user_id INTEGER, time_offset_s DOUBLE, amount_scale DOUBLE)")
    con.executemany("INSERT INTO user_map VALUES (?, ?, ?, ?)", clones)
    con.sql("CREATE TABLE max_send (pubkey VARCHAR, max_send_msat UBIGINT)")
    con.executemany("INSERT INTO max_send VALUES (?, ?)", list(max_send.items()))

    window_start = f"TIMESTAMP '{args.window_year}-01-01'"
    window_len_s = (
        con.sql(f"SELECT epoch(TIMESTAMP '{args.window_year + 1}-01-01') - epoch({window_start})").fetchone()[0]
    )
    noise = (
        f"* exp({args.payment_amount_sigma} * sqrt(-2 * ln(random())) * cos(2 * pi() * random()))"
        if args.payment_amount_sigma > 0 else ""
    )
    if args.payment_amount_sigma > 0:
        con.sql(f"SELECT setseed({(args.seed % 1000) / 1000})")
    con.sql(
        f"""
        CREATE TABLE payments AS
        SELECT
            -- Shift by the clone's offset and wrap around the window, so every clone stays within one year.
            ((epoch(t.ts) - epoch({window_start}) + c.time_offset_s) % {window_len_s} + {window_len_s})
                % {window_len_s}                                                     AS time_secs,
            c.leaf_pubkey                                                            AS source,
            m.node_pubkey                                                            AS destination,
            greatest(1, round(t.amount_usd::DOUBLE * c.amount_scale {noise} / {args.btc_usd} * {MSAT_PER_BTC}))::UBIGINT
                                                                                     AS amount_msat,
            t.user_id                                                                AS source_user,
            t.card_id                                                                AS source_card,
            t.merchant_id                                                            AS source_merchant,
            t.mcc,
            t.channel                                                                AS card_channel,
            t.ts                                                                     AS original_ts,
            t.amount_usd::DOUBLE                                                     AS original_amount_usd
        FROM user_map c
        JOIN tx t ON t.user_id = c.user_id
        JOIN merchant_map m ON m.merchant_id = t.merchant_id
        """
    )
    # 6. Inflows: top-ups and sweeps, from running totals of the card payments in time order. Their own generator keeps
    # the card payments the same whether or not inflows are on.
    cards = con.sql(
        "SELECT time_secs, source, destination, amount_msat FROM payments "
        "ORDER BY time_secs, source, original_ts, source_card, source_merchant, amount_msat"
    ).fetchall()
    inflows, home_exchange = build_inflows(
        cards, roles, stats, random.Random(f"{args.seed}-inflows"), args.topup_threshold, args.sweep_threshold,
        args.inflow_delay_s,
    )
    del cards
    # Loaded through a CSV file: executemany inserts one row at a time, which takes far too long for millions of rows.
    inflows_csv = os.path.join(out, "inflows.tmp.csv")
    with open(inflows_csv, "w") as f:
        f.write("time_secs,source,destination,amount_msat,type,seq\n")
        for i, (t, src, dst, amt, kind) in enumerate(inflows):
            f.write(f"{t!r},{src},{dst},{amt},{kind},{i}\n")
    con.sql(
        f"CREATE TABLE inflows AS SELECT * FROM read_csv('{inflows_csv}', header = true, columns = {{"
        "'time_secs': 'DOUBLE', 'source': 'VARCHAR', 'destination': 'VARCHAR', 'amount_msat': 'UBIGINT', "
        "'type': 'VARCHAR', 'seq': 'INTEGER'})"
    )
    os.remove(inflows_csv)
    con.sql(
        """
        CREATE TABLE payments_flagged AS
        SELECT p.*, p.amount_msat > s.max_send_msat AS oversize_flag
        FROM (
            SELECT time_secs, source, destination, amount_msat, 'card' AS type, source_user, source_card,
                source_merchant, mcc, card_channel, original_ts, original_amount_usd, NULL::INTEGER AS seq
            FROM payments
            UNION ALL BY NAME
            SELECT time_secs, source, destination, amount_msat, type, seq FROM inflows
        ) p JOIN max_send s ON s.pubkey = p.source
        -- A total order, so equal-time payments are always written in the same order.
        ORDER BY time_secs, source, type, original_ts, source_card, source_merchant, amount_msat, seq
        """
    )

    # Write outputs.
    con.sql(f"COPY payments_flagged TO '{os.path.join(out, 'payments.parquet')}' (FORMAT parquet, COMPRESSION zstd)")
    con.sql(
        f"COPY (SELECT round(time_secs, 3) AS time_secs, source, destination, amount_msat FROM payments_flagged) "
        f"TO '{os.path.join(out, 'trace.csv')}' (HEADER, DELIMITER ',')"
    )
    with open(os.path.join(out, "node_roles.csv"), "w") as f:
        f.write("pubkey,role,total_capacity_sat,degree\n")
        for pk in sorted(roles, key=lambda p: (-capacity[p], p)):
            f.write(f"{pk},{roles[pk]},{capacity[pk]},{degree[pk]}\n")
    con.sql(
        f"""
        COPY (SELECT t.merchant_id, string_agg(DISTINCT t.mcc::VARCHAR, ';' ORDER BY t.mcc::VARCHAR) AS mcc, m.node_pubkey, round(sum(t.amount_usd), 2) AS window_volume_usd
              FROM tx t JOIN merchant_map m USING (merchant_id) GROUP BY 1, 3 ORDER BY 4 DESC, 1)
        TO '{os.path.join(out, 'merchant_map.csv')}' (HEADER, DELIMITER ',')
        """
    )
    con.sql("CREATE TABLE home_exchange (leaf_pubkey VARCHAR, home_exchange VARCHAR)")
    con.executemany("INSERT INTO home_exchange VALUES (?, ?)", list(home_exchange.items()))
    con.sql(
        f"COPY (SELECT leaf_pubkey, user_id AS tabformer_user, round(time_offset_s, 3) AS time_offset_s, "
        f"round(amount_scale, 6) AS amount_scale, h.home_exchange "
        f"FROM user_map LEFT JOIN home_exchange h USING (leaf_pubkey) ORDER BY 1) "
        f"TO '{os.path.join(out, 'user_map.csv')}' (HEADER, DELIMITER ',')"
    )
    with open(os.path.join(out, "sim.json"), "w") as f:
        json.dump({"sim_network": sim_network, "trace": "trace.csv"}, f)

    counts = dict(zip(
        ("payments", "oversize", "senders", "receivers", "first_s", "last_s"),
        con.sql(
            "SELECT count(*), count(*) FILTER (WHERE oversize_flag), count(DISTINCT source), "
            "count(DISTINCT destination), min(time_secs), max(time_secs) FROM payments_flagged"
        ).fetchone(),
    ))
    counts["by_type"] = {
        t: {"payments": n, "volume_btc": round(v / MSAT_PER_BTC, 3), "senders": s_, "receivers": r_}
        for t, n, v, s_, r_ in con.sql(
            "SELECT type, count(*), sum(amount_msat), count(DISTINCT source), count(DISTINCT destination) "
            "FROM payments_flagged GROUP BY 1 ORDER BY 1"
        ).fetchall()
    }
    counts["leaves_that_cannot_receive_topups"] = sum(1 for pk in user_map if stats["max_recv"][pk] == 0)
    manifest = {
        "parameters": {k: v for k, v in vars(args).items() if k != "out_root"},
        "inputs": {
            "graph": {"path": os.path.abspath(args.graph), "sha256": sha256(args.graph)},
            "tabformer": {"path": os.path.abspath(args.tabformer), "sha256": sha256(args.tabformer)},
            "converter": f"scripts/describegraph_to_simln.py --disabled {args.disabled}",
        },
        "conversion_summary": conversion.strip().splitlines(),
        "directions_into_leaves_unblocked": unblocked,
        "roles": {r: len(v) for r, v in by_role.items()},
        "excluded": {
            "leaves that can't send (all outgoing directions blocked)": len(by_role["leaf"]) - len(senders),
            "hub/mid nodes that can't receive (all incoming directions blocked)": sum(
                len(by_role[r]) - len(receiving[r]) for r in ("exchange", "hub", "mid")
            ),
        },
        "window": {
            "year": args.window_year,
            "payments": con.sql("SELECT count(*) FROM tx").fetchone()[0],
            "users": len(users),
            "merchants": len(merchants),
        },
        "output": counts,
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=str)

    print(f"wrote {out}")
    print(json.dumps({k: manifest[k] for k in ("roles", "excluded", "window", "output")}, indent=1, default=str))
    return out


if __name__ == "__main__":
    main()
