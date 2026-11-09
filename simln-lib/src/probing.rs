//! Background probing for training the payment scorers of simulated nodes.
//!
//! Lightning nodes only learn about the liquidity of channels in the network by trying to send payments through them.
//! Background probing sends probes (payments that can't be settled by the recipient) to learn about the live state of
//! channels, which trains the node's scorer and improves the routes that it picks for real payments.
//!
//! This module is a port of LDK Node's background probing service (`ldk-node/src/probing.rs`) to the simulated
//! network. It provides the [`ProbingStrategy`] trait and LDK Node's two built-in strategies, [`HighDegreeStrategy`]
//! and [`RandomWalkStrategy`]. Probes are sent with [`SimNode::probe`], which updates the node's scorer in the same
//! way that LDK's background processor handles probe events.
//!
//! Differences from LDK Node, which are specific to the simulated network:
//! - Time is read from the simulation's [`Clock`], so that probing works with sped-up or virtual time.
//! - Randomness is drawn from a per-node seeded RNG, so that runs are reproducible when a seed is provided.
//! - [`HighDegreeStrategy`] breaks ties between nodes of equal degree by node id (for reproducibility) and skips the
//!   probing node itself.
//! - [`RandomWalkStrategy`] never reuses a channel within a path, because the simulated network treats the same payment
//!   hash being added to a channel twice as a critical error.

// Portions of this file are adapted from LDK Node (https://github.com/lightningdevkit/ldk-node), which is licensed
// under either of the Apache License, Version 2.0 or the MIT license, at your option.

use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use bitcoin::secp256k1::PublicKey;
use lightning::routing::gossip::NodeId;
use lightning::routing::router::{Path, RouteHop, MAX_PATH_LENGTH_ESTIMATE};
use lightning::routing::scoring::ProbabilisticScoringFeeParameters;
use lightning::types::features::{ChannelFeatures, NodeFeatures};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::clock::Clock;
use crate::sim_node::{find_payment_route_with_params, SimNetwork, SimNode};
use crate::LightningNode;

/// The CLTV expiry delta used for the final hop of probes that are built by hand. This matches the final CLTV delta
/// used for payments in the simulated network.
const FINAL_CLTV_EXPIRY_DELTA: u32 = 0;

/// Default multiplier used to avoid probing with a large share of a first hop's outbound liquidity: a path is skipped
/// if the first hop can send less than `path_value * DEFAULT_LIQUIDITY_LIMIT_MULTIPLIER`. Matches LDK Node.
pub const DEFAULT_LIQUIDITY_LIMIT_MULTIPLIER: u64 = 3;

/// A strategy that decides which path a node should probe next.
#[async_trait]
pub trait ProbingStrategy: Send + Sync {
    /// Returns the next path to probe, or `None` to skip this tick.
    async fn next_probe(&self) -> Option<Path>;
}

/// Creates the RNG used by a node's probing strategy. If a seed is provided, it is salted with the node's public key
/// (so each node gets a different sequence) and a probing-specific salt (so that probing does not share a sequence
/// with random activity generation).
pub(crate) fn probing_rng(seed: Option<u64>, node: &PublicKey) -> ChaCha8Rng {
    match seed {
        Some(seed) => {
            let mut hasher = DefaultHasher::new();
            node.serialize().hash(&mut hasher);
            seed.hash(&mut hasher);
            b"background-probing".hash(&mut hasher);
            ChaCha8Rng::seed_from_u64(hasher.finish())
        },
        None => ChaCha8Rng::from_entropy(),
    }
}

/// Returns a random value in the inclusive range `[min, max]`.
fn random_range(rng: &StdMutex<ChaCha8Rng>, min: u64, max: u64) -> u64 {
    debug_assert!(min <= max);
    rng.lock()
        .unwrap_or_else(|e| e.into_inner())
        .gen_range(min..=max)
}

/// Probes toward the most-connected nodes in the graph.
///
/// On each tick the strategy reads the graph, sorts nodes by channel count, and picks the highest-degree node from
/// the top `top_node_count` that has not been probed within `cooldown`. Nodes probed more recently are skipped so that
/// the strategy spreads across the top nodes. If all top nodes are on cooldown, the cooldown map is cleared and a new
/// cycle begins immediately.
///
/// The route to the chosen node is found with the node's own scorer, so probes follow the routes that real payments
/// would take. The probe amount is chosen uniformly at random from `[min_amount_msat, max_amount_msat]`.
pub struct HighDegreeStrategy<T: SimNetwork, C: Clock> {
    node: Arc<SimNode<T, C>>,
    /// Fee parameters used for pathfinding, which may include a probing diversity penalty. These are only used for
    /// probing, so they don't affect the routes picked for real payments.
    fee_params: ProbabilisticScoringFeeParameters,
    /// How many of the highest-degree nodes to cycle through.
    pub top_node_count: usize,
    /// Lower bound for the randomly chosen probe amount.
    pub min_amount_msat: u64,
    /// Upper bound for the randomly chosen probe amount.
    pub max_amount_msat: u64,
    /// How long a node stays ineligible after being probed.
    pub cooldown: Duration,
    /// Skip a path when the first-hop outbound liquidity is less than `path_value * liquidity_limit_multiplier`.
    pub liquidity_limit_multiplier: u64,
    /// Nodes probed recently, with the (simulation) time they were last probed.
    recently_probed: StdMutex<HashMap<NodeId, SystemTime>>,
    rng: StdMutex<ChaCha8Rng>,
}

impl<T: SimNetwork, C: Clock> HighDegreeStrategy<T, C> {
    /// Creates a new high-degree probing strategy. If `diversity_penalty_msat` is set, the scorer will penalize
    /// channels that have been recently probed when finding probe routes, encouraging path diversity.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        node: Arc<SimNode<T, C>>,
        top_node_count: usize,
        min_amount_msat: u64,
        max_amount_msat: u64,
        cooldown: Duration,
        liquidity_limit_multiplier: u64,
        diversity_penalty_msat: Option<u64>,
        seed: Option<u64>,
    ) -> Self {
        assert!(
            min_amount_msat <= max_amount_msat,
            "min_amount_msat must not exceed max_amount_msat"
        );

        let mut fee_params = ProbabilisticScoringFeeParameters::default();
        if let Some(penalty) = diversity_penalty_msat {
            fee_params.probing_diversity_penalty_msat = penalty;
        }

        let rng = StdMutex::new(probing_rng(seed, &node.get_info().pubkey));
        Self {
            node,
            fee_params,
            top_node_count,
            min_amount_msat,
            max_amount_msat,
            cooldown,
            liquidity_limit_multiplier,
            recently_probed: StdMutex::new(HashMap::new()),
            rng,
        }
    }

    /// Picks the next node to probe toward, recording it as probed.
    fn next_target(&self) -> Option<NodeId> {
        let our_node_id = NodeId::from_pubkey(&self.node.get_info().pubkey);

        let mut nodes_by_degree: Vec<(NodeId, usize)> = {
            let graph = self.node.pathfinding_graph.read_only();
            graph
                .nodes()
                .unordered_iter()
                .filter(|(id, _)| **id != our_node_id)
                .map(|(id, info)| (*id, info.channels.len()))
                .collect()
        };

        if nodes_by_degree.is_empty() {
            return None;
        }

        // Sort by degree (highest first), breaking ties by node id so that the order is reproducible.
        nodes_by_degree.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let top_node_count = self.top_node_count.min(nodes_by_degree.len());
        let now = self.node.clock.now();

        let mut probed = self
            .recently_probed
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        probed.retain(|_, probed_at| {
            now.duration_since(*probed_at)
                .map(|elapsed| elapsed < self.cooldown)
                .unwrap_or(true)
        });

        // If all top nodes are on cooldown, reset and start a new cycle.
        let final_node_id = match nodes_by_degree[..top_node_count]
            .iter()
            .find(|(node_id, _)| !probed.contains_key(node_id))
        {
            Some((node_id, _)) => *node_id,
            None => {
                probed.clear();
                nodes_by_degree[0].0
            },
        };

        probed.insert(final_node_id, now);
        Some(final_node_id)
    }
}

#[async_trait]
impl<T: SimNetwork, C: Clock> ProbingStrategy for HighDegreeStrategy<T, C> {
    async fn next_probe(&self) -> Option<Path> {
        let final_node = PublicKey::try_from(self.next_target()?).ok()?;
        let amount_msat = random_range(&self.rng, self.min_amount_msat, self.max_amount_msat);

        let path = {
            let scorer = self.node.scorer.read().ok()?;
            let route = find_payment_route_with_params(
                &self.node.get_info().pubkey,
                final_node,
                amount_msat,
                &self.node.pathfinding_graph,
                &scorer,
                &self.fee_params,
            )
            .ok()?;
            route.paths.into_iter().next()?
        };

        if path.hops.len() < 2 && path.blinded_tail.is_none() {
            return None;
        }

        // Liquidity-limit check (mirrors LDK's send_preflight_probes): skip the path when the first-hop outbound
        // liquidity is less than path_value * liquidity_limit_multiplier.
        let first_hop_scid = path.hops.first()?.short_channel_id;
        let local_channels = self.node.local_channels().await.ok()?;
        if let Some(channel) = local_channels
            .iter()
            .find(|c| c.short_channel_id == first_hop_scid)
        {
            let path_value = path.final_value_msat() + path.fee_msat();
            if channel.next_outbound_htlc_limit_msat
                < path_value.saturating_mul(self.liquidity_limit_multiplier)
            {
                return None;
            }
        }

        Some(path)
    }
}

/// Explores the graph by walking a random number (≥2) of hops outward from one of the node's own channels,
/// constructing the [`Path`] explicitly.
///
/// On each tick:
///   1. Picks one of the node's channels to start from.
///   2. Performs a random walk of a chosen depth (up to [`MAX_PATH_LENGTH_ESTIMATE`]) through the graph, skipping
///      disabled channels, dead-ends and channels already used in the path.
///
/// The probe amount is chosen uniformly at random from `[min_amount_msat, max_amount_msat]`, then clamped to the HTLC
/// limits along the path.
///
/// Because path selection ignores the scorer, this probes channels the router would never try on its own, teaching
/// the scorer about previously unknown paths.
pub struct RandomWalkStrategy<T: SimNetwork, C: Clock> {
    node: Arc<SimNode<T, C>>,
    /// Upper bound on the number of hops in a randomly constructed path.
    pub max_hops: usize,
    /// Lower bound for the randomly chosen probe amount.
    pub min_amount_msat: u64,
    /// Upper bound for the randomly chosen probe amount.
    pub max_amount_msat: u64,
    rng: StdMutex<ChaCha8Rng>,
}

impl<T: SimNetwork, C: Clock> RandomWalkStrategy<T, C> {
    /// Creates a new random-walk probing strategy. `max_hops` is clamped to `[2, MAX_PATH_LENGTH_ESTIMATE]`.
    pub fn new(
        node: Arc<SimNode<T, C>>,
        max_hops: usize,
        min_amount_msat: u64,
        max_amount_msat: u64,
        seed: Option<u64>,
    ) -> Self {
        assert!(
            min_amount_msat <= max_amount_msat,
            "min_amount_msat must not exceed max_amount_msat"
        );
        let rng = StdMutex::new(probing_rng(seed, &node.get_info().pubkey));
        Self {
            node,
            max_hops: max_hops.clamp(2, MAX_PATH_LENGTH_ESTIMATE as usize),
            min_amount_msat,
            max_amount_msat,
            rng,
        }
    }

    /// Tries to build a path of `target_hops` hops. Returns `None` if the node has no channels, or the walk
    /// terminates before reaching two hops.
    async fn try_build_path(&self, target_hops: usize, amount_msat: u64) -> Option<Path> {
        let initial_channels = self.node.local_channels().await.ok()?;
        if initial_channels.is_empty() {
            return None;
        }

        // Everything below is synchronous, so we can hold the RNG and graph locks for the rest of the walk.
        let mut rng = self.rng.lock().unwrap_or_else(|e| e.into_inner());
        let graph = self.node.pathfinding_graph.read_only();

        let first_hop = &initial_channels[rng.gen_range(0..initial_channels.len())];
        let first_hop_scid = first_hop.short_channel_id;
        let next_peer_pubkey = first_hop.counterparty;
        let next_peer_node_id = NodeId::from_pubkey(&next_peer_pubkey);

        // Track the tightest HTLC limit across all hops to cap the probe amount. The first hop limit comes from our
        // live channel state; subsequent hops use htlc_maximum_msat from the channel update in the graph.
        let mut route_least_htlc_upper_bound = first_hop.next_outbound_htlc_limit_msat;
        let mut route_greatest_htlc_lower_bound = first_hop.next_outbound_htlc_minimum_msat;

        // Walk the graph: each entry is (node_id, arrived_via_scid, pubkey); first entry is set.
        let mut route: Vec<(NodeId, u64, PublicKey)> =
            vec![(next_peer_node_id, first_hop_scid, next_peer_pubkey)];

        // Unlike LDK Node, we skip every channel already in the path (not just the one we arrived on): adding the same
        // payment hash to a simulated channel twice is a critical error that would shut down the simulation.
        let mut used_scids: HashSet<u64> = HashSet::from([first_hop_scid]);
        let mut current_node_id = next_peer_node_id;

        for _ in 1..target_hops {
            let node_info = match graph.node(&current_node_id) {
                Some(n) => n,
                None => break,
            };

            let candidates: Vec<u64> = node_info
                .channels
                .iter()
                .copied()
                .filter(|scid| !used_scids.contains(scid))
                .collect();

            if candidates.is_empty() {
                break;
            }

            let next_scid = candidates[rng.gen_range(0..candidates.len())];
            let next_channel = match graph.channel(next_scid) {
                Some(c) => c,
                None => break,
            };

            // as_directed_from validates that current_node_id is a channel endpoint and that both direction updates
            // are present.
            let Some((directed, next_node_id)) = next_channel.as_directed_from(&current_node_id)
            else {
                break;
            };
            let channel_update_info = match if directed.source() == &next_channel.node_one {
                next_channel.one_to_two.as_ref()
            } else {
                next_channel.two_to_one.as_ref()
            } {
                Some(u) => u,
                None => break,
            };

            if !channel_update_info.enabled {
                break;
            }

            route_least_htlc_upper_bound =
                route_least_htlc_upper_bound.min(channel_update_info.htlc_maximum_msat);
            route_greatest_htlc_lower_bound =
                route_greatest_htlc_lower_bound.max(channel_update_info.htlc_minimum_msat);

            let next_pubkey = match PublicKey::try_from(*next_node_id) {
                Ok(pk) => pk,
                Err(_) => break,
            };

            route.push((*next_node_id, next_scid, next_pubkey));
            used_scids.insert(next_scid);
            current_node_id = *next_node_id;
        }

        if route_greatest_htlc_lower_bound > route_least_htlc_upper_bound {
            return None;
        }
        let amount_msat = amount_msat
            .max(route_greatest_htlc_lower_bound)
            .min(route_least_htlc_upper_bound);
        if amount_msat < self.min_amount_msat || amount_msat > self.max_amount_msat {
            return None;
        }

        // Assemble hops backwards so each hop's proportional fee is computed on the amount it actually forwards.
        let mut hops = Vec::with_capacity(route.len());
        let mut forwarded = amount_msat;
        let last = route.len() - 1;

        // Resolve (node_features, channel_features) for a hop. The simulated network does not have node announcements,
        // so node features fall back to empty. All channels in the simulated network are announced.
        let hop_features =
            |node_id: &NodeId, via_scid: u64| -> Option<(NodeFeatures, ChannelFeatures)> {
                let node_features = graph
                    .node(node_id)
                    .and_then(|n| n.announcement_info.as_ref().map(|a| a.features().clone()))
                    .unwrap_or_else(NodeFeatures::empty);
                let channel_features = graph.channel(via_scid)?.features.clone();
                Some((node_features, channel_features))
            };

        // Final hop: fee_msat carries the delivery amount; cltv_expiry_delta carries the destination's final CLTV
        // (matching LDK's shifted-by-one RouteHop convention).
        {
            let (node_id, via_scid, pubkey) = route[last];
            let (node_features, channel_features) = hop_features(&node_id, via_scid)?;
            hops.push(RouteHop {
                pubkey,
                node_features,
                short_channel_id: via_scid,
                channel_features,
                fee_msat: amount_msat,
                cltv_expiry_delta: FINAL_CLTV_EXPIRY_DELTA,
                maybe_announced_channel: true,
            });
        }

        // Non-final hops, from second-to-last back to first.
        for i in (0..last).rev() {
            let (node_id, via_scid, pubkey) = route[i];
            let (node_features, channel_features) = hop_features(&node_id, via_scid)?;

            let (_, next_scid, _) = route[i + 1];
            let next_channel = graph.channel(next_scid)?;
            let (directed, _) = next_channel.as_directed_from(&node_id)?;
            let channel_update_info = if directed.source() == &next_channel.node_one {
                next_channel.one_to_two.as_ref()
            } else {
                next_channel.two_to_one.as_ref()
            }?;

            // forwarded is the amount sent over next_channel (before this hop's own fee is added on top for the
            // preceding channel), so check it against next_channel's own advertised bounds before computing the fee.
            if forwarded > channel_update_info.htlc_maximum_msat
                || forwarded < channel_update_info.htlc_minimum_msat
            {
                return None;
            }

            let proportional_fee = forwarded
                .checked_mul(channel_update_info.fees.proportional_millionths as u64)
                .map(|v| v / 1_000_000)?;
            let fee = (channel_update_info.fees.base_msat as u64).checked_add(proportional_fee)?;
            forwarded = forwarded.checked_add(fee)?;

            hops.push(RouteHop {
                pubkey,
                node_features,
                short_channel_id: via_scid,
                channel_features,
                fee_msat: fee,
                cltv_expiry_delta: channel_update_info.cltv_expiry_delta as u32,
                maybe_announced_channel: true,
            });
        }

        hops.reverse();

        if hops.len() < 2 {
            return None;
        }

        // The first-hop HTLC carries amount_msat + all intermediate fees. Verify the total fits within our live
        // outbound limit before returning.
        let total_outgoing: u64 = hops.iter().map(|h| h.fee_msat).sum();
        if total_outgoing > first_hop.next_outbound_htlc_limit_msat {
            return None;
        }

        Some(Path {
            hops,
            blinded_tail: None,
        })
    }
}

#[async_trait]
impl<T: SimNetwork, C: Clock> ProbingStrategy for RandomWalkStrategy<T, C> {
    async fn next_probe(&self) -> Option<Path> {
        let target_hops = random_range(&self.rng, 2, self.max_hops as u64) as usize;
        let amount_msat = random_range(&self.rng, self.min_amount_msat, self.max_amount_msat);

        self.try_build_path(target_hops, amount_msat).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::SimulationClock;
    use crate::sim_node::{
        ln_node_from_graph, populate_network_graph, ChannelPolicy, CustomRecords, SimGraph,
        SimulatedChannel,
    };
    use crate::test_utils::get_random_keypair;
    use crate::{PaymentOutcome, ShortChannelID};
    use tokio::sync::Mutex;
    use tokio_util::task::TaskTracker;
    use triggered::{Listener, Trigger};

    type TestNode = Arc<SimNode<SimGraph, SimulationClock>>;

    struct TestNetwork {
        pubkeys: Vec<PublicKey>,
        nodes: HashMap<PublicKey, TestNode>,
        shutdown: (Trigger, Listener),
        tasks: TaskTracker,
    }

    impl TestNetwork {
        /// Creates a simulated network with `node_count` nodes and a channel for each pair of node indices in
        /// `edges`, where the short channel id of each channel is its index in `edges`. Each channel's capacity is
        /// split evenly between its two sides.
        async fn new(node_count: usize, edges: &[(usize, usize)], capacity_msat: u64) -> Self {
            let pubkeys: Vec<PublicKey> = (0..node_count).map(|_| get_random_keypair().1).collect();
            let policy = |pubkey: PublicKey| ChannelPolicy {
                pubkey,
                alias: String::default(),
                max_htlc_count: 483,
                max_in_flight_msat: capacity_msat / 2,
                min_htlc_size_msat: 1,
                max_htlc_size_msat: capacity_msat / 2,
                cltv_expiry_delta: 40,
                base_fee: 1000,
                fee_rate_prop: 100,
            };

            let channels: Vec<SimulatedChannel> = edges
                .iter()
                .enumerate()
                .map(|(scid, (a, b))| {
                    SimulatedChannel::new(
                        capacity_msat,
                        ShortChannelID::from(scid as u64),
                        policy(pubkeys[*a]),
                        policy(pubkeys[*b]),
                        false,
                    )
                })
                .collect();

            let clock = Arc::new(SimulationClock::new(SystemTime::now()));
            let shutdown = triggered::trigger();
            let tasks = TaskTracker::new();
            let routing_graph =
                Arc::new(populate_network_graph(channels.clone(), clock.clone()).unwrap());
            let graph = Arc::new(Mutex::new(
                SimGraph::new(
                    channels,
                    tasks.clone(),
                    vec![],
                    CustomRecords::default(),
                    shutdown.clone(),
                )
                .unwrap(),
            ));
            let nodes = ln_node_from_graph(graph, routing_graph, clock)
                .await
                .unwrap();

            TestNetwork {
                pubkeys,
                nodes,
                shutdown,
                tasks,
            }
        }

        fn node(&self, index: usize) -> TestNode {
            self.nodes[&self.pubkeys[index]].clone()
        }

        async fn shutdown(self) {
            assert!(
                !self.shutdown.1.is_triggered(),
                "simulation was shut down by a critical error"
            );
            self.shutdown.0.trigger();
            self.tasks.close();
            self.tasks.wait().await;
        }
    }

    fn scids(path: &Path) -> Vec<u64> {
        path.hops.iter().map(|h| h.short_channel_id).collect()
    }

    /// Tests that random walks along a chain A -- B -- C -- D build valid paths of 2 or 3 hops that probes can
    /// traverse successfully.
    #[tokio::test]
    async fn test_random_walk_builds_valid_paths() {
        let network = TestNetwork::new(4, &[(0, 1), (1, 2), (2, 3)], 500_000_000).await;
        let node = network.node(0);
        let strategy = RandomWalkStrategy::new(node.clone(), 3, 10_000, 100_000, Some(42));

        for _ in 0..20 {
            let path = strategy.next_probe().await.unwrap();
            let hops = scids(&path);
            assert!(hops == vec![0, 1] || hops == vec![0, 1, 2], "{hops:?}");

            let amount = path.final_value_msat();
            assert!((10_000..=100_000).contains(&amount));

            let outcome = node.probe(path, network.shutdown.1.clone()).await.unwrap();
            assert_eq!(outcome, PaymentOutcome::Success);
        }

        network.shutdown().await;
    }

    /// Tests that a random walk around a cycle A -- B -- C -- A never reuses a channel. LDK Node's walk only avoids
    /// the channel it arrived on, which would go around the cycle and add the probe's HTLC to A -- B a second time,
    /// shutting down the simulation.
    #[tokio::test]
    async fn test_random_walk_does_not_reuse_channels() {
        let network = TestNetwork::new(3, &[(0, 1), (1, 2), (2, 0)], 500_000_000).await;
        let node = network.node(0);
        let strategy = RandomWalkStrategy::new(node.clone(), 8, 10_000, 100_000, Some(7));

        for _ in 0..20 {
            let path = strategy.next_probe().await.unwrap();
            let hops = scids(&path);
            let unique: HashSet<u64> = hops.iter().copied().collect();
            assert_eq!(unique.len(), hops.len(), "{hops:?}");

            let outcome = node.probe(path, network.shutdown.1.clone()).await.unwrap();
            assert_eq!(outcome, PaymentOutcome::Success);
        }

        network.shutdown().await;
    }

    /// Tests that the same seed produces the same sequence of random walks.
    #[tokio::test]
    async fn test_random_walk_is_reproducible_with_seed() {
        let network = TestNetwork::new(
            5,
            &[(0, 1), (1, 2), (2, 3), (3, 4), (1, 3), (2, 4)],
            500_000_000,
        )
        .await;

        let mut walks = vec![];
        for _ in 0..2 {
            let strategy = RandomWalkStrategy::new(network.node(0), 5, 10_000, 100_000, Some(1));
            let mut paths = vec![];
            for _ in 0..10 {
                paths.push(strategy.next_probe().await.map(|p| p.hops));
            }
            walks.push(paths);
        }
        assert_eq!(walks[0], walks[1]);

        network.shutdown().await;
    }

    /// Tests that the high-degree strategy probes towards the best-connected nodes, rotating through the top nodes
    /// with a cooldown. The network is: A -- B -- H, with H also connected to X, Y and Z.
    #[tokio::test]
    async fn test_high_degree_rotates_through_top_nodes() {
        // Indices: A = 0, B = 1, H = 2, X = 3, Y = 4, Z = 5.
        let network =
            TestNetwork::new(6, &[(0, 1), (1, 2), (2, 3), (2, 4), (2, 5)], 500_000_000).await;
        let node = network.node(0);
        let hub = network.pubkeys[2];
        let strategy = HighDegreeStrategy::new(
            node.clone(),
            2,
            10_000,
            100_000,
            Duration::from_secs(3600),
            DEFAULT_LIQUIDITY_LIMIT_MULTIPLIER,
            None,
            Some(3),
        );

        // The highest-degree node is the hub H, which is two hops away, so we probe A -> B -> H.
        let path = strategy.next_probe().await.unwrap();
        assert_eq!(scids(&path), vec![0, 1]);
        assert_eq!(path.hops.last().unwrap().pubkey, hub);
        let outcome = node.probe(path, network.shutdown.1.clone()).await.unwrap();
        assert_eq!(outcome, PaymentOutcome::Success);

        // H is now on cooldown, so the next target is B (degree 2). B is our direct peer, and a single-hop probe tells
        // us nothing, so this tick is skipped.
        assert!(strategy.next_probe().await.is_none());

        // Both top nodes are now on cooldown, so the cycle restarts with H.
        let path = strategy.next_probe().await.unwrap();
        assert_eq!(path.hops.last().unwrap().pubkey, hub);

        network.shutdown().await;
    }

    /// Tests that the high-degree strategy skips probes that would use a large share of the first hop's liquidity.
    #[tokio::test]
    async fn test_high_degree_respects_liquidity_limit() {
        // A -- B -- H, with H also connected to X and Y, so the top node H is two hops from A. A has 2_000_000 msat of
        // outbound liquidity on A -- B.
        let network = TestNetwork::new(5, &[(0, 1), (1, 2), (2, 3), (2, 4)], 4_000_000).await;
        let node = network.node(0);

        // Probing with 1_000_000 msat (plus fees) needs more than 3x that amount on the first hop.
        let strategy = HighDegreeStrategy::new(
            node.clone(),
            1,
            1_000_000,
            1_000_000,
            Duration::from_secs(3600),
            DEFAULT_LIQUIDITY_LIMIT_MULTIPLIER,
            None,
            None,
        );
        assert!(strategy.next_probe().await.is_none());

        // A probe of 100_000 msat fits within the limit.
        let strategy = HighDegreeStrategy::new(
            node.clone(),
            1,
            100_000,
            100_000,
            Duration::from_secs(3600),
            DEFAULT_LIQUIDITY_LIMIT_MULTIPLIER,
            None,
            None,
        );
        let path = strategy.next_probe().await.unwrap();
        assert_eq!(scids(&path), vec![0, 1]);

        network.shutdown().await;
    }
}
