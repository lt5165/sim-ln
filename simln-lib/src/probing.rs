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
use std::sync::atomic::{AtomicU64, Ordering};
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
use serde::{Deserialize, Serialize};
use tokio::select;
use tokio_util::task::TaskTracker;
use triggered::Listener;

use crate::clock::Clock;
use crate::event_log::{EventLog, ProbeBalanceRecord};
use crate::sim_node::{find_payment_route_with_params, ProbeResult, SimNetwork, SimNode};
use crate::{LightningNode, PaymentOutcome, SimulationError};

/// Default interval between probe attempts, in seconds. Matches LDK Node.
pub const DEFAULT_PROBING_INTERVAL_SECS: u64 = 10;
/// Default maximum amount that may be locked in a node's in-flight probes at any time. Matches LDK Node.
pub const DEFAULT_MAX_PROBE_LOCKED_MSAT: u64 = 100_000_000;
/// Default lower bound for randomly chosen probe amounts. Matches LDK Node.
pub const DEFAULT_MIN_PROBE_AMOUNT_MSAT: u64 = 1_000_000;
/// Default upper bound for randomly chosen probe amounts. Matches LDK Node.
pub const DEFAULT_MAX_PROBE_AMOUNT_MSAT: u64 = 10_000_000;
/// Default time that a node stays ineligible for [`HighDegreeStrategy`] after being probed. Matches LDK Node.
pub const DEFAULT_PROBED_NODE_COOLDOWN_SECS: u64 = 60 * 60;

/// Which built-in probing strategy to use, with its strategy-specific settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbingStrategyConfig {
    /// Use [`HighDegreeStrategy`], cycling through the `top_node_count` most-connected nodes.
    HighDegree { top_node_count: usize },
    /// Use [`RandomWalkStrategy`], with paths of at most `max_hops` hops.
    RandomWalk { max_hops: usize },
}

/// Configuration for background probing in a simulated network, read from the `probing` section of a simulation file.
/// Every field other than `strategy` is optional, with defaults that match LDK Node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbingConfig {
    /// The probing strategy to use.
    pub strategy: ProbingStrategyConfig,
    /// The nodes that run background probing. If empty, every node in the simulated network probes.
    #[serde(default)]
    pub nodes: Vec<PublicKey>,
    /// The interval between probe attempts for each probing node.
    #[serde(default = "default_interval_secs")]
    pub interval_secs: u64,
    /// The maximum amount that may be locked in each node's in-flight probes at any time.
    #[serde(default = "default_max_locked_msat")]
    pub max_locked_msat: u64,
    /// The lower bound for randomly chosen probe amounts.
    #[serde(default = "default_min_amount_msat")]
    pub min_amount_msat: u64,
    /// The upper bound for randomly chosen probe amounts.
    #[serde(default = "default_max_amount_msat")]
    pub max_amount_msat: u64,
    /// How long a node stays ineligible after being probed. Only applies to [`HighDegreeStrategy`].
    #[serde(default = "default_cooldown_secs")]
    pub cooldown_secs: u64,
    /// A penalty applied to recently probed channels when finding probe routes, which encourages path diversity. Only
    /// applies to [`HighDegreeStrategy`]. If unset, no penalty is applied.
    #[serde(default)]
    pub diversity_penalty_msat: Option<u64>,
}

fn default_interval_secs() -> u64 {
    DEFAULT_PROBING_INTERVAL_SECS
}

fn default_max_locked_msat() -> u64 {
    DEFAULT_MAX_PROBE_LOCKED_MSAT
}

fn default_min_amount_msat() -> u64 {
    DEFAULT_MIN_PROBE_AMOUNT_MSAT
}

fn default_max_amount_msat() -> u64 {
    DEFAULT_MAX_PROBE_AMOUNT_MSAT
}

fn default_cooldown_secs() -> u64 {
    DEFAULT_PROBED_NODE_COOLDOWN_SECS
}

impl ProbingConfig {
    /// Creates a config for the strategy provided, with every other setting at its default.
    pub fn new(strategy: ProbingStrategyConfig) -> Self {
        Self {
            strategy,
            nodes: vec![],
            interval_secs: DEFAULT_PROBING_INTERVAL_SECS,
            max_locked_msat: DEFAULT_MAX_PROBE_LOCKED_MSAT,
            min_amount_msat: DEFAULT_MIN_PROBE_AMOUNT_MSAT,
            max_amount_msat: DEFAULT_MAX_PROBE_AMOUNT_MSAT,
            cooldown_secs: DEFAULT_PROBED_NODE_COOLDOWN_SECS,
            diversity_penalty_msat: None,
        }
    }

    /// Validates that the config's values are usable.
    pub fn validate(&self) -> Result<(), SimulationError> {
        let err = |msg: &str| Err(SimulationError::ProbingConfigError(msg.to_string()));

        if self.interval_secs == 0 {
            return err("interval_secs must be at least 1");
        }
        if self.min_amount_msat == 0 {
            return err("min_amount_msat must be greater than 0");
        }
        if self.min_amount_msat > self.max_amount_msat {
            return err("min_amount_msat must not exceed max_amount_msat");
        }
        if let ProbingStrategyConfig::HighDegree { top_node_count: 0 } = self.strategy {
            return err("top_node_count must be at least 1");
        }

        Ok(())
    }
}

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

/// Counts of the probes sent by a [`Prober`].
#[derive(Debug, Default)]
pub struct ProbeStats {
    /// Probes dispatched into the network.
    pub sent: AtomicU64,
    /// Probes that reached their destination.
    pub reached_destination: AtomicU64,
    /// Probes that failed before reaching their destination.
    pub failed: AtomicU64,
    /// Probes that were not sent because they would have exceeded the locked-liquidity budget.
    pub skipped_over_budget: AtomicU64,
}

/// Periodically dispatches probes from a node according to a [`ProbingStrategy`]. This is a port of LDK Node's
/// `Prober` and `run_prober`.
pub struct Prober<T: SimNetwork, C: Clock> {
    node: Arc<SimNode<T, C>>,
    /// The strategy that decides what to probe.
    strategy: Box<dyn ProbingStrategy>,
    /// How often to fire a probe attempt.
    interval: Duration,
    /// Maximum total millisatoshis that may be locked in in-flight probes at any time.
    max_locked_msat: u64,
    /// Total millisatoshis currently locked in in-flight probes.
    locked_msat: Arc<AtomicU64>,
    stats: Arc<ProbeStats>,
    /// Log that probes and probe balance estimates are written to, if any.
    event_log: Option<Arc<EventLog>>,
}

impl<T: SimNetwork + 'static, C: Clock + 'static> Prober<T, C> {
    /// Creates a prober for the node provided.
    pub fn new(
        node: Arc<SimNode<T, C>>,
        strategy: Box<dyn ProbingStrategy>,
        interval: Duration,
        max_locked_msat: u64,
    ) -> Self {
        Self {
            node,
            strategy,
            interval,
            max_locked_msat,
            locked_msat: Arc::new(AtomicU64::new(0)),
            stats: Arc::new(ProbeStats::default()),
            event_log: None,
        }
    }

    /// Logs every probe that this prober sends, with the probing node's balance estimates for each channel on the
    /// probe's path, to the log provided.
    pub fn with_event_log(mut self, event_log: Arc<EventLog>) -> Self {
        self.event_log = Some(event_log);
        self
    }

    /// Returns the total millisatoshis currently locked in in-flight probes.
    pub fn locked_msat(&self) -> u64 {
        self.locked_msat.load(Ordering::SeqCst)
    }

    /// Returns the prober's probe counts, which can be read while the prober runs.
    pub fn stats(&self) -> Arc<ProbeStats> {
        self.stats.clone()
    }

    /// Runs the probing loop until `listener` is triggered. Each probe is tracked in its own task, spawned on `tasks`,
    /// so that slow probes don't hold up the loop.
    pub async fn run(self, tasks: TaskTracker, listener: Listener) {
        let pubkey = self.node.get_info().pubkey;
        log::info!(
            "Starting background probing for {pubkey} every {:?}.",
            self.interval
        );

        loop {
            select! {
                biased;
                _ = listener.clone() => break,
                _ = self.node.clock.sleep(self.interval) => {},
            }

            let path = match self.strategy.next_probe().await {
                Some(path) => path,
                None => continue,
            };

            let amount_msat: u64 = path.hops.iter().map(|h| h.fee_msat).sum();
            if self.locked_msat() + amount_msat > self.max_locked_msat {
                log::debug!("Skipping probe from {pubkey}: locked-msat budget exceeded.");
                self.stats
                    .skipped_over_budget
                    .fetch_add(1, Ordering::SeqCst);
                continue;
            }

            self.locked_msat.fetch_add(amount_msat, Ordering::SeqCst);
            self.stats.sent.fetch_add(1, Ordering::SeqCst);
            log::debug!(
                "Background probe from {pubkey} sent: locked {amount_msat} msat, path: {}",
                fmt_path(&path)
            );

            let node = self.node.clone();
            let locked_msat = self.locked_msat.clone();
            let stats = self.stats.clone();
            let listener = listener.clone();
            let event_log = self.event_log.clone();
            tasks.spawn(async move {
                let result = node.probe(path.clone(), listener).await;
                locked_msat.fetch_sub(amount_msat, Ordering::SeqCst);

                let probe = match result {
                    Ok(probe) => probe,
                    Err(e) => {
                        log::debug!("Background probe from {pubkey} not resolved: {e}");
                        return;
                    },
                };

                if probe.result.payment_outcome == PaymentOutcome::Success {
                    stats.reached_destination.fetch_add(1, Ordering::SeqCst);
                    log::debug!(
                        "Background probe from {pubkey} succeeded along the path: {}",
                        fmt_path(&path)
                    );
                } else {
                    stats.failed.fetch_add(1, Ordering::SeqCst);
                    log::debug!(
                        "Background probe from {pubkey} failed with {:?} ({}) along the path: {}",
                        probe.result.payment_outcome,
                        probe
                            .result
                            .failure
                            .as_ref()
                            .map_or("unknown reason".to_string(), |f| f.detail.clone()),
                        fmt_path(&path)
                    );
                }

                if let Some(event_log) = event_log {
                    if let Err(e) = record_probe(&node, &path, &probe, &event_log).await {
                        log::error!("Could not log probe from {pubkey}: {e}");
                    }
                }
            });
        }

        log::info!(
            "Stopped background probing for {pubkey}: {} probes sent, {} reached destination, {} failed, {} skipped \
             over budget.",
            self.stats.sent.load(Ordering::SeqCst),
            self.stats.reached_destination.load(Ordering::SeqCst),
            self.stats.failed.load(Ordering::SeqCst),
            self.stats.skipped_over_budget.load(Ordering::SeqCst),
        );
    }
}

/// Writes a resolved probe to the event log, along with a row for each channel on its path that compares the probing
/// node's (just updated) scorer estimate of the channel's liquidity with its actual balance.
async fn record_probe<T: SimNetwork, C: Clock>(
    node: &SimNode<T, C>,
    path: &Path,
    probe: &ProbeResult,
    event_log: &EventLog,
) -> Result<(), SimulationError> {
    let observer = node.get_info().pubkey;
    event_log.log_probe(observer, path, probe)?;

    let time = node.clock.now();
    let estimates: Vec<Option<(u64, u64)>> = {
        let scorer = node.scorer.read().map_err(|e| {
            SimulationError::SimulatedNetworkError(format!("scorer lock poisoned: {e}"))
        })?;
        path.hops
            .iter()
            .map(|hop| {
                scorer.estimated_channel_liquidity_range(
                    hop.short_channel_id,
                    &NodeId::from_pubkey(&hop.pubkey),
                )
            })
            .collect()
    };

    let mut from_node = observer;
    for (hop_index, (hop, estimate)) in path.hops.iter().zip(estimates).enumerate() {
        let channel = node
            .network_channels(&from_node)
            .await?
            .into_iter()
            .find(|c| c.short_channel_id == hop.short_channel_id)
            .ok_or_else(|| {
                SimulationError::SimulatedNetworkError(format!(
                    "channel {} not found for {from_node}",
                    hop.short_channel_id
                ))
            })?;

        event_log.log_balance(&ProbeBalanceRecord {
            time,
            probe_hash: Some(probe.payment_hash),
            observer,
            probe_outcome: probe.result.payment_outcome.clone(),
            hop_index,
            scid: hop.short_channel_id,
            from_node,
            to_node: hop.pubkey,
            capacity_msat: channel.capacity_msat,
            estimated_min_msat: estimate.map(|(min, _)| min),
            estimated_max_msat: estimate.map(|(_, max)| max),
            actual_balance_msat: channel.local_balance_msat,
            actual_sendable_msat: channel.next_outbound_htlc_limit_msat,
        })?;

        from_node = hop.pubkey;
    }

    Ok(())
}

fn fmt_path(path: &Path) -> String {
    path.hops
        .iter()
        .map(|h| format!("{}(scid={})", h.pubkey, h.short_channel_id))
        .collect::<Vec<_>>()
        .join(" -> ")
}

/// Starts background probing on the simulated nodes described in `config`, spawning a [`Prober`] for each on `tasks`.
/// The probers run until `listener` is triggered. If `seed` is provided, probe selection is deterministic.
pub fn start_probing<T: SimNetwork + 'static, C: Clock + 'static>(
    config: &ProbingConfig,
    nodes: &HashMap<PublicKey, Arc<SimNode<T, C>>>,
    seed: Option<u64>,
    event_log: Option<Arc<EventLog>>,
    tasks: &TaskTracker,
    listener: Listener,
) -> Result<(), SimulationError> {
    config.validate()?;

    let probing_nodes: Vec<PublicKey> = if config.nodes.is_empty() {
        let mut all: Vec<PublicKey> = nodes.keys().copied().collect();
        all.sort();
        all
    } else {
        config.nodes.clone()
    };

    for pubkey in probing_nodes {
        let node = nodes.get(&pubkey).ok_or_else(|| {
            SimulationError::ProbingConfigError(format!(
                "probing node {pubkey} not found in simulated network"
            ))
        })?;

        let strategy: Box<dyn ProbingStrategy> = match config.strategy {
            ProbingStrategyConfig::HighDegree { top_node_count } => {
                Box::new(HighDegreeStrategy::new(
                    node.clone(),
                    top_node_count,
                    config.min_amount_msat,
                    config.max_amount_msat,
                    Duration::from_secs(config.cooldown_secs),
                    DEFAULT_LIQUIDITY_LIMIT_MULTIPLIER,
                    config.diversity_penalty_msat,
                    seed,
                ))
            },
            ProbingStrategyConfig::RandomWalk { max_hops } => Box::new(RandomWalkStrategy::new(
                node.clone(),
                max_hops,
                config.min_amount_msat,
                config.max_amount_msat,
                seed,
            )),
        };

        let mut prober = Prober::new(
            node.clone(),
            strategy,
            Duration::from_secs(config.interval_secs),
            config.max_locked_msat,
        );
        if let Some(event_log) = &event_log {
            prober = prober.with_event_log(event_log.clone());
        }
        tasks.spawn(prober.run(tasks.clone(), listener.clone()));
    }

    Ok(())
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
    use std::time::UNIX_EPOCH;
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

            let outcome = node
                .probe(path, network.shutdown.1.clone())
                .await
                .unwrap()
                .result
                .payment_outcome;
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

            let outcome = node
                .probe(path, network.shutdown.1.clone())
                .await
                .unwrap()
                .result
                .payment_outcome;
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
        let outcome = node
            .probe(path, network.shutdown.1.clone())
            .await
            .unwrap()
            .result
            .payment_outcome;
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

    /// Tests that the prober sends one probe per interval, trains the node's scorer and releases its locked liquidity
    /// once probes resolve. Runs on paused (virtual) time, so the interval elapses instantly.
    #[tokio::test(start_paused = true)]
    async fn test_prober_sends_probes_on_interval() {
        let network = TestNetwork::new(4, &[(0, 1), (1, 2), (2, 3)], 500_000_000).await;
        let node = network.node(0);
        let strategy = RandomWalkStrategy::new(node.clone(), 3, 10_000, 100_000, Some(5));
        let prober = Prober::new(
            node.clone(),
            Box::new(strategy),
            Duration::from_secs(10),
            DEFAULT_MAX_PROBE_LOCKED_MSAT,
        );
        let stats = prober.stats();
        let locked_msat = prober.locked_msat.clone();
        network
            .tasks
            .spawn(prober.run(network.tasks.clone(), network.shutdown.1.clone()));

        // Probes fire at 10s, 20s, ..., 100s.
        tokio::time::sleep(Duration::from_secs(105)).await;
        assert_eq!(stats.sent.load(Ordering::SeqCst), 10);
        assert_eq!(stats.reached_destination.load(Ordering::SeqCst), 10);
        assert_eq!(stats.failed.load(Ordering::SeqCst), 0);
        assert_eq!(locked_msat.load(Ordering::SeqCst), 0);

        // Every probe crosses B -- C, so the scorer should have learned that it has at least the minimum probe amount.
        let (min_liquidity, _) = node
            .scorer
            .read()
            .unwrap()
            .estimated_channel_liquidity_range(1, &NodeId::from_pubkey(&network.pubkeys[2]))
            .unwrap();
        assert!(min_liquidity >= 10_000);

        network.shutdown().await;
    }

    /// Tests that the prober does not send probes that would exceed its locked-liquidity budget.
    #[tokio::test(start_paused = true)]
    async fn test_prober_respects_locked_budget() {
        let network = TestNetwork::new(4, &[(0, 1), (1, 2), (2, 3)], 500_000_000).await;
        let node = network.node(0);
        let strategy = RandomWalkStrategy::new(node.clone(), 3, 10_000, 100_000, Some(5));
        let prober = Prober::new(node, Box::new(strategy), Duration::from_secs(10), 1_000);
        let stats = prober.stats();
        network
            .tasks
            .spawn(prober.run(network.tasks.clone(), network.shutdown.1.clone()));

        tokio::time::sleep(Duration::from_secs(35)).await;
        assert_eq!(stats.sent.load(Ordering::SeqCst), 0);
        assert_eq!(stats.skipped_over_budget.load(Ordering::SeqCst), 3);

        network.shutdown().await;
    }

    /// Tests parsing of the probing section of a simulation file, including defaults.
    #[test]
    fn test_probing_config_parsing() {
        let config: ProbingConfig =
            serde_json::from_str(r#"{"strategy": {"random_walk": {"max_hops": 4}}}"#).unwrap();
        assert_eq!(
            config,
            ProbingConfig::new(ProbingStrategyConfig::RandomWalk { max_hops: 4 })
        );
        assert!(config.validate().is_ok());

        let config: ProbingConfig = serde_json::from_str(
            r#"{
                "strategy": {"high_degree": {"top_node_count": 50}},
                "nodes": ["0277622bf4c497475960bf91bd3c673a4cb4e9b589cebfde9700c197b3989cc1b8"],
                "interval_secs": 30,
                "max_locked_msat": 5000000,
                "min_amount_msat": 1000,
                "max_amount_msat": 2000,
                "cooldown_secs": 600,
                "diversity_penalty_msat": 250
            }"#,
        )
        .unwrap();
        assert_eq!(
            config.strategy,
            ProbingStrategyConfig::HighDegree { top_node_count: 50 }
        );
        assert_eq!(config.nodes.len(), 1);
        assert_eq!(config.interval_secs, 30);
        assert_eq!(config.max_locked_msat, 5_000_000);
        assert_eq!(
            (config.min_amount_msat, config.max_amount_msat),
            (1000, 2000)
        );
        assert_eq!(config.cooldown_secs, 600);
        assert_eq!(config.diversity_penalty_msat, Some(250));
        assert!(config.validate().is_ok());

        // A missing strategy is an error.
        assert!(serde_json::from_str::<ProbingConfig>(r#"{"interval_secs": 30}"#).is_err());
    }

    /// Tests that invalid probing configs are rejected.
    #[test]
    fn test_probing_config_validation() {
        let valid = ProbingConfig::new(ProbingStrategyConfig::HighDegree { top_node_count: 10 });

        let mut config = valid.clone();
        config.interval_secs = 0;
        assert!(config.validate().is_err());

        let mut config = valid.clone();
        config.min_amount_msat = 0;
        assert!(config.validate().is_err());

        let mut config = valid.clone();
        config.min_amount_msat = config.max_amount_msat + 1;
        assert!(config.validate().is_err());

        let mut config = valid.clone();
        config.strategy = ProbingStrategyConfig::HighDegree { top_node_count: 0 };
        assert!(config.validate().is_err());
    }

    /// Tests that probing can't be started on a node that isn't in the simulated network.
    #[tokio::test]
    async fn test_start_probing_rejects_unknown_node() {
        let network = TestNetwork::new(2, &[(0, 1)], 500_000_000).await;

        let mut config = ProbingConfig::new(ProbingStrategyConfig::RandomWalk { max_hops: 3 });
        config.nodes = vec![get_random_keypair().1];
        assert!(start_probing(
            &config,
            &network.nodes,
            None,
            None,
            &network.tasks,
            network.shutdown.1.clone()
        )
        .is_err());

        network.shutdown().await;
    }

    /// Tests that a resolved probe is written to the events log, with a balances row for every channel on its path
    /// that pairs the probing node's scorer estimate with the channel's actual state.
    #[tokio::test]
    async fn test_record_probe_logs_estimates_and_actual_balances() {
        let capacity = 500_000_000;
        let network = TestNetwork::new(4, &[(0, 1), (1, 2), (2, 3)], capacity).await;
        let node = network.node(0);
        let dir = tempfile::tempdir().unwrap();
        let event_log = EventLog::new(dir.path().to_path_buf(), 1, UNIX_EPOCH);

        let strategy = RandomWalkStrategy::new(node.clone(), 3, 10_000, 10_000, Some(9));
        let path = loop {
            let path = strategy.next_probe().await.unwrap();
            if path.hops.len() == 3 {
                break path;
            }
        };
        let probe = node
            .probe(path.clone(), network.shutdown.1.clone())
            .await
            .unwrap();
        assert_eq!(probe.result.payment_outcome, PaymentOutcome::Success);
        record_probe(&node, &path, &probe, &event_log)
            .await
            .unwrap();
        event_log.flush().unwrap();

        let mut events = csv::Reader::from_path(dir.path().join("events_0.csv")).unwrap();
        let events: Vec<csv::StringRecord> = events.records().map(|r| r.unwrap()).collect();
        assert_eq!(events.len(), 1);
        assert_eq!(&events[0][0], "probe");
        assert_eq!(&events[0][2], network.pubkeys[3].to_string());

        let mut balances = csv::Reader::from_path(dir.path().join("probe_balances_0.csv")).unwrap();
        let headers = balances.headers().unwrap().clone();
        let column = |name: &str| headers.iter().position(|h| h == name).unwrap();
        let rows: Vec<csv::StringRecord> = balances.records().map(|r| r.unwrap()).collect();
        assert_eq!(rows.len(), 3);

        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row[column("hop_index")], i.to_string());
            assert_eq!(row[column("scid")], i.to_string());
            assert_eq!(row[column("from_node")], network.pubkeys[i].to_string());
            assert_eq!(row[column("to_node")], network.pubkeys[i + 1].to_string());
            assert_eq!(row[column("observer")], network.pubkeys[0].to_string());
            // The probe reached its destination, so every channel is estimated to hold at least the probed amount.
            let estimated_min: u64 = row[column("estimated_min_msat")].parse().unwrap();
            assert!(estimated_min >= 10_000);
            // Probes never move balances, so each channel still has its starting balance of half its capacity.
            assert_eq!(
                row[column("actual_balance_msat")],
                (capacity / 2).to_string()
            );
        }

        network.shutdown().await;
    }
}
