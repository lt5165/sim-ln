//! Combined results logs for payments and probes.
//!
//! When results are written, a simulation produces two logs in addition to the payment results CSV:
//! - `events_<start time>.csv`: one row per payment and per probe, with the same columns as the payment results plus a
//!   `kind` column (`payment` or `probe`) and columns describing where and why a failed HTLC failed.
//! - `probe_balances_<start time>.csv`: one row per channel on a probe's path, written when the probe resolves, that
//!   compares the probing node's scorer estimate of the channel's liquidity with the channel's actual balance.
//!
//! Files are only created once their first row is written, so a simulation without probing has no balances log.

use std::path::PathBuf;
use std::sync::Mutex as StdMutex;
use std::time::{SystemTime, UNIX_EPOCH};

use bitcoin::secp256k1::PublicKey;
use lightning::routing::router::Path;
use lightning::types::payment::PaymentHash;
use serde::Serialize;

use crate::batched_writer::BatchedWriter;
use crate::sim_node::ProbeResult;
use crate::{serializers, Payment, PaymentOutcome, PaymentResult, SimulationError};

/// Whether a row in the events log is a payment or a probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    Payment,
    Probe,
}

/// A row in the events log.
#[derive(Debug, Clone, Serialize)]
pub struct EventRecord {
    /// Whether this row is a payment or a probe.
    pub kind: EventKind,
    /// The node that sent the payment or probe.
    pub source: PublicKey,
    /// The destination of the payment, or the final hop of the probe.
    pub destination: PublicKey,
    /// The amount delivered to the destination, excluding fees.
    pub amount_msat: u64,
    /// The payment hash, if the payment was dispatched.
    #[serde(with = "serializers::serde_option_payment_hash")]
    pub hash: Option<PaymentHash>,
    /// The simulation time at which the payment or probe was dispatched.
    #[serde(with = "serde_millis")]
    pub dispatch_time: SystemTime,
    /// The number of HTLCs used.
    pub htlc_count: usize,
    /// The outcome. For probes, `Success` means that the probe reached its final hop.
    #[serde(serialize_with = "serialize_outcome")]
    pub payment_outcome: PaymentOutcome,
    /// The short channel id of the channel that could not carry the HTLC.
    pub failed_scid: Option<u64>,
    /// The kind of failure, e.g. `InsufficientBalance`.
    pub failure_reason: Option<String>,
    /// The full failure message, including the values that caused it.
    pub failure_detail: Option<String>,
}

impl EventRecord {
    fn new(
        kind: EventKind,
        source: PublicKey,
        destination: PublicKey,
        amount_msat: u64,
        hash: Option<PaymentHash>,
        dispatch_time: SystemTime,
        result: &PaymentResult,
    ) -> Self {
        let failure = result.failure.as_ref();
        EventRecord {
            kind,
            source,
            destination,
            amount_msat,
            hash,
            dispatch_time,
            htlc_count: result.htlc_count,
            payment_outcome: result.payment_outcome.clone(),
            failed_scid: failure.and_then(|f| f.failed_scid),
            failure_reason: failure.map(|f| f.reason.clone()),
            failure_detail: failure.map(|f| f.detail.clone()),
        }
    }
}

/// A row in the probe balances log: one probing node's estimate of one channel direction's liquidity, next to that
/// channel's actual balance, recorded when a probe over the channel resolves.
#[derive(Debug, Clone, Serialize)]
pub struct ProbeBalanceRecord {
    /// The simulation time at which the row was recorded (when the probe resolved).
    #[serde(with = "serde_millis")]
    pub time: SystemTime,
    /// The payment hash of the probe, matching its row in the events log.
    #[serde(with = "serializers::serde_option_payment_hash")]
    pub probe_hash: Option<PaymentHash>,
    /// The probing node, whose scorer the estimate comes from.
    pub observer: PublicKey,
    /// The outcome of the probe.
    #[serde(serialize_with = "serialize_outcome")]
    pub probe_outcome: PaymentOutcome,
    /// The index of this channel in the probe's path.
    pub hop_index: usize,
    /// The short channel id of the channel.
    pub scid: u64,
    /// The node sending over the channel in the probe's direction.
    pub from_node: PublicKey,
    /// The node receiving over the channel in the probe's direction.
    pub to_node: PublicKey,
    /// The total capacity of the channel.
    pub capacity_msat: u64,
    /// The scorer's lower bound on the liquidity available from `from_node` to `to_node`, or empty if the scorer has
    /// no estimate for the channel.
    pub estimated_min_msat: Option<u64>,
    /// The scorer's upper bound on the liquidity available from `from_node` to `to_node`, or empty if the scorer has
    /// no estimate for the channel.
    pub estimated_max_msat: Option<u64>,
    /// `from_node`'s actual balance in the channel, excluding outgoing HTLCs that are in flight.
    pub actual_balance_msat: u64,
    /// The largest HTLC that `from_node` could actually send over the channel, which also accounts for the maximum
    /// HTLC size and in-flight limits.
    pub actual_sendable_msat: u64,
}

/// Writes an outcome in full (e.g. `IndexFailure(2)`). The CSV writer would otherwise write only the index for
/// `IndexFailure`, dropping the variant name.
fn serialize_outcome<S: serde::Serializer>(
    outcome: &PaymentOutcome,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(&format!("{outcome:?}"))
}

/// A CSV file that is created when its first row is written.
struct LazyWriter {
    directory: PathBuf,
    file_name: String,
    batch_size: u32,
    writer: Option<BatchedWriter>,
}

impl LazyWriter {
    fn queue<S: Serialize>(&mut self, record: S) -> Result<(), SimulationError> {
        if self.writer.is_none() {
            self.writer = Some(BatchedWriter::new(
                self.directory.clone(),
                self.file_name.clone(),
                self.batch_size,
            )?);
        }

        self.writer
            .as_mut()
            .expect("writer created above")
            .queue(record)
    }

    fn flush(&mut self) -> Result<(), SimulationError> {
        match self.writer.as_mut() {
            Some(writer) => writer.write(true),
            None => Ok(()),
        }
    }
}

/// Writes the events and probe balances logs. Shared between the simulation (which logs payments) and its probers.
pub struct EventLog {
    events: StdMutex<LazyWriter>,
    balances: StdMutex<LazyWriter>,
}

impl EventLog {
    /// Creates logs in `directory`, named with the simulation's start time.
    pub fn new(directory: PathBuf, batch_size: u32, start_time: SystemTime) -> Self {
        let start_secs = start_time
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_default();

        let writer = |prefix: &str| {
            StdMutex::new(LazyWriter {
                directory: directory.clone(),
                file_name: format!("{prefix}_{start_secs}.csv"),
                batch_size,
                writer: None,
            })
        };

        EventLog {
            events: writer("events"),
            balances: writer("probe_balances"),
        }
    }

    /// Writes a row to the events log.
    pub fn log_event(&self, record: &EventRecord) -> Result<(), SimulationError> {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .queue(record)
    }

    /// Writes a payment to the events log.
    pub(crate) fn log_payment(
        &self,
        payment: &Payment,
        result: &PaymentResult,
    ) -> Result<(), SimulationError> {
        self.log_event(&EventRecord::new(
            EventKind::Payment,
            payment.source,
            payment.destination,
            payment.amount_msat,
            payment.hash,
            payment.dispatch_time,
            result,
        ))
    }

    /// Writes a probe sent by `source` over `path` to the events log.
    pub fn log_probe(
        &self,
        source: PublicKey,
        path: &Path,
        probe: &ProbeResult,
    ) -> Result<(), SimulationError> {
        let destination = match path.hops.last() {
            Some(hop) => hop.pubkey,
            None => return Ok(()),
        };

        self.log_event(&EventRecord::new(
            EventKind::Probe,
            source,
            destination,
            path.final_value_msat(),
            Some(probe.payment_hash),
            probe.dispatch_time,
            &probe.result,
        ))
    }

    /// Writes a row to the probe balances log.
    pub fn log_balance(&self, record: &ProbeBalanceRecord) -> Result<(), SimulationError> {
        self.balances
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .queue(record)
    }

    /// Flushes both logs to disk.
    pub fn flush(&self) -> Result<(), SimulationError> {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .flush()?;
        self.balances
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::get_random_keypair;
    use crate::FailureDetails;
    use std::time::Duration;

    fn read_csv(path: &std::path::Path) -> Vec<Vec<String>> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(path)
            .unwrap();
        reader
            .records()
            .map(|r| r.unwrap().iter().map(String::from).collect())
            .collect()
    }

    /// Tests that payments and probes are written to the same events log, with outcomes written in full and failure
    /// details filled in when known, and that the balances log is only created once it has a row.
    #[test]
    fn test_events_log_payments_and_probes() {
        let dir = tempfile::tempdir().unwrap();
        let start = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let log = EventLog::new(dir.path().to_path_buf(), 1, start);
        let (source, destination) = (get_random_keypair().1, get_random_keypair().1);

        log.log_event(&EventRecord::new(
            EventKind::Payment,
            source,
            destination,
            1_000,
            Some(PaymentHash([1; 32])),
            start,
            &PaymentResult {
                htlc_count: 1,
                payment_outcome: PaymentOutcome::Success,
                failure: None,
            },
        ))
        .unwrap();
        log.log_event(&EventRecord::new(
            EventKind::Probe,
            source,
            destination,
            2_000,
            Some(PaymentHash([2; 32])),
            start,
            &PaymentResult {
                htlc_count: 0,
                payment_outcome: PaymentOutcome::IndexFailure(2),
                failure: Some(FailureDetails {
                    failed_scid: Some(42),
                    reason: "InsufficientBalance".to_string(),
                    detail: "InsufficientBalance: amount: 2000 > balance: 10".to_string(),
                }),
            },
        ))
        .unwrap();
        log.flush().unwrap();

        let rows = read_csv(&dir.path().join("events_1700000000.csv"));
        assert_eq!(
            rows[0],
            vec![
                "kind",
                "source",
                "destination",
                "amount_msat",
                "hash",
                "dispatch_time",
                "htlc_count",
                "payment_outcome",
                "failed_scid",
                "failure_reason",
                "failure_detail"
            ]
        );
        assert_eq!(rows[1][0], "payment");
        assert_eq!(rows[1][7], "Success");
        assert_eq!(rows[1][8..], ["", "", ""]);
        assert_eq!(rows[2][0], "probe");
        assert_eq!(rows[2][7], "IndexFailure(2)");
        assert_eq!(rows[2][8], "42");
        assert_eq!(rows[2][9], "InsufficientBalance");

        // No probe balances were logged, so there should be no balances file.
        assert!(!dir.path().join("probe_balances_1700000000.csv").exists());
    }
}
