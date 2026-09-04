use std::{
    error::Error as _,
    fmt, io,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Result, bail};
use serde::Serialize;

pub const DEFAULT_PROVIDER_MAX_ATTEMPTS: u32 = 4;
pub const DEFAULT_PROVIDER_RETRY_BASE_DELAY: Duration = Duration::from_secs(1);
pub const DEFAULT_PROVIDER_RETRY_MAX_DELAY: Duration = Duration::from_secs(30);

const MAX_PROVIDER_ATTEMPTS: u32 = 10;
const MAX_PROVIDER_RETRY_DELAY: Duration = Duration::from_secs(5 * 60);
static JITTER_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProviderRetryPolicy {
    max_attempts: u32,
    base_delay: Duration,
    max_delay: Duration,
}

impl ProviderRetryPolicy {
    pub fn new(max_attempts: u32, base_delay: Duration, max_delay: Duration) -> Result<Self> {
        if !(1..=MAX_PROVIDER_ATTEMPTS).contains(&max_attempts) {
            bail!("provider max attempts must be between 1 and {MAX_PROVIDER_ATTEMPTS}");
        }
        if base_delay > max_delay {
            bail!("provider retry base delay must not exceed the maximum delay");
        }
        if max_delay > MAX_PROVIDER_RETRY_DELAY {
            bail!("provider retry maximum delay must not exceed 300 seconds");
        }
        Ok(Self {
            max_attempts,
            base_delay,
            max_delay,
        })
    }

    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            max_attempts: 1,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
        }
    }

    pub(crate) const fn max_attempts(self) -> u32 {
        self.max_attempts
    }

    pub(crate) fn delay_after(self, failed_attempt: u32) -> Duration {
        if self.base_delay.is_zero() || self.max_delay.is_zero() {
            return Duration::ZERO;
        }
        let shift = failed_attempt.saturating_sub(1).min(31);
        let exponential = self
            .base_delay
            .checked_mul(1_u32 << shift)
            .unwrap_or(self.max_delay)
            .min(self.max_delay);
        let nanos = exponential.as_nanos();
        let spread = nanos / 4;
        if spread == 0 {
            return exponential;
        }
        let width = spread.saturating_mul(2).saturating_add(1);
        let jitter = u128::from(jitter_seed()) % width;
        let jittered = nanos.saturating_sub(spread).saturating_add(jitter);
        duration_from_nanos(jittered.min(self.max_delay.as_nanos()))
    }

    pub(crate) fn trace_value(self) -> serde_json::Value {
        serde_json::json!({
            "max_attempts": self.max_attempts,
            "base_delay_ms": duration_millis(self.base_delay),
            "max_delay_ms": duration_millis(self.max_delay),
        })
    }
}

impl Default for ProviderRetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_PROVIDER_MAX_ATTEMPTS,
            base_delay: DEFAULT_PROVIDER_RETRY_BASE_DELAY,
            max_delay: DEFAULT_PROVIDER_RETRY_MAX_DELAY,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProviderErrorClass {
    Overload,
    RateLimit,
    Server,
    ConnectionReset,
    Timeout,
    InvalidRequest,
    Authentication,
    ContextLimit,
    Schema,
    Other,
}

impl ProviderErrorClass {
    pub(crate) const fn is_transient(self) -> bool {
        matches!(
            self,
            Self::Overload | Self::RateLimit | Self::Server | Self::ConnectionReset | Self::Timeout
        )
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Overload => "overload",
            Self::RateLimit => "rate_limit",
            Self::Server => "server",
            Self::ConnectionReset => "connection_reset",
            Self::Timeout => "timeout",
            Self::InvalidRequest => "invalid_request",
            Self::Authentication => "authentication",
            Self::ContextLimit => "context_limit",
            Self::Schema => "schema",
            Self::Other => "other",
        }
    }
}

#[derive(Debug)]
pub(crate) struct ProviderRequestError {
    class: ProviderErrorClass,
}

impl ProviderRequestError {
    pub(crate) const fn new(class: ProviderErrorClass) -> Self {
        Self { class }
    }

    pub(crate) fn from_http(status: reqwest::StatusCode, body: &str) -> Self {
        Self::new(classify(Some(status.as_u16()), body))
    }

    pub(crate) fn from_provider_event(value: &serde_json::Value) -> Self {
        let error = provider_event_error(value);
        let status = error
            .get("status")
            .or_else(|| value.get("status"))
            .and_then(|status| {
                status
                    .as_u64()
                    .and_then(|status| u16::try_from(status).ok())
                    .or_else(|| status.as_str()?.parse().ok())
            });
        Self::new(classify(status, &provider_event_signature(value, error)))
    }

    pub(crate) fn from_transport(error: &reqwest::Error) -> Option<Self> {
        if error.is_timeout() {
            return Some(Self::new(ProviderErrorClass::Timeout));
        }
        if error.is_connect() || transport_source_was_reset(error) {
            return Some(Self::new(ProviderErrorClass::ConnectionReset));
        }
        Self::from_stream_transport(&error.to_string())
    }

    pub(crate) fn from_stream_transport(message: &str) -> Option<Self> {
        let normalized = message.to_ascii_lowercase();
        if contains_any(&normalized, &["timed out", "timeout"]) {
            Some(Self::new(ProviderErrorClass::Timeout))
        } else if contains_any(
            &normalized,
            &[
                "connection reset",
                "connection closed",
                "connection aborted",
                "broken pipe",
            ],
        ) {
            Some(Self::new(ProviderErrorClass::ConnectionReset))
        } else {
            None
        }
    }

    pub(crate) const fn class(&self) -> ProviderErrorClass {
        self.class
    }
}

impl fmt::Display for ProviderRequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Codex provider request failed ({})",
            self.class.as_str()
        )
    }
}

impl std::error::Error for ProviderRequestError {}

pub(crate) fn transient_provider_error(error: &anyhow::Error) -> Option<ProviderErrorClass> {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<ProviderRequestError>())
        .map(ProviderRequestError::class)
        .filter(|class| class.is_transient())
}

pub(crate) fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn classify(status: Option<u16>, signature: &str) -> ProviderErrorClass {
    let normalized = signature.to_ascii_lowercase();
    if matches!(status, Some(401 | 403))
        || contains_any(
            &normalized,
            &[
                "authentication_error",
                "invalid_api_key",
                "unauthorized",
                "forbidden",
                "not signed in",
            ],
        )
    {
        return ProviderErrorClass::Authentication;
    }
    if contains_any(
        &normalized,
        &[
            "context_length",
            "context limit",
            "context window",
            "maximum context",
            "too many tokens",
        ],
    ) {
        return ProviderErrorClass::ContextLimit;
    }
    if contains_any(
        &normalized,
        &[
            "invalid_json_schema",
            "invalid schema",
            "schema validation",
            "schema_error",
        ],
    ) {
        return ProviderErrorClass::Schema;
    }
    if matches!(status, Some(400 | 404 | 409 | 413 | 422))
        || contains_any(
            &normalized,
            &["invalid_request", "bad_request", "malformed request"],
        )
    {
        return ProviderErrorClass::InvalidRequest;
    }
    if status == Some(429)
        || contains_any(
            &normalized,
            &["rate_limit", "rate limit", "too many requests"],
        )
    {
        return ProviderErrorClass::RateLimit;
    }
    if contains_any(
        &normalized,
        &[
            "overloaded",
            "overload_error",
            "at capacity",
            "capacity temporarily",
            "insufficient capacity",
            "insufficient_capacity",
            "capacity_error",
        ],
    ) {
        return ProviderErrorClass::Overload;
    }
    if matches!(status, Some(408 | 504))
        || contains_any(&normalized, &["request_timeout", "timed out", "timeout"])
    {
        return ProviderErrorClass::Timeout;
    }
    if contains_any(
        &normalized,
        &[
            "connection reset",
            "connection closed",
            "connection aborted",
            "broken pipe",
        ],
    ) {
        return ProviderErrorClass::ConnectionReset;
    }
    if status.is_some_and(|status| (500..=599).contains(&status))
        || contains_any(
            &normalized,
            &[
                "server_error",
                "internal_server_error",
                "internal server error",
            ],
        )
    {
        return ProviderErrorClass::Server;
    }
    ProviderErrorClass::Other
}

fn provider_event_error(value: &serde_json::Value) -> &serde_json::Value {
    value
        .get("error")
        .or_else(|| {
            value
                .get("response")
                .and_then(|response| response.get("error"))
        })
        .unwrap_or(value)
}

fn provider_event_signature(value: &serde_json::Value, error: &serde_json::Value) -> String {
    ["type", "code", "message"]
        .into_iter()
        .filter_map(|field| error.get(field).and_then(serde_json::Value::as_str))
        .chain(value.get("type").and_then(serde_json::Value::as_str))
        .collect::<Vec<_>>()
        .join(" ")
}

fn transport_source_was_reset(error: &reqwest::Error) -> bool {
    let mut source = error.source();
    while let Some(cause) = source {
        if let Some(error) = cause.downcast_ref::<io::Error>()
            && matches!(
                error.kind(),
                io::ErrorKind::ConnectionReset
                    | io::ErrorKind::ConnectionAborted
                    | io::ErrorKind::BrokenPipe
                    | io::ErrorKind::TimedOut
            )
        {
            return true;
        }
        source = cause.source();
    }
    false
}

fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}

fn jitter_seed() -> u64 {
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let nonce = JITTER_NONCE.fetch_add(1, Ordering::Relaxed);
    let mut value = time ^ nonce.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn duration_from_nanos(nanos: u128) -> Duration {
    let seconds = nanos / 1_000_000_000;
    let subsecond_nanos = (nanos % 1_000_000_000) as u32;
    Duration::new(u64::try_from(seconds).unwrap_or(u64::MAX), subsecond_nanos)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn provider_error_classification_has_nontransient_precedence() {
        for (value, expected) in [
            (
                json!({"status": 400, "error": {"type": "invalid_request_error"}}),
                ProviderErrorClass::InvalidRequest,
            ),
            (
                json!({"status": 401, "error": {"message": "overloaded while authenticating"}}),
                ProviderErrorClass::Authentication,
            ),
            (
                json!({"error": {"code": "context_length_exceeded", "message": "server overloaded"}}),
                ProviderErrorClass::ContextLimit,
            ),
            (
                json!({"status": 500, "error": {"code": "invalid_json_schema"}}),
                ProviderErrorClass::Schema,
            ),
        ] {
            assert_eq!(
                ProviderRequestError::from_provider_event(&value).class(),
                expected
            );
        }
    }

    #[test]
    fn transient_http_and_stream_failures_are_classified_explicitly() {
        for (value, expected) in [
            (
                json!({"status": 429, "error": {"message": "slow down"}}),
                ProviderErrorClass::RateLimit,
            ),
            (
                json!({"status": 502, "error": {"message": "gateway failure"}}),
                ProviderErrorClass::Server,
            ),
            (
                json!({"error": {"message": "Our servers are currently overloaded. Please try again later."}}),
                ProviderErrorClass::Overload,
            ),
            (
                json!({"response": {"error": {"code": "request_timeout"}}}),
                ProviderErrorClass::Timeout,
            ),
            (
                json!({"error": {"message": "connection reset by peer"}}),
                ProviderErrorClass::ConnectionReset,
            ),
        ] {
            let class = ProviderRequestError::from_provider_event(&value).class();
            assert_eq!(class, expected);
            assert!(class.is_transient());
        }
    }

    #[test]
    fn retry_delay_is_exponential_capped_and_jittered() {
        let policy =
            ProviderRetryPolicy::new(5, Duration::from_millis(100), Duration::from_millis(350))
                .unwrap();
        let first = policy.delay_after(1);
        let third = policy.delay_after(3);
        assert!((75..=125).contains(&first.as_millis()));
        assert!((262..=350).contains(&third.as_millis()));
    }
}
