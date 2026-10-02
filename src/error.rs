//! Error type for DACE operations.
//!
//! DACE's C core uses a sticky thread-local error code (`XYY`: severity `X`,
//! number `YY`) that is polled by the caller, and calls `exit(1)` on internal
//! panic-level failures. `dace-rs` instead:
//!
//! - reports recoverable failures (initialization, parsing, blob import) as
//!   [`Result::Err`] holding a [`DaceError`] carrying the same numeric `XYY`
//!   code as the C library;
//! - **panics** with a `DaceError` payload for numeric/domain misuse during
//!   operations (division by zero, logarithm of a non-positive DA, ...), since
//!   the operator traits (`Mul`, `Div`, ...) cannot return `Result`;
//! - downgrades the C library's purely informational messages (severity
//!   `INFO`/`WARNING`) to [`log`] records with documented degraded behavior,
//!   mirroring the C library instead of failing.
//!
//! There is no sticky error state and no `exit`.

use std::fmt;

/// A DACE error: a numeric `XYY` code (severity `X` in `{1,3,6,9,10}`, number
/// `YY`) plus a human-readable message, matching the C library's error table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaceError {
    /// Numeric `XYY` error code, identical to the C library's `daceGetError`.
    pub code: u32,
    /// Human-readable error message.
    pub message: String,
}

impl DaceError {
    /// Create a new error with the given code and message.
    pub fn new(code: u32, message: impl Into<String>) -> Self {
        DaceError {
            code,
            message: message.into(),
        }
    }

    /// Severity digit `X` of the `XYY` code (1 = info, 3 = warning,
    /// 6 = error, 9 = severe, 10 = panic).
    pub fn severity(&self) -> u32 {
        self.code / 100
    }
}

impl fmt::Display for DaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DACE error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for DaceError {}

/// Panic with a `DaceError` payload (numeric/domain misuse during operations).
pub(crate) fn dace_panic(code: u32, message: &str) -> ! {
    std::panic::panic_any(DaceError::new(code, message))
}

/// Error codes reused across the crate (from the C library's `DACEerr` table).
// Error codes reused across the crate; some are only used by later-phase
// modules (elementary/special functions), so unused variants are allowed.
#[allow(dead_code)]
pub(crate) mod codes {
    pub const NOT_INITIALIZED: u32 = 1003;
    pub const ORDER_VARIABLE_TOO_LARGE: u32 = 911;
    pub const ORDER_TOO_LARGE: u32 = 622;
    pub const INVALID_ENCODED_EXPONENT: u32 = 626;
    pub const DIVIDING_BY_ZERO: u32 = 641;
    pub const INVERSE_DOES_NOT_EXIST: u32 = 642;
    pub const NON_INTEGER_POWER_NON_POSITIVE: u32 = 643;
    pub const ZERO_TH_ROOT: u32 = 644;
    pub const EVEN_ROOT_NEGATIVE: u32 = 645;
    pub const ODD_ROOT_ZERO: u32 = 646;
    pub const LOG_NON_POSITIVE: u32 = 647;
    pub const LOG_BASE_POSITIVE: u32 = 648;
    pub const COS_ZERO_IN_TANGENT: u32 = 649;
    pub const OUT_OF_DOMAIN: u32 = 650;
}
