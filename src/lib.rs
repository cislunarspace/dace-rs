//! Pure Rust implementation of DACE, the Differential Algebra Computational
//! Toolbox.
//!
//! DACE computes with truncated multivariate Taylor polynomials ("differential
//! algebra"): it propagates high-order expansions of arbitrary multivariate
//! functions through arithmetic and elementary-function composition, with a
//! configurable cut-off epsilon and truncation order. This crate is a full
//! rewrite of [DACE 2.1](https://github.com/dacelib/dace) in pure Rust.
//!
//! # Quickstart
//!
//! ```
//! dace_rs::init(20, 2).unwrap();
//! assert_eq!(dace_rs::max_order(), 20);
//! assert_eq!(dace_rs::max_variables(), 2);
//! assert_eq!(dace_rs::max_monomials(), 231);
//! ```

pub mod context;
pub mod error;
pub mod monomial;

pub use context::{
    epsilon, init, initialized, machine_epsilon, max_monomials, max_order, max_variables,
    pop_truncation_order, push_truncation_order, set_epsilon, set_truncation_order,
    truncation_order, version,
};
pub use error::DaceError;
pub use monomial::Monomial;

/// Tests mutate the process-global DACE context; serialize context-touching
/// tests with this lock.
#[cfg(test)]
pub(crate) mod test_support {
    use parking_lot::Mutex;

    pub static CONTEXT_LOCK: Mutex<()> = Mutex::new(());
}
