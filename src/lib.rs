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
//! let x = dace_rs::Da::variable(1);
//! let y = dace_rs::Da::variable(2);
//! let f = (x.clone() * x.clone() + y).sin();
//! // coefficient of x^2 in sin(x^2+y) at the origin is cos(0) = 1
//! assert!((f.get_coefficient(&[2, 0]) - 1.0).abs() < 1e-14);
//! ```

pub mod context;
pub mod error;
pub mod monomial;

mod da;
pub mod elementary;
mod io;
mod kernels;
pub mod special;

pub use context::{
    epsilon, init, initialized, machine_epsilon, max_monomials, max_order, max_variables,
    pop_truncation_order, push_truncation_order, set_epsilon, set_truncation_order,
    truncation_order, version,
};
pub use da::Da;
pub use elementary::{
    acos, acosh, asin, asinh, atan, atan2, cbrt, cos, cosh, erf, erfc, exp, hypot, icrt, isrt, log,
    log_base, log2, log10, modulo, powf, powi, root, round, sin, sinh, sqrt, tan, tanh, trunc,
};
pub use special::{bessel_i, bessel_j, bessel_k, bessel_y, gamma, log_gamma, psi};

/// The weighted sum `afac * a + bfac * b` (`daceWeightedSum`).
///
/// Unlike the C routine of the same name, this is aliasing-safe.
///
/// # Panics
///
/// Panics with [`DaceError`] when `a` and `b` belong to different DACE
/// contexts.
pub fn fma(a: &Da, afac: f64, b: &Da, bfac: f64) -> Da {
    Da::assert_same_context(a, b);
    kernels::weighted_sum(a, afac, b, bfac)
}

/// Tests mutate the process-global DACE context; serialize context-touching
/// tests with this lock.
#[cfg(test)]
pub(crate) mod test_support {
    use parking_lot::Mutex;

    pub static CONTEXT_LOCK: Mutex<()> = Mutex::new(());
}
