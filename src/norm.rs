//! Norms, order-sorted norms, estimates, and interval bounds of DA values.
//!
//! Ports `core/dacenorm.c` and the C++ `DA::convRadius`: the norm type is the
//! C `ityp` code (0 = max/infinity, 1 = sum, `p > 1` = the `p`-vector norm;
//! `p = 2` is the Euclidean norm), grouping is by monomial order (`var = 0`)
//! or by the exponent of a chosen variable.

use crate::context::{Context, pown, truncation_order};
use crate::da::Da;
use std::sync::Arc;

/// The type of a coefficient norm (the C `ityp` code).
///
/// `ityp = 0` is the maximum (infinity) norm, `ityp = 1` the sum norm, and
/// `ityp = p > 1` the `p`-th vector norm ( [`NormType::EUCLIDEAN`] is
/// `p = 2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// Maximum absolute coefficient (`ityp = 0`).
    Infinity,
    /// Sum of absolute coefficients (`ityp = 1`).
    One,
    /// `p`-th vector norm of the coefficients (`p > 1`).
    Power(u32),
}

impl NormType {
    /// The Euclidean (`L²`) norm.
    pub const EUCLIDEAN: NormType = NormType::Power(2);

    pub(crate) fn ityp(self) -> u32 {
        match self {
            NormType::Infinity => 0,
            NormType::One => 1,
            NormType::Power(p) => p,
        }
    }
}

/// A closed interval `[lo, hi]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    /// Lower bound.
    pub lo: f64,
    /// Upper bound.
    pub hi: f64,
}

/// Accumulate one coefficient into `acc` according to the norm type
/// (shared by the plain and order-sorted norms).
#[inline]
fn accumulate(acc: &mut f64, c: f64, ityp: u32) {
    if ityp == 1 {
        *acc += c.abs();
    } else if ityp > 1 {
        *acc += pown(c.abs(), ityp);
    } else {
        *acc = acc.max(c.abs());
    }
}

/// Finalize an accumulated norm (root for vector norms).
#[inline]
fn finalize(acc: f64, ityp: u32) -> f64 {
    if ityp > 1 {
        acc.powf(1.0 / f64::from(ityp))
    } else {
        acc
    }
}

impl Da {
    /// The absolute value (maximum coefficient norm) of the DA
    /// (`daceAbsoluteValue`).
    pub fn abs(&self) -> f64 {
        self.terms.iter().fold(0.0, |m, t| m.max(t.c.abs()))
    }

    /// A norm of all coefficients (`daceNorm`).
    pub fn norm(&self, ityp: NormType) -> f64 {
        let ityp = ityp.ityp();
        let mut acc = 0.0;
        for t in &self.terms {
            accumulate(&mut acc, t.c, ityp);
        }
        finalize(acc, ityp)
    }

    /// Order-sorted norms (`daceOrderedNorm`): one norm per order
    /// (`var == 0`, length `nomax + 1`) or per exponent of variable `var`
    /// (1-based, length `nomax + 1`).
    ///
    /// Out-of-range variables log a warning and return all zeros.
    pub fn order_norm(&self, var: u32, ityp: NormType) -> Vec<f64> {
        let ctx = &self.ctx;
        let ityp = ityp.ityp();
        let mut onorm = vec![0.0; ctx.nomax as usize + 1];
        if var > ctx.nvmax {
            log::warn!("DACE error 624: invalid independent variable {var} in order_norm");
            return onorm;
        }
        if var == 0 {
            for t in &self.terms {
                let io = ctx.order_of(t.idx);
                accumulate(&mut onorm[io as usize], t.c, ityp);
            }
            if ityp > 1 {
                for v in onorm.iter_mut() {
                    *v = finalize(*v, ityp);
                }
            }
        } else {
            let mut jj = vec![0u32; ctx.nvmax as usize];
            for t in &self.terms {
                ctx.decode_into(t.idx, &mut jj);
                accumulate(&mut onorm[jj[(var - 1) as usize] as usize], t.c, ityp);
            }
            if ityp > 1 {
                for v in onorm.iter_mut() {
                    *v = finalize(*v, ityp);
                }
            }
        }
        onorm
    }

    /// Estimate order-sorted norms up to order `nc` by an exponential
    /// least-squares fit (`daceEstimate`): returns the estimates `c[0..=nc]`.
    ///
    /// If fewer than two orders have non-negligible norms, the fit is
    /// impossible; a warning is logged and zeros are returned (as in C,
    /// where this is informational).
    ///
    /// # Panics
    ///
    /// Panics with [`DaceError`] code 651 when `nomax < 2`.
    pub fn estim_norm(&self, var: u32, ityp: NormType, nc: u32) -> Vec<f64> {
        self.estim_norm_impl(var, ityp, nc, false).0
    }

    /// Like [`Da::estim_norm`], also returning the fit residuals per order
    /// (length `min(nc, nomax) + 1`).
    pub fn estim_norm_err(&self, var: u32, ityp: NormType, nc: u32) -> (Vec<f64>, Vec<f64>) {
        self.estim_norm_impl(var, ityp, nc, true)
    }

    fn estim_norm_impl(
        &self,
        var: u32,
        ityp: NormType,
        nc: u32,
        with_err: bool,
    ) -> (Vec<f64>, Vec<f64>) {
        let ctx: Arc<Context> = self.ctx.clone();
        let mut c = vec![0.0; nc as usize + 1];
        let mut err = vec![0.0; nc.min(ctx.nomax) as usize + 1];
        if ctx.nomax < 2 {
            crate::error::dace_panic(651, "No estimate is possible");
        }

        let (eps, _) = crate::context::eps_nocut();
        let onorm = self.order_norm(var, ityp);

        // set up xtx and xty for the linear least squares fit
        let mut ai = [0.0f64; 2];
        let mut xtx = [[0.0f64; 2]; 2];
        for (i, &o) in onorm.iter().enumerate().skip(1) {
            // negated `<=` (as in C) so NaN norms are treated as non-zero
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(o <= eps) {
                let fi = i as f64;
                xtx[0][0] += fi * fi;
                xtx[0][1] -= fi;
                xtx[1][1] += 1.0;
                ai[0] += o.ln();
                ai[1] += fi * o.ln();
            }
        }

        if xtx[1][1] < 2.0 {
            log::warn!("DACE info 163: norm estimate not possible; returning zeros");
            return (c, err);
        }

        xtx[1][0] = xtx[0][1];
        let det = xtx[0][0] * xtx[1][1] - xtx[0][1] * xtx[1][0];

        let a = [
            (ai[0] * xtx[0][0] + ai[1] * xtx[0][1]) / det,
            (ai[0] * xtx[1][0] + ai[1] * xtx[1][1]) / det,
        ];

        for (i, ci) in c.iter_mut().enumerate() {
            *ci = (a[0] + a[1] * i as f64).exp();
        }

        if with_err {
            for i in 0..err.len() {
                let temp = onorm[i] - c[i];
                err[i] = if temp > 0.0 { temp } else { 0.0 };
            }
        }
        (c, err)
    }

    /// An upper and lower bound of the DA over `[-1, 1]^nv`
    /// (`daceGetBounds`): monomials with all-even exponents contribute
    /// their signed coefficient to one side, others their absolute value to
    /// both.
    pub fn bound(&self) -> Interval {
        let ctx = &self.ctx;
        let mut lo = 0.0;
        let mut hi = 0.0;
        let mut terms = self.terms.iter().peekable();

        // constant part is special
        if terms.peek().is_some_and(|t| t.idx == 0) {
            let c = terms.next().unwrap().c;
            lo = c;
            hi = c;
        }

        let mut jj = vec![0u32; ctx.nvmax as usize];
        for t in terms {
            ctx.decode_into(t.idx, &mut jj);
            let odd = jj.iter().any(|&e| e & 1 != 0);
            if odd {
                hi += t.c.abs();
                lo -= t.c.abs();
            } else if t.c > 0.0 {
                hi += t.c;
            } else {
                lo += t.c;
            }
        }
        Interval { lo, hi }
    }

    /// Estimate the convergence radius: the radius at which the estimated
    /// norm of the next order falls below `eps` (C++ `DA::convRadius`).
    pub fn conv_radius(&self, eps: f64, ityp: NormType) -> f64 {
        let ord = truncation_order();
        let res = self.estim_norm(0, ityp, ord + 1);
        (eps / res[(ord + 1) as usize]).powf(1.0 / f64::from(ord + 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::CONTEXT_LOCK;

    #[test]
    fn norms_and_bounds() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(4, 2).unwrap();
        let x = Da::variable(1);
        let y = Da::variable(2);
        let f = 1.0 + 2.0 * x.clone() + 3.0 * y.clone();

        assert!((f.abs() - 3.0).abs() < 1e-15);
        assert!((f.norm(NormType::One) - 6.0).abs() < 1e-15);
        assert!((f.norm(NormType::EUCLIDEAN) - 14.0f64.sqrt()).abs() < 1e-15);
        assert!((f.norm(NormType::Power(4)) - 98.0f64.powf(0.25)).abs() < 1e-14);

        let on = f.order_norm(0, NormType::One);
        assert_eq!(on.len(), 5);
        assert!((on[0] - 1.0).abs() < 1e-15);
        assert!((on[1] - 5.0).abs() < 1e-15);
        let ov = f.order_norm(1, NormType::One);
        assert!((ov[0] - 4.0).abs() < 1e-15); // constant + y
        assert!((ov[1] - 2.0).abs() < 1e-15); // x

        // bound of 1+x on [-1,1] is [0,2]
        let b = (1.0 + x.clone()).bound();
        assert!((b.lo - 0.0).abs() < 1e-15);
        assert!((b.hi - 2.0).abs() < 1e-15);
        // even monomial: x^2 contributes to one side
        let b2 = (x.clone() * x.clone() - 1.0).bound();
        assert!((b2.lo + 1.0).abs() < 1e-15);
        assert!((b2.hi - 0.0).abs() < 1e-15);

        // estim_norm of exp(x): order norms ~ 1/i!, exponential-ish fit
        let e = crate::elementary::exp(&x);
        let (c, err) = e.estim_norm_err(0, NormType::One, 6);
        assert_eq!(c.len(), 7);
        assert_eq!(err.len(), 5); // min(nc=6, nomax=4)+1
        assert!(c.iter().all(|v| v.is_finite()));

        // conv_radius is positive and finite for exp(x)
        let r = e.conv_radius(1e-6, NormType::One);
        assert!(r.is_finite() && r > 0.0);
    }
}
