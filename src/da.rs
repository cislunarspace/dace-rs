//! The DA type: a truncated multivariate Taylor polynomial.

use std::cell::Cell;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::sync::Arc;

use crate::context::Context;
use crate::elementary::minv;
use crate::error::{codes, dace_panic};
use crate::kernels::{multiply, weighted_sum};
use crate::monomial::Monomial;

/// One stored term of a [`Da`]: packed monomial index plus coefficient.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RawTerm {
    pub idx: u32,
    pub c: f64,
}

/// A truncated multivariate Taylor polynomial ("differential algebra" value).
///
/// Stored as a sparse list of `(monomial index, coefficient)` terms, sorted by
/// ascending monomial index; the index packs the exponent vector into the
/// canonical position determined by the active [context][crate::context].
///
/// Values created under one context keep working after
/// [`init`][crate::init] is called again (they hold on to their original
/// context); mixing values from different contexts in one operation panics.
#[derive(Clone, Debug)]
pub struct Da {
    pub(crate) ctx: Arc<Context>,
    /// Sorted ascending by `idx`.
    pub(crate) terms: Vec<RawTerm>,
}

impl Da {
    /// The zero polynomial (also [`Da::default`]).
    ///
    /// # Panics
    ///
    /// Panics with [`DaceError`] if DACE has not been initialized.
    pub fn new() -> Da {
        Da {
            ctx: Context::current(),
            terms: Vec::new(),
        }
    }

    /// The constant polynomial `c` (`daceCreateConstant`); `|c| <= eps` gives
    /// the zero polynomial.
    pub fn constant(c: f64) -> Da {
        Da::variable_scaled(0, c)
    }

    /// The independent DA variable number `var` (1-based), i.e. the identity
    /// in that variable (`daceCreateVariable`).
    ///
    /// Divergence from C: an out-of-range `var` logs a warning and returns the
    /// zero polynomial instead of raising C error 624.
    pub fn variable(var: u32) -> Da {
        Da::variable_scaled(var, 1.0)
    }

    /// Alias of [`Da::variable`] (`DA::identity` in the C++ interface).
    pub fn identity(var: u32) -> Da {
        Da::variable(var)
    }

    fn variable_scaled(var: u32, ckon: f64) -> Da {
        let ctx = Context::current();
        if var > ctx.nvmax {
            log::warn!("DACE error 624: invalid independent variable {var}; returning zero DA");
            return Da {
                ctx,
                terms: Vec::new(),
            };
        }
        let (eps, _nocut) = crate::context::eps_nocut();
        if ckon.abs() <= eps {
            return Da {
                ctx,
                terms: Vec::new(),
            };
        }
        // Set up the encoded exponents (dacebasic.c:80-99).
        let base = ctx.nomax + 1;
        let (ic1, ic2) = if var == 0 {
            (0, 0)
        } else if var > ctx.nv1 {
            (0, crate::context::npown_i64(base, var - 1 - ctx.nv1))
        } else {
            (crate::context::npown_i64(base, var - 1), 0)
        };
        let idx = ctx.ia1[ic1 as usize] + ctx.ia2[ic2 as usize];
        Da {
            ctx,
            terms: vec![RawTerm { idx, c: ckon }],
        }
    }

    /// The polynomial consisting of the single monomial `c * jj[0]^jj[1] * ...`
    /// (`daceCreateMonomial`).
    ///
    /// `jj` is padded with zeros or truncated to the number of DA variables
    /// (with a warning). Terms whose total order exceeds the maximum
    /// computation order, or with an exponent above it, are dropped with a
    /// warning (divergence from C, which encodes them as the constant term
    /// after raising error 622). `|c| <= eps` gives the zero polynomial.
    pub fn monomial(jj: &[u32], c: f64) -> Da {
        let ctx = Context::current();
        let (eps, _nocut) = crate::context::eps_nocut();
        if c.abs() <= eps {
            return Da {
                ctx,
                terms: Vec::new(),
            };
        }
        let jj = fix_exponent_length(&ctx, jj);
        match ctx.encode(&jj) {
            Some(idx) => Da {
                ctx,
                terms: vec![RawTerm { idx, c }],
            },
            None => {
                log::warn!(
                    "DACE error 622: monomial order too large in Da::monomial; term dropped"
                );
                Da {
                    ctx,
                    terms: Vec::new(),
                }
            }
        }
    }

    /// A DA with randomly filled coefficients (`daceCreateRandom`).
    ///
    /// `cmu` is the filling factor: `|cmu|` is the fraction of non-zero
    /// coefficients; `cmu < 0` draws coefficients in `[-1, 1]`, `cmu > 0`
    /// weights them to decay exponentially with order from 1.0 towards the
    /// machine epsilon.
    ///
    /// Divergence from C: the C library uses libc `rand()` (platform- and
    /// seed-dependent); this implementation uses a deterministic 64-bit LCG
    /// per thread, so results are reproducible everywhere.
    pub fn random(cmu: f64) -> Da {
        let ctx = Context::current();
        let (_eps, nocut) = crate::context::eps_nocut();
        let mut terms = Vec::new();
        for i in 0..ctx.nmmax {
            if ctx.ieo[i as usize] <= nocut && dace_random() < cmu.abs() {
                let c = if cmu < 0.0 {
                    2.0 * dace_random() - 1.0
                } else {
                    let w = ctx
                        .epsmac
                        .powf(f64::from(ctx.ieo[i as usize]) / f64::from(nocut));
                    w * (2.0 * dace_random() - 1.0)
                };
                terms.push(RawTerm { idx: i, c });
            }
        }
        Da { ctx, terms }
    }

    // -----------------------------------------------------------------------
    // Inspection
    // -----------------------------------------------------------------------

    /// The constant part of the polynomial (`daceGetConstant`).
    pub fn cons(&self) -> f64 {
        match self.terms.first() {
            Some(t) if t.idx == 0 => t.c,
            _ => 0.0,
        }
    }

    /// The linear coefficients, one per DA variable (`daceGetLinear`).
    pub fn linear(&self) -> Vec<f64> {
        let mut jj = vec![0u32; self.ctx.nvmax as usize];
        let mut c = vec![0.0; self.ctx.nvmax as usize];
        for (i, ci) in c.iter_mut().enumerate() {
            jj[i] = 1;
            *ci = self.get_coefficient(&jj);
            jj[i] = 0;
        }
        c
    }

    /// The gradient: derivatives with respect to all DA variables
    /// (`DA::gradient` in the C++ interface).
    pub fn gradient(&self) -> Vec<Da> {
        (1..=self.ctx.nvmax).map(|i| self.deriv(i)).collect()
    }

    /// The number of stored (non-zero) monomials (`daceGetLength`).
    pub fn size(&self) -> usize {
        self.terms.len()
    }

    /// The coefficient of the monomial with exponents `jj`
    /// (`daceGetCoefficient`). `jj` is padded/truncated to the number of DA
    /// variables; invalid exponents return 0.0 with a warning.
    pub fn get_coefficient(&self, jj: &[u32]) -> f64 {
        let jj = fix_exponent_length(&self.ctx, jj);
        match self.ctx.encode(&jj) {
            Some(ic) => self.get_coefficient0(ic),
            None => {
                log::warn!(
                    "DACE error 622: monomial order too large in get_coefficient; returning 0.0"
                );
                0.0
            }
        }
    }

    /// The coefficient of the monomial with packed index `ic`.
    pub(crate) fn get_coefficient0(&self, ic: u32) -> f64 {
        match self.terms.binary_search_by_key(&ic, |t| t.idx) {
            Ok(pos) => self.terms[pos].c,
            Err(_) => 0.0,
        }
    }

    /// Set the coefficient of the monomial with exponents `jj`
    /// (`daceSetCoefficient`): sets it, replaces it, or removes the monomial
    /// when `|c| <= eps`, keeping the term list sorted.
    pub fn set_coefficient(&mut self, jj: &[u32], c: f64) {
        let jj = fix_exponent_length(&self.ctx, jj);
        match self.ctx.encode(&jj) {
            Some(ic) => self.set_coefficient0(ic, c),
            None => {
                log::warn!("DACE error 622: monomial order too large in set_coefficient; ignored");
            }
        }
    }

    /// Set the coefficient of the monomial with packed index `ic`.
    pub(crate) fn set_coefficient0(&mut self, ic: u32, c: f64) {
        let (eps, _nocut) = crate::context::eps_nocut();
        match self.terms.binary_search_by_key(&ic, |t| t.idx) {
            Ok(pos) => {
                if crate::kernels::keep(c, eps) {
                    self.terms[pos].c = c;
                } else {
                    self.terms.remove(pos);
                }
            }
            Err(pos) => {
                if crate::kernels::keep(c, eps) {
                    self.terms.insert(pos, RawTerm { idx: ic, c });
                }
            }
        }
    }

    /// The monomial at 1-based position `pos` in the stored term list
    /// (`DA::getMonomial` in the C++ interface); `None` when out of range.
    /// The ordering is implementation-dependent.
    pub fn get_monomial(&self, pos: usize) -> Option<Monomial> {
        self.terms.get(pos.wrapping_sub(1)).map(|t| Monomial {
            jj: self.ctx.decode(t.idx),
            c: t.c,
        })
    }

    /// Iterate over all stored monomials, in stored order.
    pub fn iter_monomials(&self) -> impl Iterator<Item = Monomial> + '_ {
        let ctx = self.ctx.clone();
        self.terms.iter().map(move |t| Monomial {
            jj: ctx.decode(t.idx),
            c: t.c,
        })
    }

    /// Whether any coefficient is NaN (`daceIsNan`).
    pub fn is_nan(&self) -> bool {
        self.terms.iter().any(|t| t.c.is_nan())
    }

    /// Whether any coefficient is infinite (`daceIsInf`).
    pub fn is_inf(&self) -> bool {
        self.terms.iter().any(|t| t.c.is_infinite())
    }

    // -----------------------------------------------------------------------
    // Calculus (dacemath.c:504-649)
    // -----------------------------------------------------------------------

    /// Derivative with respect to independent variable `var` (1-based)
    /// (`daceDifferentiate`). Out-of-range variables warn and return zero.
    pub fn deriv(&self, var: u32) -> Da {
        let ctx = &self.ctx;
        if !(1..=ctx.nvmax).contains(&var) {
            log::warn!(
                "DACE error 624: invalid independent variable {var} in deriv; returning zero DA"
            );
            return Da::new();
        }
        let (_eps, nocut) = crate::context::eps_nocut();
        let ibase = ctx.nomax + 1;
        let j = if var > ctx.nv1 {
            var - 1 - ctx.nv1
        } else {
            var - 1
        };
        let idiv = crate::context::npown_i64(ibase, j);
        let in_second_half = var > ctx.nv1;
        let mut terms = Vec::with_capacity(self.terms.len());
        for t in &self.terms {
            let ic1 = ctx.ie1[t.idx as usize];
            let ic2 = ctx.ie2[t.idx as usize];
            let ipow = if in_second_half {
                (ic2 / idiv) % ibase
            } else {
                (ic1 / idiv) % ibase
            };
            if ipow == 0 || ctx.order_of(t.idx) > nocut + 1 {
                continue;
            }
            let idx = if in_second_half {
                ctx.ia1[ic1 as usize] + ctx.ia2[(ic2 - idiv) as usize]
            } else {
                ctx.ia1[(ic1 - idiv) as usize] + ctx.ia2[ic2 as usize]
            };
            terms.push(RawTerm {
                idx,
                c: t.c * f64::from(ipow),
            });
        }
        Da {
            ctx: ctx.clone(),
            terms,
        }
    }

    /// Repeated derivative with respect to `vars[0]`, then `vars[1]`, ...
    pub fn deriv_vars(&self, vars: &[u32]) -> Da {
        let mut d = self.clone();
        for &v in vars {
            d = d.deriv(v);
        }
        d
    }

    /// Integral with respect to independent variable `var` (1-based)
    /// (`daceIntegrate`); the integration constant is zero. Out-of-range
    /// variables warn and return zero.
    pub fn integ(&self, var: u32) -> Da {
        let ctx = &self.ctx;
        if !(1..=ctx.nvmax).contains(&var) {
            log::warn!(
                "DACE error 624: invalid independent variable {var} in integ; returning zero DA"
            );
            return Da::new();
        }
        let (eps, nocut) = crate::context::eps_nocut();
        let ibase = ctx.nomax + 1;
        let j = if var > ctx.nv1 {
            var - 1 - ctx.nv1
        } else {
            var - 1
        };
        let idiv = crate::context::npown_i64(ibase, j);
        let in_second_half = var > ctx.nv1;
        let mut terms = Vec::with_capacity(self.terms.len());
        for t in &self.terms {
            if ctx.order_of(t.idx) >= nocut {
                continue;
            }
            let ic1 = ctx.ie1[t.idx as usize];
            let ic2 = ctx.ie2[t.idx as usize];
            let ipow = if in_second_half {
                (ic2 / idiv) % ibase
            } else {
                (ic1 / idiv) % ibase
            };
            let ccc = t.c / f64::from(ipow + 1);
            if crate::kernels::keep(ccc, eps) {
                let idx = if in_second_half {
                    ctx.ia1[ic1 as usize] + ctx.ia2[(ic2 + idiv) as usize]
                } else {
                    ctx.ia1[(ic1 + idiv) as usize] + ctx.ia2[ic2 as usize]
                };
                terms.push(RawTerm { idx, c: ccc });
            }
        }
        Da {
            ctx: ctx.clone(),
            terms,
        }
    }

    /// Repeated integral with respect to `vars[0]`, then `vars[1]`, ...
    pub fn integ_vars(&self, vars: &[u32]) -> Da {
        let mut d = self.clone();
        for &v in vars {
            d = d.integ(v);
        }
        d
    }

    /// Keep only terms of order between `min_order` and `max_order`
    /// inclusive (`daceTrim`).
    pub fn trim(&self, min_order: u32, max_order: u32) -> Da {
        let terms = self
            .terms
            .iter()
            .filter(|t| {
                let io = self.ctx.order_of(t.idx);
                io >= min_order && io <= max_order
            })
            .copied()
            .collect();
        Da {
            ctx: self.ctx.clone(),
            terms,
        }
    }

    /// The multiplicative inverse `1/self` (`daceMultiplicativeInverse`):
    /// direct alternating series below truncation order 5, Newton iteration
    /// above.
    ///
    /// # Panics
    ///
    /// Panics with [`DaceError`] code 641 ("Dividing by zero") when the
    /// constant part of `self` is zero.
    pub fn minv(&self) -> Da {
        minv(self)
    }

    /// The square `self * self` (`daceSquare`).
    pub fn sqr(&self) -> Da {
        multiply(self, self)
    }

    pub(crate) fn assert_same_context(a: &Da, b: &Da) {
        if !Arc::ptr_eq(&a.ctx, &b.ctx) {
            std::panic::panic_any(crate::error::DaceError::new(
                codes::NOT_INITIALIZED,
                "mixed DACE contexts (was init() called again?)",
            ));
        }
    }
}

impl Default for Da {
    fn default() -> Da {
        Da::new()
    }
}

/// Pad with zeros or truncate an exponent vector to the context's number of
/// variables, warning on length mismatch.
fn fix_exponent_length(ctx: &Context, jj: &[u32]) -> Vec<u32> {
    let nvar = ctx.nvmax as usize;
    if jj.len() == nvar {
        return jj.to_vec();
    }
    if jj.len() > nvar {
        log::warn!("DACE info: exponent vector longer than the number of variables; truncating");
        jj[..nvar].to_vec()
    } else {
        log::warn!("DACE info: exponent vector shorter than the number of variables; zero-padding");
        let mut v = vec![0u32; nvar];
        v[..jj.len()].copy_from_slice(jj);
        v
    }
}

// ---------------------------------------------------------------------------
// Operator traits
// ---------------------------------------------------------------------------

impl Add for Da {
    type Output = Da;
    fn add(self, rhs: Da) -> Da {
        Da::assert_same_context(&self, &rhs);
        weighted_sum(&self, 1.0, &rhs, 1.0)
    }
}

impl Sub for Da {
    type Output = Da;
    fn sub(self, rhs: Da) -> Da {
        Da::assert_same_context(&self, &rhs);
        weighted_sum(&self, 1.0, &rhs, -1.0)
    }
}

impl Mul for Da {
    type Output = Da;
    fn mul(self, rhs: Da) -> Da {
        Da::assert_same_context(&self, &rhs);
        multiply(&self, &rhs)
    }
}

impl Div for Da {
    type Output = Da;
    /// # Panics
    ///
    /// Panics with [`DaceError`] code 641 when `rhs` has a zero constant
    /// part (division by zero), via the multiplicative inverse.
    fn div(self, rhs: Da) -> Da {
        Da::assert_same_context(&self, &rhs);
        multiply(&self, &rhs.minv())
    }
}

impl Neg for Da {
    type Output = Da;
    fn neg(self) -> Da {
        weighted_sum(&self, -1.0, &self, 0.0)
    }
}

impl Add<f64> for Da {
    type Output = Da;
    fn add(self, rhs: f64) -> Da {
        weighted_sum(&self, 1.0, &Da::constant(rhs), 1.0)
    }
}

impl Sub<f64> for Da {
    type Output = Da;
    fn sub(self, rhs: f64) -> Da {
        weighted_sum(&self, 1.0, &Da::constant(rhs), -1.0)
    }
}

impl Mul<f64> for Da {
    type Output = Da;
    fn mul(self, rhs: f64) -> Da {
        weighted_sum(&self, rhs, &self, 0.0)
    }
}

impl Div<f64> for Da {
    type Output = Da;
    /// # Panics
    ///
    /// Panics with [`DaceError`] code 641 when `rhs == 0.0`.
    fn div(self, rhs: f64) -> Da {
        if rhs == 0.0 {
            dace_panic(codes::DIVIDING_BY_ZERO, "Dividing by zero");
        }
        weighted_sum(&self, 1.0 / rhs, &self, 0.0)
    }
}

impl Add<Da> for f64 {
    type Output = Da;
    fn add(self, rhs: Da) -> Da {
        weighted_sum(&Da::constant(self), 1.0, &rhs, 1.0)
    }
}

impl Sub<Da> for f64 {
    type Output = Da;
    fn sub(self, rhs: Da) -> Da {
        weighted_sum(&Da::constant(self), 1.0, &rhs, -1.0)
    }
}

impl Mul<Da> for f64 {
    type Output = Da;
    fn mul(self, rhs: Da) -> Da {
        weighted_sum(&rhs, self, &rhs, 0.0)
    }
}

impl Div<Da> for f64 {
    type Output = Da;
    fn div(self, rhs: Da) -> Da {
        Da::constant(self) / rhs
    }
}

impl AddAssign for Da {
    fn add_assign(&mut self, rhs: Da) {
        *self = self.clone() + rhs;
    }
}

impl SubAssign for Da {
    fn sub_assign(&mut self, rhs: Da) {
        *self = self.clone() - rhs;
    }
}

impl MulAssign for Da {
    fn mul_assign(&mut self, rhs: Da) {
        *self = self.clone() * rhs;
    }
}

impl DivAssign for Da {
    fn div_assign(&mut self, rhs: Da) {
        *self = self.clone() / rhs;
    }
}

impl AddAssign<f64> for Da {
    fn add_assign(&mut self, rhs: f64) {
        *self = self.clone() + rhs;
    }
}

impl SubAssign<f64> for Da {
    fn sub_assign(&mut self, rhs: f64) {
        *self = self.clone() - rhs;
    }
}

impl MulAssign<f64> for Da {
    fn mul_assign(&mut self, rhs: f64) {
        *self = self.clone() * rhs;
    }
}

impl DivAssign<f64> for Da {
    fn div_assign(&mut self, rhs: f64) {
        *self = self.clone() / rhs;
    }
}

// ---------------------------------------------------------------------------
// Deterministic PRNG for Da::random (documented divergence: C uses libc rand)
// ---------------------------------------------------------------------------

thread_local! {
    static RAND_STATE: Cell<u64> = const { Cell::new(0x9E3779B97F4A7C15) };
}

/// Deterministic pseudo-random number in `[0, 1)` from a per-thread 64-bit LCG.
pub(crate) fn dace_random() -> f64 {
    RAND_STATE.with(|s| {
        let state = s
            .get()
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s.set(state);
        (state >> 11) as f64 / (1u64 << 53) as f64
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::CONTEXT_LOCK;

    #[test]
    fn arithmetic_and_calculus_basics() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(3, 2).unwrap();
        let x = Da::variable(1);
        let y = Da::variable(2);

        // (x+y) + (x-y) == 2x
        let s = (x.clone() + y.clone()) + (x.clone() - y.clone());
        assert_eq!(s.size(), 1);
        assert!((s.get_coefficient(&[1, 0]) - 2.0).abs() == 0.0);
        assert_eq!(s.get_coefficient(&[0, 1]), 0.0);

        // deriv / integ roundtrip
        assert_eq!(x.clone().deriv(1).cons(), 1.0);
        assert_eq!(x.clone().deriv(1).size(), 1);
        let xi = x.clone().integ(1);
        assert!((xi.get_coefficient(&[2, 0]) - 0.5).abs() < 1e-15);
        let xd = xi.deriv(1);
        assert_eq!(xd.size(), 1);
        assert!((xd.get_coefficient(&[1, 0]) - 1.0).abs() < 1e-15);

        // multiplication
        let xx = x.clone() * x.clone();
        assert_eq!(xx.size(), 1);
        assert!((xx.get_coefficient(&[2, 0]) - 1.0).abs() < 1e-15);
        let xy = x.clone() * y.clone();
        assert!((xy.get_coefficient(&[1, 1]) - 1.0).abs() < 1e-15);

        // division by scalar and inverse
        let d = (x.clone() * 2.0) / 2.0;
        assert!((d.get_coefficient(&[1, 0]) - 1.0).abs() < 1e-15);
        let inv = (1.0 + x.clone()).minv();
        // 1/(1+x) = 1 - x + x^2 - x^3 at order 3
        assert!((inv.cons() - 1.0).abs() < 1e-15);
        assert!((inv.get_coefficient(&[1, 0]) + 1.0).abs() < 1e-15);
        assert!((inv.get_coefficient(&[2, 0]) - 1.0).abs() < 1e-15);
        assert!((inv.get_coefficient(&[3, 0]) + 1.0).abs() < 1e-15);

        // division a/b*b ~ a
        let b = 2.0 + x.clone() * y.clone();
        let q = (1.0 + x.clone()) / b.clone();
        let r = q * b;
        for m in r.iter_monomials() {
            let expect = if m.jj == vec![0, 0] || m.jj == vec![1, 0] {
                1.0
            } else {
                0.0
            };
            assert!(
                (m.c - expect).abs() <= 1e-13 * expect.abs().max(1.0),
                "coefficient of {:?} = {}",
                m.jj,
                m.c
            );
        }

        // trim keeps only constant/linear part
        let f = (1.0 + x.clone() + y.clone()) * (x.clone() - y.clone());
        let ft = f.trim(0, 1);
        assert_eq!(ft.size(), 2);
        assert!((ft.get_coefficient(&[1, 0]) - 1.0).abs() < 1e-15);
        assert!((ft.get_coefficient(&[0, 1]) + 1.0).abs() < 1e-15);
        assert_eq!(f.trim(2, 3).size(), 2); // x^2 - y^2

        // constructors
        assert_eq!(Da::constant(5.0).cons(), 5.0);
        assert_eq!(Da::new().size(), 0);
        assert_eq!(Da::default().size(), 0);
        assert_eq!(Da::identity(2).get_coefficient(&[0, 1]), 1.0);
        assert_eq!(Da::monomial(&[2, 1], 3.0).get_coefficient(&[2, 1]), 3.0);
        assert_eq!(Da::variable(3).size(), 0); // out of range -> warn + zero

        // inspection
        let f = 1.0 + 2.0 * x.clone() + 3.0 * y.clone();
        assert_eq!(f.cons(), 1.0);
        assert_eq!(f.linear(), vec![2.0, 3.0]);
        assert_eq!(f.size(), 3);
        let mut g = f.clone();
        g.set_coefficient(&[1, 1], 7.0);
        assert_eq!(g.get_coefficient(&[1, 1]), 7.0);
        g.set_coefficient(&[1, 1], 0.0); // |0| <= eps -> removed
        assert_eq!(g.get_coefficient(&[1, 1]), 0.0);
        assert_eq!(g.size(), 3);
        assert_eq!(f.get_monomial(1).unwrap().jj, vec![0, 0]);
        assert!(f.get_monomial(4).is_none());
        assert_eq!(f.iter_monomials().count(), 3);
        assert!(!f.is_nan());
        assert!(!f.is_inf());
        assert!(Da::constant(f64::NAN).is_nan());
        assert!(Da::constant(f64::INFINITY).is_inf());

        // gradient
        let grad = f.gradient();
        assert_eq!(grad.len(), 2);
        assert_eq!(grad[0].cons(), 2.0);
        assert_eq!(grad[1].cons(), 3.0);
    }

    #[test]
    fn eps_flush_and_operators() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(3, 2).unwrap();

        let old = crate::context::set_epsilon(0.5);
        let z = Da::constant(0.5) + Da::constant(0.25); // both <= eps -> flushed
        assert_eq!(z.size(), 0);
        crate::context::set_epsilon(old);
        assert_eq!(Da::constant(0.5).cons(), 0.5);

        // scalar operators
        let f = Da::variable(1);
        assert!(((2.0 * f.clone()).get_coefficient(&[1, 0]) - 2.0).abs() < 1e-15);
        assert!(((f.clone() + 1.0).cons() - 1.0).abs() < 1e-15);
        assert!(((f.clone() - 1.0).cons() + 1.0).abs() < 1e-15);
        assert!(((1.0 + f.clone()).cons() - 1.0).abs() < 1e-15);
        assert!(((1.0 - f.clone()).cons() - 1.0).abs() < 1e-15);
        assert_eq!((1.0 / (1.0 + f.clone())).cons(), 1.0);

        // assign operators
        let mut a = Da::variable(1);
        a += 1.0;
        a *= 2.0;
        a -= Da::constant(1.0);
        a /= 2.0;
        assert_eq!(a.cons(), 0.5);
        assert!((a.get_coefficient(&[1, 0]) - 1.0).abs() < 1e-15);

        // neg
        assert_eq!((-f.clone()).get_coefficient(&[1, 0]), -1.0);
    }
}
