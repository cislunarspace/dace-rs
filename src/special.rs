//! Special functions of DA values: Bessel functions, gamma, and polygamma.
//!
//! Ports the special-function layer of `core/dacemath.c` (lines 1523-1995):
//! the Bessel coefficient tables are built from scalar Bessel values of
//! orders `n - nocut .. n + nocut` at the constant part (integer orders,
//! using `J_{-k} = (-1)^k J_k`, `Y_{-k} = (-1)^k Y_k`, `I_{-k} = I_k`,
//! `K_{-k} = K_k`), then composed with the Kahan-summation derivative
//! recurrences of DLMF 10.6. Scalar Bessel values come from
//! [`puruspe`] (`besseljy`/`besselik`), the gamma function from
//! the `puruspe::gamma` function, and the polygamma/Hurwitz zeta from a
//! transcription of DACE's `contrib/psi.c` (netlib PSIFN, f2c translation)
//! and `contrib/zeta.c` (Cephes 2.8) in the `netlib_psi_zeta` module.

use crate::context::truncation_order;
use crate::da::Da;
use crate::elementary::{evaluate_series, exp};
use crate::error::{codes, dace_panic};

pub(crate) mod netlib_psi_zeta;

// ---------------------------------------------------------------------------
// Scalar providers
// ---------------------------------------------------------------------------

/// Values of `J_k(x)` (kind `J`) or `Y_k(x)` (kind `Y`) for integer orders
/// `n0..=n1`, with the negative-order reflections (`BesselWrapper`,
/// dacemath.c:1523). Returns `None` on evaluation failure.
fn bessel_jy_orders(x: f64, n0: i32, n1: i32, bessel_y: bool) -> Option<Vec<f64>> {
    let max_order = n0.unsigned_abs().max(n1.unsigned_abs());
    let mut b = Vec::with_capacity(max_order as usize + 1);
    for k in 0..=max_order {
        // Integer-order dedicated routines; note puruspe::besseljy returns a
        // wrong Y at integer nu (verified against libm), so it is not used.
        b.push(if bessel_y {
            puruspe::Yn(k, x)
        } else {
            puruspe::Jn(k, x)
        });
    }
    let mut out = Vec::with_capacity((n1 - n0 + 1) as usize);
    let mut s = if n0 % 2 == 0 { 1.0 } else { -1.0 };
    for i in n0..=n1 {
        if i >= 0 {
            out.push(b[i as usize]);
        } else {
            out.push(s * b[i.unsigned_abs() as usize]);
            s *= -1.0;
        }
    }
    Some(out)
}

/// Values of `I_k(x)` (unscaled), `e^{-x}·I_k(x)` (scaled I), `K_k(x)`
/// (unscaled), or `e^{x}·K_k(x)` (scaled K) for integer orders `n0..=n1`
/// (`ModifiedBesselWrapper`, dacemath.c:1578). Returns `None` on failure.
///
/// Note: the scaled variants are computed from the unscaled values, so the
/// scaled K function loses accuracy once the unscaled `K` underflows
/// (arguments beyond roughly 690); the C library computes it scaled
/// internally. Divergence documented at [`bessel_k`].
fn bessel_ik_orders(x: f64, n0: i32, n1: i32, bessel_k: bool, scaled: bool) -> Option<Vec<f64>> {
    let max_order = n0.unsigned_abs().max(n1.unsigned_abs());
    let mut b = Vec::with_capacity(max_order as usize + 1);
    for k in 0..=max_order {
        if bessel_k {
            b.push(if scaled {
                puruspe::Kn(k, x) * x.exp()
            } else {
                puruspe::Kn(k, x)
            });
        } else {
            b.push(if scaled {
                puruspe::In(k, x) * (-x).exp()
            } else {
                puruspe::In(k, x)
            });
        }
    }
    // I_{-k} = I_k and K_{-k} = K_k for all k.
    Some((n0..=n1).map(|i| b[i.unsigned_abs() as usize]).collect())
}

// ---------------------------------------------------------------------------
// Bessel composition (dacemath.c:1776-1875)
// ---------------------------------------------------------------------------

/// Compose a Bessel function from its order-`n±nocut` values `bz`
/// (`daceEvaluateBesselFunction`): `type` is -1 for ordinary (J/Y) and +1
/// for modified (I) functions, `ktype` is -1 for K and +1 otherwise.
fn evaluate_bessel_function(a: &Da, bz: &[f64], kind: f64, kkind: f64) -> Da {
    let nocut = truncation_order();
    let mut xf = vec![0.0; nocut as usize + 1];
    let mut binomial = vec![0.0; nocut as usize + 1];

    xf[0] = bz[nocut as usize];
    binomial[0] = 1.0;
    let mut factor = 1.0;
    for i in 1..=nocut as usize {
        factor *= kkind * 0.5 / i as f64;
        // binomial coefficients i choose j from i-1 choose j
        binomial[i] = 1.0;
        for j in (1..i).rev() {
            binomial[j] += binomial[j - 1];
        }
        // n-th derivative of the Bessel function (DLMF 10.6), Kahan-summed.
        let mut sign = 1.0;
        let mut c = 0.0;
        xf[i] = 0.0;
        for j in 0..=i {
            let y = binomial[j] * sign * bz[nocut as usize - i + 2 * j] - c;
            let t = xf[i] + y;
            c = (t - xf[i]) - y;
            xf[i] = t;
            sign *= kind;
        }
        xf[i] *= factor;
    }

    evaluate_series(a, &xf)
}

/// Compose a scaled modified Bessel function from its order-`n±nocut`
/// values `bz` (`daceEvaluateScaledModifiedBesselFunction`); `kkind` is +1
/// for scaled I and -1 for scaled K.
fn evaluate_scaled_modified_bessel_function(a: &Da, bz: &[f64], kkind: f64) -> Da {
    let nocut = truncation_order();
    let mut xf = vec![0.0; nocut as usize + 1];
    let mut binomial = vec![0.0; 2 * nocut as usize + 1];

    xf[0] = bz[nocut as usize];
    binomial[0] = 1.0;
    let mut factor = 1.0;
    for i in 1..=nocut as usize {
        factor *= kkind * 0.5 / i as f64;
        // binomial coefficients 2i-1 choose j, then 2i choose j
        binomial[2 * i - 1] = 1.0;
        for j in (1..2 * i - 1).rev() {
            binomial[j] += binomial[j - 1];
        }
        binomial[2 * i] = 1.0;
        for j in (1..2 * i).rev() {
            binomial[j] += binomial[j - 1];
        }
        let mut sign = 1.0;
        let mut c = 0.0;
        xf[i] = 0.0;
        for j in 0..=2 * i {
            let y = binomial[j] * sign * bz[nocut as usize - i + j] - c;
            let t = xf[i] + y;
            c = (t - xf[i]) - y;
            xf[i] = t;
            sign *= -1.0;
        }
        xf[i] *= factor;
    }

    evaluate_series(a, &xf)
}

// ---------------------------------------------------------------------------
// Public Bessel functions of DA values (dacemath.c:1619-1766)
// ---------------------------------------------------------------------------

/// The Bessel function of the first kind `J_n(a)` (`daceBesselJFunction`).
pub fn bessel_j(a: &Da, n: i32) -> Da {
    bessel_common(a, n, false, "bessel_j")
}

/// The Bessel function of the second kind `Y_n(a)`
/// (`daceBesselYFunction`).
pub fn bessel_y(a: &Da, n: i32) -> Da {
    bessel_common(a, n, false, "bessel_y")
}

/// The modified Bessel function of the first kind `I_n(a)`
/// (`daceBesselIFunction`); with `scaled`, computes `e^{-a}·I_n(a)`.
pub fn bessel_i(a: &Da, n: i32, scaled: bool) -> Da {
    bessel_common(a, n, scaled, "bessel_i")
}

/// The modified Bessel function of the second kind `K_n(a)`
/// (`daceBesselKFunction`); with `scaled`, computes `e^{a}·K_n(a)`.
///
/// Divergence from C: the scaled variant is derived from the unscaled one,
/// so it loses accuracy (and eventually returns 0/∞ products) once the
/// unscaled `K` underflows, around arguments of 690; the C library computes
/// the scaled function directly and remains accurate there.
pub fn bessel_k(a: &Da, n: i32, scaled: bool) -> Da {
    bessel_common(a, n, scaled, "bessel_k")
}

fn bessel_common(a: &Da, n: i32, scaled: bool, kind: &str) -> Da {
    let a0 = a.cons();
    if a0 <= 0.0 {
        dace_panic(codes::OUT_OF_DOMAIN, "Out of domain");
    }
    let nocut = truncation_order() as i32;
    let n0 = n - nocut;
    let n1 = n + nocut;
    let bz = match kind {
        "bessel_i" => bessel_ik_orders(a0, n0, n1, false, scaled),
        "bessel_k" => bessel_ik_orders(a0, n0, n1, true, scaled),
        "bessel_j" => bessel_jy_orders(a0, n0, n1, false),
        _ => bessel_jy_orders(a0, n0, n1, true),
    };
    match bz {
        Some(bz) => match kind {
            "bessel_j" | "bessel_y" => evaluate_bessel_function(a, &bz, -1.0, 1.0),
            "bessel_k" if scaled => evaluate_scaled_modified_bessel_function(a, &bz, -1.0),
            "bessel_k" => evaluate_bessel_function(a, &bz, 1.0, -1.0),
            _ if scaled => evaluate_scaled_modified_bessel_function(a, &bz, 1.0),
            _ => evaluate_bessel_function(a, &bz, 1.0, 1.0),
        },
        None => dace_panic(codes::OUT_OF_DOMAIN, "Out of domain"),
    }
}

// ---------------------------------------------------------------------------
// Gamma and polygamma (dacemath.c:1884-1995)
// ---------------------------------------------------------------------------

/// Partial logarithmic gamma: the series without the constant term
/// (`daceLogGammaFunction0`).
fn log_gamma0(a: &Da, a0: f64) -> Da {
    let nocut = truncation_order();
    let mut xf = vec![0.0; nocut as usize + 1];
    xf[0] = 0.0;
    xf[1] = netlib_psi_zeta::psi(a0);
    let mut s = 1.0;
    for (i, item) in xf.iter_mut().enumerate().skip(2) {
        *item = (s / i as f64) * netlib_psi_zeta::zeta(i as f64, a0);
        s *= -1.0;
    }
    evaluate_series(a, &xf)
}

/// The logarithmic gamma function `ln Γ(a)` (`daceLogGammaFunction`).
///
/// # Panics
///
/// Panics with [`crate::DaceError`] code 650 ("Out of domain") when the constant
/// part is zero or a negative integer.
pub fn log_gamma(a: &Da) -> Da {
    let a0 = a.cons();
    if a0 <= 0.0 && a0.trunc() == a0 {
        dace_panic(codes::OUT_OF_DOMAIN, "Out of domain");
    }
    let mut r = log_gamma0(a, a0);
    let c = puruspe::gamma(a0).ln();
    r.set_coefficient0(0, c);
    r
}

/// The gamma function `Γ(a)` (`daceGammaFunction`).
///
/// # Panics
///
/// Panics with [`crate::DaceError`] code 650 ("Out of domain") when the constant
/// part is zero or a negative integer.
pub fn gamma(a: &Da) -> Da {
    let a0 = a.cons();
    if a0 <= 0.0 && a0.trunc() == a0 {
        dace_panic(codes::OUT_OF_DOMAIN, "Out of domain");
    }
    let lg = log_gamma0(a, a0);
    let mut r = exp(&lg);
    // multiply by gamma(a0) via a scalar multiply of every term
    let g = puruspe::gamma(a0);
    r = crate::kernels::weighted_sum(&r, g, &r, 0.0);
    r
}

/// The polygamma function of order `n`: the `(n+1)`-th derivative of
/// `ln Γ` (`dacePsiFunction`).
///
/// # Panics
///
/// Panics with [`crate::DaceError`] code 650 ("Out of domain") when the constant
/// part is zero or a negative integer.
pub fn psi(a: &Da, n: u32) -> Da {
    let a0 = a.cons();
    if a0 <= 0.0 && a0.trunc() == a0 {
        dace_panic(codes::OUT_OF_DOMAIN, "Out of domain");
    }

    let nocut = truncation_order();
    let mut xf = vec![0.0; nocut as usize + 1];

    if n == 0 {
        xf[0] = netlib_psi_zeta::psi(a0);
        let mut s = 1.0;
        for (i, item) in xf.iter_mut().enumerate().skip(1) {
            *item = s * netlib_psi_zeta::zeta(f64::from(i as u32 + 1), a0);
            s *= -1.0;
        }
    } else {
        // fac = (-1)^(n+1) * n!  (C: sign then 2..=n in order)
        let mut fac = if n % 2 != 0 { 1.0 } else { -1.0 };
        for i in 2..=n {
            fac *= i as f64;
        }
        for (i, item) in xf.iter_mut().enumerate() {
            *item = fac * netlib_psi_zeta::zeta(f64::from(n + i as u32 + 1), a0);
            fac = -(fac / (i as f64 + 1.0)) * f64::from(n + i as u32 + 1);
        }
    }

    evaluate_series(a, &xf)
}

impl Da {
    /// Method form of [`bessel_j`](crate::special::bessel_j).
    pub fn bessel_j(&self, n: i32) -> Da {
        bessel_j(self, n)
    }

    /// Method form of [`bessel_y`](crate::special::bessel_y).
    pub fn bessel_y(&self, n: i32) -> Da {
        bessel_y(self, n)
    }

    /// Method form of [`bessel_i`](crate::special::bessel_i).
    pub fn bessel_i(&self, n: i32, scaled: bool) -> Da {
        bessel_i(self, n, scaled)
    }

    /// Method form of [`bessel_k`](crate::special::bessel_k).
    pub fn bessel_k(&self, n: i32, scaled: bool) -> Da {
        bessel_k(self, n, scaled)
    }

    /// Method form of [`log_gamma`](crate::special::log_gamma).
    pub fn log_gamma(&self) -> Da {
        log_gamma(self)
    }

    /// Method form of [`gamma`](crate::special::gamma).
    pub fn gamma(&self) -> Da {
        gamma(self)
    }

    /// Method form of [`psi`](crate::special::psi).
    pub fn psi(&self, n: u32) -> Da {
        psi(self, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::CONTEXT_LOCK;

    #[test]
    fn netlib_psi_zeta_values() {
        // psi(1) = -gamma
        assert!((netlib_psi_zeta::psi(1.0) + 0.5772156649015329).abs() < 1e-13);
        // psi'(1) = zeta(2,1) = pi^2/6
        assert!(
            (netlib_psi_zeta::zeta(2.0, 1.0) - std::f64::consts::PI.powi(2) / 6.0).abs() < 1e-12
        );
        // psi(1/2) = -gamma - 2 ln 2
        assert!((netlib_psi_zeta::psi(0.5) + 0.5772156649015329 + 2.0 * 2.0f64.ln()).abs() < 1e-12);
        // zeta(4,1) = pi^4/90
        assert!(
            (netlib_psi_zeta::zeta(4.0, 1.0) - std::f64::consts::PI.powi(4) / 90.0).abs() < 1e-10
        );
    }

    #[test]
    fn gamma_and_bessel_constants() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(6, 2).unwrap();

        // Gamma(1/2) = sqrt(pi): constant coefficient matches puruspe
        let g = gamma(&(0.5 + Da::variable(1)));
        assert!((g.cons() - std::f64::consts::PI.sqrt()).abs() < 1e-13);

        // log_gamma constant matches ln of puruspe gamma
        let lg = log_gamma(&(0.5 + Da::variable(1)));
        assert!((lg.cons() - puruspe::gamma(0.5).ln()).abs() < 1e-14);

        // bessel_j(0) of constant 1 matches scalar J0(1) in the constant term
        let j = bessel_j(&(1.0 + Da::variable(1)), 0);
        assert!((j.cons() - puruspe::besseljy(0.0, 1.0).0).abs() < 1e-13);

        // J0'(x) = -J1(x): linear coefficient of bessel_j(x, 0) at a0=1
        let lin = j.get_coefficient(&[1, 0]);
        assert!((lin + puruspe::besseljy(1.0, 1.0).0).abs() < 1e-12);

        // psi(0) of a DA == netlib psi at the constant; linear term = zeta(2,a0)
        let p0 = psi(&(1.0 + Da::variable(1)), 0);
        assert!((p0.cons() - netlib_psi_zeta::psi(1.0)).abs() < 1e-14);
        assert!((p0.get_coefficient(&[1, 0]) - netlib_psi_zeta::zeta(2.0, 1.0)).abs() < 1e-12);

        // psi(1) of a DA: xf[i] = fac*zeta(n+i+1)
        let p1 = psi(&(1.0 + Da::variable(1)), 1);
        assert!((p1.cons() - netlib_psi_zeta::zeta(2.0, 1.0)).abs() < 1e-12);
        assert!(
            (p1.get_coefficient(&[1, 0]) + 2.0 * netlib_psi_zeta::zeta(3.0, 1.0)).abs() < 1e-11
        );

        // bessel_k scaled / unscaled relation at the constant part
        let kx = 2.0 + Da::variable(1);
        let ku = bessel_k(&kx, 1, false);
        let ks = bessel_k(&kx, 1, true);
        assert!((ks.cons() - ku.cons() * 2.0f64.exp()).abs() < 1e-6 * ks.cons().abs());

        // bessel_i scaled / unscaled
        let iu = bessel_i(&kx, 1, false);
        let is_ = bessel_i(&kx, 1, true);
        assert!((is_.cons() - iu.cons() * (-2.0f64).exp()).abs() < 1e-13);
    }

    #[test]
    fn special_domain_panics() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(6, 2).unwrap();

        fn expect_code(f: impl FnOnce() -> Da + std::panic::UnwindSafe, code: u32) {
            let err = std::panic::catch_unwind(f).expect_err("must panic");
            let e = err
                .downcast_ref::<crate::error::DaceError>()
                .expect("DaceError payload");
            assert_eq!(e.code, code, "{}", e);
        }

        expect_code(|| gamma(&Da::constant(0.0)), codes::OUT_OF_DOMAIN);
        expect_code(|| gamma(&Da::constant(-3.0)), codes::OUT_OF_DOMAIN);
        expect_code(|| log_gamma(&Da::constant(-2.0)), codes::OUT_OF_DOMAIN);
        expect_code(|| psi(&Da::constant(-1.0), 2), codes::OUT_OF_DOMAIN);
        expect_code(|| bessel_j(&Da::constant(0.0), 1), codes::OUT_OF_DOMAIN);
        expect_code(|| bessel_y(&Da::constant(-1.0), 1), codes::OUT_OF_DOMAIN);
        expect_code(
            || bessel_i(&Da::constant(-1.0), 1, false),
            codes::OUT_OF_DOMAIN,
        );
        expect_code(
            || bessel_k(&Da::constant(0.0), 1, true),
            codes::OUT_OF_DOMAIN,
        );
    }
}
