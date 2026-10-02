//! Elementary-function composition engine: series evaluation and the
//! multiplicative inverse.
//!
//! Ports `daceEvaluateSeries` (dacemath.c:2003-2026) and
//! `daceMultiplicativeInverse` (dacemath.c:859-918). The full set of
//! elementary functions is layered on top of [`evaluate_series`] in later
//! phases.

use crate::context::{set_truncation_order, truncation_order};
use crate::da::Da;
use crate::error::{codes, dace_panic};
use crate::kernels::{multiply, weighted_sum};

/// Evaluate the polynomial with coefficients `xf` (length `nocut + 1`) on the
/// non-constant part of `a`, by Horner's rule over orders
/// (`daceEvaluateSeries`).
///
/// Exactly as in C, the truncation order is temporarily lowered per Horner
/// step (`nocut = 1` first, then `nocut - i`) and restored at the end.
pub(crate) fn evaluate_series(a: &Da, xf: &[f64]) -> Da {
    let nocut = truncation_order();
    assert!(
        xf.len() > nocut as usize,
        "series coefficient table too short"
    );

    // Non-constant part of a.
    let mut inon = a.clone();
    inon.set_coefficient0(0, 0.0);

    set_truncation_order(1);
    let mut inc = weighted_sum(&inon, xf[nocut as usize], &inon, 0.0);
    inc = add_double(&inc, xf[(nocut - 1) as usize]);

    let mut i = nocut as i64 - 2;
    while i >= 0 {
        set_truncation_order(nocut - i as u32);
        inc = multiply(&inon, &inc);
        inc = add_double(&inc, xf[i as usize]);
        i -= 1;
    }

    set_truncation_order(nocut);
    inc
}

/// Add a constant to a DA (`daceAddDouble`): copy, then set the constant
/// coefficient to `cons + c` (flushing per epsilon as usual).
pub(crate) fn add_double(a: &Da, c: f64) -> Da {
    let mut r = a.clone();
    let cons = r.cons();
    r.set_coefficient0(0, cons + c);
    r
}

/// Multiplicative inverse `1/a` (`daceMultiplicativeInverse`).
///
/// # Panics
///
/// Panics with [`DaceError`] code 641 when the constant part of `a` is zero.
pub(crate) fn minv(a: &Da) -> Da {
    let a0 = a.cons();
    if a0 == 0.0 {
        dace_panic(codes::DIVIDING_BY_ZERO, "Dividing by zero");
    }

    let nocut = truncation_order();
    if nocut < 5 {
        // Lower orders: compute the series directly.
        minv0(a, a0)
    } else {
        // Higher orders: Newton iteration.
        set_truncation_order(2);
        let mut inc = minv0(a, a0);
        let mut ord: u32 = 3;
        while ord <= nocut {
            set_truncation_order(nocut.min(2 * ord - 1));
            let temp = multiply(a, &inc);
            // temp = 2.0 - temp (daceDoubleSubtract)
            let temp = weighted_sum(&temp, -1.0, &temp, 0.0);
            let temp = add_double(&temp, 2.0);
            inc = multiply(&inc, &temp);
            ord *= 2;
        }
        set_truncation_order(nocut);
        inc
    }
}

/// Series-expansion multiplicative inverse (`daceMultiplicativeInverse0`),
/// for orders below 5 (or as the Newton seed).
fn minv0(a: &Da, a0: f64) -> Da {
    let scaled = weighted_sum(a, 1.0 / a0, a, 0.0);
    let nocut = truncation_order();
    let mut xf = vec![0.0; nocut as usize + 1];
    xf[0] = 1.0 / a0;
    for i in 1..xf.len() {
        xf[i] = -xf[i - 1];
    }
    evaluate_series(&scaled, &xf)
}
