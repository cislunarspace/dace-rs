//! Elementary functions of DA values.
//!
//! Ports the intrinsic-function layer of `core/dacemath.c` (lines 656-1504):
//! each function computes the divided derivatives of the scalar function at
//! the constant part (`xf[k]`, the Taylor coefficients in the non-constant
//! part) and composes them via the Horner engine `evaluate_series`.
//! Domain violations panic with the numeric code and message of the C error
//! table (e.g. 647 "Negative constant part in logarithm").

use crate::context::truncation_order;
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

    crate::context::set_truncation_order(1);
    let mut inc = weighted_sum(&inon, xf[nocut as usize], &inon, 0.0);
    inc = add_double(&inc, xf[(nocut - 1) as usize]);

    let mut i = nocut as i64 - 2;
    while i >= 0 {
        crate::context::set_truncation_order(nocut - i as u32);
        inc = multiply(&inon, &inc);
        inc = add_double(&inc, xf[i as usize]);
        i -= 1;
    }

    crate::context::set_truncation_order(nocut);
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

/// `c - a` (`daceDoubleSubtract`): negate, then add the constant `c`.
pub(crate) fn double_subtract(a: &Da, c: f64) -> Da {
    let mut r = weighted_sum(a, -1.0, a, 0.0);
    let cons = r.cons();
    r.set_coefficient0(0, cons + c);
    r
}

/// Subtract a constant from a DA (`daceSubtractDouble`).
pub(crate) fn subtract_double(a: &Da, c: f64) -> Da {
    add_double(a, -c)
}

/// Divide a DA by a scalar (`daceDivideDouble`).
///
/// # Panics
///
/// Panics with [`crate::DaceError`] code 641 when `c == 0.0`.
pub(crate) fn divide_double(a: &Da, c: f64) -> Da {
    if c == 0.0 {
        dace_panic(codes::DIVIDING_BY_ZERO, "Dividing by zero");
    }
    weighted_sum(a, 1.0 / c, a, 0.0)
}

/// Multiplicative inverse `1/a` (`daceMultiplicativeInverse`).
///
/// # Panics
///
/// Panics with [`crate::DaceError`] code 641 when the constant part of `a` is zero.
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
        crate::context::set_truncation_order(2);
        let mut inc = minv0(a, a0);
        let mut ord: u32 = 3;
        while ord <= nocut {
            crate::context::set_truncation_order(nocut.min(2 * ord - 1));
            let temp = multiply(a, &inc);
            let temp = double_subtract(&temp, 2.0);
            inc = multiply(&inc, &temp);
            ord *= 2;
        }
        crate::context::set_truncation_order(nocut);
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

// ---------------------------------------------------------------------------
// Constant-term-only operations (dacemath.c:661-688)
// ---------------------------------------------------------------------------

/// Truncate the constant part to an integer (round half to even, as C
/// `rint`), keeping higher-order terms (`daceTruncate`).
pub fn trunc(a: &Da) -> Da {
    let mut r = a.clone();
    let c = r.cons().round_ties_even();
    r.set_coefficient0(0, c);
    r
}

/// Round the constant part to an integer (half away from zero, as C `round`),
/// keeping higher-order terms (`daceRound`).
pub fn round(a: &Da) -> Da {
    let mut r = a.clone();
    let c = r.cons().round();
    r.set_coefficient0(0, c);
    r
}

/// Modulo of the constant part by `p`, keeping higher-order terms
/// (`daceModulo`).
pub fn modulo(a: &Da, p: f64) -> Da {
    let mut r = a.clone();
    let c = r.cons() % p;
    r.set_coefficient0(0, c);
    r
}

// ---------------------------------------------------------------------------
// Powers and roots (dacemath.c:690-918)
// ---------------------------------------------------------------------------
/// Raise `a` to the real power `p` (`dacePowerDouble`): integer powers go
/// through [`powi`], otherwise a series in `(a - a0)/a0`.
///
/// # Panics
///
/// Panics with [`crate::DaceError`] code 643 for a non-integer power of a DA with
/// non-positive constant part.
pub fn powf(a: &Da, p: f64) -> Da {
    if p == 0.0 {
        return Da::constant(1.0);
    }
    if p.fract() == 0.0 && p.abs() <= i32::MAX as f64 {
        return powi(a, p as i32);
    }

    let a0 = a.cons();
    if a0 <= 0.0 {
        dace_panic(
            codes::NON_INTEGER_POWER_NON_POSITIVE,
            "Non-integer power of non-positive DA",
        );
    }

    let nocut = truncation_order();
    let mut xf = vec![0.0; nocut as usize + 1];
    xf[0] = a0.powf(p);
    for i in 1..xf.len() {
        xf[i] = xf[i - 1] / i as f64 * (p - (i - 1) as f64);
    }

    let scaled = divide_double(a, a0);
    evaluate_series(&scaled, &xf)
}

/// Raise `a` to the integer power `np` (`dacePower`), by squaring for
/// `|np| > 4` and hard-coded small cases, with the inverse for negative
/// powers.
///
/// Divergence from C: the C implementation computes negative powers by
/// calling `daceMultiplicativeInverse(inc, inc)` on the aliased result, and
/// that Newton iteration is not aliasing safe despite its documentation
/// note - C returns wrong coefficients for negative powers of non-constant
/// DAs (verified: C's `pow(A, -2)` disagrees with C's own `minv(sqr(A))`).
/// This implementation returns the correct value.
pub fn powi(a: &Da, np: i32) -> Da {
    match np {
        0 => Da::constant(1.0),
        1 => a.clone(),
        -1 => minv(a),
        _ => {
            let abs_np = np.unsigned_abs();
            let mut result = match abs_np {
                2 => a.sqr(),
                3 => multiply(a, &a.sqr()),
                4 => a.sqr().sqr(),
                _ => {
                    // Binary exponentiation (dacePower default branch).
                    let mut itemp = a.clone();
                    let mut inc = Da::constant(1.0);
                    let mut inp = abs_np;
                    while inp > 0 {
                        if inp & 1 != 0 {
                            inc = multiply(&inc, &itemp);
                        }
                        inp >>= 1;
                        if inp > 0 {
                            itemp = itemp.sqr();
                        }
                    }
                    inc
                }
            };
            if np < 0 {
                result = minv(&result);
            }
            result
        }
    }
}

/// Take the `np`-th root of `a` (`daceRoot`).
///
/// # Panics
///
/// Panics with [`crate::DaceError`] code 644 for `np == 0`, 645 for an even root of
/// a DA with non-positive constant part, 646 for an odd root of a zero DA.
pub fn root(a: &Da, np: i32) -> Da {
    if np == 0 {
        dace_panic(codes::ZERO_TH_ROOT, "Zero-th root does not exist");
    }

    let a0 = a.cons();
    let iodd = np.unsigned_abs() & 1;
    if iodd == 0 && a0 <= 0.0 {
        dace_panic(codes::EVEN_ROOT_NEGATIVE, "Even root of negative DA");
    } else if iodd == 1 && a0 == 0.0 {
        dace_panic(codes::ODD_ROOT_ZERO, "Odd root of zero DA");
    }

    let nocut = truncation_order();
    let mut xf = vec![0.0; nocut as usize + 1];
    let mut cr = 1.0 / f64::from(np);
    xf[0] = a0.abs().powf(cr).copysign(a0);
    for i in 1..xf.len() {
        xf[i] = xf[i - 1] / i as f64 * cr;
        cr -= 1.0;
    }

    let scaled = divide_double(a, a0);
    evaluate_series(&scaled, &xf)
}

/// The square root (`daceSquareRoot`, i.e. `root(a, 2)`).
///
/// # Panics
///
/// Panics with [`crate::DaceError`] code 645 when the constant part is negative.
pub fn sqrt(a: &Da) -> Da {
    root(a, 2)
}

/// The inverse square root `1/sqrt(a)` (`daceInverseSquareRoot`).
pub fn isrt(a: &Da) -> Da {
    root(a, -2)
}

/// The cubic root (`daceCubicRoot`).
pub fn cbrt(a: &Da) -> Da {
    root(a, 3)
}

/// The inverse cubic root (`daceInverseCubicRoot`).
pub fn icrt(a: &Da) -> Da {
    root(a, -3)
}

/// The hypotenuse `sqrt(a² + b²)` (`daceHypotenuse`).
pub fn hypot(a: &Da, b: &Da) -> Da {
    Da::assert_same_context(a, b);
    root(&(a.sqr() + b.sqr()), 2)
}

// ---------------------------------------------------------------------------
// Exponentials and logarithms (dacemath.c:985-1075)
// ---------------------------------------------------------------------------

/// The exponential (`daceExponential`).
pub fn exp(a: &Da) -> Da {
    let nocut = truncation_order();
    let mut xf = vec![0.0; nocut as usize + 1];
    xf[0] = a.cons().exp();
    for i in 1..xf.len() {
        xf[i] = xf[i - 1] / i as f64;
    }
    evaluate_series(a, &xf)
}

/// The natural logarithm (`daceLogarithm`), as a series in `(a - a0)/a0`.
///
/// # Panics
///
/// Panics with [`crate::DaceError`] code 647 when the constant part is not positive.
pub fn log(a: &Da) -> Da {
    let a0 = a.cons();
    if a0 <= 0.0 {
        dace_panic(
            codes::LOG_NON_POSITIVE,
            "Negative constant part in logarithm",
        );
    }

    let nocut = truncation_order();
    let mut xf = vec![0.0; nocut as usize + 1];
    let scaled = divide_double(a, a0);
    xf[0] = a0.ln();
    xf[1] = 1.0;
    for i in 2..xf.len() {
        xf[i] = -xf[i - 1] / i as f64 * (i - 1) as f64;
    }
    evaluate_series(&scaled, &xf)
}

/// The logarithm in base `b` (`daceLogarithmBase`).
///
/// # Panics
///
/// Panics with [`crate::DaceError`] code 648 when `b` is not positive.
pub fn log_base(a: &Da, b: f64) -> Da {
    if b <= 0.0 {
        dace_panic(
            codes::LOG_BASE_POSITIVE,
            "Base of logarithm must be positive",
        );
    }
    let l = log(a);
    weighted_sum(&l, 1.0 / b.ln(), &l, 0.0)
}

/// The decadic logarithm (`daceLogarithm10`).
pub fn log10(a: &Da) -> Da {
    log_base(a, 10.0)
}

/// The binary logarithm (`daceLogarithm2`).
pub fn log2(a: &Da) -> Da {
    log_base(a, 2.0)
}

// ---------------------------------------------------------------------------
// Trigonometry (dacemath.c:1082-1280)
// ---------------------------------------------------------------------------

/// The sine (`daceSine`).
pub fn sin(a: &Da) -> Da {
    let nocut = truncation_order();
    let a0 = a.cons();
    let mut xf = vec![0.0; nocut as usize + 1];
    xf[0] = a0.sin();
    xf[1] = a0.cos();
    for i in 2..xf.len() {
        xf[i] = -xf[i - 2] / (i * (i - 1)) as f64;
    }
    evaluate_series(a, &xf)
}

/// The cosine (`daceCosine`).
pub fn cos(a: &Da) -> Da {
    let nocut = truncation_order();
    let a0 = a.cons();
    let mut xf = vec![0.0; nocut as usize + 1];
    xf[0] = a0.cos();
    xf[1] = -a0.sin();
    for i in 2..xf.len() {
        xf[i] = -xf[i - 2] / (i * (i - 1)) as f64;
    }
    evaluate_series(a, &xf)
}

/// The tangent (`daceTangent`): `sin(a)/cos(a)`.
///
/// # Panics
///
/// Panics with [`crate::DaceError`] code 649 when the cosine of the constant part
/// is zero.
pub fn tan(a: &Da) -> Da {
    if a.cons().cos() == 0.0 {
        dace_panic(codes::COS_ZERO_IN_TANGENT, "Cosine is zero in tangent");
    }
    let s = sin(a);
    let c = cos(a);
    divide_da(&s, &c)
}

/// The arcsine (`daceArcSine`): `atan(a / sqrt(1 - a²))`.
///
/// # Panics
///
/// Panics with [`crate::DaceError`] code 650 when `|constant part| >= 1`.
pub fn asin(a: &Da) -> Da {
    if a.cons().abs() >= 1.0 {
        dace_panic(codes::OUT_OF_DOMAIN, "Out of domain");
    }
    let d = double_subtract(&a.sqr(), 1.0);
    let d = sqrt(&d);
    let q = divide_da(a, &d);
    atan(&q)
}

/// The arccosine (`daceArcCosine`): `π/2 - asin(a)`.
///
/// # Panics
///
/// Panics with [`crate::DaceError`] code 650 when `|constant part| >= 1`.
pub fn acos(a: &Da) -> Da {
    if a.cons().abs() >= 1.0 {
        dace_panic(codes::OUT_OF_DOMAIN, "Out of domain");
    }
    let s = asin(a);
    double_subtract(&s, std::f64::consts::FRAC_PI_2)
}

/// The arctangent (`daceArcTangent`): integrates `1/(1+x²)` at the constant
/// part, i.e. a series in `(a - a0)/(1 + a0·a)`.
pub fn atan(a: &Da) -> Da {
    let nocut = truncation_order();
    let a0 = a.cons();
    let mut xf = vec![0.0; nocut as usize + 1];

    let iarg = {
        let denom = add_double(&weighted_sum(a, a0, a, 0.0), 1.0);
        let num = subtract_double(a, a0);
        divide_da(&num, &denom)
    };

    let mut s = 1.0;
    xf[0] = a0.atan();
    let mut i = 1;
    while i < xf.len() {
        xf[i] = s / i as f64;
        s = -s;
        i += 2;
    }
    evaluate_series(&iarg, &xf)
}

/// The four-quadrant arctangent `atan2(y, x)` of two DAs, with the sign
/// correction placing the result in `(-π, π]` (`daceArcTangent2`).
pub fn atan2(y: &Da, x: &Da) -> Da {
    Da::assert_same_context(y, x);
    let cx = x.cons();
    let cy = y.cons();

    if cx == 0.0 && cy == 0.0 {
        return Da::constant(0.0);
    }
    if cy.abs() > cx.abs() {
        let t = atan(&divide_da(x, y));
        if cy < 0.0 {
            double_subtract(&t, -std::f64::consts::FRAC_PI_2)
        } else {
            double_subtract(&t, std::f64::consts::FRAC_PI_2)
        }
    } else {
        let t = atan(&divide_da(y, x));
        if cx < 0.0 {
            if cy > 0.0 {
                add_double(&t, std::f64::consts::PI)
            } else {
                add_double(&t, -std::f64::consts::PI)
            }
        } else {
            t
        }
    }
}

/// Division of two DAs: `a * minv(b)` (`daceDivide`).
///
/// # Panics
///
/// Panics with [`crate::DaceError`] code 641 when `b` has a zero constant part.
pub(crate) fn divide_da(a: &Da, b: &Da) -> Da {
    multiply(a, &minv(b))
}

// ---------------------------------------------------------------------------
// Hyperbolic functions (dacemath.c:1287-1434)
// ---------------------------------------------------------------------------

/// The hyperbolic sine (`daceHyperbolicSine`).
pub fn sinh(a: &Da) -> Da {
    let nocut = truncation_order();
    let a0 = a.cons();
    let mut xf = vec![0.0; nocut as usize + 1];
    xf[0] = a0.sinh();
    xf[1] = a0.cosh();
    for i in 2..xf.len() {
        xf[i] = xf[i - 2] / (i * (i - 1)) as f64;
    }
    evaluate_series(a, &xf)
}

/// The hyperbolic cosine (`daceHyperbolicCosine`).
pub fn cosh(a: &Da) -> Da {
    let nocut = truncation_order();
    let a0 = a.cons();
    let mut xf = vec![0.0; nocut as usize + 1];
    xf[0] = a0.cosh();
    xf[1] = a0.sinh();
    for i in 2..xf.len() {
        xf[i] = xf[i - 2] / (i * (i - 1)) as f64;
    }
    evaluate_series(a, &xf)
}

/// The hyperbolic tangent (`daceHyperbolicTangent`), via the stable
/// exponential form depending on the sign of the constant part.
pub fn tanh(a: &Da) -> Da {
    let a0 = a.cons();
    if a0 > 0.0 {
        let t = exp(&weighted_sum(a, -2.0, a, 0.0));
        let denom = add_double(&t, 1.0);
        let num = double_subtract(&t, 1.0);
        divide_da(&num, &denom)
    } else {
        let t = exp(&weighted_sum(a, 2.0, a, 0.0));
        let denom = add_double(&t, 1.0);
        let num = add_double(&t, -1.0);
        divide_da(&num, &denom)
    }
}

/// The hyperbolic arcsine (`daceHyperbolicArcSine`): `log(a + sqrt(a²+1))`.
pub fn asinh(a: &Da) -> Da {
    let s = sqrt(&add_double(&a.sqr(), 1.0));
    log(&(a.clone() + s))
}

/// The hyperbolic arccosine (`daceHyperbolicArcCosine`): `log(a + sqrt(a²-1))`.
///
/// # Panics
///
/// Panics with [`crate::DaceError`] code 650 when the constant part is `<= 1`.
pub fn acosh(a: &Da) -> Da {
    if a.cons() <= 1.0 {
        dace_panic(codes::OUT_OF_DOMAIN, "Out of domain");
    }
    let s = sqrt(&subtract_double(&a.sqr(), 1.0));
    log(&(a.clone() + s))
}

/// The hyperbolic arctangent (`daceHyperbolicArcTangent`):
/// `log((1+a)/(1-a))/2`.
///
/// # Panics
///
/// Panics with [`crate::DaceError`] code 650 when `|constant part| >= 1`.
pub fn atanh(a: &Da) -> Da {
    if a.cons().abs() >= 1.0 {
        dace_panic(codes::OUT_OF_DOMAIN, "Out of domain");
    }
    let num = add_double(a, 1.0);
    let den = double_subtract(a, 1.0);
    let q = log(&divide_da(&num, &den));
    weighted_sum(&q, 0.5, &q, 0.0)
}

// ---------------------------------------------------------------------------
// Error functions (dacemath.c:1441-1504); scalar values from puruspe
// ---------------------------------------------------------------------------

/// The error function (`daceErrorFunction`), via the Hermite-polynomial
/// derivative recursion.
pub fn erf(a: &Da) -> Da {
    let nocut = truncation_order();
    let a0 = a.cons();
    let mut xf = vec![0.0; nocut as usize + 1];
    let mut factor = 2.0 * (-a0 * a0).exp() / std::f64::consts::PI.sqrt();
    xf[0] = puruspe::erf(a0);
    xf[1] = factor;
    let mut hi2 = 1.0; // Hermite H_0
    let mut hi1 = 2.0 * a0; // Hermite H_1
    for (i, item) in xf.iter_mut().enumerate().skip(2) {
        factor /= -(i as f64);
        *item = factor * hi1;
        let temp = 2.0 * a0 * hi1 - 2.0 * (i - 1) as f64 * hi2;
        hi2 = hi1;
        hi1 = temp;
    }
    evaluate_series(a, &xf)
}

/// The complementary error function (`daceComplementaryErrorFunction`).
pub fn erfc(a: &Da) -> Da {
    let nocut = truncation_order();
    let a0 = a.cons();
    let mut xf = vec![0.0; nocut as usize + 1];
    let mut factor = -2.0 * (-a0 * a0).exp() / std::f64::consts::PI.sqrt();
    xf[0] = puruspe::erfc(a0);
    xf[1] = factor;
    let mut hi2 = 1.0;
    let mut hi1 = 2.0 * a0;
    for (i, item) in xf.iter_mut().enumerate().skip(2) {
        factor /= -(i as f64);
        *item = factor * hi1;
        let temp = 2.0 * a0 * hi1 - 2.0 * (i - 1) as f64 * hi2;
        hi2 = hi1;
        hi1 = temp;
    }
    evaluate_series(a, &xf)
}

// ---------------------------------------------------------------------------
// Method forms on Da
// ---------------------------------------------------------------------------

macro_rules! method_form {
    ($($name:ident),* $(,)?) => {
        $(
            #[doc = concat!("Method form of [`", stringify!($name), "`](crate::elementary::", stringify!($name), ").")]
            pub fn $name(&self) -> Da {
                $name(self)
            }
        )*
    };
}

impl Da {
    method_form!(
        exp, log, log10, log2, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh,
        atanh, erf, erfc, sqrt, isrt, cbrt, icrt,
    );

    /// Method form of [`powf`](crate::elementary::powf).
    pub fn powf(&self, p: f64) -> Da {
        powf(self, p)
    }

    /// Method form of [`powi`](crate::elementary::powi).
    pub fn powi(&self, np: i32) -> Da {
        powi(self, np)
    }

    /// Method form of [`root`](crate::elementary::root).
    pub fn root(&self, np: i32) -> Da {
        root(self, np)
    }

    /// Method form of [`log_base`](crate::elementary::log_base).
    pub fn log_base(&self, b: f64) -> Da {
        log_base(self, b)
    }

    /// Method form of [`hypot`](crate::elementary::hypot).
    pub fn hypot(&self, b: &Da) -> Da {
        hypot(self, b)
    }

    /// Method form of [`atan2`](crate::elementary::atan2).
    pub fn atan2(&self, x: &Da) -> Da {
        atan2(self, x)
    }

    /// Method form of [`modulo`](crate::elementary::modulo).
    pub fn modulo(&self, p: f64) -> Da {
        modulo(self, p)
    }

    /// Method form of [`trunc`](crate::elementary::trunc).
    pub fn trunc(&self) -> Da {
        trunc(self)
    }

    /// Method form of [`round`](crate::elementary::round).
    pub fn round(&self) -> Da {
        round(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::CONTEXT_LOCK;

    #[test]
    fn taylor_coefficients() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(6, 1).unwrap();
        let x = Da::variable(1);

        // sin(x) = x - x^3/6 + x^5/120
        let s = sin(&x);
        assert!((s.get_coefficient(&[1]) - 1.0).abs() < 1e-15);
        assert!((s.get_coefficient(&[3]) + 1.0 / 6.0).abs() < 1e-15);
        assert!((s.get_coefficient(&[5]) - 1.0 / 120.0).abs() < 1e-15);
        assert_eq!(s.size(), 3);

        // exp(x) = 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120 + x^6/720
        let e = exp(&x);
        let mut fact = 1.0;
        for k in 0..=6u32 {
            if k > 0 {
                fact *= f64::from(k);
            }
            assert!(
                (e.get_coefficient(&[k]) - 1.0 / fact).abs() < 1e-15,
                "exp coeff {k}"
            );
        }

        // atan(x) matches its integrated form 1/(1+x^2)
        let a = atan(&x);
        assert!((a.get_coefficient(&[1]) - 1.0).abs() < 1e-15);
        assert!((a.get_coefficient(&[3]) + 1.0 / 3.0).abs() < 1e-15);
        assert!((a.get_coefficient(&[5]) - 1.0 / 5.0).abs() < 1e-15);

        // sqrt(1+u) and powi
        let u = 1.0 + x.clone();
        let r = sqrt(&(u.clone() * u.clone()));
        assert!((r.cons() - 1.0).abs() < 1e-12);
        assert!((r.get_coefficient(&[1]) - 1.0).abs() < 1e-12);
        let p3 = x.clone().powi(3);
        assert!((p3.get_coefficient(&[3]) - 1.0).abs() < 1e-15);
        assert_eq!(p3.size(), 1);
    }

    #[test]
    fn identities_two_vars() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(8, 2).unwrap();
        let x = Da::variable(1);
        let y = Da::variable(2);
        let f = 0.7 + 1.3 * x.clone() - 0.4 * y.clone();

        // sin^2 + cos^2 == 1 (rtol 1e-12)
        let s2 = sin(&f).sqr() + cos(&f).sqr();
        assert!((s2.cons() - 1.0).abs() < 1e-12);
        for m in s2.iter_monomials() {
            if m.order() == 0 {
                continue;
            }
            assert!(m.c.abs() < 1e-12, "sin^2+cos^2 residual at {:?}", m.jj);
        }

        // log(exp(f)) == f (rtol 1e-12)
        let lf = log(&exp(&f));
        for m in lf.iter_monomials() {
            let expect = f.get_coefficient(&m.jj);
            assert!(
                (m.c - expect).abs() <= 1e-12 * expect.abs().max(1.0),
                "log(exp) at {:?}: {} vs {}",
                m.jj,
                m.c,
                expect
            );
        }

        // tanh via exp, sinh/cosh
        let th = tanh(&f);
        let sh = sinh(&f);
        let ch = cosh(&f);
        let q = divide_da(&sh, &ch);
        for (m1, m2) in th.iter_monomials().zip(q.iter_monomials()) {
            assert_eq!(m1.jj, m2.jj);
            assert!((m1.c - m2.c).abs() <= 1e-11 * m1.c.abs().max(1.0));
        }

        // asin/acos derivative check: d/dx asin(x) = 1/sqrt(1-x^2)
        let g = 0.3 * x.clone();
        let asg = asin(&g);
        let d = asg.deriv(1);
        let expect = weighted_sum(&isrt(&(1.0 - g.clone() * g.clone())), 0.3, &Da::new(), 0.0);
        for (m1, m2) in d.iter_monomials().zip(expect.iter_monomials()) {
            assert_eq!(m1.jj, m2.jj);
            assert!((m1.c - m2.c).abs() <= 1e-11 * m1.c.abs().max(1.0));
        }

        // atan2 roundtrip
        let ang = 0.5 + 0.2 * x.clone();
        let r = 1.2 + 0.3 * y.clone();
        let yv = r.clone() * sin(&ang.clone());
        let xv = r.clone() * cos(&ang.clone());
        let a2 = atan2(&yv, &xv);
        for m in a2.iter_monomials() {
            let expect = ang.get_coefficient(&m.jj);
            assert!(
                (m.c - expect).abs() <= 1e-11 * expect.abs().max(1.0),
                "atan2 at {:?}: {} vs {}",
                m.jj,
                m.c,
                expect
            );
        }

        // erf: erf(0) = 0, derivative at 0 is 2/sqrt(pi)
        let z = 0.0 + x.clone();
        let ez = erf(&z);
        assert_eq!(ez.cons(), 0.0);
        assert!((ez.get_coefficient(&[1, 0]) - 2.0 / std::f64::consts::PI.sqrt()).abs() < 1e-14);
        let ec = erfc(&z);
        assert!((ec.cons() - 1.0).abs() < 1e-14);
        assert!((ec.get_coefficient(&[1, 0]) + 2.0 / std::f64::consts::PI.sqrt()).abs() < 1e-14);

        // constant-part-only ops
        // C daceTruncate uses rint (round half to even): 2.7 -> 3, 2.5 -> 2;
        // daceRound rounds half away from zero: 2.5 -> 3.
        let w = 2.5 + x.clone();
        assert!((w.trunc().cons() - 2.0).abs() < 1e-15);
        assert!((w.round().cons() - 3.0).abs() < 1e-15);
        assert!((w.modulo(2.0).cons() - 0.5).abs() < 1e-12);

        // logs in bases
        assert!((log10(&(Da::constant(100.0))).cons() - 2.0).abs() < 1e-12);
        assert!((log2(&(Da::constant(8.0))).cons() - 3.0).abs() < 1e-12);
        assert!((log_base(&(Da::constant(8.0)), 2.0).cons() - 3.0).abs() < 1e-12);

        // roots
        assert!((cbrt(&(Da::constant(27.0))).cons() - 3.0).abs() < 1e-12);
        assert!((isrt(&(Da::constant(4.0))).cons() - 0.5).abs() < 1e-12);
        assert!((icrt(&(Da::constant(8.0))).cons() - 0.5).abs() < 1e-12);
        assert!((hypot(&(Da::constant(3.0)), &Da::constant(4.0)).cons() - 5.0).abs() < 1e-12);
        assert!((powf(&(Da::constant(2.0)), 10.0).cons() - 1024.0).abs() < 1e-9);

        // acosh / asinh / atanh sanity
        assert!((acosh(&(Da::constant(2.0))).cons() - 2.0f64.acosh()).abs() < 1e-12);
        assert!((asinh(&(Da::constant(1.5))).cons() - 1.5f64.asinh()).abs() < 1e-12);
        assert!((atanh(&(Da::constant(0.5))).cons() - 0.5f64.atanh()).abs() < 1e-12);
    }

    #[test]
    fn domain_panics() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(6, 2).unwrap();

        fn expect_code(f: impl FnOnce() -> Da + std::panic::UnwindSafe, code: u32) {
            let err = std::panic::catch_unwind(f).expect_err("must panic");
            let e = err
                .downcast_ref::<crate::error::DaceError>()
                .expect("DaceError payload");
            assert_eq!(e.code, code, "{}", e);
        }

        expect_code(|| log(&Da::constant(-1.0)), codes::LOG_NON_POSITIVE);
        expect_code(|| log(&Da::constant(0.0)), codes::LOG_NON_POSITIVE);
        // Error 649 (cos(const) == 0 in tan) cannot be triggered by a double
        // constant: libm cos never returns exact 0.0 (cos(π/2) ~ 6.1e-17),
        // matching the C library's behavior on the same inputs.
        expect_code(|| Da::constant(0.0).minv(), codes::DIVIDING_BY_ZERO);
        expect_code(|| sqrt(&Da::constant(-4.0)), codes::EVEN_ROOT_NEGATIVE);
        expect_code(|| root(&Da::constant(1.0), 0), codes::ZERO_TH_ROOT);
        expect_code(|| root(&Da::constant(0.0), 3), codes::ODD_ROOT_ZERO);
        expect_code(
            || powf(&Da::constant(-2.0), 0.5),
            codes::NON_INTEGER_POWER_NON_POSITIVE,
        );
        expect_code(|| asin(&Da::constant(1.0)), codes::OUT_OF_DOMAIN);
        expect_code(|| acos(&Da::constant(-1.5)), codes::OUT_OF_DOMAIN);
        expect_code(|| acosh(&Da::constant(1.0)), codes::OUT_OF_DOMAIN);
        expect_code(|| atanh(&Da::constant(-1.0)), codes::OUT_OF_DOMAIN);
        expect_code(
            || log_base(&Da::constant(2.0), -1.0),
            codes::LOG_BASE_POSITIVE,
        );
        expect_code(
            || Da::constant(1.0) / Da::variable(1),
            codes::DIVIDING_BY_ZERO,
        );
    }
}
