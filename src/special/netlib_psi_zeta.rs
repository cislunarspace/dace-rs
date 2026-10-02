// Mechanical transcription: literals and expression shapes are kept as close
// to the original sources as the language allows.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

//! Polygamma (`psi`) and Hurwitz zeta (`zeta`) scalar functions.
//!
//! Mechanical transcriptions of DACE's `core/contrib/psi.c` (netlib PSIFN,
//! f2c translation of the FORTRAN original by W. J. Cody et al., Argonne
//! National Laboratory) and `core/contrib/zeta.c` (Cephes Math Library
//! Release 2.8), with the original comments preserved. Attribution in
//! `THIRD_PARTY_NOTICES.md`.

// ---------------------------------------------------------------------------
// psi.c: netlib/psi.f, translated by f2c (version 20100827).
// ---------------------------------------------------------------------------

// This function program evaluates the logarithmic derivative of the
// gamma function,
//
//     psi(x) = d/dx (gamma(x)) / gamma(x) = d/dx (ln gamma(x))
//
// for real x, where either
//
//         -xmax1 < x < -xmin (x not a negative integer), or
//           xmin < x.
//
// The main computation uses rational Chebyshev approximations
// published in Math. Comp. 27, 123-127 (1973) by Cody, Strecok and
// Thacher.  This transportable program is patterned after the
// machine-dependent FUNPACK program PSI(X), but cannot match that
// version for efficiency or accuracy.  This version uses rational
// approximations that are theoretically accurate to 20 significant
// decimal digits.  The accuracy achieved depends on the arithmetic
// system, the compiler, the intrinsic functions, and proper selection
// of the machine-dependent constants.
//
// Error Returns
//
// The program returns XINF for  X < -XMAX1, for X zero or a negative
// integer, or when X lies in (-XMIN1, 0), and returns -XINF
// when X lies in (0, XMIN1).
//
// Author: W. J. Cody
//         Mathematics and Computer Science Division
//         Argonne National Laboratory
//         Argonne, IL 60439
//
// Latest modification: June 8, 1988
//
// Machine-dependent constants (IEEE 754 double precision):
//   XINF   = largest positive machine number
//   XMAX1  = beta ** (p-1): upper bound on non-integral floats and the
//            negative of the lower bound on acceptable negative arguments
//   XMIN1  = the smallest in magnitude acceptable argument
//   XSMALL = absolute argument below which  PI*COTAN(PI*X)  may be
//            represented by 1/X
//   XLARGE = argument beyond which PSI(X) may be represented by LOG(X)

/// Digamma `ψ(x) = d/dx ln Γ(x)` (netlib PSIFN).
pub(crate) fn psi(xx: f64) -> f64 {
    const XMAX1: f64 = 4.5e15;
    const XSMALL: f64 = 5.8e-9;
    const XLARGE: f64 = 2.71e14;
    const X01: f64 = 187.0;
    const X01D: f64 = 128.0;
    const X02: f64 = 6.9464496836234126266e-4;
    const P1: [f64; 9] = [
        0.004510468124576293416,
        5.4932855833000385356,
        376.46693175929276856,
        7952.5490849151998065,
        71451.59581895193321,
        306559.76301987365674,
        636069.97788964458797,
        580413.12783537569993,
        165856.95029761022321,
    ];
    const Q1: [f64; 8] = [
        96.141654774222358525,
        2628.771579058119333,
        29862.49702225027792,
        162065.66091533671639,
        434878.80712768329037,
        542563.84537269993733,
        242421.85002017985252,
        6.4155223783576225996e-8,
    ];
    const P2: [f64; 7] = [
        -2.7103228277757834192,
        -15.166271776896121383,
        -19.784554148719218667,
        -8.8100958828312219821,
        -1.4479614616899842986,
        -0.073689600332394549911,
        -6.5135387732718171306e-21,
    ];
    const Q2: [f64; 6] = [
        44.992760373789365846,
        202.40955312679931159,
        247.36979003315290057,
        107.42543875702278326,
        17.463965060678569906,
        0.88427520398873480342,
    ];
    const PIOV4: f64 = 0.78539816339744830962;
    const XINF: f64 = 1.79e308;
    const XMIN1: f64 = 2.23e-308;

    let mut x = xx;
    let mut w = x.abs();
    let mut aug: f64 = 0.0;

    // Check for valid arguments, then branch to appropriate algorithm
    if -x >= XMAX1 || w < XMIN1 {
        // L410: error return
        return if x > 0.0 { -XINF } else { XINF };
    }
    if x >= 0.5 {
        // L200
    } else {
        // X < 0.5, use reflection formula: psi(1-x) = psi(x) + pi*cot(pi*x)
        // Use 1/X for PI*COTAN(PI*X) when XMIN1 < |X| <= XSMALL.
        if w <= XSMALL {
            aug = -1.0 / x;
        } else {
            // Argument reduction for cot
            let mut sgn = if x < 0.0 { PIOV4 } else { -PIOV4 };
            w -= w.trunc();
            let nq = (w * 4.0) as i64;
            w = 4.0 * (w - nq as f64 * 0.25);

            // W is now related to the fractional part of 4.0*X.
            // Adjust argument to correspond to values in the first
            // quadrant and determine the sign.
            let mut n = nq / 2;
            if n + n != nq {
                w = 1.0 - w;
            }
            let z = PIOV4 * w;
            if n % 2 != 0 {
                sgn = -sgn;
            }

            // determine the final value for -pi * cotan(pi*x)
            n = (nq + 1) / 2;
            if n % 2 == 0 {
                // Check for singularity
                if z == 0.0 {
                    return if x > 0.0 { -XINF } else { XINF };
                }
                aug = sgn * (4.0 / z.tan());
            } else {
                aug = sgn * (4.0 * z.tan());
            }
        }
        x = 1.0 - x;
    }

    if x > 3.0 {
        // L300: 3.0 < X
        if x < XLARGE {
            w = 1.0 / (x * x);
            let mut den = w;
            let mut upper = P2[0] * w;
            for i in 1..=5 {
                den = (den + Q2[i - 1]) * w;
                upper = (upper + P2[i]) * w;
            }
            aug += (upper + P2[6]) / (den + Q2[5]) - 0.5 / x;
        }
        aug + x.ln()
    } else {
        // 0.5 <= X <= 3.0
        let mut den = x;
        let mut upper = P1[0] * x;
        for i in 1..=7 {
            den = (den + Q1[i - 1]) * x;
            upper = (upper + P1[i]) * x;
        }
        den = (upper + P1[8]) / (den + Q1[7]);
        x -= X01 / X01D + X02;
        den * x + aug
    }
}

// ---------------------------------------------------------------------------
// zeta.c: Cephes Math Library Release 2.8: June, 2000
// Copyright 1984, 1995, 2000 Stephen L. Moshier
// ---------------------------------------------------------------------------

// Riemann zeta function of two arguments
//
//                 inf.
//                  -        -x
//   zeta(x,q)  =   >   (k+q)
//                  -
//                 k=0
//
// where x > 1 and q is not a negative integer or zero.
// The Euler-Maclaurin summation formula is used to obtain
// the expansion
//
//                n
//                -       -x
// zeta(x,q)  =   >  (k+q)
//                -
//               k=1
//
//           1-x                 inf.  B   x(x+1)...(x+2j)
//      (n+q)           1         -     2j
//  +  ---------  -  -------  +   >    --------------------
//        x-1              x      -                   x+2j+1
//                   2(n+q)      j=1       (2j)! (n+q)
//
// where the B2j are Bernoulli numbers.
//
// REFERENCE: Gradshteyn, I. S., and I. M. Ryzhik, Tables of Integrals,
// Series, and Products, p. 1073; Academic Press, 1980.

// Expansion coefficients for the Euler-Maclaurin summation formula:
// (2k)! / B2k where B2k are Bernoulli numbers.
// 30 Nov 86 -- error in third coefficient fixed
const A: [f64; 12] = [
    12.0,
    -720.0,
    30240.0,
    -1209600.0,
    47900160.0,
    -1.8924375803183791606e9, /*1.307674368e12/691*/
    7.47242496e10,
    -2.950130727918164224e12,  /*1.067062284288e16/3617*/
    1.1646782814350067249e14,  /*5.109094217170944e18/43867*/
    -4.5979787224074726105e15, /*8.028576626982912e20/174611*/
    1.8152105401943546773e17,  /*1.5511210043330985984e23/854513*/
    -7.1661652561756670113e18, /*1.6938241367317436694528e27/236364091*/
];

const MACHEP: f64 = 1.11022302462515654042e-16;

/// Hurwitz zeta `ζ(x, q) = Σ_{k≥0} (k+q)^{-x}` for `x > 1` (Cephes).
///
/// Follows the C routine exactly, including the error sentinels: `NaN` for
/// domain errors, infinities at singular arguments.
pub(crate) fn zeta(x: f64, q: f64) -> f64 {
    if x == 1.0 {
        return f64::INFINITY;
    }
    if x < 1.0 {
        return f64::NAN;
    }
    if q <= 0.0 {
        if q == q.floor() {
            return f64::INFINITY;
        }
        if x != x.floor() {
            return f64::NAN; // because q^-x not defined
        }
    }

    // Euler-Maclaurin summation formula. Permit negative q but continue
    // sum until n+q > +9. This case should be handled by a reflection
    // formula. If q<0 and x is an integer, there is a relation to
    // the polygamma function.
    let mut s = q.powf(-x);
    let mut a = q;
    let mut b = 0.0;
    let mut i = 0u32;
    while i < 9 || a <= 9.0 {
        a += 1.0;
        b = a.powf(-x);
        s += b;
        if (b / s).abs() < MACHEP {
            return s;
        }
        i += 1;
    }

    let w = a;
    s += b * w / (x - 1.0);
    s -= 0.5 * b;
    let mut a2 = 1.0;
    let mut k = 0.0;
    for coeff in A.iter() {
        a2 *= x + k;
        b /= w;
        let t = a2 * b / coeff;
        s += t;
        if (t / s).abs() < MACHEP {
            return s;
        }
        k += 1.0;
        a2 *= x + k;
        b /= w;
        k += 1.0;
    }
    s
}
