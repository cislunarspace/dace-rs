//! Vector-of-DA operations: the [`DaVector`] extension trait, inverse maps,
//! and scalar-vector helpers.
//!
//! Ports the `AlgebraicVector<DA>` surface used in practice: elementwise
//! calculus and evaluation, the linear-part matrix, and the fixed-point map
//! inversion of `AlgebraicVector.cpp:221-280` (Gauss-Jordan inverse of the
//! linear part with full pivoting, then iteration in rising truncation
//! order). The `AlgebraicMatrix` type itself is not ported (experimental
//! upstream, off by default).

use crate::context::{max_variables, set_truncation_order, truncation_order};
use crate::da::Da;
use crate::error::{codes, dace_panic};
use crate::eval::CompiledDa;

/// Elementwise operations on vectors of DAs (C++ `AlgebraicVector<DA>`).
pub trait DaVector {
    /// Elementwise constant parts.
    fn cons(&self) -> Vec<f64>;

    /// The linear-part matrix (row per component, column per variable).
    fn linear(&self) -> Vec<Vec<f64>>;

    /// Elementwise derivative with respect to variable `var` (1-based).
    fn deriv(&self, var: u32) -> Vec<Da>;

    /// Elementwise integral with respect to variable `var` (1-based).
    fn integ(&self, var: u32) -> Vec<Da>;

    /// Elementwise evaluation at a point.
    fn eval(&self, args: &[f64]) -> Vec<f64>;

    /// Elementwise partial evaluation of variable `var` (1-based).
    fn plug(&self, var: u32, val: f64) -> Vec<Da>;

    /// Elementwise trim to orders `min_order..=max_order`.
    fn trim(&self, min_order: u32, max_order: u32) -> Vec<Da>;

    /// Invert the polynomial map (C++ `AlgebraicVector<DA>::invert`).
    ///
    /// # Panics
    ///
    /// Panics with [`crate::DaceError`] when the vector dimension exceeds the
    /// number of DA variables or the linear part is singular.
    fn invert(&self) -> Vec<Da>;
}

impl DaVector for [Da] {
    fn cons(&self) -> Vec<f64> {
        self.iter().map(|d| d.cons()).collect()
    }

    fn linear(&self) -> Vec<Vec<f64>> {
        self.iter().map(|d| d.linear()).collect()
    }

    fn deriv(&self, var: u32) -> Vec<Da> {
        self.iter().map(|d| d.deriv(var)).collect()
    }

    fn integ(&self, var: u32) -> Vec<Da> {
        self.iter().map(|d| d.integ(var)).collect()
    }

    fn eval(&self, args: &[f64]) -> Vec<f64> {
        self.iter().map(|d| d.eval(args)).collect()
    }

    fn plug(&self, var: u32, val: f64) -> Vec<Da> {
        self.iter().map(|d| d.plug(var, val)).collect()
    }

    fn trim(&self, min_order: u32, max_order: u32) -> Vec<Da> {
        self.iter().map(|d| d.trim(min_order, max_order)).collect()
    }

    fn invert(&self) -> Vec<Da> {
        let ord = truncation_order();
        let nvar = self.len();
        if nvar > max_variables() as usize {
            dace_panic(
                codes::TOO_MANY_VARIABLES,
                "dimension of vector exceeds maximum number of DA variables",
            );
        }

        // DA identity
        let dda: Vec<Da> = (1..=nvar as u32).map(Da::variable).collect();

        // Split map into constant part AC, non-constant part M, and
        // non-linear part AN.
        let ac: Vec<f64> = self.cons();
        let m: Vec<Da> = self.iter().map(|d| d.trim(1, u32::MAX)).collect();
        let an: Vec<Da> = m.iter().map(|d| d.trim(2, u32::MAX)).collect();

        // Inverse of the linear coefficients matrix (Gauss-Jordan with full
        // pivoting, AlgebraicVector.cpp:180-217).
        let mut ai = m.linear();
        matrix_inverse(&mut ai);

        // AI*AN, compiled; and Linv = AI*DDA.
        let aloan: Vec<Da> = (0..nvar)
            .map(|i| (0..nvar).fold(Da::constant(0.0), |acc, j| acc + ai[i][j] * an[j].clone()))
            .collect();
        let aioan = CompiledDa::from_das(&aloan);
        let linv: Vec<Da> = (0..nvar)
            .map(|i| (0..nvar).fold(Da::constant(0.0), |acc, j| acc + ai[i][j] * dda[j].clone()))
            .collect();

        // Iterate to obtain the inverse map.
        let mut mi = linv.clone();
        for i in 1..ord {
            set_truncation_order(i + 1);
            let correction = aioan.eval_da(&mi);
            mi = linv
                .iter()
                .zip(correction)
                .map(|(l, c)| l.clone() - c)
                .collect();
        }
        set_truncation_order(ord);

        // Evaluate at the shifted identity.
        let args: Vec<Da> = dda.iter().zip(&ac).map(|(d, &c)| d.clone() - c).collect();
        mi.iter().map(|m| m.eval_da(&args)).collect()
    }
}

impl DaVector for Vec<Da> {
    fn cons(&self) -> Vec<f64> {
        self.as_slice().cons()
    }

    fn linear(&self) -> Vec<Vec<f64>> {
        self.as_slice().linear()
    }

    fn deriv(&self, var: u32) -> Vec<Da> {
        self.as_slice().deriv(var)
    }

    fn integ(&self, var: u32) -> Vec<Da> {
        self.as_slice().integ(var)
    }

    fn eval(&self, args: &[f64]) -> Vec<f64> {
        self.as_slice().eval(args)
    }

    fn plug(&self, var: u32, val: f64) -> Vec<Da> {
        self.as_slice().plug(var, val)
    }

    fn trim(&self, min_order: u32, max_order: u32) -> Vec<Da> {
        self.as_slice().trim(min_order, max_order)
    }

    fn invert(&self) -> Vec<Da> {
        self.as_slice().invert()
    }
}

/// In-place Gauss-Jordan matrix inverse with full pivoting
/// (`AlgebraicVector<DA>::matrix_inverse`).
///
/// # Panics
///
/// Panics with [`crate::DaceError`] when the matrix is singular.
fn matrix_inverse(a: &mut [Vec<f64>]) {
    let n = a.len();
    let mut indexc = vec![0usize; n];
    let mut indexr = vec![0usize; n];
    let mut ipiv = vec![0usize; n];

    for i in 0..n {
        let mut icol = 0usize;
        let mut irow = 0usize;
        let mut big = 0.0f64;
        for (j, jp) in ipiv.iter().enumerate() {
            if *jp != 0 {
                continue;
            }
            for (k, kp) in ipiv.iter().enumerate() {
                if *kp == 0 && a[j][k].abs() >= big {
                    big = a[j][k].abs();
                    irow = j;
                    icol = k;
                }
            }
        }
        ipiv[icol] = 1;
        if irow != icol {
            a.swap(irow, icol);
        }
        indexr[i] = irow;
        indexc[i] = icol;
        if a[icol][icol] == 0.0 {
            dace_panic(
                codes::INVERSE_DOES_NOT_EXIST,
                "linear matrix inverse does not exist",
            );
        }
        let pivinv = 1.0 / a[icol][icol];
        a[icol][icol] = 1.0;
        for v in a[icol].iter_mut() {
            *v *= pivinv;
        }
        for ll in 0..n {
            if ll != icol {
                let temp = a[ll][icol];
                a[ll][icol] = 0.0;
                let src = a[icol].clone();
                for (v, s) in a[ll].iter_mut().zip(&src) {
                    *v -= s * temp;
                }
            }
        }
    }

    // Unscramble the column permutation.
    for i in (0..n).rev() {
        if indexr[i] != indexc[i] {
            for row in a.iter_mut() {
                row.swap(indexr[i], indexc[i]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar vector helpers (C++ AlgebraicVector<double> free functions)
// ---------------------------------------------------------------------------

/// Dot product of two scalar vectors.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Dot product of a vector of DAs.
pub fn dot_da(a: &[Da]) -> Da {
    a.iter()
        .skip(1)
        .fold(a[0].clone(), |acc, d| acc + d.clone())
}

/// Cross product of two 3-vectors.
///
/// # Panics
///
/// Panics when either argument is not of length 3.
pub fn cross(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), 3, "cross: not a 3-vector");
    assert_eq!(b.len(), 3, "cross: not a 3-vector");
    vec![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Euclidean norm of a scalar vector.
pub fn vnorm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// The normalized vector; panics on a zero vector.
pub fn normalize(a: &[f64]) -> Vec<f64> {
    let n = vnorm(a);
    assert!(n > 0.0, "normalize: zero vector");
    a.iter().map(|x| x / n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::CONTEXT_LOCK;

    #[test]
    fn invert_roundtrip() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(8, 3).unwrap();
        let x = Da::variable(1);
        let y = Da::variable(2);
        let z = Da::variable(3);

        // A mildly nonlinear map around the origin.
        let map = vec![
            x.clone() + 0.3 * (x.clone() * y.clone()),
            y.clone() - 0.2 * (y.clone() * z.clone()),
            z.clone() + 0.1 * x.clone() * z.clone(),
        ];

        let inv = map.invert();
        assert_eq!(inv.len(), 3);

        // v.invert().eval(v.eval(unit)) == unit at several points.
        for &(px, py, pz) in &[(0.05, -0.03, 0.04), (-0.08, 0.06, 0.02), (0.0, 0.0, 0.0)] {
            let img = map.eval(&[px, py, pz]);
            let back = inv.eval(&img);
            assert!((back[0] - px).abs() < 1e-10, "x: {} vs {px}", back[0]);
            assert!((back[1] - py).abs() < 1e-10, "y: {} vs {py}", back[1]);
            assert!((back[2] - pz).abs() < 1e-10, "z: {} vs {pz}", back[2]);
        }

        // Constant part of the inverse at the image of the origin.
        assert_eq!(inv.cons(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn scalar_helpers() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, -5.0, 6.0]), 12.0);
        assert_eq!(
            cross(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]),
            vec![0.0, 0.0, 1.0]
        );
        assert!((vnorm(&[3.0, 4.0]) - 5.0).abs() < 1e-15);
        assert_eq!(normalize(&[3.0, 4.0]), vec![0.6, 0.8]);
    }
}
