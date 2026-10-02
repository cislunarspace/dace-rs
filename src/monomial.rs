//! A decoded monomial: exponent vector plus coefficient.

/// One monomial of a Taylor polynomial: the exponents `jj` (one per DA
/// variable, in variable order) and its coefficient `c`.
///
/// This is the decoded, human-facing form; [`Da`][crate::Da] values store
/// monomials packed as `(index, coefficient)` pairs (see
/// [`Da::iter_monomials`][crate::Da::iter_monomials]).
#[derive(Debug, Clone, PartialEq)]
pub struct Monomial {
    /// Exponent of each variable (length = number of DA variables).
    pub jj: Vec<u32>,
    /// Coefficient of the monomial.
    pub c: f64,
}

impl Monomial {
    /// Total order of the monomial (sum of all exponents).
    pub fn order(&self) -> u32 {
        self.jj.iter().sum()
    }
}
