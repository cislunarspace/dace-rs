//! Text formatting of [`Da`] values.
//!
//! Ports the format of `daceWrite` (core/daceio.c:55-110): a header line, one
//! line per monomial (ordered by order, then list position) showing index,
//! coefficient, order, and exponents, and a footer of dashes. A zero DA
//! prints the "ALL COEFFICIENTS ZERO" line instead of monomials.

use std::fmt;

use crate::da::Da;

const BEGSTR: &str = "     I  COEFFICIENT              ORDER EXPONENTS";
const ENDSTR: &str = "------------------------------------------------";
const ZEROSTR: &str = "        ALL COEFFICIENTS ZERO";

/// Format a coefficient exactly like C's `%24.16e`: 16 fractional digits and
/// a signed two-digit exponent (e.g. `1.0000000000000000e+00`).
fn fmt_c_e(c: f64) -> String {
    let s = format!("{c:.16e}");
    let (mantissa, exponent) = s.split_once('e').expect("scientific notation");
    let exp: i32 = exponent.parse().unwrap_or(0);
    format!(
        "{mantissa}e{}{:02}",
        if exp < 0 { '-' } else { '+' },
        exp.abs()
    )
}

impl fmt::Display for Da {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            writeln!(f, "{ZEROSTR}")?;
            writeln!(f, "{ENDSTR}")?;
            return Ok(());
        }

        writeln!(f, "{BEGSTR}")?;
        let ctx = &self.ctx;
        let mut jj = vec![0u32; ctx.nvmax as usize];
        let mut iout: usize = 1;
        for ioa in 0..=ctx.nomax {
            for t in &self.terms {
                if ctx.order_of(t.idx) != ioa {
                    continue;
                }
                ctx.decode_into(t.idx, &mut jj);
                write!(f, "{iout:6}  {:>24}", fmt_c_e(t.c))?;
                write!(f, "{ioa:4} ")?;
                for &e in &jj {
                    write!(f, " {e:2}")?;
                }
                writeln!(f)?;
                iout += 1;
            }
        }
        writeln!(f, "{ENDSTR}")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::CONTEXT_LOCK;

    #[test]
    fn display_matches_c_format() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(3, 2).unwrap();
        let x = Da::variable(1);
        let y = Da::variable(2);
        let f = 1.0 + 2.0 * x.clone() - 0.5 * y.clone() * y.clone();
        let s = f.to_string();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines[0], "     I  COEFFICIENT              ORDER EXPONENTS");
        assert_eq!(lines[1], "     1    1.0000000000000000e+00   0   0  0");
        assert_eq!(lines[2], "     2    2.0000000000000000e+00   1   1  0");
        assert_eq!(lines[3], "     3   -5.0000000000000000e-01   2   0  2");
        assert_eq!(lines[4], "------------------------------------------------");
        assert_eq!(lines.len(), 5);

        // Zero DA prints the special line plus footer.
        let z = Da::new().to_string();
        assert_eq!(
            z,
            "        ALL COEFFICIENTS ZERO\n------------------------------------------------\n"
        );
    }
}
