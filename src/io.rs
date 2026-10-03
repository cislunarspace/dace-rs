//! Text and binary I/O of [`Da`] values: the `daceWrite` display format and
//! its parser (`daceRead`), the binary blob format (`daceExportBlob` /
//! `daceImportBlob`), and the code-generation formatter
//! (`DASimpleFormatter`).
//!
//! The blob format preserves the C layout: magic `0x1E304144`, then `no`,
//! `nv1`, `nv2`, `len` as little-endian `u32`s, then `len` packed
//! `{i1, i2, cc}` records (`u32, u32, f64`, little-endian). All supported
//! targets are little-endian, matching the C library on the reference
//! platforms.

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

impl std::str::FromStr for Da {
    type Err = crate::error::DaceError;

    /// Parse the text format emitted by [`Display`][fmt::Display] (the
    /// `daceRead` parser, including its tolerance: line-number and order
    /// mismatches warn, order/truncation-excess rows are skipped, duplicate
    /// monomials accumulate).
    ///
    /// # Panics
    ///
    /// Panics with [`crate::DaceError`] if DACE has not been initialized.
    fn from_str(s: &str) -> Result<Da, crate::error::DaceError> {
        let lines: Vec<&str> = s.lines().collect();
        if lines.is_empty() {
            return Err(crate::error::DaceError::new(
                634,
                "Not enough lines to read",
            ));
        }
        let first = lines[0];
        if first.starts_with(ZEROSTR) || first.starts_with("        ALL COMPONENTS ZERO") {
            return Ok(Da::new());
        }
        // DACE and COSY header differ only in the coefficient field width;
        // dace-rs emits only the DACE one but accepts both.
        let cosy = first.starts_with("     I  COEFFICIENT            ORDER EXPONENTS");
        let dace = first.starts_with(BEGSTR);
        if !cosy && !dace {
            return Err(crate::error::DaceError::new(632, "Unknown format"));
        }
        let coefflen = if cosy { 22 } else { 24 };

        let ctx = crate::context::Context::current();
        let (_eps, nocut) = crate::context::eps_nocut();
        let mut cc = vec![0.0; ctx.nmmax as usize];
        let mut jj = vec![0u32; ctx.nvmax as usize];

        for (iin, line) in lines.iter().enumerate().skip(1) {
            if line.len() < 4 {
                return Err(crate::error::DaceError::new(632, "Unknown format"));
            }
            if line[4..].starts_with(ENDSTR) {
                break;
            }
            let b = line.as_bytes();
            if line.len() < 37 {
                return Err(crate::error::DaceError::new(632, "Unknown format"));
            }
            // line number (columns 0-5)
            let ii: u32 = std::str::from_utf8(&b[..6])
                .ok()
                .and_then(|t| t.trim().parse().ok())
                .unwrap_or(0);
            // coefficient (columns 8..8+coefflen)
            let c: f64 = std::str::from_utf8(&b[8..8 + coefflen])
                .ok()
                .and_then(|t| t.trim().parse().ok())
                .unwrap_or(0.0);
            // order (columns 32-35)
            let io1: u32 = std::str::from_utf8(&b[32..36])
                .ok()
                .and_then(|t| t.trim().parse().ok())
                .unwrap_or(0);
            // exponents: walk sequentially (1 space, then 2 digits per
            // variable; in COSY format only every other variable has the
            // leading space)
            let mut pos = 37usize;
            for (i, slot) in jj.iter_mut().enumerate() {
                *slot = 0;
                if line.len() > pos && (!cosy || i % 2 == 0) {
                    pos += 1;
                }
                if line.len() >= pos + 2 {
                    *slot = std::str::from_utf8(&b[pos..pos + 2])
                        .ok()
                        .and_then(|t| t.trim().parse().ok())
                        .unwrap_or(0);
                    pos += 2;
                }
            }

            // check line numbers (informational in C)
            if ii != iin as u32 {
                log::warn!("DACE info 164: numbering out of order while reading");
            }
            // check order and hence number of variables
            let io: u32 = jj.iter().sum();
            if io != io1 {
                log::warn!("DACE info 165: inaccurate estimate while reading; line skipped");
                continue;
            }
            // check cutoff order
            if io > nocut {
                continue;
            }
            let icc = ctx.encode(&jj).expect("validated order");
            if cc[icc as usize] != 0.0 {
                log::warn!("DACE info 166: duplicate monomial while reading");
            }
            cc[icc as usize] += c;
        }

        Ok(crate::eval::pack(&ctx, &mut cc))
    }
}

/// Binary magic `0x1E304144` (little-endian "DA0" + record separator).
const DACE_BINARY_MAGIC: u32 = 0x1E30_4144;

impl Da {
    /// Export in the C binary blob format (`daceExportBlob`): magic, `no`,
    /// `nv1`, `nv2`, `len` as little-endian `u32`s, then `len` packed
    /// `{i1, i2, cc}` records. As in C, the buffer always includes one
    /// record slot (zeroed when the DA is empty).
    ///
    /// All supported targets are little-endian, so blobs interoperate with
    /// the C library on the reference platforms.
    pub fn to_blob(&self) -> Vec<u8> {
        let ctx = &self.ctx;
        let len = self.terms.len();
        let mut out = Vec::with_capacity(20 + 16 * len.max(1));
        out.extend_from_slice(&DACE_BINARY_MAGIC.to_le_bytes());
        out.extend_from_slice(&ctx.nomax.to_le_bytes());
        out.extend_from_slice(&ctx.nv1.to_le_bytes());
        out.extend_from_slice(&ctx.nv2.to_le_bytes());
        out.extend_from_slice(&(len as u32).to_le_bytes());
        for t in &self.terms {
            out.extend_from_slice(&ctx.ie1[t.idx as usize].to_le_bytes());
            out.extend_from_slice(&ctx.ie2[t.idx as usize].to_le_bytes());
            out.extend_from_slice(&t.c.to_le_bytes());
        }
        // C always reserves one record slot in the header struct.
        if len == 0 {
            out.extend_from_slice(&[0u8; 16]);
        }
        out
    }

    /// Import from the C binary blob format (`daceImportBlob`): silently
    /// truncates orders above the current maximum computation order and any
    /// extra variables present, as in C.
    ///
    /// # Errors
    ///
    /// Returns [`crate::DaceError`] code 631 ("Invalid data") when the magic is
    /// wrong or the buffer is too short.
    pub fn from_blob(blob: &[u8]) -> Result<Da, crate::error::DaceError> {
        let invalid = || crate::error::DaceError::new(631, "Invalid data");
        if blob.len() < 20 {
            return Err(invalid());
        }
        let magic = u32::from_le_bytes(blob[0..4].try_into().unwrap());
        if magic != DACE_BINARY_MAGIC {
            return Err(invalid());
        }
        let no = u32::from_le_bytes(blob[4..8].try_into().unwrap());
        let nv1 = u32::from_le_bytes(blob[8..12].try_into().unwrap());
        let nv2 = u32::from_le_bytes(blob[12..16].try_into().unwrap());
        let len = u32::from_le_bytes(blob[16..20].try_into().unwrap()) as usize;
        if blob.len() < 20 + 16 * len {
            return Err(invalid());
        }

        let ctx = crate::context::Context::current();
        let nv = nv1 + nv2;
        let mut p = vec![0u32; nv.max(ctx.nvmax) as usize];
        let mut cc = vec![0.0; ctx.nmmax as usize];

        for i in 0..len {
            let off = 20 + 16 * i;
            let i1 = u32::from_le_bytes(blob[off..off + 4].try_into().unwrap());
            let i2 = u32::from_le_bytes(blob[off + 4..off + 8].try_into().unwrap());
            let c = f64::from_le_bytes(blob[off + 8..off + 16].try_into().unwrap());

            // decode with the blob's parameters
            let base = no + 1;
            let mut order = 0u32;
            let mut ic = i1;
            for slot in p[..nv1 as usize].iter_mut() {
                *slot = ic % base;
                ic /= base;
                order += *slot;
            }
            let mut ic = i2;
            for slot in p[nv1 as usize..nv as usize].iter_mut() {
                *slot = ic % base;
                ic /= base;
                order += *slot;
            }

            // order of variables outside the current setup
            let extravar: u32 = p[ctx.nvmax as usize..nv as usize].iter().sum();

            if order <= ctx.nomax && extravar == 0 {
                let idx = ctx.encode(&p[..ctx.nvmax as usize]).expect("order checked");
                cc[idx as usize] = c;
            }
        }

        Ok(crate::eval::pack(&ctx, &mut cc))
    }
}

// ---------------------------------------------------------------------------
// Code-generation formatter (C++ DASimpleFormatter)
// ---------------------------------------------------------------------------

/// The elements of a simple code-generation format (C++
/// `DASimpleFormat`): strings used around signs, products, variables, and
/// powers, plus line wrapping.
#[derive(Debug, Clone)]
pub struct SimpleFormat {
    /// Prefix of a positive term (e.g. `"+"`).
    pub pos: String,
    /// Prefix of a negative term with the coefficient negated (e.g. `"-"`).
    pub neg: String,
    /// Multiplication separator (e.g. `"*"`).
    pub mul: String,
    /// Power prefix (e.g. `"pow("`).
    pub pre_pow: String,
    /// Variable name prefix (e.g. `"p"`).
    pub var: String,
    /// Before the variable index (e.g. `"["`).
    pub pre_var: String,
    /// After the variable index (e.g. `"]"`).
    pub post_var: String,
    /// Power separator (e.g. `","`).
    pub pow: String,
    /// Power suffix (e.g. `")"`).
    pub post_pow: String,
    /// Line break inserted between wrapped lines.
    pub linebreak: String,
    /// Offset added to the 0-based variable position.
    pub first_var: i64,
    /// Offset added to each exponent.
    pub first_pow: i64,
    /// Monomials per line before wrapping.
    pub monperline: u32,
    /// Skip the power syntax for exponent 1.
    pub shorten: bool,
}

impl SimpleFormat {
    // A 14-field constructor mirroring the C++ aggregate initializer.
    #[allow(clippy::too_many_arguments)]
    fn new(
        pos: &str,
        neg: &str,
        mul: &str,
        pre_pow: &str,
        var: &str,
        pre_var: &str,
        post_var: &str,
        pow: &str,
        post_pow: &str,
        linebreak: &str,
        first_var: i64,
        first_pow: i64,
        monperline: u32,
        shorten: bool,
    ) -> SimpleFormat {
        SimpleFormat {
            pos: pos.to_string(),
            neg: neg.to_string(),
            mul: mul.to_string(),
            pre_pow: pre_pow.to_string(),
            var: var.to_string(),
            pre_var: pre_var.to_string(),
            post_var: post_var.to_string(),
            pow: pow.to_string(),
            post_pow: post_pow.to_string(),
            linebreak: linebreak.to_string(),
            first_var,
            first_pow,
            monperline,
            shorten,
        }
    }

    /// C array-output preset (`p[0]*p[1]`).
    pub fn c() -> SimpleFormat {
        Self::new(
            "+", "-", "*", "", "p", "[", "", "][", "]", " \\\n\t", 0, -1, 20, false,
        )
    }

    /// C preset using `pow(...)`.
    pub fn c_pow() -> SimpleFormat {
        Self::new(
            "+", "-", "*", "pow(", "x", "[", "]", ",", ")", " \\\n\t", 0, 0, 20, true,
        )
    }

    /// Fortran preset (`p(1)*p(2)`).
    pub fn fortran() -> SimpleFormat {
        Self::new(
            "+",
            "-",
            "*",
            "",
            "p",
            "(",
            "",
            ",",
            ")",
            " &\n     &",
            1,
            0,
            20,
            false,
        )
    }

    /// Fortran preset using `**( )`.
    pub fn fortran_pow() -> SimpleFormat {
        Self::new(
            "+",
            "-",
            "*",
            "",
            "x",
            "(",
            ")",
            "**(",
            ")",
            " &\n     &",
            1,
            0,
            20,
            true,
        )
    }

    /// MATLAB preset (`p(1).*p(2)`).
    pub fn matlab() -> SimpleFormat {
        Self::new(
            "+", "-", ".*", "", "p", "(", "", ",", ")", " ...\n\t", 1, 0, 20, false,
        )
    }

    /// MATLAB preset using `.( )`.
    pub fn matlab_pow() -> SimpleFormat {
        Self::new(
            "+", "-", ".*", "", "x", "(", ")", ".^(", ")", " ...\n\t", 1, 0, 20, true,
        )
    }

    /// LaTeX preset (`x_{1} \cdot x_{2}^{3}`).
    pub fn latex() -> SimpleFormat {
        Self::new(
            " +", " -", " \\cdot ", "", "x", "_{", "}", "^{", "}", " \n\t", 1, 0, 20, true,
        )
    }
}

/// Format a coefficient like a C++ ostream with `precision(16)` (16
/// significant digits, trailing zeros trimmed, scientific notation outside
/// `1e-4..1e16`).
fn fmt_g16(c: f64) -> String {
    if c == 0.0 {
        return "0".to_string();
    }
    let sci = format!("{:.15e}", c);
    let (mantissa, exponent) = sci.split_once('e').expect("scientific notation");
    let exp: i32 = exponent.parse().unwrap_or(0);
    if !(-4..16).contains(&exp) {
        let mut m = mantissa
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        if m.is_empty() {
            m = "0".to_string();
        }
        format!("{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    } else {
        let decimals = (15 - exp).max(0) as usize;
        let mut s = format!("{c:.decimals$}");
        if s.contains('.') {
            s = s.trim_end_matches('0').trim_end_matches('.').to_string();
        }
        s
    }
}

/// Format a DA as source code in the given [`SimpleFormat`]
/// (`DASimpleFormatter::format`).
pub fn format_da(da: &Da, sf: &SimpleFormat) -> String {
    let monomials = da.iter_monomials();
    let all: Vec<_> = monomials.collect();
    let mut res = String::new();
    for (i, m) in all.iter().enumerate() {
        if m.c < 0.0 {
            res.push_str(&sf.neg);
            res.push_str(&fmt_g16(-m.c));
        } else {
            res.push_str(&sf.pos);
            res.push_str(&fmt_g16(m.c));
        }
        for (j, &e) in m.jj.iter().enumerate() {
            if e == 0 {
                continue;
            } else if sf.shorten && e == 1 {
                res.push_str(&sf.mul);
                res.push_str(&sf.var);
                res.push_str(&sf.pre_var);
                res.push_str(&(j as i64 + sf.first_var).to_string());
                res.push_str(&sf.post_var);
            } else {
                res.push_str(&sf.mul);
                res.push_str(&sf.pre_pow);
                res.push_str(&sf.var);
                res.push_str(&sf.pre_var);
                res.push_str(&(j as i64 + sf.first_var).to_string());
                res.push_str(&sf.post_var);
                res.push_str(&sf.pow);
                res.push_str(&(e as i64 + sf.first_pow).to_string());
                res.push_str(&sf.post_pow);
            }
        }
        if (i + 1) % sf.monperline as usize == 0 && i + 1 < all.len() {
            res.push_str(&sf.linebreak);
        }
    }
    res
}

/// Format a vector of DAs, one per line (`DASimpleFormatter::format`).
pub fn format_das(das: &[Da], sf: &SimpleFormat) -> String {
    das.iter().map(|da| format_da(da, sf) + "\n").collect()
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
    #[test]
    fn from_str_roundtrip() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(4, 2).unwrap();
        let x = Da::variable(1);
        let y = Da::variable(2);
        let f = 0.75 - 1.25 * x.clone() + 3.5 * (x.clone() * y.clone()) - 0.125 * y.clone();
        let parsed: Da = f.to_string().parse().expect("parses");
        assert_eq!(parsed.size(), f.size());
        for m in f.iter_monomials() {
            assert!(
                (parsed.get_coefficient(&m.jj) - m.c).abs() == 0.0,
                "{:?}: {} vs {}",
                m.jj,
                parsed.get_coefficient(&m.jj),
                m.c
            );
        }

        let z: Da = Da::new().to_string().parse().unwrap();
        assert_eq!(z.size(), 0);

        assert!("garbage".parse::<Da>().is_err());
        assert!("".parse::<Da>().is_err());
    }

    #[test]
    fn blob_roundtrip_and_layout() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(5, 3).unwrap();
        let x = Da::variable(1);
        let y = Da::variable(2);
        let z = Da::variable(3);
        let f = 1.5 + 0.5 * (x.clone() * z.clone()) - 2.0 * y.clone();

        let blob = f.to_blob();
        // header (20) + one slot per term; empty DAs still reserve a slot
        assert_eq!(blob.len(), 20 + 16 * f.size().max(1));
        assert_eq!(
            u32::from_le_bytes(blob[0..4].try_into().unwrap()),
            0x1E304144
        );
        assert_eq!(u32::from_le_bytes(blob[4..8].try_into().unwrap()), 5);
        assert_eq!(u32::from_le_bytes(blob[8..12].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(blob[12..16].try_into().unwrap()), 1);
        assert_eq!(
            u32::from_le_bytes(blob[16..20].try_into().unwrap()) as usize,
            f.size()
        );

        let back = Da::from_blob(&blob).expect("imports");
        for m in f.iter_monomials() {
            assert!((back.get_coefficient(&m.jj) - m.c).abs() == 0.0);
        }

        // empty DA roundtrip
        let zb = Da::new().to_blob();
        assert_eq!(zb.len(), 36);
        assert_eq!(Da::from_blob(&zb).unwrap().size(), 0);

        // error paths
        assert!(Da::from_blob(&[0u8; 36]).is_err());
        assert!(Da::from_blob(&[0u8; 10]).is_err());

        // truncation on import: order-4 term dropped at nomax=3 context
        crate::context::init(3, 3).unwrap();
        let g = Da::monomial(&[2, 1, 0], 2.0) + Da::monomial(&[1, 1, 1], 1.0);
        let gblob = g.to_blob();
        let gback = Da::from_blob(&gblob).unwrap();
        assert_eq!(gback.get_coefficient(&[1, 1, 1]), 1.0);
        // extra variable: blob made with 3 vars, read with fewer
    }

    #[test]
    fn formatter_presets() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(3, 2).unwrap();
        let x = Da::variable(1);
        let y = Da::variable(2);
        let f =
            1.5 + 2.0 * x.clone() - 0.5 * (x.clone() * y.clone()) + 0.25 * (x.clone() * x.clone());

        let c = format_da(&f, &SimpleFormat::c());
        assert_eq!(c, "+1.5+2*p[0][0]-0.5*p[0][0]*p[1][0]+0.25*p[0][1]");

        let cp = format_da(&f, &SimpleFormat::c_pow());
        assert_eq!(cp, "+1.5+2*x[0]-0.5*x[0]*x[1]+0.25*pow(x[0],2)");

        let fo = format_da(&f, &SimpleFormat::fortran());
        assert_eq!(fo, "+1.5+2*p(1,1)-0.5*p(1,1)*p(2,1)+0.25*p(1,2)");

        let fp = format_da(&f, &SimpleFormat::fortran_pow());
        assert_eq!(fp, "+1.5+2*x(1)-0.5*x(1)*x(2)+0.25*x(1)**(2)");

        let ml = format_da(&f, &SimpleFormat::matlab());
        assert_eq!(ml, "+1.5+2.*p(1,1)-0.5.*p(1,1).*p(2,1)+0.25.*p(1,2)");

        let lx = format_da(&f, &SimpleFormat::latex());
        assert_eq!(
            lx,
            " +1.5 +2 \\cdot x_{1} -0.5 \\cdot x_{1} \\cdot x_{2} +0.25 \\cdot x_{1}^{2}"
        );
        let multi = format_das(&[f.clone(), f.clone()], &SimpleFormat::c());
        assert_eq!(multi.matches('\n').count(), 2);
    }
}
