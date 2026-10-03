//! Polynomial evaluation: compiled evaluation trees, partial evaluation,
//! and variable substitution.
//!
//! Ports `core/daceeval.c`: the Horner-style evaluation tree of
//! `daceEvalTree` (as the C++ `compiledDA`), partial evaluation of one
//! variable (`daceEvalVariable`), variable replacement/scaling/translation,
//! and the monomial-wise dot product `daceEvalMonomials`.

use std::sync::Arc;

use crate::context::{Context, npown_i64};
use crate::da::{Da, RawTerm};
use crate::kernels::{keep, multiply, weighted_sum};

/// Pack a dense coefficient array (length `nmmax`) into a sparse `Da`,
/// flushing `|c| <= eps` and re-zeroing the array (`dacePack`, fast path:
/// the truncation order is enforced by the callers, as in C).
pub(crate) fn pack(ctx: &Arc<Context>, cc: &mut [f64]) -> Da {
    let (eps, _nocut) = crate::context::eps_nocut();
    let mut terms = Vec::new();
    for (i, c) in cc.iter_mut().enumerate() {
        if keep(*c, eps) {
            terms.push(RawTerm {
                idx: i as u32,
                c: *c,
            });
        }
        *c = 0.0;
    }
    Da {
        ctx: ctx.clone(),
        terms,
    }
}

impl Da {
    /// Partial evaluation: replace independent variable `var` (1-based) by
    /// the value `val` (`daceEvalVariable`). Out-of-range variables warn
    /// and return zero.
    pub fn plug(&self, var: u32, val: f64) -> Da {
        let ctx = &self.ctx;
        if !(1..=ctx.nvmax).contains(&var) {
            log::warn!("DACE error 624: invalid independent variable {var} in plug");
            return Da::new();
        }
        let (_eps, nocut) = crate::context::eps_nocut();
        let ibase = ctx.nomax + 1;
        let j = if var > ctx.nv1 {
            var - 1 - ctx.nv1
        } else {
            var - 1
        };
        let idiv = npown_i64(ibase, j);
        let in_second_half = var > ctx.nv1;

        let mut p = vec![1.0; ctx.nomax as usize + 1];
        for i in 1..p.len() {
            p[i] = p[i - 1] * val;
        }
        let mut cc = vec![0.0; ctx.nmmax as usize];

        for t in &self.terms {
            let ic1 = ctx.ie1[t.idx as usize];
            let ic2 = ctx.ie2[t.idx as usize];
            let ipow = if in_second_half {
                (ic2 / idiv) % ibase
            } else {
                (ic1 / idiv) % ibase
            };
            let j = if in_second_half {
                ctx.ia1[ic1 as usize] + ctx.ia2[(ic2 - ipow * idiv) as usize]
            } else {
                ctx.ia1[(ic1 - ipow * idiv) as usize] + ctx.ia2[ic2 as usize]
            };
            if ctx.order_of(j) <= nocut {
                cc[j as usize] += t.c * p[ipow as usize];
            }
        }

        pack(ctx, &mut cc)
    }

    /// Replace independent variable `from` by `val` times independent
    /// variable `to` (`daceReplaceVariable`); `from == to` scales the
    /// variable by `val`. Out-of-range variables warn and return zero.
    ///
    /// Divergence from C: the C implementation indexes its 0-based exponent
    /// array with the 1-based variable numbers, so it actually replaces
    /// variable `from + 1` by `val · (variable to + 1)` and silently does
    /// nothing when `from == nvmax`. This implementation follows the
    /// documented (1-based) semantics instead.
    pub fn replace_variable(&self, from: u32, to: u32, val: f64) -> Da {
        let ctx = &self.ctx;
        if !(1..=ctx.nvmax).contains(&from) || !(1..=ctx.nvmax).contains(&to) {
            log::warn!("DACE error 624: invalid independent variable in replace_variable");
            return Da::new();
        }
        if from == to {
            return self.scale_variable(from, val);
        }

        let mut pows = vec![1.0; ctx.nomax as usize + 1];
        for i in 0..ctx.nomax as usize {
            pows[i + 1] = pows[i] * val;
        }
        let mut p = vec![0u32; ctx.nvmax as usize];
        let mut cc = vec![0.0; ctx.nmmax as usize];
        for t in &self.terms {
            ctx.decode_into(t.idx, &mut p);
            p[to as usize - 1] += p[from as usize - 1];
            let c = pows[p[from as usize - 1] as usize] * t.c;
            p[from as usize - 1] = 0;
            let idx = ctx.encode(&p).expect("order preserved by replacement");
            cc[idx as usize] += c;
        }
        pack(ctx, &mut cc)
    }

    /// Scale independent variable `var` by `val`: `x_var -> val * x_var`
    /// (`daceScaleVariable`). Out-of-range variables warn and return zero.
    pub fn scale_variable(&self, var: u32, val: f64) -> Da {
        let ctx = &self.ctx;
        if !(1..=ctx.nvmax).contains(&var) {
            log::warn!("DACE error 624: invalid independent variable {var} in scale_variable");
            return Da::new();
        }
        let mut pows = vec![1.0; ctx.nomax as usize + 1];
        for i in 0..ctx.nomax as usize {
            pows[i + 1] = pows[i] * val;
        }
        let ibase = ctx.nomax + 1;
        let j = if var > ctx.nv1 {
            var - 1 - ctx.nv1
        } else {
            var - 1
        };
        let idiv = npown_i64(ibase, j);
        let in_second_half = var > ctx.nv1;

        let mut terms = self.terms.clone();
        for t in terms.iter_mut() {
            let ipow = if in_second_half {
                (ctx.ie2[t.idx as usize] / idiv) % ibase
            } else {
                (ctx.ie1[t.idx as usize] / idiv) % ibase
            };
            t.c *= pows[ipow as usize];
        }
        Da {
            ctx: ctx.clone(),
            terms,
        }
    }

    /// Translate independent variable `var` to `a*x + c`
    /// (`daceTranslateVariable`). Out-of-range variables warn and return
    /// zero.
    pub fn translate_variable(&self, var: u32, a: f64, c: f64) -> Da {
        let ctx = &self.ctx;
        if !(1..=ctx.nvmax).contains(&var) {
            log::warn!("DACE error 624: invalid independent variable {var} in translate_variable");
            return Da::new();
        }
        let n1 = ctx.nomax as usize;

        let mut powa = vec![1.0; n1 + 1];
        let mut powc = vec![1.0; n1 + 1];
        for i in 0..n1 {
            powa[i + 1] = powa[i] * a;
            powc[i + 1] = powc[i] * c;
        }

        // binomial coefficients n choose k
        let mut binomial = vec![0.0; (n1 + 1) * (n1 + 1)];
        for n in 0..=n1 {
            binomial[n * (n1 + 1)] = 1.0;
            binomial[n * (n1 + 1) + n] = 1.0;
            for k in 1..n {
                binomial[n * (n1 + 1) + k] =
                    binomial[(n - 1) * (n1 + 1) + k - 1] + binomial[(n - 1) * (n1 + 1) + k];
            }
        }

        let mut p = vec![0u32; ctx.nvmax as usize];
        let mut cc = vec![0.0; ctx.nmmax as usize];
        for t in &self.terms {
            ctx.decode_into(t.idx, &mut p);
            let n = p[(var - 1) as usize];

            // shortcut the case when the monomial doesn't depend on var
            if n == 0 {
                cc[t.idx as usize] += t.c;
                continue;
            }

            for k in 0..=n {
                let idx = ctx.encode(&p).expect("order preserved by translation");
                cc[idx as usize] += t.c
                    * binomial[n as usize * (n1 + 1) + k as usize]
                    * powa[(n - k) as usize]
                    * powc[k as usize];
                // C decrements unconditionally (unsigned wrap), but the
                // value is only read after the next decode; guard instead.
                if p[(var - 1) as usize] > 0 {
                    p[(var - 1) as usize] -= 1;
                }
            }
        }

        pack(ctx, &mut cc)
    }

    /// Evaluate by providing the value of each monomial in `values`: the
    /// monomial-wise dot product of the two DAs (`daceEvalMonomials`).
    pub fn eval_monomials(&self, values: &Da) -> f64 {
        Da::assert_same_context(self, values);
        let mut res = 0.0;
        let mut ib = values.terms.iter().peekable();
        for ta in &self.terms {
            while let Some(tb) = ib.peek() {
                if tb.idx < ta.idx {
                    ib.next();
                } else {
                    break;
                }
            }
            match ib.peek() {
                Some(tb) if tb.idx == ta.idx => res += tb.c * ta.c,
                Some(_) => {}
                None => break,
            }
        }
        res
    }

    /// Evaluate at a point: compile and evaluate the tree, returning the
    /// first component (C++ `DA::eval`).
    pub fn eval(&self, args: &[f64]) -> f64 {
        self.compile().eval(args)[0]
    }

    /// Evaluate with DA arguments (contraction): substitute each argument
    /// DA for the corresponding variable.
    pub fn eval_da(&self, args: &[Da]) -> Da {
        self.compile().eval_da(args)[0].clone()
    }

    /// Compile into a reusable evaluation tree (C++ `compiledDA`).
    pub fn compile(&self) -> CompiledDa {
        CompiledDa::from_das(std::slice::from_ref(self))
    }
}

/// A compiled evaluation tree over one or more DAs (C++ `compiledDA`):
/// precomputed Horner-tree coefficients that can be evaluated repeatedly
/// at low cost.
#[derive(Debug, Clone)]
pub struct CompiledDa {
    /// Number of component DAs.
    pub dim: u32,
    /// Maximum order of the tree.
    pub ord: u32,
    /// Number of variables used.
    pub vars: u32,
    /// Number of terms (including the root).
    pub terms: u32,
    /// Packed coefficients: two unused slots, `dim` constants, then per
    /// term `(level, variable, dim coefficients)` with 1-based indices.
    pub(crate) ac: Vec<f64>,
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CompiledDa>();
};

impl CompiledDa {
    /// Compile several DAs (same context) into one shared tree
    /// (`daceEvalTree`).
    ///
    /// # Panics
    ///
    /// Panics with [`crate::DaceError`] when the DAs belong to different contexts.
    pub fn from_das(das: &[Da]) -> CompiledDa {
        for da in das {
            Da::assert_same_context(&das[0], da);
        }
        let ctx = das[0].ctx.clone();
        let count = das.len();
        let mut nc = vec![0u32; ctx.nmmax as usize];

        // mark all used monomials as new
        for da in das {
            for t in &da.terms {
                nc[t.idx as usize] = 2;
            }
        }

        // make sure each term has a parent
        nc[0] = 1; // constant part is the root, doesn't need a parent
        let mut p = vec![0u32; ctx.nvmax as usize];
        for i in 1..ctx.nmmax as usize {
            if nc[i] != 2 {
                continue;
            }
            nc[i] = 1;
            ctx.decode_into(i as u32, &mut p);
            // generate an ancestor tree for this entry
            let mut parent: i64;
            loop {
                parent = -1;
                // find a parent
                for j in 0..ctx.nvmax as usize {
                    if p[j] == 0 {
                        continue;
                    }
                    p[j] -= 1;
                    if nc[ctx.encode(&p).expect("valid parent") as usize] != 0 {
                        // parent already exists => done
                        parent = -1;
                        break;
                    }
                    p[j] += 1;
                    parent = j as i64;
                }
                // no parent found => create foster parent
                if parent >= 0 {
                    p[parent as usize] -= 1;
                    nc[ctx.encode(&p).expect("valid foster parent") as usize] = 1;
                } else {
                    break;
                }
            }
        }

        // constant terms are always stored
        nc[0] = 3;
        let mut nord = 0u32;
        let mut nvar = 0u32;
        let mut nterm = 1u32;
        let mut ac: Vec<f64> = Vec::new();
        ac.push(0.0);
        ac.push(0.0);
        for da in das {
            ac.push(da.cons());
        }

        // higher order terms
        p[0] = 1;
        for slot in p.iter_mut().skip(1) {
            *slot = 0;
        }
        let mut stack = vec![0u32; ctx.nomax as usize];
        let mut sp: i64 = 0;
        stack[0] = 0;
        while sp >= 0 {
            let ic = ctx.encode(&p).expect("valid monomial");
            if nc[ic as usize] == 1 {
                // store entry
                nc[ic as usize] = 3;
                nord = nord.max(sp as u32 + 1);
                nvar = nvar.max(stack[sp as usize] + 1);
                nterm += 1;
                ac.push((sp + 1) as f64); // +1 for 1-based indices (as in C)
                ac.push(f64::from(stack[sp as usize] + 1));
                for da in das {
                    ac.push(da.get_coefficient0(ic));
                }

                // step forward if we can
                if sp < ctx.nomax as i64 - 1 {
                    sp += 1;
                    stack[sp as usize] = 0;
                    p[0] += 1;
                    continue;
                }
            }
            if stack[sp as usize] < ctx.nvmax - 1 {
                // step sideways
                let s = stack[sp as usize];
                p[s as usize] -= 1;
                stack[sp as usize] = s + 1;
                p[(s + 1) as usize] += 1;
            } else {
                // step back
                let s = stack[sp as usize];
                p[s as usize] -= 1;
                sp -= 1;
            }
        }

        CompiledDa {
            dim: count as u32,
            ord: nord,
            vars: nvar,
            terms: nterm,
            ac,
        }
    }

    /// Evaluate the tree at a point (`compiledDA::eval<double>`): one
    /// result per component DA.
    pub fn eval(&self, args: &[f64]) -> Vec<f64> {
        let narg = args.len();
        let mut p = self.ac.iter().skip(2);
        let mut xm = vec![0.0; self.ord as usize + 1];

        // prepare temporary powers
        xm[0] = 1.0;
        // constant part
        let mut res: Vec<f64> = p.by_ref().take(self.dim as usize).copied().collect();
        // higher order terms
        for _ in 1..self.terms {
            let jl = *p.next().expect("tree level") as usize;
            let jv = *p.next().expect("tree variable") as usize - 1;
            xm[jl] = if jv < narg {
                xm[jl - 1] * args[jv]
            } else {
                0.0
            };
            for r in res.iter_mut().take(self.dim as usize) {
                *r += xm[jl] * p.next().expect("tree coefficient");
            }
        }
        res
    }

    /// Evaluate the tree with DA arguments (contraction,
    /// `compiledDA::eval<DA>`), including the skip logic for unused
    /// variables.
    pub fn eval_da(&self, args: &[Da]) -> Vec<Da> {
        let narg = args.len();
        let mut jlskip = self.ord + 1;
        let mut p = self.ac.iter().skip(2);
        let mut xm: Vec<Da> = (0..=self.ord).map(|_| Da::new()).collect();

        // prepare temporary powers
        xm[0] = Da::constant(1.0);
        // constant part
        let mut res: Vec<Da> = p
            .by_ref()
            .take(self.dim as usize)
            .map(|c| Da::constant(*c))
            .collect();
        // higher order terms
        for _ in 1..self.terms {
            let jl = *p.next().expect("tree level") as u32;
            let jv = *p.next().expect("tree variable") as u32 - 1;
            if jl > jlskip {
                p.by_ref().take(self.dim as usize).for_each(drop);
                continue;
            }
            if jv as usize >= narg {
                jlskip = jl;
                p.by_ref().take(self.dim as usize).for_each(drop);
                continue;
            }
            jlskip = self.ord + 1;
            xm[jl as usize] = multiply(&xm[(jl - 1) as usize], &args[jv as usize]);
            for r in res.iter_mut().take(self.dim as usize) {
                let coef = *p.next().expect("tree coefficient");
                if coef != 0.0 {
                    *r = weighted_sum(r, 1.0, &xm[jl as usize], coef);
                }
            }
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::CONTEXT_LOCK;

    #[test]
    fn eval_matches_direct_evaluation() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(5, 3).unwrap();
        let x = Da::variable(1);
        let y = Da::variable(2);
        let z = Da::variable(3);

        let f = 1.0 + 2.0 * x.clone() * y.clone() - 0.5 * z.clone() * z.clone() + 0.25 * x.clone();

        // direct polynomial evaluation at a point
        let (px, py, pz) = (1.3, -0.7, 0.9);
        let direct = 1.0 + 2.0 * px * py - 0.5 * pz * pz + 0.25 * px;
        assert!((f.eval(&[px, py, pz]) - direct).abs() < 1e-12);

        // CompiledDa eval agrees with Da::eval on random points
        let compiled = f.compile();
        for trial in 0..10 {
            let t = 0.1 * trial as f64;
            let args = [t, -0.3 + 0.05 * t, 0.7 - 0.1 * t];
            assert!((compiled.eval(&args)[0] - f.eval(&args)).abs() < 1e-13);
        }

        // multi-component compilation
        let g = y.clone() - z.clone();
        let c2 = CompiledDa::from_das(&[f.clone(), g.clone()]);
        let r = c2.eval(&[px, py, pz]);
        assert_eq!(r.len(), 2);
        assert!((r[1] - (py - pz)).abs() < 1e-13);

        // eval_da contraction: f(x, y, x) with z := x
        let sub = f.eval_da(&[x.clone(), y.clone(), x.clone()]);
        // f(x,y,x) coefficient checks: 2xy term unchanged, -0.5x^2 replaces -0.5z^2
        assert!((sub.get_coefficient(&[1, 1, 0]) - 2.0).abs() < 1e-13);
        assert!((sub.get_coefficient(&[2, 0, 0]) + 0.5).abs() < 1e-13);

        // plug then eval == eval with that coordinate fixed
        let plugged = f.plug(2, py);
        assert!((plugged.eval(&[px, 0.0, pz]) - direct).abs() < 1e-12);

        // translate_variable on x by c gives cons() == c
        let t = x.clone().translate_variable(1, 1.0, 0.75);
        assert!((t.cons() - 0.75).abs() < 1e-15);
        assert!((t.get_coefficient(&[1, 0, 0]) - 1.0).abs() < 1e-15);

        // scale_variable multiplies the right exponents
        let s = (x.clone() * y.clone()).scale_variable(1, 3.0);
        assert!((s.get_coefficient(&[1, 1, 0]) - 3.0).abs() < 1e-15);

        // replace_variable y -> 2x
        let r = (x.clone() * y.clone()).replace_variable(2, 1, 2.0);
        assert!((r.get_coefficient(&[2, 0, 0]) - 2.0).abs() < 1e-15);
        assert_eq!(r.size(), 1);

        // eval_monomials == monomial-wise dot product
        let a = 1.0 + x.clone();
        let b = 2.0 + 3.0 * x.clone();
        assert!((a.eval_monomials(&b) - (1.0 * 2.0 + 1.0 * 3.0)).abs() < 1e-15);
    }
}
