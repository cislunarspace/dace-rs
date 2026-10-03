//! Sparse computation kernels: weighted sums and multiplication.
//!
//! Ports the merge kernels of `core/dacemath.c` (`daceWeightedSum`,
//! `daceMultiply`, `dacePack`) with identical loop structure, accumulation
//! order, and flush rules (`|c| <= eps` dropped, `ieo > nocut` skipped), so
//! results are bit-identical to the C library on the same platform.
//!
//! Unlike the C kernels, results are built into fresh `Vec`s, so all of these
//! are aliasing-safe by construction (C's `daceWeightedSum` is not).

use std::cell::RefCell;

use crate::context::{count_monomials, eps_nocut};
use crate::da::{Da, RawTerm};

/// Flush threshold check exactly as in C: keep when `|c| > eps`.
///
/// Written as the negated `<=` (not `>`) to preserve the C library's
/// NaN behavior: `!(NaN <= eps)` is true, so NaN coefficients are kept.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
#[inline]
pub(crate) fn keep(c: f64, eps: f64) -> bool {
    !(c.abs() <= eps)
}

/// Weighted sum `afac * a + bfac * b` over the sorted sparse lists
/// (`daceWeightedSum`, dacemath.c:2037). Aliasing-safe.
pub(crate) fn weighted_sum(a: &Da, afac: f64, b: &Da, bfac: f64) -> Da {
    let ctx = a.ctx.clone();
    let (eps, nocut) = eps_nocut();
    let nocut = nocut.min(ctx.nomax);
    let mut terms: Vec<RawTerm> = Vec::with_capacity(a.terms.len() + b.terms.len());

    let mut ia = a.terms.iter().peekable();
    let mut ib = b.terms.iter().peekable();
    loop {
        match (ia.peek(), ib.peek()) {
            (Some(&ta), Some(&tb)) => {
                if ta.idx == tb.idx {
                    if ctx.order_of(ta.idx) <= nocut {
                        let ccc = ta.c * afac + tb.c * bfac;
                        if keep(ccc, eps) {
                            terms.push(RawTerm {
                                idx: ta.idx,
                                c: ccc,
                            });
                        }
                    }
                    ia.next();
                    ib.next();
                } else if ta.idx < tb.idx {
                    if ctx.order_of(ta.idx) <= nocut {
                        let ccc = ta.c * afac;
                        if keep(ccc, eps) {
                            terms.push(RawTerm {
                                idx: ta.idx,
                                c: ccc,
                            });
                        }
                    }
                    ia.next();
                } else {
                    if ctx.order_of(tb.idx) <= nocut {
                        let ccc = tb.c * bfac;
                        if keep(ccc, eps) {
                            terms.push(RawTerm {
                                idx: tb.idx,
                                c: ccc,
                            });
                        }
                    }
                    ib.next();
                }
            }
            (Some(&ta), None) => {
                if ctx.order_of(ta.idx) <= nocut {
                    let ccc = ta.c * afac;
                    if keep(ccc, eps) {
                        terms.push(RawTerm {
                            idx: ta.idx,
                            c: ccc,
                        });
                    }
                }
                ia.next();
            }
            (None, Some(&tb)) => {
                if ctx.order_of(tb.idx) <= nocut {
                    let ccc = tb.c * bfac;
                    if keep(ccc, eps) {
                        terms.push(RawTerm {
                            idx: tb.idx,
                            c: ccc,
                        });
                    }
                }
                ib.next();
            }
            (None, None) => break,
        }
    }

    Da { ctx, terms }
}

/// A `b`-term bucketed by order, carrying its encoded exponent halves
/// (the C `extended_monomial`).
struct BucketEntry {
    i1: u32,
    i2: u32,
    cc: f64,
}

/// Thread-local multiply scratch (replaces the C static TLS buffers of
/// `daceMultiply`, dacemath.c:129-153), invalidated by context generation.
struct MulScratch {
    generation: u64,
    /// Dense coefficient accumulator, length `nmmax`, zero between calls.
    cc: Vec<f64>,
    /// Order-bucketed copy of `b`'s terms (the C `emb` array).
    emb: Vec<BucketEntry>,
    /// Bucket start offsets by order (the C `ipbeg`).
    ipbeg: Vec<usize>,
}

thread_local! {
    static MULSCRATCH: RefCell<Option<MulScratch>> = const { RefCell::new(None) };
}

/// Multiply two DAs (`daceMultiply`, dacemath.c:107-203): bucket `b` by order,
/// then for each `a` term accumulate `cc[ic] += cia*cib` over order-descending
/// buckets; finally pack (iterate ascending, keep `|c| > eps`, re-zero).
/// Aliasing-safe.
pub(crate) fn multiply(a: &Da, b: &Da) -> Da {
    let ctx = a.ctx.clone();
    let (eps, nocut) = eps_nocut();
    let nocut = nocut.min(ctx.nomax);
    let nomax = ctx.nomax as usize;
    let nmmax = ctx.nmmax as usize;

    // Sort so that a is the shorter list (as in C).
    let (a, b) = if a.terms.len() > b.terms.len() {
        (b, a)
    } else {
        (a, b)
    };

    MULSCRATCH.with(|s| {
        let mut slot = s.borrow_mut();
        let scratch = slot.get_or_insert_with(|| MulScratch {
            generation: 0,
            cc: Vec::new(),
            emb: Vec::new(),
            ipbeg: Vec::new(),
        });
        if scratch.generation != ctx.generation {
            scratch.generation = ctx.generation;
            scratch.cc = vec![0.0; nmmax];
            scratch.emb = (0..nmmax)
                .map(|_| BucketEntry {
                    i1: 0,
                    i2: 0,
                    cc: 0.0,
                })
                .collect();
            scratch.ipbeg = Vec::with_capacity(nomax + 1);
            scratch.ipbeg.push(0usize);
            // ipbeg[i] = number of monomials of order < i (absolute offset,
            // exactly C's emb + daceCountMonomials(i-1, nv)).
            for i in 1..=ctx.nomax {
                scratch
                    .ipbeg
                    .push(count_monomials(i - 1, ctx.nvmax) as usize);
            }
        }
        // Zero the accumulator defensively (C maintains this invariant via
        // dacePack; an explicit fill also stays correct after a panic).
        scratch.cc.iter_mut().for_each(|c| *c = 0.0);

        // Bucket b's terms by order, appended in b's sorted order.
        let mut ipend: Vec<usize> = scratch.ipbeg.clone();
        for tb in &b.terms {
            let noib = ctx.order_of(tb.idx);
            if noib > nocut {
                continue;
            }
            let slot = &mut scratch.emb[ipend[noib as usize]];
            slot.i1 = ctx.ie1[tb.idx as usize];
            slot.i2 = ctx.ie2[tb.idx as usize];
            slot.cc = tb.c;
            ipend[noib as usize] += 1;
        }

        // Perform the multiplication: for each a term, walk b's buckets from
        // order nocut - order(a) down to 0.
        for ta in &a.terms {
            let i1ia = ctx.ie1[ta.idx as usize];
            let i2ia = ctx.ie2[ta.idx as usize];
            let ccia = ta.c;
            let noia = ctx.order_of(ta.idx);
            let mut noib = nocut as i64 - noia as i64;
            while noib >= 0 {
                for ib in scratch.ipbeg[noib as usize]..ipend[noib as usize] {
                    let tb = &scratch.emb[ib];
                    let ic = ctx.ia1[(i1ia + tb.i1) as usize] + ctx.ia2[(i2ia + tb.i2) as usize];
                    scratch.cc[ic as usize] += ccia * tb.cc;
                }
                noib -= 1;
            }
        }

        // Pack (ascending) and leave the accumulator zeroed, as dacePack does.
        let mut terms = Vec::new();
        for i in 0..nmmax {
            let c = scratch.cc[i];
            if keep(c, eps) {
                terms.push(RawTerm { idx: i as u32, c });
            }
            scratch.cc[i] = 0.0;
        }
        Da { ctx, terms }
    })
}
