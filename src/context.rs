//! DACE computation context: initialization, monomial index encoding, and
//! thread-local computation settings.
//!
//! A DACE computation is parameterized by a maximum computation order `no` and
//! a number of variables `nv`. Monomials are stored as sparse lists of
//! `{coefficient, index}` pairs, where the index is a canonical packed
//! position `0..nmmax-1` built from a base-`(no+1)` digit encoding of the
//! exponent vector, split into two halves of `nv1 = (nv+1)/2` and
//! `nv2 = nv - nv1` variables (this is what makes the reverse lookup tables
//! fit in 32 bits for useful `(no, nv)` combinations).
//!
//! This module ports the C library's `daceInitialize` (`core/daceinit.c`) and
//! the encoding machinery of `core/daceaux.c`. Divergences from the C library:
//!
//! - [`init`] may be called again at any time; previously created [`Da`][crate::Da]
//!   values keep working against their original context (in C,
//!   `daceInitialize` purges all existing DA objects).
//! - the truncation-order stack ([`push_truncation_order`]/[`pop_truncation_order`])
//!   is thread-local, like all other computation settings.
//! - informational messages go through the [`log`] crate instead of stderr.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};

use parking_lot::RwLock;

use crate::error::{DaceError, codes};

/// Immutable computation context: lookup tables and limits.
///
/// Built by [`init`]; shared (via `Arc`) by every [`Da`][crate::Da] created
/// while it is the active context.
// Encoding/decoding machinery is wired into `Da` in Phase 2 and the
// computation kernels in Phase 3; until then the fields are only exercised
// by the tests below.
#[allow(dead_code)]
pub(crate) struct Context {
    pub nomax: u32,
    pub nvmax: u32,
    pub nv1: u32,
    pub nv2: u32,
    pub nmmax: u32,
    pub epsmac: f64,
    /// Encoded exponents of the first `nv1` variables, by monomial index
    /// (length `nmmax`).
    pub ie1: Vec<u32>,
    /// Encoded exponents of the last `nv2` variables, by monomial index
    /// (length `nmmax`).
    pub ie2: Vec<u32>,
    /// Total order of each monomial index (length `nmmax`).
    pub ieo: Vec<u32>,
    /// Base monomial index by encoded exponents of the first half
    /// (length `lia+1`).
    pub ia1: Vec<u32>,
    /// Monomial offset by encoded exponents of the second half
    /// (length `lia+1`).
    pub ia2: Vec<u32>,
    /// Generation of the global context this snapshot belongs to; used to
    /// invalidate thread-local scratch buffers after re-initialization.
    pub generation: u64,
}

#[allow(dead_code)] // encode/decode/order_of are wired into Da in Phase 2
impl Context {
    /// Encode an exponent vector (length `nv`) into its monomial index.
    ///
    /// Returns `None` if the slice length differs from `nv`, any single
    /// exponent exceeds `nomax` (unrepresentable in base `nomax+1`), or the
    /// total order exceeds `nomax` (C error 622).
    pub(crate) fn encode(&self, jj: &[u32]) -> Option<u32> {
        if jj.len() != self.nvmax as usize {
            return None;
        }
        let base = self.nomax + 1;
        let mut io: u32 = 0;
        let mut ic1: u32 = 0;
        let mut ic2: u32 = 0;
        for &e in jj[self.nv1 as usize..].iter().rev() {
            if e > self.nomax {
                return None;
            }
            ic2 = ic2 * base + e;
            io += e;
        }
        for &e in jj[..self.nv1 as usize].iter().rev() {
            if e > self.nomax {
                return None;
            }
            ic1 = ic1 * base + e;
            io += e;
        }
        if io > self.nomax {
            return None;
        }
        Some(self.ia1[ic1 as usize] + self.ia2[ic2 as usize])
    }

    /// Decode a monomial index into its exponent vector (length `nv`).
    ///
    /// # Panics
    ///
    /// Panics if `ii >= nmmax` (C error 626, invalid encoded exponent).
    pub(crate) fn decode(&self, ii: u32) -> Vec<u32> {
        let mut jj = vec![0u32; self.nvmax as usize];
        self.decode_into(ii, &mut jj);
        jj
    }

    /// Decode a monomial index into a caller-provided exponent vector.
    pub(crate) fn decode_into(&self, ii: u32, jj: &mut [u32]) {
        assert!(jj.len() >= self.nvmax as usize, "decode buffer too short");
        if ii >= self.nmmax {
            crate::error::dace_panic(codes::INVALID_ENCODED_EXPONENT, "Invalid encoded exponent");
        }
        let base = self.nomax + 1;
        let mut ic = self.ie1[ii as usize];
        for slot in jj[..self.nv1 as usize].iter_mut() {
            *slot = ic % base;
            ic /= base;
        }
        let mut ic = self.ie2[ii as usize];
        for slot in jj[self.nv1 as usize..self.nvmax as usize].iter_mut() {
            *slot = ic % base;
            ic /= base;
        }
    }

    /// Total order of the monomial with the given index.
    pub(crate) fn order_of(&self, ii: u32) -> u32 {
        self.ieo[ii as usize]
    }
}

static CONTEXT: LazyLock<RwLock<Option<Arc<Context>>>> = LazyLock::new(|| RwLock::new(None));
static GENERATION: AtomicU64 = AtomicU64::new(0);

impl Context {
    /// The active context.
    ///
    /// # Panics
    ///
    /// Panics with [`DaceError`] code 1003 if DACE has not been initialized.
    pub(crate) fn current() -> Arc<Context> {
        match CONTEXT.read().clone() {
            Some(ctx) => ctx,
            None => {
                crate::error::dace_panic(codes::NOT_INITIALIZED, "DACE has not been initialized")
            }
        }
    }
}

/// Whether DACE has been initialized on this process.
pub fn initialized() -> bool {
    CONTEXT.read().is_some()
}

/// DACE version this crate reproduces (`"2.1.0-rs"`).
pub fn version() -> &'static str {
    "2.1.0-rs"
}

/// Initialize DACE with computation order `order` and `nvars` variables,
/// replacing any previous context.
///
/// Values of `order`/`nvars` below 1 are clamped to 1 with a warning (C
/// informational messages 167/168). Returns an error if the required lookup
/// tables do not fit in a 32-bit index space (C error 911).
///
/// Unlike the C library, re-initializing does **not** invalidate existing
/// [`Da`][crate::Da] values: they keep operating on their original context.
/// Computation settings (epsilon cutoff, truncation order) are re-initialized
/// on the calling thread only, matching C's thread model.
pub fn init(order: u32, nvars: u32) -> Result<(), DaceError> {
    let mut no = order;
    let mut nv = nvars;
    if no < 1 {
        log::warn!("DACE info 167: computation order increased to 1");
        no = 1;
    }
    if nv < 1 {
        log::warn!("DACE info 168: number of variables increased to 1");
        nv = 1;
    }

    // Machine epsilon, computed as in the C library.
    let mut epsmac = 1.0f64;
    while 1.0 + epsmac > 1.0 {
        epsmac /= 2.0;
    }
    epsmac *= 2.0;

    // Length of the reverse lookup arrays must fit the 32 bit index space.
    let nv1 = nv.div_ceil(2);
    let clia = pown(f64::from(no + 1), nv1);
    if clia >= pown(2.0, 32) {
        return Err(DaceError::new(
            codes::ORDER_VARIABLE_TOO_LARGE,
            "Order and/or variable too large",
        ));
    }
    let lia = clia as u32;
    let nmmax = count_monomials(no, nv);

    let mut ie1 = vec![0u32; nmmax as usize];
    let mut ie2 = vec![0u32; nmmax as usize];
    let mut ieo = vec![0u32; nmmax as usize];
    let mut ia1 = vec![0u32; lia as usize + 1];
    let mut ia2 = vec![0u32; lia as usize + 1];

    // Fill the addressing arrays, enumerating ordered monomials exactly like
    // the C implementation (core/daceinit.c:139-152).
    let nv2 = nv - nv1;
    let mut p1 = vec![0u32; nv1 as usize];
    let mut p2 = vec![0u32; nv2 as usize];
    let mut i: u32 = 0;
    let mut no1: u32;
    let mut no2: u32;
    loop {
        let exp1 = encode_exponents(&p1, no);
        let i0 = i;
        ia1[exp1 as usize] = i0;
        no1 = p1.iter().sum();
        loop {
            ie1[i as usize] = exp1;
            let exp2 = encode_exponents(&p2, no);
            ie2[i as usize] = exp2;
            ieo[i as usize] = no1 + p2.iter().sum::<u32>();
            ia2[exp2 as usize] = i - i0;
            i += 1;
            no2 = next_ordered_monomial(&mut p2, no - no1);
            if no2 == 0 {
                break;
            }
        }
        no1 = next_ordered_monomial(&mut p1, no);
        if no1 == 0 {
            break;
        }
    }

    // Cross-checks mirroring the C PANIC 5/6 internal invariants.
    if i != nmmax {
        crate::error::dace_panic(1005, "Incorrect number of monomials");
    }
    for i in 0..nmmax as usize {
        let nn = ia1[ie1[i] as usize] + ia2[ie2[i] as usize];
        if nn != i as u32 {
            crate::error::dace_panic(1006, "Incorrect DA coding arrays");
        }
    }

    let generation = GENERATION.fetch_add(1, Ordering::SeqCst) + 1;
    let ctx = Arc::new(Context {
        nomax: no,
        nvmax: nv,
        nv1,
        nv2,
        nmmax,
        epsmac,
        ie1,
        ie2,
        ieo,
        ia1,
        ia2,
        generation,
    });
    *CONTEXT.write() = Some(ctx);

    // Re-initialize the calling thread's settings (C daceInitializeThread0).
    SETTINGS.with(|s| {
        s.eps.set(0.0);
        s.nocut.set(no);
        s.ready.set(true);
        s.stack.borrow_mut().clear();
    });
    Ok(())
}

/// Raise `a` to the positive integer power `b` (binary exponentiation, as in
/// the C library's `pown`).
pub(crate) fn pown(a: f64, b: u32) -> f64 {
    let mut res = 1.0;
    let mut a = a;
    let mut b = b;
    while b > 0 {
        if b & 1 != 0 {
            res *= a;
        }
        a *= a;
        b >>= 1;
    }
    res
}

/// Number of monomials of maximum order `no` in `nv` variables, i.e.
/// `C(no+nv, min(no,nv))` (the C library's `daceCountMonomials`).
pub(crate) fn count_monomials(no: u32, nv: u32) -> u32 {
    let mut dnumda = 1.0f64;
    let mm = nv.max(no);
    for i in 1..=nv.min(no) {
        dnumda = dnumda * f64::from(mm + i) / f64::from(i);
    }
    dnumda as u32
}

/// Encode `nv` exponents (each at most `no`) into one base-`(no+1)` integer
/// (the C library's `daceEncodeExponents`).
fn encode_exponents(p: &[u32], no: u32) -> u32 {
    if p.is_empty() {
        return 0;
    }
    let base = no + 1;
    let mut res = p[p.len() - 1];
    for &e in p[..p.len() - 1].iter().rev() {
        res = res * base + e;
    }
    res
}

/// Advance `p` to the next monomial of `nv` variables in arbitrary order
/// (the C library's `daceNextMonomial`); returns the new order, or 0 when
/// wrapping back to the constant monomial.
fn next_monomial(p: &mut [u32], no: u32) -> u32 {
    let mut o: u32 = p.iter().sum();
    for e in p.iter_mut() {
        if o < no {
            *e += 1;
            return o + 1;
        }
        o -= *e;
        *e = 0;
    }
    0
}

/// Advance `p` to the next monomial in order-sorted enumeration (the C
/// library's `daceNextOrderedMonomial`).
fn next_ordered_monomial(p: &mut [u32], no: u32) -> u32 {
    if p.is_empty() || no == 0 {
        return 0;
    }
    let mut o: u32 = p.iter().sum();
    let oo = next_monomial(&mut p[1..], o);
    if oo == 0 {
        o = (o + 1) % (no + 1); // jump to next order
    }
    p[0] = o - oo; // complete the monomial up to order o
    o
}

// ---------------------------------------------------------------------------
// Thread-local computation settings (C DACECom_t)
// ---------------------------------------------------------------------------

struct Settings {
    eps: Cell<f64>,
    nocut: Cell<u32>,
    ready: Cell<bool>,
    stack: RefCell<Vec<u32>>,
}

thread_local! {
    static SETTINGS: Settings = const {
        Settings {
            eps: Cell::new(0.0),
            nocut: Cell::new(0),
            ready: Cell::new(false),
            stack: RefCell::new(Vec::new()),
        }
    };
}

/// Lazily initialize this thread's settings from the active context on first
/// use (C `daceInitializeThread0`: eps = 0, nocut = nomax).
fn with_settings<R>(f: impl FnOnce(&Settings) -> R) -> R {
    SETTINGS.with(|s| {
        if !s.ready.get() {
            let ctx = Context::current();
            s.eps.set(0.0);
            s.nocut.set(ctx.nomax);
            s.ready.set(true);
        }
        f(s)
    })
}

/// Current coefficient cutoff epsilon (coefficients with `|c| <= eps` are
/// flushed to zero).
///
/// Initialized to `0.0` (cutoff disabled).
pub fn epsilon() -> f64 {
    with_settings(|s| s.eps.get())
}

/// Set the coefficient cutoff epsilon to `eps` (its absolute value is used)
/// and return the previous value.
///
/// # Warning
///
/// Flushing occurs for any intermediate result also within the engine, and
/// can produce wrong results whenever DA coefficients become very small
/// relative to epsilon (e.g. a division by a large DA divisor can flush the
/// internally computed inverse entirely to zero).
pub fn set_epsilon(eps: f64) -> f64 {
    with_settings(|s| {
        let old = s.eps.get();
        s.eps.set(eps.abs());
        old
    })
}

/// The experimentally determined machine epsilon of the active context.
///
/// # Panics
///
/// Panics if DACE has not been initialized.
pub fn machine_epsilon() -> f64 {
    Context::current().epsmac
}

/// The maximum computation order of the active context.
///
/// # Panics
///
/// Panics if DACE has not been initialized.
pub fn max_order() -> u32 {
    Context::current().nomax
}

/// The number of variables of the active context.
///
/// # Panics
///
/// Panics if DACE has not been initialized.
pub fn max_variables() -> u32 {
    Context::current().nvmax
}

/// The total number of monomials of the active context.
///
/// # Panics
///
/// Panics if DACE has not been initialized.
pub fn max_monomials() -> u32 {
    Context::current().nmmax
}

/// The current truncation order (order above which computed terms are dropped).
pub fn truncation_order() -> u32 {
    with_settings(|s| s.nocut.get())
}

/// Set the truncation order, clamped to `[1, nomax]` (with a warning when
/// clamping, C informational message 162), and return the previous value.
pub fn set_truncation_order(order: u32) -> u32 {
    with_settings(|s| {
        let ctx = Context::current();
        if order > ctx.nomax {
            log::warn!(
                "DACE info 162: truncation order too high, clamping to {}",
                ctx.nomax
            );
        }
        let old = s.nocut.get();
        s.nocut.set(order.min(ctx.nomax).max(1));
        old
    })
}

/// Push the current truncation order on this thread's stack and set a new one
/// (clamped to `[1, nomax]`).
pub fn push_truncation_order(order: u32) {
    with_settings(|s| {
        let ctx = Context::current();
        if order > ctx.nomax {
            log::warn!(
                "DACE info 162: truncation order too high, clamping to {}",
                ctx.nomax
            );
        }
        s.stack.borrow_mut().push(s.nocut.get());
        s.nocut.set(order.min(ctx.nomax).max(1));
    });
}

/// Pop the truncation order stack, restoring the value saved by the matching
/// [`push_truncation_order`].
///
/// # Panics
///
/// Panics (C error 161) if the stack is empty.
pub fn pop_truncation_order() {
    with_settings(|s| match s.stack.borrow_mut().pop() {
        Some(nocut) => s.nocut.set(nocut),
        None => crate::error::dace_panic(161, "Free or invalid variable"),
    });
}

/// Read the active settings (eps, nocut) in one go; internal helper for the
/// computation kernels.
#[allow(dead_code)] // used by the computation kernels from Phase 3
pub(crate) fn eps_nocut() -> (f64, u32) {
    with_settings(|s| (s.eps.get(), s.nocut.get()))
}

/// Current context generation; internal helper for scratch invalidation.
#[allow(dead_code)] // used by thread-local scratch invalidation from Phase 3
pub(crate) fn generation() -> u64 {
    GENERATION.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::CONTEXT_LOCK;
    fn binom(n: u64, k: u64) -> u64 {
        let mut r = 1u64;
        for i in 1..=k {
            r = r * (n - k + i) / i;
        }
        r
    }

    #[test]
    fn encoding_roundtrip_all_indices() {
        let _g = CONTEXT_LOCK.lock();
        for &(no, nv) in &[(3u32, 2u32), (5, 3), (10, 7), (1, 1)] {
            init(no, nv).unwrap();
            let ctx = Context::current();
            assert_eq!(ctx.nv1 + ctx.nv2, nv);
            assert_eq!(ctx.nv1, nv.div_ceil(2));
            assert_eq!(ctx.nmmax as u64, binom(u64::from(no + nv), u64::from(nv)));
            for ii in 0..ctx.nmmax {
                let jj = ctx.decode(ii);
                assert_eq!(jj.len(), nv as usize);
                let re = ctx.encode(&jj).expect("valid monomial must encode");
                assert_eq!(
                    re, ii,
                    "encode(decode({ii})) mismatch at (no,nv)=({no},{nv})"
                );
                let order: u32 = jj.iter().sum();
                assert_eq!(ctx.order_of(ii), order);
                assert!(order <= no);
                // decode_into agrees with decode
                let mut buf = vec![0u32; nv as usize];
                ctx.decode_into(ii, &mut buf);
                assert_eq!(buf, jj);
            }
        }
    }

    #[test]
    fn encode_rejects_invalid() {
        let _g = CONTEXT_LOCK.lock();
        init(3, 2).unwrap();
        let ctx = Context::current();
        assert_eq!(ctx.encode(&[0, 0]), Some(0));
        assert_eq!(ctx.encode(&[4, 0]), None); // exponent above nomax
        assert_eq!(ctx.encode(&[2, 2]), None); // total order above nomax
        assert_eq!(ctx.encode(&[1]), None); // wrong length
    }

    #[test]
    fn init_clamps_and_errors() {
        let _g = CONTEXT_LOCK.lock();
        init(0, 0).unwrap();
        assert_eq!(max_order(), 1);
        assert_eq!(max_variables(), 1);
        let err = init(100, 21).unwrap_err();
        assert_eq!(err.code, codes::ORDER_VARIABLE_TOO_LARGE);
        // previous context still active after failed init
        assert_eq!(max_order(), 1);
        assert!(initialized());
    }

    #[test]
    fn settings_roundtrip() {
        let _g = CONTEXT_LOCK.lock();
        init(5, 2).unwrap();
        assert_eq!(epsilon(), 0.0);
        assert_eq!(truncation_order(), 5);
        assert_eq!(set_epsilon(-0.5), 0.0);
        assert_eq!(epsilon(), 0.5);
        assert_eq!(set_epsilon(0.0), 0.5);
        assert_eq!(set_truncation_order(3), 5);
        assert_eq!(truncation_order(), 3);
        assert_eq!(set_truncation_order(99), 3); // clamps to nomax=5
        assert_eq!(truncation_order(), 5);
        assert_eq!(set_truncation_order(0), 5); // clamps to 1
        assert_eq!(truncation_order(), 1);
        push_truncation_order(2);
        assert_eq!(truncation_order(), 2);
        pop_truncation_order();
        assert_eq!(truncation_order(), 1);
        // init resets settings
        init(4, 3).unwrap();
        assert_eq!(truncation_order(), 4);
        assert_eq!(epsilon(), 0.0);
        assert_eq!(version(), "2.1.0-rs");
        assert!(machine_epsilon() > 0.0 && machine_epsilon() <= f64::EPSILON * 2.0);
    }

    #[test]
    fn truncation_stack_empty_pop_panics() {
        let _g = CONTEXT_LOCK.lock();
        init(5, 2).unwrap();
        let result = std::panic::catch_unwind(|| {
            pop_truncation_order();
        });
        assert!(result.is_err());
    }
}
