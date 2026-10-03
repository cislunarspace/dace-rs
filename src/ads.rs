//! Automatic domain splitting (ADS): recursive bisection and re-expansion
//! of a DA map over a large uncertainty box.
//!
//! A single truncated Taylor expansion loses accuracy as the uncertainty
//! box grows or the mapped function becomes more nonlinear. The [`split`]
//! driver restores it the way the ADS literature describes (Losacco et
//! al. 2022; Wittig 2015): recursively bisect the box in the DA
//! variables, re-expand the caller's map at the center of every sub-box,
//! and stop splitting a sub-box once the polynomial bounds of the target
//! outputs are narrower than caller-supplied tolerances.
//!
//! This is a Rust-only extension: upstream C DACE 2.1 has no ADS
//! counterpart to port. The split criteria reuse the existing
//! [`Da::bound`](crate::Da::bound) and
//! [`Da::deriv`](crate::Da::deriv) primitives; no interval arithmetic is
//! introduced.
//!
//! # Algorithm
//!
//! The domain is one interval per DA variable: `domain[i]` bounds
//! variable `i + 1`. A sub-box is described by a per-variable `center`
//! and `half_width`. To expand the map `f` on a sub-box, the driver calls
//! `f` with the substituted inputs `x_i = c_i + h_i * variable(i + 1)`
//! (built with [`Da::translate_variable`]) rather than composing with the
//! already truncated polynomial: a fresh call re-expands the underlying
//! function around `c_i`, which is what recovers accuracy under strong
//! nonlinearity. The map must build its outputs from the inputs it is
//! given. Outputs are polynomials in the unit variables of the sub-box,
//! so [`Da::bound`](crate::Da::bound) is a rigorous enclosure over the
//! sub-box; bounds containing NaN never meet a tolerance.
//!
//! A sub-box becomes a leaf when every target component's bound width is
//! within tolerance (see [`AdsConfig`]). Otherwise the driver splits
//! along the eligible variable with the largest contribution, measured as
//! the largest absolute bound of the target components' partial
//! derivatives with respect to that sub-box variable — a rigorous
//! first-order width proxy over the sub-box. A variable stays eligible
//! while it has been split fewer than `max_splits_per_var` times and has
//! positive half-width.
//!
//! When the leaf budget `max_leaves` is reached, or no variable is
//! eligible, the sub-box still becomes a leaf but is flagged
//! `met = false`: an unmet leaf is reported, never hidden. The traversal
//! is depth-first with an explicit stack, children are visited lower half
//! first, and ties in the direction choice go to the lowest variable
//! index, so identical inputs produce identical results.
//!
//! # Parameter semantics and defaults
//!
//! | Parameter | Meaning | Default |
//! |---|---|---|
//! | `tolerances` | One tolerance per target component (see `targets`); positive and finite. | `[1e-8]` |
//! | `tolerance_kind` | `Absolute`: `hi - lo <= tol`. `Relative`: `hi - lo <= tol * max(\|lo\|, \|hi\|)` of the component's bound. | `Absolute` |
//! | `targets` | Output components checked against `tolerances`; empty means all. | `[]` (all) |
//! | `max_splits_per_var` | Per-direction bisection cap; a variable at the cap is no longer eligible. `0` forbids splitting. | `32` |
//! | `max_leaves` | Total leaf budget of the whole tree, met or not. | `1024` |
//!
//! The defaults are starting points: the tolerance is problem-dependent,
//! `32` splits per direction let half-widths shrink by a factor `2^-32`,
//! and `1024` leaves bound the cost of a default run.
//!
//! # Example
//!
//! ```
//! use dace_rs::Interval;
//! use dace_rs::ads::{AdsConfig, split};
//!
//! dace_rs::init(16, 1).unwrap();
//! // sin over a box spanning most of a half period: a single expansion
//! // varies by ~2, so the driver splits until every leaf varies by <= 1e-2.
//! let cfg = AdsConfig { tolerances: vec![1e-2], ..Default::default() };
//! let result = split(
//!     |x: &[dace_rs::Da]| vec![dace_rs::sin(&x[0])],
//!     &[Interval { lo: -1.5, hi: 1.5 }],
//!     &cfg,
//! );
//! assert!(result.leaves.len() > 1);
//! assert_eq!(result.met_leaves, result.leaves.len());
//! for leaf in &result.leaves {
//!     assert!(leaf.bounds[0].hi - leaf.bounds[0].lo <= 1e-2);
//! }
//! ```

use crate::da::Da;
use crate::error::{codes, dace_panic};
use crate::norm::Interval;

/// How a tolerance is compared against a component's bound width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToleranceKind {
    /// `hi - lo <= tol`.
    Absolute,
    /// `hi - lo <= tol * max(|lo|, |hi|)`; a `[0, 0]` bound only meets the
    /// zero width.
    Relative,
}

/// Configuration of one [`split`] run. See the [module docs](self) for
/// the semantics of each field and the rationale for the defaults.
#[derive(Debug, Clone)]
pub struct AdsConfig {
    /// One tolerance per target component; positive and finite.
    pub tolerances: Vec<f64>,
    /// Absolute or relative comparison against the bound width.
    pub tolerance_kind: ToleranceKind,
    /// Output components checked against `tolerances`; empty means all.
    pub targets: Vec<usize>,
    /// Maximum bisections per variable; `0` forbids splitting.
    pub max_splits_per_var: u32,
    /// Total leaf budget of the whole tree, met or not.
    pub max_leaves: usize,
}

impl Default for AdsConfig {
    /// `[1e-8]` absolute tolerance over all components, `32` splits per
    /// variable, `1024` leaves.
    fn default() -> Self {
        AdsConfig {
            tolerances: vec![1e-8],
            tolerance_kind: ToleranceKind::Absolute,
            targets: Vec::new(),
            max_splits_per_var: 32,
            max_leaves: 1024,
        }
    }
}

/// One sub-box of the finished split, with the map re-expanded on it.
#[derive(Debug, Clone)]
pub struct AdsLeaf {
    /// Sub-box center, one entry per domain variable.
    pub center: Vec<f64>,
    /// Sub-box half-width, one entry per domain variable.
    pub half_width: Vec<f64>,
    /// The map's outputs re-expanded on the sub-box, in the sub-box's
    /// unit variables `x_i = c_i + h_i * variable(i + 1)`.
    pub values: Vec<Da>,
    /// Rigorous enclosure over the sub-box, one per output component.
    pub bounds: Vec<Interval>,
    /// Whether every target component's bound width is within tolerance.
    pub met: bool,
}

/// The outcome of a [`split`] run.
#[derive(Debug, Clone)]
pub struct AdsResult {
    /// The leaves, in deterministic lower-half-first depth-first order.
    pub leaves: Vec<AdsLeaf>,
    /// Bisections performed per domain variable.
    pub splits_per_var: Vec<u32>,
    /// Leaves whose target bounds are all within tolerance.
    pub met_leaves: usize,
}

/// Split `domain` until the target outputs of the map `f` are enclosed to
/// the configured tolerances on every leaf, and return the leaves with
/// the map re-expanded on each.
///
/// `f` receives one input per domain variable, already substituted to the
/// sub-box (`x_i = c_i + h_i * variable(i + 1)`), and must build its
/// outputs from those inputs. See the [module docs](self) for the split
/// criterion, the direction choice, and the parameter semantics.
///
/// # Panics
///
/// Panics with [`crate::DaceError`] code 650 (`OUT_OF_DOMAIN`) when the
/// domain is empty or holds an interval with `lo > hi` or a NaN endpoint,
/// when `tolerances` is empty or holds a non-positive or non-finite
/// entry, when `max_leaves` is zero, when `tolerances` does not align
/// with `targets` (or with all components when `targets` is empty), when
/// a target index is out of range, or when `f` returns no components.
pub fn split<F>(f: F, domain: &[Interval], config: &AdsConfig) -> AdsResult
where
    F: Fn(&[Da]) -> Vec<Da>,
{
    validate_config(config);
    if domain.is_empty() {
        dace_panic(
            codes::OUT_OF_DOMAIN,
            "ADS: the domain must cover at least one variable",
        );
    }
    if domain
        .iter()
        .any(|iv| iv.lo.is_nan() || iv.hi.is_nan() || iv.lo > iv.hi)
    {
        dace_panic(
            codes::OUT_OF_DOMAIN,
            "ADS: domain interval with lo > hi or a NaN endpoint",
        );
    }

    let center: Vec<f64> = domain.iter().map(|iv| 0.5 * (iv.lo + iv.hi)).collect();
    let half_width: Vec<f64> = domain.iter().map(|iv| 0.5 * (iv.hi - iv.lo)).collect();

    let mut leaves: Vec<AdsLeaf> = Vec::new();
    let mut splits_per_var = vec![0u32; half_width.len()];
    let mut met_leaves = 0usize;
    let mut stack = vec![Node {
        center,
        half_width,
        path_splits: vec![0; domain.len()],
    }];

    while let Some(Node {
        center,
        half_width,
        path_splits,
    }) = stack.pop()
    {
        let values = expand(&f, &center, &half_width);
        let targets = resolve_targets(&config.targets, values.len());
        if config.tolerances.len() != targets.len() {
            dace_panic(
                codes::OUT_OF_DOMAIN,
                "ADS: tolerances must align with targets (or all components when targets is empty)",
            );
        }
        let bounds: Vec<Interval> = values.iter().map(Da::bound).collect();
        let met = targets
            .iter()
            .zip(&config.tolerances)
            .all(|(&j, &tol)| within_tol(bounds[j], tol, config.tolerance_kind));

        let direction = if met || leaves.len() + stack.len() + 2 > config.max_leaves {
            None
        } else {
            choose_direction(&values, &targets, &half_width, &path_splits, config)
        };
        if let Some(dir) = direction {
            splits_per_var[dir] += 1;
            let h = half_width[dir] / 2.0;
            let mut upper = Node {
                center: center.clone(),
                half_width: half_width.clone(),
                path_splits: path_splits.clone(),
            };
            upper.center[dir] += h;
            upper.half_width[dir] = h;
            upper.path_splits[dir] += 1;
            let mut lower = Node {
                center,
                half_width,
                path_splits,
            };
            lower.center[dir] -= h;
            lower.half_width[dir] = h;
            lower.path_splits[dir] += 1;
            // The lower half is pushed last so it is visited first.
            stack.push(upper);
            stack.push(lower);
        } else {
            met_leaves += usize::from(met);
            leaves.push(AdsLeaf {
                center,
                half_width,
                values,
                bounds,
                met,
            });
        }
    }

    AdsResult {
        leaves,
        splits_per_var,
        met_leaves,
    }
}

/// A pending sub-box of the traversal: geometry plus the bisections per
/// variable along the path from the root (the per-direction cap is a
/// path property).
struct Node {
    center: Vec<f64>,
    half_width: Vec<f64>,
    path_splits: Vec<u32>,
}

/// The eligible variable with the largest contribution to the target
/// bound widths: the largest absolute bound of the targets' partial
/// derivatives with respect to the sub-box variable. Ties go to the
/// lowest variable index.
fn choose_direction(
    values: &[Da],
    targets: &[usize],
    half_width: &[f64],
    path_splits: &[u32],
    config: &AdsConfig,
) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, &h) in half_width.iter().enumerate() {
        if path_splits[i] >= config.max_splits_per_var || h <= 0.0 || h.is_nan() {
            continue;
        }
        let var = i as u32 + 1;
        let mut contrib = 0.0f64;
        for &j in targets {
            let b = values[j].deriv(var).bound();
            contrib = contrib.max(b.lo.abs()).max(b.hi.abs());
        }
        if contrib > 0.0 && best.is_none_or(|(_, c)| contrib > c) {
            best = Some((i, contrib));
        }
    }
    best.map(|(i, _)| i)
}

/// Call `f` with the sub-box substituted inputs `x_i = c_i + h_i * var_i`.
fn expand<F>(f: &F, center: &[f64], half_width: &[f64]) -> Vec<Da>
where
    F: Fn(&[Da]) -> Vec<Da>,
{
    let vars: Vec<Da> = (0..center.len())
        .map(|i| {
            let var = i as u32 + 1;
            Da::variable(var).translate_variable(var, half_width[i], center[i])
        })
        .collect();
    let values = f(&vars);
    if values.is_empty() {
        dace_panic(codes::OUT_OF_DOMAIN, "ADS: the map returned no components");
    }
    values
}

/// Which output components the tolerances apply to.
fn resolve_targets(targets: &[usize], nvals: usize) -> Vec<usize> {
    if targets.is_empty() {
        (0..nvals).collect()
    } else {
        if targets.iter().any(|&j| j >= nvals) {
            dace_panic(codes::OUT_OF_DOMAIN, "ADS: target index out of range");
        }
        targets.to_vec()
    }
}

/// Whether a component's bound width is within tolerance.
fn within_tol(b: Interval, tol: f64, kind: ToleranceKind) -> bool {
    let width = b.hi - b.lo;
    match kind {
        ToleranceKind::Absolute => width <= tol,
        ToleranceKind::Relative => width <= tol * b.lo.abs().max(b.hi.abs()),
    }
}

/// Reject malformed configurations up front.
fn validate_config(config: &AdsConfig) {
    if config.max_leaves == 0 {
        dace_panic(codes::OUT_OF_DOMAIN, "ADS: max_leaves must be at least 1");
    }
    if config.tolerances.is_empty() {
        dace_panic(codes::OUT_OF_DOMAIN, "ADS: tolerances must not be empty");
    }
    if config
        .tolerances
        .iter()
        .any(|&t| !t.is_finite() || t <= 0.0)
    {
        dace_panic(
            codes::OUT_OF_DOMAIN,
            "ADS: tolerances must be positive and finite",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::CONTEXT_LOCK;

    fn cfg(tolerances: Vec<f64>) -> AdsConfig {
        AdsConfig {
            tolerances,
            ..Default::default()
        }
    }

    #[test]
    fn constant_map_is_one_met_leaf() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(4, 2).unwrap();
        let result = split(
            |_: &[Da]| vec![Da::constant(0.5)],
            &[
                Interval { lo: -1.0, hi: 1.0 },
                Interval { lo: -2.0, hi: 2.0 },
            ],
            &cfg(vec![1e-6]),
        );
        assert_eq!(result.leaves.len(), 1);
        assert_eq!(result.met_leaves, 1);
        assert!(result.leaves[0].met);
        assert_eq!(result.leaves[0].bounds[0], Interval { lo: 0.5, hi: 0.5 });
        assert_eq!(result.leaves[0].center, vec![0.0, 0.0]);
        assert_eq!(result.leaves[0].half_width, vec![1.0, 2.0]);
        assert_eq!(result.splits_per_var, vec![0, 0]);
    }

    #[test]
    fn linear_map_meets_tolerance_without_splitting() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(4, 1).unwrap();
        // 0.1 * x on [-1, 1] has width 0.2 <= 0.3.
        let result = split(
            |x: &[Da]| vec![0.1 * x[0].clone()],
            &[Interval { lo: -1.0, hi: 1.0 }],
            &cfg(vec![0.3]),
        );
        assert_eq!(result.leaves.len(), 1);
        assert!(result.leaves[0].met);
        assert_eq!(result.met_leaves, 1);
        let b = result.leaves[0].bounds[0];
        assert!((b.hi - b.lo - 0.2).abs() < 1e-15, "width {b:?}");
    }

    #[test]
    fn relative_tolerance_uses_bound_scale() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(4, 1).unwrap();
        // 1000 + 0.001 * x on [-1, 1]: width 0.002, scale ~1000.
        let domain = &[Interval { lo: -1.0, hi: 1.0 }];
        let map = |x: &[Da]| vec![Da::constant(1000.0) + 0.001 * x[0].clone()];

        // Absolute 1e-3 misses (0.002 > 1e-3); splitting is forbidden.
        let abs = AdsConfig {
            tolerances: vec![1e-3],
            max_splits_per_var: 0,
            ..Default::default()
        };
        let r = split(map, domain, &abs);
        assert_eq!(r.leaves.len(), 1);
        assert!(!r.leaves[0].met);
        assert_eq!(r.met_leaves, 0);

        // Relative 1e-3 meets: 0.002 <= 1e-3 * 1000.001.
        let rel = AdsConfig {
            tolerances: vec![1e-3],
            tolerance_kind: ToleranceKind::Relative,
            max_splits_per_var: 0,
            ..Default::default()
        };
        let r = split(map, domain, &rel);
        assert_eq!(r.leaves.len(), 1);
        assert!(r.leaves[0].met);
        assert_eq!(r.met_leaves, 1);
    }

    #[test]
    fn sin_on_big_box_splits_with_rigorous_leaf_bounds() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(16, 1).unwrap();
        let cfg = AdsConfig {
            tolerances: vec![1e-2],
            ..Default::default()
        };
        let result = split(
            |x: &[Da]| vec![crate::elementary::sin(&x[0])],
            &[Interval { lo: -1.5, hi: 1.5 }],
            &cfg,
        );
        assert!(result.leaves.len() > 1, "a near-half-period box must split");
        assert_eq!(result.met_leaves, result.leaves.len());
        assert!(result.splits_per_var[0] > 0);
        for leaf in &result.leaves {
            let (lo, hi) = (
                leaf.center[0] - leaf.half_width[0],
                leaf.center[0] + leaf.half_width[0],
            );
            let (true_lo, true_hi) = sin_range(lo, hi);
            // Order-16 re-expansions of sin on |x| <= 1.5 carry a
            // remainder below 1.5^17/17! ~ 6e-14; 1e-10 is a safe margin.
            assert!(
                leaf.bounds[0].lo <= true_lo + 1e-10,
                "leaf bound {leaf:?} below true range on [{lo}, {hi}]"
            );
            assert!(
                leaf.bounds[0].hi >= true_hi - 1e-10,
                "leaf bound {leaf:?} above true range on [{lo}, {hi}]"
            );
        }
    }

    /// True range of sin on [lo, hi] from endpoints and interior extrema
    /// (independent of the DA machinery under test).
    fn sin_range(lo: f64, hi: f64) -> (f64, f64) {
        let mut lo_v = lo.sin().min(hi.sin());
        let mut hi_v = lo.sin().max(hi.sin());
        let k_min = ((lo - std::f64::consts::FRAC_PI_2) / std::f64::consts::PI).ceil() as i64;
        let k_max = ((hi - std::f64::consts::FRAC_PI_2) / std::f64::consts::PI).floor() as i64;
        for k in k_min..=k_max {
            let x = std::f64::consts::FRAC_PI_2 + k as f64 * std::f64::consts::PI;
            if (lo..=hi).contains(&x) {
                let v = x.sin();
                lo_v = lo_v.min(v);
                hi_v = hi_v.max(v);
            }
        }
        (lo_v, hi_v)
    }

    #[test]
    fn leaf_budget_caps_the_tree_and_flags_unmet() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(16, 1).unwrap();
        // 1e-6 would need millions of leaves; the budget of 8 stops early.
        let cfg = AdsConfig {
            tolerances: vec![1e-6],
            max_leaves: 8,
            ..Default::default()
        };
        let result = split(
            |x: &[Da]| vec![crate::elementary::sin(&x[0])],
            &[Interval { lo: -1.5, hi: 1.5 }],
            &cfg,
        );
        assert_eq!(result.leaves.len(), 8);
        assert_eq!(result.met_leaves, 0);
        assert!(result.leaves.iter().all(|l| !l.met));
        // The leaves still tile the domain, lower half first.
        assert!((result.leaves[0].center[0] - result.leaves[0].half_width[0] + 1.5).abs() < 1e-12);
        for w in result.leaves.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            let a_hi = a.center[0] + a.half_width[0];
            let b_lo = b.center[0] - b.half_width[0];
            assert!((a_hi - b_lo).abs() < 1e-12, "leaves must tile in order");
        }
        let last = result.leaves.last().unwrap();
        assert!((last.center[0] + last.half_width[0] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn per_direction_cap_limits_half_width() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(16, 1).unwrap();
        // Three bisections along the only variable: 8 leaves of half-width
        // 1.5/8, none meeting the unreachable tolerance.
        let cfg = AdsConfig {
            tolerances: vec![1e-9],
            max_splits_per_var: 3,
            ..Default::default()
        };
        let result = split(
            |x: &[Da]| vec![crate::elementary::sin(&x[0])],
            &[Interval { lo: -1.5, hi: 1.5 }],
            &cfg,
        );
        assert_eq!(result.leaves.len(), 8);
        assert_eq!(result.splits_per_var, vec![7]);
        assert_eq!(result.met_leaves, 0);
        for leaf in &result.leaves {
            assert!(
                (leaf.half_width[0] - 1.5 / 8.0).abs() < 1e-12,
                "cap must bound the half-width: {leaf:?}"
            );
        }
    }

    #[test]
    fn direction_choice_follows_the_largest_contribution() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(8, 2).unwrap();
        // 0.001 * x + 10 * y on [-1, 1]^2 with absolute tolerance 1:
        // a leaf's bound width is 0.002 + 20 * h_y, met once h_y = 2^-5.
        // Only y ever contributes enough to be chosen, so x is never split.
        let cfg = AdsConfig {
            tolerances: vec![1.0],
            ..Default::default()
        };
        let result = split(
            |x: &[Da]| vec![0.001 * x[0].clone() + 10.0 * x[1].clone()],
            &[
                Interval { lo: -1.0, hi: 1.0 },
                Interval { lo: -1.0, hi: 1.0 },
            ],
            &cfg,
        );
        // All 32 leaves sit at y-depth 5: a full binary tree of depth 5
        // has 2^5 - 1 = 31 internal bisections, all in y, none in x.
        assert_eq!(result.splits_per_var, vec![0, 31]);
        assert_eq!(result.leaves.len(), 32);
        assert_eq!(result.met_leaves, 32);
        for leaf in &result.leaves {
            assert!((leaf.half_width[0] - 1.0).abs() < 1e-12);
            assert!((leaf.half_width[1] - 1.0 / 32.0).abs() < 1e-12);
        }
    }

    #[test]
    fn invalid_inputs_panic_with_code_650() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(4, 2).unwrap();

        fn expect_code(f: impl FnOnce() -> AdsResult + std::panic::UnwindSafe, code: u32) {
            let err = std::panic::catch_unwind(f).expect_err("must panic");
            let e = err
                .downcast_ref::<crate::error::DaceError>()
                .expect("DaceError payload");
            assert_eq!(e.code, code, "{}", e);
        }

        let unit = &[Interval { lo: -1.0, hi: 1.0 }];
        let ok = || AdsConfig {
            tolerances: vec![1e-3],
            ..Default::default()
        };
        let lin = |x: &[Da]| vec![x[0].clone()];

        expect_code(|| split(lin, &[], &ok()), codes::OUT_OF_DOMAIN);
        expect_code(
            || split(lin, &[Interval { lo: 1.0, hi: -1.0 }], &ok()),
            codes::OUT_OF_DOMAIN,
        );
        expect_code(
            || {
                split(
                    lin,
                    &[Interval {
                        lo: f64::NAN,
                        hi: 1.0,
                    }],
                    &ok(),
                )
            },
            codes::OUT_OF_DOMAIN,
        );
        expect_code(
            || {
                split(
                    lin,
                    unit,
                    &AdsConfig {
                        tolerances: vec![],
                        ..Default::default()
                    },
                )
            },
            codes::OUT_OF_DOMAIN,
        );
        for bad in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
            expect_code(
                || {
                    split(
                        lin,
                        unit,
                        &AdsConfig {
                            tolerances: vec![bad],
                            ..Default::default()
                        },
                    )
                },
                codes::OUT_OF_DOMAIN,
            );
        }
        expect_code(
            || {
                split(
                    lin,
                    unit,
                    &AdsConfig {
                        max_leaves: 0,
                        ..Default::default()
                    },
                )
            },
            codes::OUT_OF_DOMAIN,
        );
        // Two tolerances but a single output component.
        expect_code(
            || {
                split(
                    lin,
                    unit,
                    &AdsConfig {
                        tolerances: vec![1e-3, 1e-3],
                        ..Default::default()
                    },
                )
            },
            codes::OUT_OF_DOMAIN,
        );
        expect_code(
            || {
                split(
                    lin,
                    unit,
                    &AdsConfig {
                        tolerances: vec![1e-3],
                        targets: vec![1],
                        ..Default::default()
                    },
                )
            },
            codes::OUT_OF_DOMAIN,
        );
        expect_code(
            || split(|_: &[Da]| vec![], unit, &ok()),
            codes::OUT_OF_DOMAIN,
        );
    }

    #[test]
    fn identical_runs_produce_identical_results() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(12, 2).unwrap();
        let cfg = AdsConfig {
            tolerances: vec![1e-3, 1e-3],
            ..Default::default()
        };
        let domain = &[
            Interval { lo: -1.2, hi: 1.0 },
            Interval { lo: -0.8, hi: 1.4 },
        ];
        let map = |x: &[Da]| {
            vec![
                crate::elementary::sin(&x[0]) * crate::elementary::cos(&x[1]),
                x[0].clone() * x[1].clone(),
            ]
        };
        let a = split(map, domain, &cfg);
        let b = split(map, domain, &cfg);
        assert_eq!(a.leaves.len(), b.leaves.len());
        assert_eq!(a.splits_per_var, b.splits_per_var);
        assert_eq!(a.met_leaves, b.met_leaves);
        for (la, lb) in a.leaves.iter().zip(&b.leaves) {
            assert_eq!(la.center, lb.center);
            assert_eq!(la.half_width, lb.half_width);
            assert_eq!(la.bounds, lb.bounds);
            assert_eq!(la.met, lb.met);
        }
    }

    #[test]
    fn zero_width_variables_never_split() {
        let _g = CONTEXT_LOCK.lock();
        crate::context::init(8, 2).unwrap();
        // Variable 1 is fixed at 0.5 (zero width): only variable 2 splits.
        let cfg = AdsConfig {
            tolerances: vec![1e-2],
            ..Default::default()
        };
        let result = split(
            |x: &[Da]| vec![x[0].clone() * crate::elementary::sin(&x[1])],
            &[
                Interval { lo: 0.5, hi: 0.5 },
                Interval { lo: -1.5, hi: 1.5 },
            ],
            &cfg,
        );
        assert!(result.leaves.len() > 1);
        assert_eq!(result.splits_per_var[0], 0);
        assert!(result.leaves.iter().all(|l| l.half_width[0] == 0.0));
        assert_eq!(result.met_leaves, result.leaves.len());
    }
}
