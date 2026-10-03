//! ADS acceptance tests (issue #11): self-built numerical baselines, no
//! C-parity counterpart. Expected values come from independent truth
//! sources — analytic function ranges, Monte Carlo sampling — never from
//! the machinery under test.

use std::sync::{LazyLock, Mutex};

use dace_rs::ads::{AdsConfig, split};
use dace_rs::{Da, Interval};

/// Serialize tests sharing the process-global DACE context.
static CONTEXT_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

/// Deterministic LCG matching the crate's internal generator.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

fn unit() -> Interval {
    Interval { lo: -1.0, hi: 1.0 }
}

fn box_(half_width: f64) -> Interval {
    Interval {
        lo: -half_width,
        hi: half_width,
    }
}

fn sin_map() -> impl Fn(&[Da]) -> Vec<Da> {
    |x: &[Da]| vec![dace_rs::sin(&x[0])]
}

fn inv_map(a: f64) -> impl Fn(&[Da]) -> Vec<Da> {
    move |x: &[Da]| vec![1.0 / (x[0].clone() - a)]
}

/// Sum of the leaf bound widths of target component 0.
fn total_width(result: &dace_rs::ads::AdsResult) -> f64 {
    result
        .leaves
        .iter()
        .map(|l| l.bounds[0].hi - l.bounds[0].lo)
        .sum()
}

/// True range of 1/(x - a) on [lo, hi] for a pole `a` outside the
/// interval: the map is monotone there, so the range is the image of the
/// endpoints.
fn inv_range(lo: f64, hi: f64, a: f64) -> (f64, f64) {
    let flo = 1.0 / (lo - a);
    let fhi = 1.0 / (hi - a);
    (flo.min(fhi), flo.max(fhi))
}

#[test]
fn trig_total_width_converges_monotonically_with_budget() {
    let _g = CONTEXT_LOCK.lock().unwrap();
    dace_rs::init(16, 1).unwrap();

    let mut prev = f64::INFINITY;
    for budget in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
        let cfg = AdsConfig {
            tolerances: vec![1e-2],
            max_leaves: budget,
            ..Default::default()
        };
        let result = split(sin_map(), &[box_(1.5)], &cfg);
        let w = total_width(&result);
        assert!(
            w <= prev + 1e-12,
            "total width must not increase with the budget: budget {budget} gave {w} after {prev}"
        );
        prev = w;
    }
    // With enough budget every leaf meets the tolerance and the total
    // width approaches the total variation of sin over the box, 2*sin(1.5).
    let cfg = AdsConfig {
        tolerances: vec![1e-2],
        max_leaves: 1024,
        ..Default::default()
    };
    let result = split(sin_map(), &[box_(1.5)], &cfg);
    assert_eq!(result.met_leaves, result.leaves.len());
    assert!(
        total_width(&result) >= 2.0 * 1.5_f64.sin() - 1e-9,
        "rigorous bounds cannot undercut the total variation"
    );
}

#[test]
fn rational_near_pole_bounds_stay_rigorous() {
    let _g = CONTEXT_LOCK.lock().unwrap();
    dace_rs::init(16, 1).unwrap();

    // Pole at 1.2, 0.2 outside the domain edge: the root expansion around
    // 0 barely converges there, only re-expansion on sub-boxes keeps the
    // enclosures honest.
    let cfg = AdsConfig {
        tolerances: vec![3e-2],
        max_leaves: 4096,
        ..Default::default()
    };
    let result = split(inv_map(1.2), &[unit()], &cfg);
    assert_eq!(result.met_leaves, result.leaves.len());
    assert!(result.leaves.len() > 1);
    for leaf in &result.leaves {
        let lo = leaf.center[0] - leaf.half_width[0];
        let hi = leaf.center[0] + leaf.half_width[0];
        let (true_lo, true_hi) = inv_range(lo, hi, 1.2);
        // Re-expansions at leaf centers converge geometrically (ratio
        // h / |c - 1.2| < 0.1 at order 16): the remainder is far below
        // this margin.
        assert!(
            leaf.bounds[0].lo <= true_lo + 1e-9,
            "false convergence: bound misses truth on [{lo}, {hi}]: {:?}",
            leaf.bounds[0]
        );
        assert!(
            leaf.bounds[0].hi >= true_hi - 1e-9,
            "false convergence: bound misses truth on [{lo}, {hi}]: {:?}",
            leaf.bounds[0]
        );
    }

    // The enclosures converge as the budget grows.
    let mut prev = f64::INFINITY;
    for budget in [1, 4, 16, 64, 256] {
        let cfg = AdsConfig {
            tolerances: vec![3e-2],
            max_leaves: budget,
            ..Default::default()
        };
        let result = split(inv_map(1.2), &[unit()], &cfg);
        let w = total_width(&result);
        assert!(
            w <= prev + 1e-12,
            "total width must not increase with the budget: budget {budget} gave {w} after {prev}"
        );
        prev = w;
    }
}

#[test]
fn monte_carlo_samples_fall_into_exactly_one_rigorous_leaf() {
    let _g = CONTEXT_LOCK.lock().unwrap();
    dace_rs::init(12, 2).unwrap();

    // Several nonlinear analytic maps, per the acceptance criteria, each
    // paired with its independent f64 evaluation of the same function.
    struct Case<'a> {
        name: &'static str,
        map: &'a dyn Fn(&[Da]) -> Vec<Da>,
        truth: &'a dyn Fn(f64, f64) -> f64,
    }
    let cases = [
        Case {
            name: "exp*sin + xy",
            map: &|x: &[Da]| {
                vec![
                    dace_rs::exp(&(0.5 * x[0].clone())) * dace_rs::sin(&x[1])
                        + 0.3 * (x[0].clone() * x[1].clone()),
                ]
            },
            truth: &|sx, sy| (0.5 * sx).exp() * sy.sin() + 0.3 * sx * sy,
        },
        Case {
            name: "cos*cos + x^2 y",
            map: &|x: &[Da]| {
                let sq = x[0].clone() * x[0].clone();
                vec![dace_rs::cos(&x[0]) * dace_rs::cos(&x[1]) + sq * x[1].clone()]
            },
            truth: &|sx, sy| sx.cos() * sy.cos() + sx * sx * sy,
        },
    ];
    // Unreachable tolerance with per-direction caps: a full depth-4 tree
    // of 256 leaves whose tiling is checked against random samples.
    let cfg = AdsConfig {
        tolerances: vec![1e-9],
        max_splits_per_var: 4,
        max_leaves: 4096,
        ..Default::default()
    };

    fn check_tiling(name: &str, result: &dace_rs::ads::AdsResult, truth: &dyn Fn(f64, f64) -> f64) {
        assert_eq!(result.leaves.len(), 256, "{name}");

        // Leaf volumes tile the domain exactly.
        let volume: f64 = result
            .leaves
            .iter()
            .map(|l| (2.0 * l.half_width[0]) * (2.0 * l.half_width[1]))
            .sum();
        assert!((volume - 4.0).abs() < 1e-9, "{name}: volume {volume}");

        let contains = |leaf: &dace_rs::ads::AdsLeaf, sx: f64, sy: f64| -> bool {
            // Half-open edges [lo, hi); the domain's upper edges stay closed.
            let (lo, hi) = (
                leaf.center[0] - leaf.half_width[0],
                leaf.center[0] + leaf.half_width[0],
            );
            let x_ok = sx >= lo && (sx < hi || hi >= 1.0);
            let (lo, hi) = (
                leaf.center[1] - leaf.half_width[1],
                leaf.center[1] + leaf.half_width[1],
            );
            let y_ok = sy >= lo && (sy < hi || hi >= 1.0);
            x_ok && y_ok
        };

        let mut rng = Lcg(0xAD5);
        for _ in 0..5000 {
            let (sx, sy) = (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0));
            let owners: Vec<&dace_rs::ads::AdsLeaf> = result
                .leaves
                .iter()
                .filter(|l| contains(l, sx, sy))
                .collect();
            assert_eq!(
                owners.len(),
                1,
                "{name}: sample ({sx}, {sy}) must own exactly one leaf"
            );
            let leaf = owners[0];
            let value = truth(sx, sy);
            // The leaf polynomial is the order-12 re-expansion at the leaf
            // center with half-width 1/16: its remainder is far below 1e-9.
            assert!(
                leaf.bounds[0].lo <= value + 1e-9 && leaf.bounds[0].hi >= value - 1e-9,
                "{name}: sample value {value} escapes the owning leaf bound {:?} at ({sx}, {sy})",
                leaf.bounds[0]
            );
        }
    }

    for case in cases {
        let result = split(case.map, &[unit(), unit()], &cfg);
        check_tiling(case.name, &result, case.truth);
    }
}

#[test]
fn split_count_shrinks_with_the_box_and_recedes_with_the_pole() {
    let _g = CONTEXT_LOCK.lock().unwrap();
    dace_rs::init(16, 1).unwrap();

    // Same tolerance, shrinking boxes: fewer leaves, monotonically.
    let mut prev = usize::MAX;
    for h in [1.5_f64, 1.0, 0.5, 0.2, 0.1, 0.05] {
        let cfg = AdsConfig {
            tolerances: vec![2e-2],
            max_leaves: 8192,
            ..Default::default()
        };
        let result = split(sin_map(), &[box_(h)], &cfg);
        assert_eq!(result.met_leaves, result.leaves.len(), "box half-width {h}");
        assert!(
            result.leaves.len() <= prev,
            "leaf count must not grow as the box shrinks: h={h} gave {} after {prev}",
            result.leaves.len()
        );
        prev = result.leaves.len();
    }

    // Same tolerance, pole receding from the domain edge: fewer leaves.
    let mut prev = usize::MAX;
    for a in [1.05_f64, 1.3, 2.0] {
        let cfg = AdsConfig {
            tolerances: vec![4e-2],
            max_leaves: 8192,
            ..Default::default()
        };
        let result = split(inv_map(a), &[unit()], &cfg);
        assert_eq!(result.met_leaves, result.leaves.len(), "pole at {a}");
        assert!(
            result.leaves.len() <= prev,
            "leaf count must not grow as the pole recedes: a={a} gave {} after {prev}",
            result.leaves.len()
        );
        prev = result.leaves.len();
    }
}
