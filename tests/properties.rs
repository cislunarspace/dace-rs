//! Deterministic property tests (fixed-seed LCG inputs, no proptest
//! dependency): algebraic identities that must hold for random DAs.

use std::sync::{LazyLock, Mutex};

use dace_rs::{Da, NormType};

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

fn random_da(rng: &mut Lcg) -> Da {
    // Sparse polynomial with constant in [0.5, 1.5] and nonlinear terms
    // (orders 2-4) in [-1, 1]; linear parts are added by callers when
    // needed so they stay well-conditioned.
    let mut d = Da::constant(rng.uniform(0.5, 1.5));
    for _ in 0..6 {
        let e1 = (rng.next_f64() * 3.0) as u32;
        let e2 = (rng.next_f64() * 3.0) as u32;
        if e1 + e2 < 2 || e1 + e2 > 4 {
            continue;
        }
        d += Da::monomial(&[e1, e2], rng.uniform(-0.5, 0.5));
    }
    d
}

fn assert_close(a: &Da, b: &Da, rtol: f64, label: &str) {
    // Compare over the union of both supports: series results may carry
    // tiny residual terms beyond the inputs' orders.
    let mut jj = Vec::new();
    for m in a.iter_monomials() {
        jj.push(m.jj.clone());
    }
    for m in b.iter_monomials() {
        jj.push(m.jj.clone());
    }
    for j in jj {
        let (x, y) = (a.get_coefficient(&j), b.get_coefficient(&j));
        assert!(
            (x - y).abs() <= rtol * y.abs().max(1.0),
            "{label}: {j:?} {x} vs {y}"
        );
    }
}

#[test]
fn exp_log_roundtrip() {
    let _g = CONTEXT_LOCK.lock().unwrap();
    dace_rs::init(10, 2).unwrap();
    let mut rng = Lcg(0xDEADBEEF);
    for _ in 0..10 {
        let d = random_da(&mut rng);
        assert_close(&dace_rs::log(&dace_rs::exp(&d)), &d, 1e-11, "log(exp(d))");
        assert_close(&dace_rs::exp(&dace_rs::log(&d)), &d, 1e-11, "exp(log(d))");
    }
}

#[test]
fn deriv_integ_identity() {
    let _g = CONTEXT_LOCK.lock().unwrap();
    dace_rs::init(10, 2).unwrap();
    let mut rng = Lcg(0xC0FFEE);
    for _ in 0..10 {
        let d = random_da(&mut rng);
        for var in 1..=2 {
            assert_close(&d.integ(var).deriv(var), &d, 1e-12, "deriv(integ)");
        }
    }
}

#[test]
fn eval_matches_monomial_sum() {
    let _g = CONTEXT_LOCK.lock().unwrap();
    dace_rs::init(8, 2).unwrap();
    let mut rng = Lcg(0x5EED);
    for _ in 0..10 {
        let d = random_da(&mut rng);
        let (px, py) = (rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5));
        let mut direct = 0.0;
        for m in d.iter_monomials() {
            let mut t = m.c;
            for (j, &e) in m.jj.iter().enumerate() {
                let base = if j == 0 { px } else { py };
                t *= base.powi(e as i32);
            }
            direct += t;
        }
        let ev = d.eval(&[px, py]);
        assert!(
            (ev - direct).abs() <= 1e-10 * direct.abs().max(1.0),
            "eval {ev} vs {direct}"
        );
    }
}

#[test]
fn sin_cos_pythagoras() {
    let _g = CONTEXT_LOCK.lock().unwrap();
    dace_rs::init(8, 2).unwrap();
    let mut rng = Lcg(0xABCD);
    for _ in 0..5 {
        let d = random_da(&mut rng);
        let s = dace_rs::sin(&d).sqr() + dace_rs::cos(&d).sqr();
        for m in s.iter_monomials() {
            let expect = if m.order() == 0 { 1.0 } else { 0.0 };
            assert!(
                (m.c - expect).abs() <= 1e-11,
                "sin^2+cos^2 at {:?}: {}",
                m.jj,
                m.c
            );
        }
    }
}

#[test]
fn binomial_square() {
    let _g = CONTEXT_LOCK.lock().unwrap();
    dace_rs::init(8, 2).unwrap();
    let mut rng = Lcg(0x1234);
    for _ in 0..5 {
        let a = random_da(&mut rng);
        let b = random_da(&mut rng);
        let lhs = (a.clone() + b.clone()).sqr();
        let rhs = a.clone().sqr() + (2.0 * a.clone() * b.clone()) + b.clone().sqr();
        assert_close(&lhs, &rhs, 1e-11, "(a+b)^2");
    }
}

#[test]
fn invert_roundtrip_random_maps() {
    let _g = CONTEXT_LOCK.lock().unwrap();
    use dace_rs::vector::DaVector;
    dace_rs::init(10, 2).unwrap();
    let mut rng = Lcg(0xFEED);
    for trial in 0..5 {
        // Ensure a nonsingular linear part (component 0 gets x, 1 gets y).
        let mut map = vec![random_da(&mut rng), random_da(&mut rng)];
        map[0] = map[0].clone() + 0.7 * Da::variable(1);
        map[1] = map[1].clone() + 0.6 * Da::variable(2);
        let inv = map.invert();
        for k in 0..5 {
            let px = rng.uniform(-0.05, 0.05);
            let py = rng.uniform(-0.05, 0.05);
            let img = map.eval(&[px, py]);
            let back = inv.eval(&img);
            for i in 0..2 {
                // residual limited by the order-10 truncation of the
                // inverse map with O(1) coefficients
                assert!(
                    (back[i] - [px, py][i]).abs() < 1e-6,
                    "trial {trial} point {k} coord {i}: {} vs {}",
                    back[i],
                    [px, py][i]
                );
            }
        }
    }
}

#[test]
fn norms_are_consistent() {
    let _g = CONTEXT_LOCK.lock().unwrap();
    dace_rs::init(8, 2).unwrap();
    let mut rng = Lcg(0x9999);
    for _ in 0..10 {
        let d = random_da(&mut rng);
        let inf = d.norm(NormType::Infinity);
        let one = d.norm(NormType::One);
        let eu = d.norm(NormType::EUCLIDEAN);
        assert!(inf <= eu + 1e-15 && eu <= one + 1e-12, "norm monotonicity");
        let b = d.bound();
        assert!(b.lo <= b.hi + 1e-15, "bound ordering");
        assert!(b.lo - 1e-12 <= inf || b.lo <= inf, "bound vs norm");
    }
}
