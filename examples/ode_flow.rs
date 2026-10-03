//! DA-arithmetic ODE integration: from initial-value offsets to a polynomial
//! flow (dace-rs issue #13).
//!
//! A fixed-step classical RK4 integrator operates directly on
//! [`dace_rs::Da`] values: each initial condition is written as (nominal
//! constant) + (DA variables standing for the offsets), and the integrator
//! carries those variables through the flow, so the terminal state comes out
//! as a truncated Taylor polynomial in the offsets — the *polynomial flow*
//! of the ODE.
//!
//! Run with `cargo run --release --example ode_flow`. `--release` matters:
//! the Kepler section multiplies order-8 polynomials in 4 variables (up to
//! 495 monomials each) thousands of times, which is impractically slow in a
//! debug build.
//!
//! Three demonstrations:
//!
//! 1. **Harmonic oscillator** — the flow of ẍ = −x is exactly linear, so the
//!    computed polynomial must reproduce the analytic flow matrix
//!    [[cos T, sin T], [−sin T, cos T]]; RK4's O(h⁴) discretization error is
//!    the only error left, and no monomial above order 1 can appear.
//! 2. **Planar two-body Kepler orbit** — the energy of the terminal
//!    polynomial is compared with the initial energy; the residual's Taylor
//!    coefficients shrink by ≈2⁴ when the step h is halved, exposing RK4's
//!    global error order.
//! 3. **Validity domain of the polynomial flow** — evaluating the terminal
//!    polynomial at sampled offsets and re-integrating the offset initial
//!    condition in plain f64 (same scheme, same step) isolates the pure DA
//!    truncation error, which grows with the offset-box radius.
//!
//! # The h–p–ρ picture
//!
//! Three independent knobs control what the printed tables show:
//!
//! * **h (integrator step).** RK4's O(h⁴) *global* error bounds the constant
//!   part and, because the scheme is applied to polynomial data, every Taylor
//!   coefficient of the result: halving h shrinks the whole energy-residual
//!   polynomial by ≈2⁴ = 16 (printed and asserted in the Kepler section).
//! * **p (DA truncation order).** The polynomial flow is a truncated Taylor
//!   series of the true flow. Its remainder over an offset box of radius s
//!   grows like (s/ρ)^p, where ρ is the distance from the nominal point to
//!   the flow's nearest singularity in (possibly complex) offset space. For
//!   this Kepler setup ρ is far smaller than the real collision distance:
//!   the terminal polynomial's order norms *grow* by ×10–15 per order (a
//!   genuine property of the flow map, not rounding noise), so the flow is
//!   usable only for offsets ≲ 1e-2, and the validity-domain table shows
//!   the deviation climbing from ~2e-7 at s = 0.01 to ~5e8 at s = 0.4. A
//!   polynomial flow must be validated before use — that table is how.
//! * **ρ (convergence radius).** [`Da::conv_radius`] estimates from the
//!   order-norm decay the offset radius at which the truncation remainder
//!   drops below a given eps; its value is printed for comparison against
//!   the empirical deviation table (informational only: the underlying
//!   exponential fit may warn, as in C).

use dace_rs::{Da, NormType};

fn main() {
    harmonic_oscillator();
    let (y0, y_final) = kepler();

    let mut rng = Lcg(0x853c49e6748fea9b);
    validity_domain(&y0, &y_final, &mut rng);
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// One classical RK4 step in DA arithmetic: elementwise
/// y ← y + (h/6)(k1 + 2k2 + 2k3 + k4) with k's from the vector field `f`.
///
/// Because the tableau only combines stages with scalars, an exactly linear
/// vector field stays linear: DA·DA products in the *right-hand side* are
/// what create higher-order terms, never the stepper itself.
fn rk4_step_da(f: &dyn Fn(&[Da]) -> Vec<Da>, y: &[Da], h: f64) -> Vec<Da> {
    let axpy = |y: &[Da], k: &[Da], a: f64| -> Vec<Da> {
        y.iter()
            .zip(k)
            .map(|(yi, ki)| yi.clone() + a * ki.clone())
            .collect()
    };

    let k1 = f(y);
    let k2 = f(&axpy(y, &k1, 0.5 * h));
    let k3 = f(&axpy(y, &k2, 0.5 * h));
    let k4 = f(&axpy(y, &k3, h));

    y.iter()
        .enumerate()
        .map(|(i, yi)| {
            yi.clone()
                + (h / 6.0)
                    * (k1[i].clone() + 2.0 * k2[i].clone() + 2.0 * k3[i].clone() + k4[i].clone())
        })
        .collect()
}

/// One classical RK4 step on plain f64, same tableau as [`rk4_step_da`].
///
/// A separate f64 stepper is needed because the crate has no generic
/// axpy-style combinator shared between `f64` and `Da` states.
fn rk4_step_f64(f: &dyn Fn(&[f64; 4]) -> [f64; 4], y: &[f64; 4], h: f64) -> [f64; 4] {
    let axpy = |y: &[f64; 4], k: [f64; 4], a: f64| -> [f64; 4] {
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = y[i] + a * k[i];
        }
        out
    };

    let k1 = f(y);
    let k2 = f(&axpy(y, k1, 0.5 * h));
    let k3 = f(&axpy(y, k2, 0.5 * h));
    let k4 = f(&axpy(y, k3, h));

    let mut out = [0.0; 4];
    for i in 0..4 {
        out[i] = y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    out
}

/// A 64-bit linear congruational generator for reproducible box sampling
/// (`Da::random` draws random *polynomials*, not scalars).
struct Lcg(u64);

impl Lcg {
    /// Uniform draw in [−1, 1).
    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        2.0 * u - 1.0
    }
}

// ---------------------------------------------------------------------------
// 1. Harmonic oscillator: analytic flow-matrix check
// ---------------------------------------------------------------------------

/// Integrate ẍ = −x (ω = 1) from x(0) = 1 + δx, v(0) = δv over T = 1.7 with
/// 4096 RK4 steps and compare every Taylor coefficient against the analytic
/// flow matrix of ẍ = −x:
///
/// ```text
/// [x(T)]   [ cos T  sin T] [x(0)]
/// [v(T)] = [−sin T  cos T] [v(0)]
/// ```
fn harmonic_oscillator() {
    dace_rs::init(20, 2).expect("init(20, 2)");
    println!("== harmonic oscillator: x'' = -x, order 20, 2 vars ==");

    let rhs = |y: &[Da]| vec![y[1].clone(), -y[0].clone()];
    let y0 = vec![1.0 + Da::variable(1), Da::variable(2)];

    let t = 1.7_f64;
    let n = 4096;
    let h = t / n as f64;
    let mut y = y0;
    for _ in 0..n {
        y = rk4_step_da(&rhs, &y, h);
    }
    let (x, v) = (&y[0], &y[1]);

    // Analytic Taylor coefficients of the flow at T, given x(0) = 1 + δx,
    // v(0) = δv: the flow is linear, so the polynomial must terminate at
    // order 1 with exactly these coefficients.
    let (c, s) = (t.cos(), t.sin());
    let lin_x = x.linear();
    let lin_v = v.linear();
    let rows = [
        ("x(T) const", x.cons(), c),
        ("x(T) d/dx0", lin_x[0], c),
        ("x(T) d/dv0", lin_x[1], s),
        ("v(T) const", v.cons(), -s),
        ("v(T) d/dx0", lin_v[0], -s),
        ("v(T) d/dv0", lin_v[1], c),
    ];
    println!("\nT = {t}, h = {h:.6e}, n = {n} RK4 steps");
    println!("coefficient      DA value            analytic            deviation");
    let mut max_dev = 0.0_f64;
    for (label, da_val, analytic) in rows {
        let dev = (da_val - analytic).abs();
        max_dev = max_dev.max(dev);
        println!("{label:<12} {da_val:+19.15} {analytic:+19.15} {dev:.3e}");
        assert!(dev < 1e-12, "{label}: deviation {dev:.3e} >= 1e-12");
    }
    println!("max deviation from the analytic flow: {max_dev:.3e}");

    // The RK4 tableau only ever forms scalar·DA and DA+DA, and the RHS is
    // linear in (x, v): no mechanism exists to create a second-order
    // monomial, so these coefficients are exactly zero.
    for (label, q) in [
        ("x(T) dx^2", x.get_coefficient(&[2, 0])),
        ("x(T) dx dv", x.get_coefficient(&[1, 1])),
        ("x(T) dv^2", x.get_coefficient(&[0, 2])),
        ("v(T) dx^2", v.get_coefficient(&[2, 0])),
        ("v(T) dx dv", v.get_coefficient(&[1, 1])),
        ("v(T) dv^2", v.get_coefficient(&[0, 2])),
    ] {
        assert_eq!(q, 0.0, "{label} must be exactly 0.0, got {q}");
    }
    println!("all second-order coefficients are exactly 0.0 (linear flow)");

    println!("\nx(T) polynomial:\n{x}");
}

// ---------------------------------------------------------------------------
// 2. Planar two-body Kepler orbit: energy-residual scaling in h
// ---------------------------------------------------------------------------

/// Specific orbital energy of the planar two-body state (x, y, vx, vy) with
/// μ = 1: E = |v|²/2 − 1/r, as a polynomial in the initial offsets.
fn energy_da(y: &[Da]) -> Da {
    0.5 * (y[2].sqr() + y[3].sqr()) - (y[0].sqr() + y[1].sqr()).sqrt().minv()
}

/// Integrate one period of a planar two-body orbit (μ = 1, a = 1, e = 0.2)
/// from the perigee state (0.8, 0, 0, √1.5) + DA offsets in all four
/// components, with 128 and then 256 RK4 steps. Returns the initial DA
/// state and the 256-step terminal flow.
///
/// The energy residual E(y_final) − E(y_initial) is a polynomial whose
/// coefficients contain only numerical error; because RK4's global error is
/// O(h⁴), every order-norm row must shrink by ≈2⁴ = 16 when h is halved.
fn kepler() -> (Vec<Da>, Vec<Da>) {
    dace_rs::init(8, 4).expect("init(8, 4)");
    println!("\n== planar two-body Kepler orbit: order 8, 4 vars ==");

    let rhs = |y: &[Da]| -> Vec<Da> {
        let r2 = y[0].sqr() + y[1].sqr();
        // 1/r³ via one Newton inverse of r²·r instead of four Da/Da
        // divisions in the accelerations.
        let r3inv = (r2.clone() * r2.sqrt()).minv();
        vec![
            y[2].clone(),
            y[3].clone(),
            -(y[0].clone() * r3inv.clone()),
            -(y[1].clone() * r3inv),
        ]
    };

    // Perigee initial state: r0 = a(1-e) = 0.8, v0^2 = 2/r0 - 1/a = 1.5.
    let base = [0.8, 0.0, 0.0, 1.5_f64.sqrt()];
    let y0: Vec<Da> = (0..4)
        .map(|i| Da::constant(base[i]) + Da::variable(i as u32 + 1))
        .collect();
    let e0 = energy_da(&y0).cons();
    println!("E(y0) = {e0:+.15} (expected -0.5)");

    let t = std::f64::consts::TAU;
    let mut y_final = Vec::new();
    let mut metrics = Vec::new();
    let mut residuals = Vec::new();
    for &n in &[128_usize, 256] {
        let h = t / n as f64;
        let mut y = y0.clone();
        for _ in 0..n {
            y = rk4_step_da(&rhs, &y, h);
        }

        let p = energy_da(&y) - energy_da(&y0);
        let onorm = p.order_norm(0, NormType::Infinity);
        // Metric: the largest order-norm among orders 1..=4 (skip order 0:
        // the constant part is checked separately below).
        let m = onorm[1..=4].iter().copied().fold(0.0_f64, f64::max);

        println!(
            "\nenergy residual P = E(y_final) - E(y0), n = {n} steps (h = {h:.6e}):"
        );
        println!("  P.const = {:+.3e}", p.cons());
        println!("  order-norm rows (infinity norm per order):");
        for (order, norm) in onorm.iter().enumerate().skip(1) {
            println!("    order {order}: {norm:.3e}");
        }
        metrics.push((n, m));
        residuals.push(p.cons());
        y_final = y;
    }

    let (m1, m2) = (metrics[0].1, metrics[1].1);
    let ratio = m1 / m2;
    println!("\nmetric m = max order-norm over orders 1..=4 of P:");
    println!("  m(h)      = {:.3e}", m1);
    println!("  m(h/2)    = {:.3e}", m2);
    println!("  m(h)/m(h/2) = {ratio:.3} (RK4 global error O(h^4) -> expected ~16)");
    assert!(
        (10.0..=26.0).contains(&ratio),
        "h^4 scaling broken: m(h)/m(h/2) = {ratio}"
    );
    assert!(
        residuals[1].abs() <= 1e-5,
        "constant-part energy drift too large: {}",
        residuals[1]
    );

    // Order norms of the terminal flow itself. They grow steeply (×10–15
    // per order) — a genuine property of the flow map (verified: converged
    // in the step count and identical across higher DA truncation orders),
    // not rounding noise. The fixed-time Kepler flow map has a nearby
    // singularity in complex offset space, so the polynomial flow diverges
    // once the offset box approaches it (next section).
    let ox = y_final[0].order_norm(0, NormType::Infinity);
    let ovx = y_final[2].order_norm(0, NormType::Infinity);
    println!("\norder norms of the 256-step terminal flow (they grow: small convergence radius):");
    println!("  order     x(T)              vx(T)");
    for order in 0..=8 {
        println!("  {order:>5}  {:>15.3e}  {:>15.3e}", ox[order], ovx[order]);
    }
    let rho = y_final[0].conv_radius(1e-10, NormType::Infinity);
    println!(
        "\nconv_radius(x(T), eps = 1e-10) = {rho:.6}   \
         (estimated offset radius at which the next-order remainder of the\n\
         \x20        flow polynomial drops below eps; informational only, the\n\
         \x20        exponential fit behind it may warn)"
    );

    (y0, y_final)
}

// ---------------------------------------------------------------------------
// 3. Validity domain of the polynomial flow
// ---------------------------------------------------------------------------

/// Compare the 256-step DA flow of the Kepler orbit against re-integration
/// of offset initial conditions in plain f64 (same RK4 scheme, same step),
/// which isolates the pure DA truncation error: the deviation is the tail
/// of the Taylor series, growing like (s/ρ)^p in the offset-box radius s.
/// Empirically ρ ≈ a few 1e-2 for this base orbit — far inside the real
/// collision distance — so the table shows validity only for s ≲ 0.01.
fn validity_domain(y0: &[Da], y_final: &[Da], rng: &mut Lcg) {
    println!("\n== validity domain of the polynomial flow ==");

    let rhs = |y: &[f64; 4]| -> [f64; 4] {
        let r2 = y[0] * y[0] + y[1] * y[1];
        let r3 = r2 * r2.sqrt();
        [y[2], y[3], -y[0] / r3, -y[1] / r3]
    };

    // Same base state and step count as the 256-step DA run.
    let base = [y0[0].cons(), y0[1].cons(), y0[2].cons(), y0[3].cons()];
    let n = 256_usize;
    let h = std::f64::consts::TAU / n as f64;

    println!("evaluating the flow polynomial vs re-integration, 16 samples per radius:");
    println!("  radius s    max deviation");
    let mut first = 0.0_f64;
    let mut last = 0.0_f64;
    for &s in &[0.01, 0.05, 0.1, 0.2, 0.4] {
        let mut max_dev = 0.0_f64;
        for _ in 0..16 {
            let delta = [rng.next_unit(), rng.next_unit(), rng.next_unit(), rng.next_unit()];
            let delta = delta.map(|u| s * u);

            // Polynomial flow: evaluate each terminal component at δ.
            let poly = [
                y_final[0].eval(&delta),
                y_final[1].eval(&delta),
                y_final[2].eval(&delta),
                y_final[3].eval(&delta),
            ];

            // Reference: RK4 in f64 from the offset initial condition.
            let mut y = base;
            for i in 0..4 {
                y[i] += delta[i];
            }
            for _ in 0..n {
                y = rk4_step_f64(&rhs, &y, h);
            }

            let dev: f64 = poly
                .iter()
                .zip(&y)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            max_dev = max_dev.max(dev);
        }
        println!("  {s:>8.2}  {max_dev:.3e}");
        first = if s == 0.01 { max_dev } else { first };
        last = if s == 0.4 { max_dev } else { last };
    }
    assert!(
        last > first,
        "deviation must grow with the box radius: dev(0.01) = {first:.3e}, dev(0.4) = {last:.3e}"
    );
    println!("the deviation climbs steeply with s: the polynomial flow is an");
    println!("order-8 truncation of a Taylor series whose radius rho lies well");
    println!("inside the real collision distance, so it is valid only for s <~ 1e-2.");
}
