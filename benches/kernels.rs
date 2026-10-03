//! Criterion benchmarks for the core DA kernels (issue #14).
//!
//! Each bench function initializes its own global DACE context: the two
//! configurations (order 6 / 2 vars, order 20 / 6 vars) cannot coexist in one
//! process at a time, and criterion executes bench functions serially, so
//! re-initializing at the top of every function is safe.
//!
//! Run everything with `cargo bench --bench kernels`, or filter a single
//! benchmark with e.g. `cargo bench mul/20x6`.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use dace_rs::vector::DaVector;
use dace_rs::{Da, elementary, init};

/// Low-config first input: 1 + x1 + x1*x2.
fn poly_x(vars: &[Da]) -> Da {
    vars[0].clone() + vars[0].clone() * vars[1].clone() + 1.0
}

/// High-config first input: 1 + x1 + x2*x3.
fn poly_x6(vars: &[Da]) -> Da {
    vars[0].clone() + vars[1].clone() * vars[2].clone() + 1.0
}

/// Low-config second input: 1 + x2 - x1*x2.
fn poly_y(vars: &[Da]) -> Da {
    vars[1].clone() - vars[0].clone() * vars[1].clone() + 1.0
}

/// High-config second input: 1 + x4 - x5*x6.
fn poly_y6(vars: &[Da]) -> Da {
    vars[3].clone() - vars[4].clone() * vars[5].clone() + 1.0
}

/// Near-identity map on `n` variables (1-based component i):
/// xi + 0.1 * x((i mod n) + 1) * x(((i+1) mod n) + 1), indices wrapping.
/// The linear part is the identity matrix, so `DaVector::invert` applies.
fn near_identity_map(n: u32) -> Vec<Da> {
    let vars: Vec<Da> = (1..=n).map(Da::variable).collect();
    (1..=n)
        .map(|i| {
            vars[(i as usize) - 1].clone()
                + 0.1
                    * vars[(i as usize) % n as usize].clone()
                    * vars[(i as usize + 1) % n as usize].clone()
        })
        .collect()
}

fn mul_6x2(c: &mut Criterion) {
    init(6, 2).unwrap();
    let vars: Vec<Da> = (1..=2).map(Da::variable).collect();
    let x = poly_x(&vars);
    let y = poly_y(&vars);
    c.bench_function("mul/6x2", |b| b.iter(|| black_box(x.clone() * y.clone())));
}

fn mul_20x6(c: &mut Criterion) {
    init(20, 6).unwrap();
    let vars: Vec<Da> = (1..=6).map(Da::variable).collect();
    let x = poly_x6(&vars);
    let y = poly_y6(&vars);
    c.bench_function("mul/20x6", |b| b.iter(|| black_box(x.clone() * y.clone())));
}

fn sin_6x2(c: &mut Criterion) {
    init(6, 2).unwrap();
    let vars: Vec<Da> = (1..=2).map(Da::variable).collect();
    let x = poly_x(&vars);
    c.bench_function("sin/6x2", |b| {
        b.iter(|| black_box(elementary::sin(black_box(&x))))
    });
}

fn sin_20x6(c: &mut Criterion) {
    init(20, 6).unwrap();
    let vars: Vec<Da> = (1..=6).map(Da::variable).collect();
    let x = poly_x6(&vars);
    c.bench_function("sin/20x6", |b| {
        b.iter(|| black_box(elementary::sin(black_box(&x))))
    });
}

fn sqrt_6x2(c: &mut Criterion) {
    init(6, 2).unwrap();
    let vars: Vec<Da> = (1..=2).map(Da::variable).collect();
    let x = poly_x(&vars);
    c.bench_function("sqrt/6x2", |b| {
        b.iter(|| black_box(elementary::sqrt(black_box(&x))))
    });
}

fn sqrt_20x6(c: &mut Criterion) {
    init(20, 6).unwrap();
    let vars: Vec<Da> = (1..=6).map(Da::variable).collect();
    let x = poly_x6(&vars);
    c.bench_function("sqrt/20x6", |b| {
        b.iter(|| black_box(elementary::sqrt(black_box(&x))))
    });
}

fn exp_6x2(c: &mut Criterion) {
    init(6, 2).unwrap();
    let vars: Vec<Da> = (1..=2).map(Da::variable).collect();
    let x = poly_x(&vars);
    c.bench_function("exp/6x2", |b| {
        b.iter(|| black_box(elementary::exp(black_box(&x))))
    });
}

fn exp_20x6(c: &mut Criterion) {
    init(20, 6).unwrap();
    let vars: Vec<Da> = (1..=6).map(Da::variable).collect();
    let x = poly_x6(&vars);
    c.bench_function("exp/20x6", |b| {
        b.iter(|| black_box(elementary::exp(black_box(&x))))
    });
}

fn compile_6x2(c: &mut Criterion) {
    init(6, 2).unwrap();
    let vars: Vec<Da> = (1..=2).map(Da::variable).collect();
    let z = poly_x(&vars) * poly_y(&vars);
    c.bench_function("compile/6x2", |b| b.iter(|| black_box(z.compile())));
}

fn compile_20x6(c: &mut Criterion) {
    init(20, 6).unwrap();
    let vars: Vec<Da> = (1..=6).map(Da::variable).collect();
    let z = poly_x6(&vars) * poly_y6(&vars);
    c.bench_function("compile/20x6", |b| b.iter(|| black_box(z.compile())));
}

fn eval_6x2(c: &mut Criterion) {
    init(6, 2).unwrap();
    let vars: Vec<Da> = (1..=2).map(Da::variable).collect();
    let z = poly_x(&vars) * poly_y(&vars);
    let compiled = z.compile();
    let args = vec![0.1; 2];
    c.bench_function("eval/6x2", |b| {
        b.iter(|| black_box(compiled.eval(black_box(&args))))
    });
}

fn eval_20x6(c: &mut Criterion) {
    init(20, 6).unwrap();
    let vars: Vec<Da> = (1..=6).map(Da::variable).collect();
    let z = poly_x6(&vars) * poly_y6(&vars);
    let compiled = z.compile();
    let args = vec![0.1; 6];
    c.bench_function("eval/20x6", |b| {
        b.iter(|| black_box(compiled.eval(black_box(&args))))
    });
}

fn invert_6x2(c: &mut Criterion) {
    init(6, 2).unwrap();
    let map = near_identity_map(2);
    c.bench_function("invert/6x2", |b| b.iter(|| black_box(map.invert())));
}

fn invert_10x4(c: &mut Criterion) {
    init(10, 4).unwrap();
    let map = near_identity_map(4);
    c.bench_function("invert/10x4", |b| b.iter(|| black_box(map.invert())));
}

criterion_group!(
    benches,
    mul_6x2,
    mul_20x6,
    sin_6x2,
    sin_20x6,
    sqrt_6x2,
    sqrt_20x6,
    exp_6x2,
    exp_20x6,
    compile_6x2,
    compile_20x6,
    eval_6x2,
    eval_20x6,
    invert_6x2,
    invert_10x4
);
criterion_main!(benches);
