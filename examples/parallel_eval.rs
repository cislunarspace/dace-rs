//! Rayon-parallel evaluation of a compiled polynomial.
//!
//! `Da` and `CompiledDa` are `Send + Sync`: a polynomial built on the main
//! thread can be evaluated from many worker threads without extra setup —
//! each thread derives its computation settings lazily from the active
//! context. Per-point evaluation is deterministic, so parallel and serial
//! results must match exactly.
//!
//! Run with `cargo run --example parallel_eval`.

use rayon::prelude::*;

fn main() {
    dace_rs::init(20, 2).expect("init");

    let x = dace_rs::Da::variable(1);
    let y = dace_rs::Da::variable(2);
    let f = (x.clone() * x.clone() + y).sin();
    let g = f.compile();

    // 400x400 grid over [-1, 1]^2, row-major.
    let n = 400;
    let grid: Vec<[f64; 2]> = (0..n)
        .flat_map(|i| {
            let a = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
            (0..n).map(move |j| [a, -1.0 + 2.0 * j as f64 / (n - 1) as f64])
        })
        .collect();

    let par: Vec<f64> = grid.par_iter().map(|&[a, b]| g.eval(&[a, b])[0]).collect();
    let ser: Vec<f64> = grid.iter().map(|&[a, b]| g.eval(&[a, b])[0]).collect();

    assert_eq!(par, ser);
    println!("Parallel Rayon evaluation of sin(x^2+y) over 160000 points matches serial.");
    println!("checksum = {:e}", ser.iter().sum::<f64>());
}
