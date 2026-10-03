//! Quickstart: elementary expansion and a simple map iteration.
//!
//! Run with `cargo run --example quickstart`.

fn main() {
    dace_rs::init(20, 2).expect("init");

    let x = dace_rs::Da::variable(1);
    let y = dace_rs::Da::variable(2);

    // Taylor expansion of sin(1 + x*y).
    let f = (1.0 + x.clone() * y.clone()).sin();
    println!("sin(1 + x*y) =\n{f}");

    // Iterate a rotation-like map (Henon-esque) three times and track the
    // constant part of the result.
    let a: f64 = 0.4;
    let (mut xi, mut yi) = (x.clone(), y.clone());
    for k in 0..3 {
        let xn = xi.clone() * a.cos() - yi.clone() * a.sin() + 0.1 * xi.clone() * yi.clone();
        let yn = xi * a.sin() + yi * a.cos() + 0.05 * k as f64;
        xi = xn;
        yi = yn;
    }
    println!("\nafter 3 map iterations:");
    println!("x' =\n{xi}");
    println!("y' =\n{yi}");

    // Taylor check: coefficient of x^3 in sin(x) is -1/6.
    dace_rs::init(20, 2).unwrap();
    let x = dace_rs::Da::variable(1);
    let s = x.sin();
    let c3 = s.get_coefficient(&[3, 0]);
    assert!((c3 + 1.0 / 6.0).abs() < 1e-15, "coefficient of x^3 is {c3}");
    println!(
        "sin(x) x^3 coefficient = {c3:.17} (-1/6 = {})",
        -1.0f64 / 6.0
    );
}
