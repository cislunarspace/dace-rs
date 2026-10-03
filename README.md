# dace-rs

Pure Rust implementation of [DACE](https://github.com/dacelib/dace), the Differential Algebra Computational Toolbox.

`dace-rs` computes with truncated multivariate Taylor polynomials ("differential algebra"): it propagates high-order expansions of arbitrary multivariate functions through arithmetic, elementary-function composition, and full polynomial map inversion, with a configurable cut-off epsilon and truncation order. It is a full rewrite of DACE 2.1 in safe Rust — no C toolchain at build time, no unsafe code in the computation kernels.

**English** | [简体中文](README.zh-CN.md)

[![CI](https://github.com/cislunarspace/dace-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/cislunarspace/dace-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/dace-rs)](https://crates.io/crates/dace-rs)
[![Docs.rs](https://img.shields.io/docsrs/dace-rs)](https://docs.rs/dace-rs)
[![License](https://img.shields.io/crates/l/dace-rs)](#license)

## Contents

- [Why dace-rs](#why-dace-rs)
- [Quickstart](#quickstart)
- [Platform Support](#platform-support)
- [API Map (C++ → Rust)](#api-map-c--rust)
- [Parity Methodology](#parity-methodology)
- [License](#license)

## Why dace-rs

The existing [`dace`](https://crates.io/crates/dace) crate is a C-binding wrapper around the upstream library: it requires a C toolchain and links the C core. `dace-rs` is an independent pure-Rust implementation of the same algorithms and numerics, verified coefficient-for-coefficient against the C reference (see [Parity Methodology](#parity-methodology)). It adds Rust-native ergonomics:

- operator traits (`Add`/`Sub`/`Mul`/`Div`/`Neg` + assignments) for `Da × Da`, `Da × f64`, `f64 × Da`;
- values survive re-initialization (each `Da` keeps its context; C purges everything);
- recoverable failures return `Result`, domain misuse panics with the C error code, informational degradation goes through the [`log`] crate — no sticky error state, no `exit(1)`;
- automatic domain splitting ([`ads::split`], a Rust-only extension with no C counterpart) keeps polynomial enclosures rigorous over large uncertainty boxes via recursive bisection and re-expansion.

## Quickstart

```rust
use dace_rs::Da;

fn main() {
    dace_rs::init(20, 2).unwrap();                 // order 20, 2 variables
    let x = Da::variable(1);
    let y = Da::variable(2);

    let f = (1.0 + x.clone() * y.clone()).sin();   // Taylor expansion of sin(1+xy)
    println!("{f}");                               // daceWrite-style listing

    // Coefficients: coefficient of x*y in sin(1+xy) is cos(1)
    assert!((f.get_coefficient(&[1, 1]) - 1.0f64.cos()).abs() < 1e-14);
}
```

See [`examples/quickstart.rs`](examples/quickstart.rs) for a map-iteration example, and the [crate documentation](https://docs.rs/dace-rs) for the full API.

## Platform Support

All dependencies are pure Rust; there is no `build.rs` and no C toolchain requirement.

| Target | CI |
|---|---|
| Linux x86-64 | `ubuntu-latest` |
| Linux aarch64 | `ubuntu-24.04-arm` |
| Windows x86-64 | `windows-latest` |
| Windows aarch64 | `windows-11-arm` |

Arithmetic kernels use a fixed accumulation order and default floating-point settings (no fast-math, no FMA contraction), so results are bit-identical across the four platforms; only scalar transcendentals (`f64::sin`, ...) may differ by a few ulps between libm implementations.

## API Map (C++ → Rust)

| DACE C++ | dace-rs |
|---|---|
| `DA::DA(var)`, `DA::identity` | [`Da::variable`] / [`Da::identity`] |
| `DA::cons`, `DA::linear`, `DA::gradient` | [`Da::cons`] / [`Da::linear`] / [`Da::gradient`] |
| `DA::getCoefficient`, `setCoefficient` | [`Da::get_coefficient`] / [`Da::set_coefficient`] |
| `DA::deriv`, `DA::integ`, `DA::trim` | [`Da::deriv`] / [`Da::integ`] / [`Da::trim`] |
| `exp, log, sin, tan, asin, ...` | [`dace_rs::exp`] etc. (free fns) and `Da` methods |
| `BesselJFunction` etc. | [`dace_rs::bessel_j`] etc. |
| `GammaFunction`, `PsiFunction` | [`dace_rs::gamma`] / [`dace_rs::psi`] |
| `DA::norm`, `orderNorm`, `estimNorm`, `bound`, `convRadius` | [`Da::norm`] etc. |
| `DA::compile`, `compiledDA::eval` | [`Da::compile`] / [`CompiledDa::eval`] / [`CompiledDa::eval_da`] |
| `DA::plug`, `DA::eval` | [`Da::plug`] / [`Da::eval`] / [`Da::eval_da`] |
| `DA::replaceVariable`, `scaleVariable`, `translateVariable` | [`Da::replace_variable`] etc. |
| `DA::read`/`write` (blob), `operator>>`/`<<` | [`Da::to_blob`]/[`Da::from_blob`], [`Display`]/[`FromStr`] |
| `DASimpleFormatter` | [`SimpleFormat`] presets and [`format_da`] |
| `AlgebraicVector<DA>::invert` | [`DaVector::invert`] |
| — (no C counterpart; Rust-only extension) | [`ads::split`] with [`AdsConfig`]/[`AdsResult`] |

Not ported (upstream experimental/off by default): `AlgebraicMatrix`, `dacecompat` aliases, the MATLAB interface.

## Parity Methodology

The C library has no test suite, so `dace-rs` is verified against the C reference directly:

- **Goldens**: `dev/golden/gen.sh` builds the C library and runs a fixed case table (`dev/golden/main.c`) covering arithmetic, all elementary and special functions, calculus, norms, evaluation, and blob export at `(6,3)`, `(20,6)`, and `(6,2)` contexts with several constant-term choices. The committed `tests/golden/cases.txt` is replayed by `tests/parity.rs` in CI (no C toolchain needed); coefficients agree within rtol `1e-10` (arithmetic within `1e-13`, verified bit-identical in many cases).
- **Bit-level checks during development**: the monomial encoding tables, the multiply/inverse kernels, the text format, and the elementary-function layer were compared byte-for-byte against the C library on this machine.

Two upstream C bugs are deliberately not reproduced; both divergences are documented in the rustdoc of the affected functions:

1. `daceReplaceVariable` indexes its exponent array with 1-based variable numbers (documented semantics say 1-based replacement; implemented is effectively `from+1 → val·(to+1)`, a silent no-op for `from == nvmax`). dace-rs implements the documented semantics.
2. `dacePower` with negative powers calls the multiplicative inverse on an aliased result, and the Newton iteration is not aliasing safe; C returns wrong coefficients (C's `pow(A,-2)` disagrees with C's own `minv(sqr(A))`). dace-rs returns the correct value.

## License

Apache-2.0. `dace-rs` is a derivative work of DACE (Copyright 2016 Politecnico di Milano and contributors; Copyright 2014 Dinamica Srl). See [LICENSE](LICENSE), [NOTICE](NOTICE), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

[`Da::variable`]: https://docs.rs/dace-rs/latest/dace_rs/struct.Da.html#method.variable
[`Da::cons`]: https://docs.rs/dace-rs/latest/dace_rs/struct.Da.html#method.cons
[`Da::compile`]: https://docs.rs/dace-rs/latest/dace_rs/struct.Da.html#method.compile
[`CompiledDa::eval`]: https://docs.rs/dace-rs/latest/dace_rs/struct.CompiledDa.html#method.eval
[`SimpleFormat`]: https://docs.rs/dace-rs/latest/dace_rs/io/struct.SimpleFormat.html
[`format_da`]: https://docs.rs/dace-rs/latest/dace_rs/io/fn.format_da.html
[`DaVector::invert`]: https://docs.rs/dace-rs/latest/dace_rs/vector/trait.DaVector.html#tymethod.invert
[`ads::split`]: https://docs.rs/dace-rs/latest/dace_rs/ads/fn.split.html
[`AdsConfig`]: https://docs.rs/dace-rs/latest/dace_rs/ads/struct.AdsConfig.html
[`AdsResult`]: https://docs.rs/dace-rs/latest/dace_rs/ads/struct.AdsResult.html
[`log`]: https://docs.rs/log
