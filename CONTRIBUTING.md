# Contributing to dace-rs

Thanks for your interest in contributing! This document covers the development
environment, the checks CI runs on every pull request, and the conventions the
codebase follows.

## Development Environment

dace-rs is pure Rust: a stable Rust toolchain is all you need (MSRV 1.85, per
`rust-version` in `Cargo.toml`). There are no system dependencies.

Optional: a C toolchain plus cmake, with the DACE C sources available at
`DACE_C_SRC` (default `/home/ouyangjiahong/codes/dace`), is needed only to
regenerate the golden parity data via `bash dev/golden/gen.sh`. CI never needs
it — CI only replays the committed `tests/golden/cases.txt`.

## Local Verification

Run the same gates CI runs before opening a pull request:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

Also verify behavior with the quickstart example:

```bash
cargo run --example quickstart
```

It expands `sin(1 + x·y)` and asserts that the `x³` coefficient of `sin(x)` is
`-1/6`.

## Commit Convention

Use an imperative English subject line, e.g. "Add norms, bounds, and
polynomial evaluation". The explanatory body may be in Chinese. No
conventional-commit prefixes are required.

## Code Conventions

- Write rustdoc comments and other code comments in English.
- Ports of C functions keep the C symbol name in their doc (e.g.
  `daceMultiply`), so the correspondence with upstream stays traceable.
- Divergences from the upstream C library MUST be documented in rustdoc.
  Existing examples: `Da::replace_variable` (C off-by-one bug, faithfully
  documented) and `Da::powi` (C aliasing bug in negative powers, corrected
  here).
- Numeric parity with the C reference is anchored by `tests/golden/cases.txt`.
  Regenerating that file requires the C reference (see Development
  Environment); changes to it go through a pull request containing the updated
  file.

## Pull Request Process

`master` is protected: changes land through pull requests with the strict CI
(required on all four target platforms: linux-x64, linux-arm64, windows-x64,
windows-arm64). Include your local verification output in the PR description
and complete the checklist in the PR template.
