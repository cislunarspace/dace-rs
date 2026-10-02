# Third-party notices

## netlib psi (dace-rs `special::netlib_psi_zeta`)

The polygamma (`psi`) implementation transcribed into
`src/special/netlib_psi_zeta.rs` derives from the netlib FORTRAN routine
`PSIFN` by W. J. Cody et al. (Mathematics and Computer Science Division,
Argonne National Laboratory, 1988; rational Chebyshev approximations from
Math. Comp. 27, 123-127, 1973), as vendored (f2c-translated) in DACE's
`contrib/psi.c`. Netlib/Slatec public-domain-style terms.

## Cephes zeta (dace-rs `special::netlib_psi_zeta`)

The Hurwitz zeta implementation in the same file derives from the Cephes
Math Library Release 2.8 (June 2000), `zeta.c`, Copyright 1984, 1995, 2000
Stephen L. Moshier, as vendored in DACE's `contrib/zeta.c`. The original
Fortran/C comments are preserved in the transcription.

## puruspe

Scalar special functions (Bessel J/Y/I/K, gamma, error function) are
provided by the `puruspe` crate (MIT OR Apache-2.0), itself based on the
Cephes mathematical library by Stephen L. Moshier and Numerical Recipes
routines.

## DACE

This crate is a pure Rust rewrite of DACE 2.1
(https://github.com/dacelib/dace), Apache-2.0, Copyright 2016 Politecnico
di Milano and contributors (Copyright 2014 Dinamica Srl). See LICENSE and
NOTICE.
