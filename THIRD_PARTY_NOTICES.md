# Third-party notices

## netlib psi/zeta (dace-rs `special::netlib_psi_zeta`)

The polygamma (`psi`) and Riemann zeta implementations transcribed into
`src/special.rs` derive from the FORTRAN 77 routines `PSIFN` and `DCSEVL`
by Amos, Daniel; and `ZETA` references the netlib implementation
(`https://netlib.org/slatec`), as vendored (f2c-translated) in DACE's
`contrib/psi.c` and `contrib/zeta.c`:

- psi.c: "Computes the polygamma functions (derivatives of the log gamma
  function), translated by f2c from the FORTRAN routine PSIFN by
  Donald E. Amos and Sharon L. Daniel (Sandia National Laboratories,
  1984)". Licensed under the netlib/Slatec public-domain-style terms.
- zeta.c: Riemann zeta function, f2c translation of the netlib routine,
  public domain.

The original Fortran comments are preserved in the transcription.

## puruspe

Scalar special functions (Bessel J/Y/I/K, gamma, error function) are
provided by the `puruspe` crate (MIT OR Apache-2.0), itself based on the
Cephes mathematical library by Stephen L. Moshier.

## DACE

This crate is a pure Rust rewrite of DACE 2.1
(https://github.com/dacelib/dace), Apache-2.0, Copyright 2016 Politecnico
di Milano and contributors (Copyright 2014 Dinamica Srl). See LICENSE and
NOTICE.
