# Security Policy

## Reporting a Vulnerability

Please report vulnerabilities through GitHub's private security advisories:
Security → Advisories → "Report a vulnerability". Do not open public issues
for suspected vulnerabilities.

We offer credit in the advisory unless you prefer to remain anonymous.

## Scope of Support

This policy covers the Rust code in this repository only. Defects in the DACE
C library itself belong upstream: report them at
<https://github.com/dacelib/dace>.

Documented intentional numeric divergences from the C reference (see the
rustdoc of `Da::replace_variable` and `Da::powi`) are not vulnerabilities.

## Response Timeline

- P0 (memory safety issues, exploitable undefined behavior): initial response
  within 3 days.
- Everything else: initial response within 7 days.
