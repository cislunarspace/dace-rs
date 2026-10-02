//! Uninitialized-access behavior: runs in its own test binary (fresh process)
//! so nothing has called [`dace_rs::init`] yet.

use std::panic::catch_unwind;

use dace_rs::max_order;

#[test]
fn accessors_panic_before_init() {
    assert!(!dace_rs::initialized());
    assert!(catch_unwind(max_order).is_err());
    assert!(catch_unwind(dace_rs::machine_epsilon).is_err());
}
