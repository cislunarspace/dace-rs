//! Cross-thread behavior of `Da`: lazy per-thread initialization of
//! computation settings and scratch buffers, invalidation of those settings
//! when the global context is re-initialized, and mixing values from
//! different context generations.

use std::sync::LazyLock;
use std::sync::mpsc;
use std::thread;

use dace_rs::Da;
use parking_lot::Mutex;

/// Serialize tests sharing the process-global DACE context.
static CONTEXT_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

#[test]
fn worker_first_use_and_alternating() {
    let _g = CONTEXT_LOCK.lock();
    dace_rs::init(3, 2).unwrap();
    let x = Da::variable(1);
    let coeff = thread::spawn(move || {
        // First Da use on this thread: settings and multiply scratch are
        // derived lazily from the active context, no explicit init needed.
        let y = Da::variable(2);
        (x * y).get_coefficient(&[1, 1])
    })
    .join()
    .unwrap();
    assert_eq!(coeff, 1.0);
}

#[test]
fn reinit_same_shape_warmed_thread() {
    let _g = CONTEXT_LOCK.lock();
    dace_rs::init(5, 2).unwrap();

    let (to_worker, worker_rx) = mpsc::channel::<Da>();
    let (to_main, main_rx) = mpsc::channel::<f64>();

    let handle = thread::spawn(move || {
        // Round 1: warm this thread's settings and multiply scratch under G1.
        let y = Da::variable(2);
        let warm = y.clone() * y.clone();
        to_main.send(warm.get_coefficient(&[0, 2])).unwrap();
        // Round 2: value created under the re-initialized context.
        let z = worker_rx.recv().unwrap();
        let z2 = z.clone() * z.clone();
        to_main.send(z2.get_coefficient(&[2, 0])).unwrap();
        z2
    });

    assert_eq!(main_rx.recv().unwrap(), 1.0); // worker warmed under G1
    dace_rs::init(5, 2).unwrap(); // G2: same shape, new generation
    to_worker.send(Da::variable(1)).unwrap();

    assert_eq!(main_rx.recv().unwrap(), 1.0);
    let joined = handle.join().unwrap();
    assert_eq!(joined.size(), 1);
}

#[test]
fn reinit_smaller_nomax_no_panic_correct() {
    let _g = CONTEXT_LOCK.lock();
    dace_rs::init(5, 2).unwrap();

    let (to_worker, worker_rx) = mpsc::channel::<Da>();
    let (to_main, main_rx) = mpsc::channel::<f64>();

    let handle = thread::spawn(move || {
        // Warm this thread under (5,2): settings nocut = 5.
        let y = Da::variable(2);
        let warm = y.clone() * y.clone();
        to_main.send(warm.get_coefficient(&[0, 2])).unwrap();
        let z = worker_rx.recv().unwrap();
        let z2 = z.clone() * z.clone();
        to_main.send(z2.get_coefficient(&[2, 0])).unwrap();
        z2
    });

    assert_eq!(main_rx.recv().unwrap(), 1.0);
    dace_rs::init(3, 2).unwrap(); // smaller nomax on re-init
    to_worker.send(Da::variable(1)).unwrap();

    // Pre-fix: worker's stale nocut = 5 indexes ipbeg out of bounds for the
    // (3,2) scratch and panics; must instead re-derive and compute x^2 = 1.
    assert_eq!(main_rx.recv().unwrap(), 1.0);
    let joined = handle.join().unwrap();
    assert_eq!(joined.get_coefficient(&[2, 0]), 1.0);
    assert_eq!(joined.size(), 1);
}

#[test]
fn reinit_larger_nomax_no_silent_truncation() {
    let _g = CONTEXT_LOCK.lock();
    dace_rs::init(3, 2).unwrap();

    let (to_worker, worker_rx) = mpsc::channel::<Da>();
    let (to_main, main_rx) = mpsc::channel::<f64>();

    let handle = thread::spawn(move || {
        // Warm this thread under (3,2): settings nocut = 3.
        let y = Da::variable(2);
        let warm = y.clone() * y.clone();
        to_main.send(warm.get_coefficient(&[0, 2])).unwrap();
        let z2 = worker_rx.recv().unwrap();
        let z4 = z2.clone() * z2.clone();
        to_main.send(z4.get_coefficient(&[4, 0])).unwrap();
        z4
    });

    assert_eq!(main_rx.recv().unwrap(), 1.0);
    dace_rs::init(5, 2).unwrap(); // larger nomax on re-init
    let x = Da::variable(1);
    let x2 = x.clone() * x.clone(); // x^2 under the new context
    to_worker.send(x2).unwrap();

    // Pre-fix: worker's stale nocut = 3 silently truncates x^4 to 0; must
    // instead re-derive nocut = 5 and keep the x^4 coefficient.
    assert_eq!(main_rx.recv().unwrap(), 1.0);
    let joined = handle.join().unwrap();
    assert_eq!(joined.size(), 1);
}

#[test]
fn mixed_generation_values_no_panic() {
    let _g = CONTEXT_LOCK.lock();
    dace_rs::init(3, 2).unwrap();
    let old = Da::variable(1);
    dace_rs::init(5, 2).unwrap();
    // `old` keeps its original (3,2) context while this thread's truncation
    // order now derives from the new context; the kernel must clamp to the
    // operand context instead of indexing out of bounds.
    let sq = old.clone() * old.clone();
    assert_eq!(sq.get_coefficient(&[2, 0]), 1.0);
}

#[test]
fn truncation_settings_persist_within_generation() {
    let _g = CONTEXT_LOCK.lock();
    dace_rs::init(5, 2).unwrap();

    let (to_worker, worker_rx) = mpsc::channel::<()>();
    let (to_main, main_rx) = mpsc::channel::<()>();

    let handle = thread::spawn(move || {
        // First use initializes this thread's settings, then set a custom
        // truncation order.
        let _ = Da::variable(1) + Da::constant(1.0);
        dace_rs::set_truncation_order(2);
        to_main.send(()).unwrap();
        worker_rx.recv().unwrap();
        // An unrelated Da op on this thread must not reset the setting: no
        // re-initialization happened in between.
        let _ = Da::variable(2) * Da::constant(2.0);
        assert_eq!(dace_rs::truncation_order(), 2);
        to_main.send(()).unwrap();
    });

    main_rx.recv().unwrap();
    to_worker.send(()).unwrap();
    main_rx.recv().unwrap();
    handle.join().unwrap();
}
