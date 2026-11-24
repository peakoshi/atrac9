#![forbid(unsafe_code)]
#![allow(clippy::expect_used)]

mod support;

#[test]
fn persistent_worker_answers_multiple_requests_and_exits_on_drop() {
    let config = support::fixtures::config(7, 2, 3, 2048);
    let frames = support::fixtures::advanced_frames(true, 4, None);
    let mut worker = support::oracle::Timer::new(config, &frames).expect("worker handshake");
    for repeats in [1, 3, 2] {
        assert!(
            !worker
                .timed(repeats)
                .expect("worker timing response")
                .is_zero()
        );
    }
    assert!(worker.timed(0).is_err());
    assert!(!worker.timed(1).expect("worker remains usable").is_zero());
    drop(worker);
}

#[test]
fn worker_reports_initialization_failure_without_hanging() {
    assert!(support::oracle::Timer::new([0; 4], &[vec![0; 8]]).is_err());
}
