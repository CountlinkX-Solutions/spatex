use super::*;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * 1.0_f64.max(b.abs())
}

/// a..e = 0..4; asymmetric, e is an island (same graph as the asymmetric_island fixture).
fn hand() -> Csr {
    Csr::from_arrays(
        5,
        &[0, 0, 1, 2, 3, 3, 4],
        &[1, 2, 0, 3, 2, 0, 4],
        &[1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 0.0],
    )
    .unwrap()
}

#[test]
fn canonical_order_and_arrays_roundtrip() {
    let g = Csr::from_arrays(3, &[2, 0, 0, 1], &[0, 2, 1, 2], &[4.0, 2.0, 1.0, 3.0]).unwrap();
    assert_eq!(g.offsets, vec![0, 2, 3, 4]);
    assert_eq!(g.neighbors, vec![1, 2, 2, 0]);
    assert_eq!(g.weights, vec![1.0, 2.0, 3.0, 4.0]);
    let (f, n, w) = g.to_arrays();
    assert_eq!(f, vec![0, 0, 1, 2]);
    assert_eq!(Csr::from_arrays(3, &f, &n, &w).unwrap(), g);
}

#[test]
fn islands_become_zero_weight_self_loops() {
    let g = hand();
    assert_eq!(g.nnz(), 7);
    assert_eq!(g.nonzero_count(), 6);
    assert_eq!(g.row(4), 6..7);
    assert_eq!((g.neighbors[6], g.weights[6]), (4, 0.0));
    assert_eq!(g.isolates(), vec![4]);
    assert_eq!(g.cardinalities(), vec![2, 1, 1, 2, 0]);

    let lone = Csr::from_arrays(2, &[], &[], &[]).unwrap();
    assert_eq!(lone.neighbors, vec![0, 1]);
    assert_eq!(lone.isolates(), vec![0, 1]);
}

#[test]
fn rejects_invalid_input() {
    let e = |r: Result<Csr, String>| r.unwrap_err();
    assert!(e(Csr::from_arrays(2, &[0], &[2], &[1.0])).contains("out of bounds"));
    assert!(e(Csr::from_arrays(2, &[5], &[0], &[1.0])).contains("out of bounds"));
    assert!(e(Csr::from_arrays(2, &[0], &[1], &[f64::NAN])).contains("not finite"));
    assert!(e(Csr::from_arrays(2, &[0], &[1], &[f64::INFINITY])).contains("not finite"));
    assert!(e(Csr::from_arrays(2, &[0, 0], &[1, 1], &[1.0, 2.0])).contains("duplicate edge (0, 1)"));
    assert!(e(Csr::from_arrays(2, &[0], &[1], &[])).contains("equal lengths"));
}

#[test]
fn negative_weights_are_kept() {
    let g = Csr::from_arrays(2, &[0, 1], &[1, 0], &[-1.5, 2.0]).unwrap();
    assert_eq!(g.weights, vec![-1.5, 2.0]);
    assert_eq!(g.cardinalities(), vec![1, 1]);
}

#[test]
fn transforms_match_libpysal_fixture_values() {
    let g = hand();
    let r = g.transform(Transform::R);
    let expect_r = [1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 0.625, 0.375, 0.0];
    assert!(r.weights.iter().zip(expect_r).all(|(&a, b)| close(a, b)));
    assert!(close(r.s0(), 4.0));

    let b = g.transform(Transform::B);
    assert_eq!(b.weights, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]);

    let d = g.transform(Transform::D);
    assert!(close(d.weights[0], 1.0 / 10.5));
    assert!(close(d.s0(), 1.0));

    let v = g.transform(Transform::V);
    assert!(close(v.weights[0], 0.47438348577266526));
    assert!(close(v.weights[5], 0.5457532873125134));
    assert!(close(v.s0(), 5.0));

    assert_eq!(g.transform(Transform::O), g);
}

#[test]
fn transform_parse_is_case_insensitive() {
    assert_eq!(Transform::parse("r").unwrap(), Transform::R);
    assert_eq!(Transform::parse("V").unwrap(), Transform::V);
    assert!(Transform::parse("x").is_err());
}

#[test]
fn lag_and_length_check() {
    let g = hand();
    let y = [1.0, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(g.lag(&y).unwrap(), vec![8.0, 3.0, 2.0, 7.0, 0.0]);
    let lr = g.transform(Transform::R).lag(&y).unwrap();
    assert!(close(lr[0], 2.6666666666666665));
    assert!(close(lr[4], 0.0));
    assert!(g.lag(&y[..4]).unwrap_err().contains("length 4"));
}

#[test]
fn s_statistics_match_fixture() {
    let g = hand();
    assert!(close(g.s0(), 10.5));
    assert!(close(g.s1(), 30.25));
    assert!(close(g.s2(), 124.5));
    let b = g.transform(Transform::B);
    assert!(close(b.s1(), 10.0));
    assert!(close(b.s2(), 38.0));
}

#[test]
fn symmetry_with_tolerance() {
    assert!(!hand().is_symmetric(1e-12));
    let g = Csr::from_arrays(2, &[0, 1], &[1, 0], &[1.0, 1.0 + 1e-14]).unwrap();
    assert!(g.is_symmetric(1e-12));
    assert!(!g.is_symmetric(0.0));
    // one-directional non-zero edge is asymmetric even with a zero reverse entry stored
    let h = Csr::from_arrays(2, &[0, 1], &[1, 0], &[1.0, 0.0]).unwrap();
    assert!(!h.is_symmetric(1e-12));
}

#[test]
fn indptr_is_the_offsets_and_keeps_island_self_loops() {
    let g = hand();
    assert_eq!(g.indptr(), vec![0_i64, 2, 3, 4, 6, 7]);
    // the island (e) is one explicit zero entry, so the CSR holds every stored entry
    assert_eq!(g.indptr().last().copied(), Some(g.nnz() as i64));
    assert_eq!(g.neighbors[g.row(4)].to_vec(), vec![4]);
    assert_eq!(g.weights[g.row(4)].to_vec(), vec![0.0]);
}
