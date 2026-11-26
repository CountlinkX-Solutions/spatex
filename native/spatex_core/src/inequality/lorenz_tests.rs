use super::lorenz::{atkinson, lorenz_curve, schutz, wolfson};

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} vs {b}");
}

#[test]
fn atkinson_matches_the_docstring_values() {
    let y = [10.0, 20.0, 30.0, 40.0, 50.0];
    let half = atkinson(&y, 0.5).unwrap();
    close(half.a, 0.06315, 5e-6);
    close(half.ede, 28.1054, 5e-5);
    let one = atkinson(&y, 1.0).unwrap();
    close(one.a, 0.13161, 5e-6);
    close(one.ede, 26.05171, 5e-6);
}

#[test]
fn atkinson_is_zero_without_aversion_or_inequality() {
    let y = [3.0, 1.0, 8.0];
    close(atkinson(&y, 0.0).unwrap().a, 0.0, 1e-15);
    close(atkinson(&[4.0, 4.0, 4.0], 2.0).unwrap().a, 0.0, 1e-15);
}

#[test]
fn atkinson_grows_with_aversion() {
    let y = [1.0, 2.0, 9.0, 30.0];
    let a: Vec<f64> = [0.5, 1.0, 2.0]
        .iter()
        .map(|&e| atkinson(&y, e).unwrap().a)
        .collect();
    assert!(a[0] < a[1] && a[1] < a[2]);
}

#[test]
fn atkinson_rejects_bad_input() {
    assert!(atkinson(&[1.0, 0.0], 0.5).is_err());
    assert!(atkinson(&[1.0, -2.0], 0.5).is_err());
    assert!(atkinson(&[1.0, 2.0], -0.1).is_err());
    assert!(atkinson(&[], 0.5).is_err());
}

#[test]
fn lorenz_curve_of_a_known_distribution() {
    let (pop, share) = lorenz_curve(&[3.0, 1.0, 2.0, 4.0]);
    assert_eq!(pop, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    let expected = [0.0, 0.1, 0.3, 0.6, 1.0];
    for (s, e) in share.iter().zip(expected) {
        close(*s, e, 1e-15);
    }
}

#[test]
fn schutz_matches_the_docstring_values() {
    let s = schutz(&[1000.0, 2000.0, 1500.0, 3000.0, 2500.0]).unwrap();
    close(s.distance, 0.15, 1e-12);
    close(s.intersection_point, 0.6, 1e-12);
    close(s.coefficient, 7.5, 1e-12);
}

#[test]
fn schutz_of_equal_values_is_zero_at_the_first_point() {
    let s = schutz(&[2.0; 5]).unwrap();
    close(s.distance, 0.0, 1e-15);
    close(s.intersection_point, 0.2, 1e-15);
    close(s.coefficient, 0.0, 1e-12);
}

#[test]
fn schutz_rejects_an_all_zero_vector() {
    assert!(schutz(&[0.0, 0.0]).is_err());
    assert!(schutz(&[]).is_err());
}

#[test]
fn wolfson_matches_the_docstring_values() {
    let w = wolfson(&[
        20000.0, 25000.0, 27000.0, 30000.0, 35000.0, 45000.0, 60000.0, 75000.0, 80000.0, 120000.0,
    ])
    .unwrap();
    close(w, 0.2013, 5e-5);
    let b = wolfson(&[6.0, 6.0, 8.0, 8.0, 10.0, 10.0, 12.0, 12.0]).unwrap();
    close(b, 0.0833, 5e-5);
}

#[test]
fn wolfson_of_one_value_is_zero_and_empty_input_is_an_error() {
    close(wolfson(&[7.0]).unwrap(), 0.0, 1e-15);
    assert!(wolfson(&[]).is_err());
}
