//! Distance-band graphs, mirroring `libpysal.graph.Graph.build_distance_band` (Euclidean, or
//! haversine kilometres on lon/lat, see [`super::metric`]).
//!
//! The candidate set matches `scipy.spatial.KDTree.sparse_distance_matrix`: ordered pairs
//! with `dx^2 + dy^2 <= threshold^2` (inclusive), self pairs removed, coincident distinct
//! points kept.

use super::knn::{build_tree, split_points};
use super::metric::{search_chord, unit_vectors, Metric};
use crate::graph::Csr;
use crate::limits;

/// Ordered `(focal, neighbor, distance)` candidate pairs with `distance <= threshold` under
/// `metric` (inclusive, self pairs removed, coincident distinct points kept). Collection stops
/// with an error naming `hint` as soon as more than `max_edges` pairs are found.
pub(super) fn band_edges(
    pts: &[[f64; 2]],
    threshold: f64,
    metric: Metric,
    max_edges: usize,
    hint: &str,
) -> Result<Vec<(u32, u32, f64)>, String> {
    if !(threshold.is_finite() && threshold > 0.0) {
        return Err(format!(
            "threshold must be a positive finite number, got {threshold}"
        ));
    }
    let mut edges = Vec::new();
    let mut keep = |i: usize, j: usize, inside: bool, d: f64| -> Result<(), String> {
        if j != i && inside {
            edges.push((i as u32, j as u32, d));
            if edges.len() > max_edges {
                limits::ensure_edges("distance band", edges.len(), max_edges, hint)?;
            }
        }
        Ok(())
    };
    match metric {
        Metric::Euclidean => {
            let t2 = threshold * threshold;
            let tree = build_tree(pts);
            for (i, p) in pts.iter().enumerate() {
                for item in tree.locate_within_distance(*p, t2) {
                    let j = item.data as usize;
                    let (dx, dy) = (p[0] - pts[j][0], p[1] - pts[j][1]);
                    let d2 = dx * dx + dy * dy;
                    keep(i, j, d2 <= t2, d2.sqrt())?;
                }
            }
        }
        Metric::Haversine => {
            // Search a slightly larger chord on the sphere, then keep the exact km distances.
            let v = unit_vectors(pts);
            let tree = build_tree(&v);
            let chord = search_chord(threshold);
            for (i, p) in v.iter().enumerate() {
                for item in tree.locate_within_distance(*p, chord * chord) {
                    let j = item.data as usize;
                    let d = metric.distance(pts[i], pts[j]);
                    keep(i, j, d <= threshold, d)?;
                }
            }
        }
    }
    Ok(edges)
}

const BAND_HINT: &str = "stopped collecting; use a smaller threshold";

/// Builds the distance-band graph: points within `threshold` (inclusive) are neighbors.
///
/// `binary` gives every neighbor weight 1; otherwise the weight is `distance.powf(alpha)`
/// (inverse distance for `alpha = -1`). Zero weights are dropped, as libpysal's taper does.
/// A non-binary band with negative `alpha` rejects coincident points (their weight would be
/// infinite). Points without neighbors become islands.
pub fn distance_band(
    points: &[f64],
    threshold: f64,
    binary: bool,
    alpha: f64,
) -> Result<Csr, String> {
    distance_band_limited(
        points,
        threshold,
        binary,
        alpha,
        Metric::Euclidean,
        limits::DEFAULT_MAX_EDGES,
    )
}

/// [`distance_band`] under `metric` (haversine thresholds and distances in kilometres) that
/// errors once the band holds more than `max_edges` edges.
pub fn distance_band_limited(
    points: &[f64],
    threshold: f64,
    binary: bool,
    alpha: f64,
    metric: Metric,
    max_edges: usize,
) -> Result<Csr, String> {
    let pts = split_points(points)?;
    metric.validate(&pts)?;
    if !binary && !alpha.is_finite() {
        return Err(format!("alpha must be finite, got {alpha}"));
    }
    let (mut focal, mut neighbor, mut weight) = (Vec::new(), Vec::new(), Vec::new());
    for (i, j, d) in band_edges(&pts, threshold, metric, max_edges, BAND_HINT)? {
        let w = if binary {
            1.0
        } else {
            if d == 0.0 && alpha < 0.0 {
                return Err(format!(
                    "points {i} and {j} are coincident, so their inverse-distance weight \
                     (alpha = {alpha}) is infinite; use binary: true or remove duplicates"
                ));
            }
            d.powf(alpha)
        };
        if w == 0.0 {
            continue;
        }
        focal.push(i);
        neighbor.push(j);
        weight.push(w);
    }
    Csr::from_arrays(pts.len(), &focal, &neighbor, &weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINE: [f64; 10] = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.5, 0.0, 10.0, 0.0];

    fn row(g: &Csr, i: usize) -> (Vec<u32>, Vec<f64>) {
        let r = g.offsets[i]..g.offsets[i + 1];
        (g.neighbors[r.clone()].to_vec(), g.weights[r].to_vec())
    }

    #[test]
    fn threshold_is_inclusive_and_far_points_are_islands() {
        let g = distance_band(&LINE, 1.0, true, -1.0).unwrap();
        assert_eq!(row(&g, 0), (vec![1], vec![1.0]));
        assert_eq!(row(&g, 1), (vec![0, 2], vec![1.0, 1.0]));
        assert_eq!(row(&g, 2), (vec![1], vec![1.0]));
        assert_eq!(g.isolates(), vec![3, 4]);
    }

    #[test]
    fn inverse_distance_and_alpha() {
        let g = distance_band(&LINE, 1.5, false, -1.0).unwrap();
        let (n, w) = row(&g, 2);
        assert_eq!(n, vec![1, 3]);
        assert!((w[0] - 1.0).abs() < 1e-15 && (w[1] - 1.0 / 1.5).abs() < 1e-15);
        let g = distance_band(&LINE, 1.5, false, -2.0).unwrap();
        assert!((row(&g, 2).1[1] - 1.0 / 2.25).abs() < 1e-15);
    }

    #[test]
    fn lattice_diagonal_is_inside_a_sqrt2_threshold() {
        let pts = [0.0, 0.0, 1.0, 1.0];
        let g = distance_band(&pts, 2.0_f64.sqrt(), true, -1.0).unwrap();
        assert_eq!(row(&g, 0).0, vec![1]);
        let g = distance_band(&pts, 1.414, true, -1.0).unwrap();
        assert_eq!(g.isolates(), vec![0, 1]);
    }

    #[test]
    fn binary_band_is_symmetric() {
        let pts: Vec<f64> = (0..100)
            .map(|i| ((i * 7919) % 1000) as f64 / 100.0)
            .collect();
        let g = distance_band(&pts, 2.5, true, -1.0).unwrap();
        assert!(g.is_symmetric(0.0));
    }

    #[test]
    fn coincident_points_are_linked_when_binary() {
        let pts = [0.0, 0.0, 0.0, 0.0, 5.0, 5.0];
        let g = distance_band(&pts, 1.0, true, -1.0).unwrap();
        assert_eq!(row(&g, 0), (vec![1], vec![1.0]));
        assert_eq!(g.isolates(), vec![2]);
    }

    #[test]
    fn coincident_points_with_negative_alpha_are_rejected() {
        let pts = [0.0, 0.0, 0.0, 0.0];
        let err = distance_band(&pts, 1.0, false, -1.0).unwrap_err();
        assert!(err.contains("coincident"), "{err}");
    }

    #[test]
    fn zero_alpha_gives_unit_weights_and_positive_alpha_drops_coincident_pairs() {
        let pts = [0.0, 0.0, 0.0, 0.0, 0.5, 0.0];
        let g = distance_band(&pts, 1.0, false, 0.0).unwrap();
        assert!(g.weights.iter().all(|&w| w == 1.0));
        let g = distance_band(&pts, 1.0, false, 1.0).unwrap();
        assert_eq!(row(&g, 0).0, vec![2]);
    }

    #[test]
    fn rejects_bad_threshold_and_alpha() {
        for t in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(distance_band(&LINE, t, true, -1.0)
                .unwrap_err()
                .contains("threshold"));
        }
        assert!(distance_band(&LINE, 1.0, false, f64::NAN)
            .unwrap_err()
            .contains("alpha"));
    }

    #[test]
    fn rejects_malformed_points() {
        assert!(distance_band(&[0.0, 0.0, 1.0], 1.0, true, -1.0).is_err());
        assert!(distance_band(&[0.0, f64::INFINITY], 1.0, true, -1.0).is_err());
    }

    #[test]
    fn matches_brute_force() {
        let pts: Vec<f64> = (0..120)
            .map(|i| ((i * 104729) % 997) as f64 / 99.7)
            .collect();
        let t = 1.7;
        let g = distance_band(&pts, t, true, -1.0).unwrap();
        let n = pts.len() / 2;
        for i in 0..n {
            let expected: Vec<u32> = (0..n)
                .filter(|&j| {
                    let dx = pts[2 * i] - pts[2 * j];
                    let dy = pts[2 * i + 1] - pts[2 * j + 1];
                    j != i && dx * dx + dy * dy <= t * t
                })
                .map(|j| j as u32)
                .collect();
            let (got, _) = row(&g, i);
            if expected.is_empty() {
                assert_eq!(got, vec![i as u32]);
            } else {
                assert_eq!(got, expected, "row {i}");
            }
        }
    }

    #[test]
    fn edge_budget_stops_an_oversized_band() {
        let err =
            distance_band_limited(&LINE, 100.0, true, -1.0, Metric::Euclidean, 5).unwrap_err();
        assert!(
            err.contains("max_edges") && err.contains("threshold"),
            "{err}"
        );
        assert!(distance_band_limited(&LINE, 100.0, true, -1.0, Metric::Euclidean, 20).is_ok());
    }

    #[test]
    fn haversine_band_uses_kilometres() {
        use super::super::metric::haversine_km;
        // Madrid, Paris, London: Paris - London is about 344 km, Madrid - Paris about 1053 km.
        let pts = [-3.70, 40.42, 2.35, 48.86, -0.13, 51.51];
        let max = crate::limits::DEFAULT_MAX_EDGES;
        let g = distance_band_limited(&pts, 400.0, true, -1.0, Metric::Haversine, max).unwrap();
        let (f, t, _) = g.to_arrays();
        // Madrid is an island (listed as a zero self pair, as in libpysal).
        assert_eq!((f, t), (vec![0, 1, 2], vec![0, 2, 1]));
        assert_eq!(g.isolates(), vec![0]);
        // Inclusive at exactly the pair distance; inverse-distance weights in 1 / km.
        let d = haversine_km([2.35, 48.86], [-0.13, 51.51]);
        let g = distance_band_limited(&pts, d, false, -1.0, Metric::Haversine, max).unwrap();
        let (_, _, w) = g.to_arrays();
        assert_eq!(w.len(), 3);
        assert!((w[1] - 1.0 / d).abs() < 1e-15 && (w[2] - 1.0 / d).abs() < 1e-15);
    }
}
