//! K-nearest-neighbor graphs, mirroring `libpysal.graph.Graph.build_knn` (Euclidean or
//! haversine metric, see [`super::metric`]).
//!
//! Neighbors come from an exact R*-tree nearest-neighbor search (`rstar`). Equidistant
//! candidates are resolved by ascending point index, a deliberate deviation from libpysal,
//! whose scikit-learn KD-tree resolves ties in tree order (see `docs/parity.md`).

use std::collections::HashMap;

use rstar::primitives::GeomWithData;
use rstar::RTree;

use super::metric::{unit_vectors, Metric};
use crate::graph::Csr;
use crate::limits;
use crate::perm::replicate_rng;
use crate::pointpats::unit;

/// How coincident (duplicated) points are handled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coincident {
    /// Error when any two points share the same coordinates.
    Raise,
    /// Connect all points on one site to each other and to the neighbors of the site.
    Clique,
    /// Displace the points of each coincident group by a tiny seeded amount (see [`jitter`]).
    Jitter { seed: u64 },
}

/// Relative size of a jitter displacement: `2^-30` of the coordinate magnitude (at least 1),
/// far above float resolution and far below any meaningful distance.
const JITTER_SCALE: f64 = 1.0 / (1u64 << 30) as f64;
/// Redraws allowed before jittering gives up.
const JITTER_ATTEMPTS: u64 = 16;

/// `coincident: :jitter`: moves every point of a coincident group by a uniform displacement in
/// a disc of radius `max(|x|, |y|, 1) * 2^-30`, drawn from the stream `(seed, attempt)`, and
/// redraws until all points are distinct. Distinct points are left untouched (libpysal moves
/// every point, unseeded, by about one coordinate unit; see `docs/parity.md`).
pub fn jitter(pts: &[[f64; 2]], seed: u64) -> Result<Vec<[f64; 2]>, String> {
    let mut out = pts.to_vec();
    for attempt in 0..JITTER_ATTEMPTS {
        let groups = coincident_groups(&out);
        if groups.is_empty() {
            return Ok(out);
        }
        let mut rng = replicate_rng(seed, attempt);
        for &i in groups.iter().flatten() {
            let p = out[i];
            let radius = p[0].abs().max(p[1].abs()).max(1.0) * JITTER_SCALE;
            let r = unit(&mut rng).sqrt() * radius;
            let theta = unit(&mut rng) * std::f64::consts::TAU;
            out[i] = [p[0] + r * theta.cos(), p[1] + r * theta.sin()];
        }
    }
    Err(format!(
        "jittering did not separate the coincident points after {JITTER_ATTEMPTS} attempts"
    ))
}

/// The points a builder works on: jittered for [`Coincident::Jitter`], unchanged otherwise.
pub(super) fn resolve_coincident(
    pts: Vec<[f64; 2]>,
    coincident: Coincident,
) -> Result<Vec<[f64; 2]>, String> {
    match coincident {
        Coincident::Jitter { seed } => jitter(&pts, seed),
        _ => Ok(pts),
    }
}

pub(super) type Indexed<const D: usize> = GeomWithData<[f64; D], u32>;

/// Splits interleaved `x, y` coordinates into points, rejecting odd lengths and non-finite values.
pub(super) fn split_points(points: &[f64]) -> Result<Vec<[f64; 2]>, String> {
    let (pairs, rest) = points.as_chunks::<2>();
    if !rest.is_empty() {
        return Err(format!(
            "points must be x, y pairs, got {} coordinates",
            points.len()
        ));
    }
    if let Some(i) = pairs
        .iter()
        .position(|p| !(p[0].is_finite() && p[1].is_finite()))
    {
        return Err(format!(
            "point {i} has a non-finite coordinate; coordinates must be finite"
        ));
    }
    if u32::try_from(pairs.len()).is_err() {
        return Err(format!("{} points exceed the u32 index range", pairs.len()));
    }
    Ok(pairs.to_vec())
}

pub(super) fn build_tree<const D: usize>(points: &[[f64; D]]) -> RTree<Indexed<D>>
where
    [f64; D]: rstar::Point<Scalar = f64>,
{
    RTree::bulk_load(
        points
            .iter()
            .enumerate()
            .map(|(i, &p)| GeomWithData::new(p, i as u32))
            .collect(),
    )
}

/// Builds the KNN graph of interleaved `x0, y0, x1, y1, ...` points. All weights are 1.
///
/// Errors when `k` is zero or exceeds the number of other points (distinct locations for
/// [`Coincident::Clique`]), or when [`Coincident::Raise`] meets duplicated coordinates.
pub fn knn(points: &[f64], k: usize, coincident: Coincident) -> Result<Csr, String> {
    knn_limited(
        points,
        k,
        coincident,
        Metric::Euclidean,
        limits::DEFAULT_MAX_EDGES,
    )
}

/// [`knn`] under `metric` that errors before building when `n * k` exceeds `max_edges`.
pub fn knn_limited(
    points: &[f64],
    k: usize,
    coincident: Coincident,
    metric: Metric,
    max_edges: usize,
) -> Result<Csr, String> {
    let pts = split_points(points)?;
    metric.validate(&pts)?;
    let pts = resolve_coincident(pts, coincident)?;
    let edges = limits::checked_product("knn graph", pts.len(), k)?;
    limits::ensure_edges("knn graph", edges, max_edges, "use a smaller k")?;
    let edges = knn_edges(&pts, k, coincident, metric)?;
    csr_from_edges(pts.len(), edges.into_iter())
}

/// The directed `(focal, neighbor)` edges of the KNN graph, shared with the kernel builder.
pub(super) fn knn_edges(
    pts: &[[f64; 2]],
    k: usize,
    coincident: Coincident,
    metric: Metric,
) -> Result<Vec<(u32, u32)>, String> {
    if k == 0 {
        return Err("k must be at least 1".into());
    }
    let groups = coincident_groups(pts);
    if groups.is_empty() {
        let rows = rows_under(pts, k, metric)?;
        return Ok(rows
            .iter()
            .enumerate()
            .flat_map(|(i, row)| row.iter().map(move |&j| (i as u32, j)))
            .collect());
    }
    match coincident {
        Coincident::Raise => Err(format!(
            "{} points share a location with another point (coincident points); \
             use coincident: :clique to connect them",
            groups.iter().map(Vec::len).sum::<usize>()
        )),
        Coincident::Clique => clique(pts, k, &groups, metric),
        Coincident::Jitter { .. } => Err("jittered points are still coincident".into()),
    }
}

/// Groups of two or more points with identical coordinates (indices ascending).
pub(super) fn coincident_groups(pts: &[[f64; 2]]) -> Vec<Vec<usize>> {
    let mut by_site: HashMap<(u64, u64), Vec<usize>> = HashMap::new();
    for (i, p) in pts.iter().enumerate() {
        // `+ 0.0` maps -0.0 to 0.0 so both hash to the same site.
        by_site
            .entry(((p[0] + 0.0).to_bits(), (p[1] + 0.0).to_bits()))
            .or_default()
            .push(i);
    }
    let mut groups: Vec<Vec<usize>> = by_site.into_values().filter(|g| g.len() > 1).collect();
    groups.sort_unstable();
    groups
}

fn csr_from_edges(n: usize, edges: impl Iterator<Item = (u32, u32)>) -> Result<Csr, String> {
    let (focal, neighbor): (Vec<u32>, Vec<u32>) = edges.unzip();
    let weight = vec![1.0; focal.len()];
    Csr::from_arrays(n, &focal, &neighbor, &weight)
}

/// [`neighbor_rows`] under `metric`: haversine searches the unit-sphere vectors.
fn rows_under(pts: &[[f64; 2]], k: usize, metric: Metric) -> Result<Vec<Vec<u32>>, String> {
    match metric {
        Metric::Euclidean => neighbor_rows(pts, k),
        Metric::Haversine => neighbor_rows(&unit_vectors(pts), k),
    }
}

/// The `k` nearest other points of every point, ascending by index. Points must be distinct.
fn neighbor_rows<const D: usize>(pts: &[[f64; D]], k: usize) -> Result<Vec<Vec<u32>>, String>
where
    [f64; D]: rstar::Point<Scalar = f64>,
{
    if k >= pts.len() {
        return Err(format!(
            "k = {k} must be at most {} (the number of other points)",
            pts.len().saturating_sub(1)
        ));
    }
    let tree = build_tree(pts);
    Ok(pts
        .iter()
        .enumerate()
        .map(|(i, p)| nearest(&tree, p, i as u32, k))
        .collect())
}

fn nearest<const D: usize>(tree: &RTree<Indexed<D>>, p: &[f64; D], me: u32, k: usize) -> Vec<u32>
where
    [f64; D]: rstar::Point<Scalar = f64>,
{
    let mut found: Vec<(f64, u32)> = Vec::with_capacity(k + 1);
    for (item, d2) in tree.nearest_neighbor_iter_with_distance_2(*p) {
        if item.data == me {
            continue;
        }
        // The iterator is non-decreasing in distance: stop once past the k-th distance.
        if found.len() >= k && d2 > found[k - 1].0 {
            break;
        }
        found.push((d2, item.data));
    }
    found.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    found.truncate(k);
    let mut row: Vec<u32> = found.into_iter().map(|(_, j)| j).collect();
    row.sort_unstable();
    row
}

/// `coplanar="clique"`: KNN over distinct locations, then every point on a site takes the
/// site's edges and all points of a site are linked. Mirrors libpysal's choice of site
/// representative (the second point of a pair, the first of larger groups), which only
/// matters for tie-breaking.
fn clique(
    pts: &[[f64; 2]],
    k: usize,
    groups: &[Vec<usize>],
    metric: Metric,
) -> Result<Vec<(u32, u32)>, String> {
    let mut dropped = vec![false; pts.len()];
    let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
    for g in groups {
        let rep = if g.len() == 2 { g[1] } else { g[0] };
        for &i in g {
            dropped[i] = i != rep;
        }
        members.insert(rep, g.clone());
    }
    let remaining: Vec<usize> = (0..pts.len()).filter(|&i| !dropped[i]).collect();
    let unique: Vec<[f64; 2]> = remaining.iter().map(|&i| pts[i]).collect();
    let rows = rows_under(&unique, k, metric).map_err(|e| {
        format!("{e}; with coincident points k is limited by the number of distinct locations")
    })?;
    let site = |u: usize| -> Vec<usize> {
        members
            .get(&remaining[u])
            .cloned()
            .unwrap_or_else(|| vec![remaining[u]])
    };
    let mut edges: Vec<(u32, u32)> = Vec::new();
    for (u, row) in rows.iter().enumerate() {
        let from = site(u);
        for &v in row {
            let to = site(v as usize);
            for &a in &from {
                edges.extend(to.iter().map(|&b| (a as u32, b as u32)));
            }
        }
    }
    for g in groups {
        for &a in g {
            edges.extend(g.iter().filter(|&&b| b != a).map(|&b| (a as u32, b as u32)));
        }
    }
    Ok(edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edges(g: &Csr) -> Vec<(u32, u32, f64)> {
        let (f, t, w) = g.to_arrays();
        f.into_iter()
            .zip(t)
            .zip(w)
            .map(|((f, t), w)| (f, t, w))
            .collect()
    }

    fn row(g: &Csr, i: usize) -> Vec<u32> {
        g.neighbors[g.offsets[i]..g.offsets[i + 1]].to_vec()
    }

    fn lcg_points(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..2 * n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 11) as f64 / (1u64 << 53) as f64 * 10.0
            })
            .collect()
    }

    #[test]
    fn nearest_neighbor_on_a_line() {
        let pts = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.5, 0.0, 10.0, 0.0];
        let g = knn(&pts, 1, Coincident::Raise).unwrap();
        assert_eq!(
            edges(&g),
            vec![
                (0, 1, 1.0),
                (1, 0, 1.0),
                (2, 1, 1.0),
                (3, 2, 1.0),
                (4, 3, 1.0)
            ]
        );
    }

    #[test]
    fn rows_are_in_position_order_not_distance_order() {
        let pts = [0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 2.0, 0.0];
        let g = knn(&pts, 2, Coincident::Raise).unwrap();
        assert_eq!(row(&g, 0), vec![2, 3]);
        assert_eq!(row(&g, 1), vec![2, 3]);
    }

    #[test]
    fn ties_break_by_lowest_index() {
        // Point 0 has three neighbors at distance 1; indices 1 and 2 win for k = 2.
        let pts = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];
        let g = knn(&pts, 2, Coincident::Raise).unwrap();
        assert_eq!(row(&g, 0), vec![1, 2]);
        let g = knn(&pts, 1, Coincident::Raise).unwrap();
        assert_eq!(row(&g, 0), vec![1]);
    }

    #[test]
    fn k_equal_n_minus_one_is_the_complete_graph() {
        let pts = lcg_points(6, 7);
        let g = knn(&pts, 5, Coincident::Raise).unwrap();
        for i in 0..6 {
            let expected: Vec<u32> = (0..6).filter(|&j| j != i).collect();
            assert_eq!(row(&g, i as usize), expected);
        }
    }

    #[test]
    fn every_row_has_k_neighbors_and_no_self() {
        let pts = lcg_points(120, 42);
        let g = knn(&pts, 7, Coincident::Raise).unwrap();
        for i in 0..120 {
            let r = row(&g, i);
            assert_eq!(r.len(), 7);
            assert!(!r.contains(&(i as u32)));
        }
    }

    #[test]
    fn matches_brute_force() {
        let pts = lcg_points(80, 99);
        let g = knn(&pts, 5, Coincident::Raise).unwrap();
        for i in 0..80 {
            let mut d: Vec<(f64, usize)> = (0..80)
                .filter(|&j| j != i)
                .map(|j| {
                    let dx = pts[2 * i] - pts[2 * j];
                    let dy = pts[2 * i + 1] - pts[2 * j + 1];
                    (dx * dx + dy * dy, j)
                })
                .collect();
            d.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut expected: Vec<u32> = d[..5].iter().map(|&(_, j)| j as u32).collect();
            expected.sort_unstable();
            assert_eq!(row(&g, i), expected, "row {i}");
        }
    }

    #[test]
    fn rejects_bad_k() {
        let pts = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0];
        assert!(knn(&pts, 0, Coincident::Raise).unwrap_err().contains("k"));
        let err = knn(&pts, 3, Coincident::Raise).unwrap_err();
        assert!(err.contains("at most 2"), "{err}");
    }

    #[test]
    fn rejects_malformed_points() {
        assert!(knn(&[0.0, 0.0, 1.0], 1, Coincident::Raise)
            .unwrap_err()
            .contains("pairs"));
        assert!(knn(&[0.0, 0.0, f64::NAN, 1.0], 1, Coincident::Raise)
            .unwrap_err()
            .contains("finite"));
    }

    #[test]
    fn raise_reports_coincident_points() {
        let pts = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let err = knn(&pts, 1, Coincident::Raise).unwrap_err();
        assert!(err.contains("coincident"), "{err}");
    }

    #[test]
    fn negative_zero_equals_zero_for_coincidence() {
        let pts = [0.0, 0.0, -0.0, 0.0, 1.0, 0.0];
        assert!(knn(&pts, 1, Coincident::Raise).is_err());
    }

    #[test]
    fn clique_expands_sites_like_libpysal() {
        let pts = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 3.0, 1.0];
        let g = knn(&pts, 2, Coincident::Clique).unwrap();
        assert_eq!(row(&g, 0), vec![1, 2, 3]);
        assert_eq!(row(&g, 1), vec![0, 2, 3]);
        assert_eq!(row(&g, 2), vec![0, 1, 3]);
        let g = knn(&pts, 1, Coincident::Clique).unwrap();
        assert_eq!(row(&g, 0), vec![1, 2]);
        assert_eq!(row(&g, 1), vec![0, 2]);
        assert!(g.weights.iter().all(|&w| w == 1.0));
    }

    #[test]
    fn clique_with_three_points_on_a_site() {
        let pts = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let g = knn(&pts, 1, Coincident::Clique).unwrap();
        assert_eq!(row(&g, 0), vec![1, 2, 3]);
        assert_eq!(row(&g, 2), vec![0, 1, 3]);
        assert_eq!(row(&g, 1), vec![0, 2, 3]);
    }

    #[test]
    fn clique_needs_enough_unique_sites() {
        let pts = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        assert!(knn(&pts, 2, Coincident::Clique).is_err());
    }

    #[test]
    fn clique_without_duplicates_equals_raise() {
        let pts = lcg_points(40, 5);
        assert_eq!(
            knn(&pts, 3, Coincident::Clique).unwrap(),
            knn(&pts, 3, Coincident::Raise).unwrap()
        );
    }

    #[test]
    fn edge_budget_is_checked_before_building() {
        let pts: Vec<f64> = (0..20)
            .flat_map(|i| [i as f64, (i * i % 7) as f64])
            .collect();
        let err = knn_limited(&pts, 10, Coincident::Raise, Metric::Euclidean, 50).unwrap_err();
        assert!(
            err.contains("200 edges") && err.contains("max_edges"),
            "{err}"
        );
        assert!(err.contains("smaller k"), "{err}");
        assert!(knn_limited(&pts, 10, Coincident::Raise, Metric::Euclidean, 200).is_ok());
    }

    // Lon/lat points spread over western Europe, in general position.
    const LONLAT: [f64; 16] = [
        -3.70, 40.42, 2.35, 48.86, -0.13, 51.51, 13.40, 52.52, 12.50, 41.90, -9.14, 38.72, 4.90,
        52.37, 16.37, 48.21,
    ];

    fn brute_knn(pts: &[f64], k: usize) -> Vec<Vec<u32>> {
        let p: Vec<[f64; 2]> = pts.chunks(2).map(|c| [c[0], c[1]]).collect();
        (0..p.len())
            .map(|i| {
                let mut d: Vec<(f64, u32)> = (0..p.len())
                    .filter(|&j| j != i)
                    .map(|j| (super::super::metric::haversine_km(p[i], p[j]), j as u32))
                    .collect();
                d.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut row: Vec<u32> = d[..k].iter().map(|x| x.1).collect();
                row.sort_unstable();
                row
            })
            .collect()
    }

    #[test]
    fn haversine_knn_matches_great_circle_brute_force() {
        let max = crate::limits::DEFAULT_MAX_EDGES;
        for k in 1..=3 {
            let g = knn_limited(&LONLAT, k, Coincident::Raise, Metric::Haversine, max).unwrap();
            let rows: Vec<Vec<u32>> = (0..8)
                .map(|i| edges(&g).iter().filter(|e| e.0 == i).map(|e| e.1).collect())
                .collect();
            assert_eq!(rows, brute_knn(&LONLAT, k), "k = {k}");
        }
        // Near the antimeridian the planar neighbor is wrong and the spherical one right.
        let pts = [179.0, 0.0, -179.0, 0.0, 175.0, 0.0];
        let g = knn_limited(&pts, 1, Coincident::Raise, Metric::Haversine, max).unwrap();
        assert_eq!(edges(&g)[0].1, 1);
        assert!(knn_limited(
            &[0.0, 95.0, 1.0, 1.0],
            1,
            Coincident::Raise,
            Metric::Haversine,
            max
        )
        .unwrap_err()
        .contains("haversine range"));
    }

    #[test]
    fn jitter_separates_only_coincident_points_deterministically() {
        let pts = [
            [0.0, 0.0],
            [5.0, 1.0],
            [0.0, 0.0],
            [3.0, 3.0],
            [3.0, 3.0],
            [3.0, 3.0],
        ];
        let a = jitter(&pts, 7).unwrap();
        assert_eq!(a, jitter(&pts, 7).unwrap());
        assert_ne!(a, jitter(&pts, 8).unwrap());
        assert!(coincident_groups(&a).is_empty());
        assert_eq!(a[1], pts[1]);
        for (p, q) in pts.iter().zip(&a) {
            let moved = (p[0] - q[0]).hypot(p[1] - q[1]);
            assert!(moved < 3.0 * JITTER_SCALE * 3.0, "{moved}");
        }
        assert!(a[0] != pts[0] && a[2] != pts[2] && a[3] != pts[3]);
    }

    #[test]
    fn jittered_knn_has_k_neighbors_per_point() {
        let flat = [0.0, 0.0, 5.0, 1.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0, 9.0, 2.0];
        let max = crate::limits::DEFAULT_MAX_EDGES;
        let j = Coincident::Jitter { seed: 1 };
        let g = knn_limited(&flat, 2, j, Metric::Euclidean, max).unwrap();
        assert_eq!(g.cardinalities(), vec![2; 6]);
        // The coincident partner is the nearest neighbor of each jittered point.
        let e = edges(&g);
        assert!(e.contains(&(0, 2, 1.0)) && e.contains(&(2, 0, 1.0)) && e.contains(&(3, 4, 1.0)));
        assert_eq!(
            g.to_arrays(),
            knn_limited(&flat, 2, j, Metric::Euclidean, max)
                .unwrap()
                .to_arrays()
        );
        assert!(knn_limited(&flat, 2, Coincident::Raise, Metric::Euclidean, max).is_err());
    }
}
