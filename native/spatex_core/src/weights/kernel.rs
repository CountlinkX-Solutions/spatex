//! Kernel-weighted graphs, mirroring `libpysal.graph.Graph.build_kernel` (Euclidean or
//! haversine metric, see [`super::metric`]).
//!
//! Kernel functions use libpysal's names and formulas (`libpysal/graph/_kernel.py`). Distances are
//! Euclidean, candidate pairs come from the `rstar` tree shared with KNN and distance bands.

use std::f64::consts::PI;

use super::distance_band::band_edges;
use super::knn::{build_tree, knn_edges, resolve_coincident, split_points, Coincident};
use super::metric::Metric;
use crate::graph::Csr;
use crate::limits;

const BANDWIDTH_HINT: &str = "stopped collecting; use a smaller bandwidth";

/// Kernel functions, with libpysal's names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Triangular,
    Parabolic,
    Gaussian,
    Bisquare,
    Tricube,
    Cosine,
    Boxcar,
    Exponential,
    Identity,
}

/// How the bandwidth is chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Bandwidth {
    Default,
    Fixed(f64),
    Adaptive,
}

/// Options of [`kernel_graph`].
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub kernel: Kernel,
    pub bandwidth: Bandwidth,
    pub k: Option<usize>,
    pub taper: bool,
    pub decay: bool,
    pub coincident: Coincident,
    /// Distance metric; haversine bandwidths are kilometres.
    pub metric: Metric,
    /// Largest number of edges built; more is an error raised before (or while) collecting.
    pub max_edges: usize,
}

impl Kernel {
    /// Parses libpysal's kernel names (`discrete` is accepted as a boxcar alias).
    pub fn parse(name: &str) -> Result<Kernel, String> {
        Ok(match name {
            "triangular" => Kernel::Triangular,
            "parabolic" => Kernel::Parabolic,
            "gaussian" => Kernel::Gaussian,
            "bisquare" => Kernel::Bisquare,
            "tricube" => Kernel::Tricube,
            "cosine" => Kernel::Cosine,
            "boxcar" | "discrete" => Kernel::Boxcar,
            "exponential" => Kernel::Exponential,
            "identity" => Kernel::Identity,
            other => return Err(format!("unknown kernel {other:?}")),
        })
    }

    /// Kernels that are exactly zero beyond the bandwidth.
    fn is_compact(self) -> bool {
        !matches!(
            self,
            Kernel::Gaussian | Kernel::Exponential | Kernel::Identity
        )
    }

    /// `libpysal.kernels` value at distance `d` for bandwidth `h` (no taper, no decay).
    pub fn eval(self, d: f64, h: f64) -> f64 {
        let clip = |x: f64| x.clamp(0.0, 1.0);
        match self {
            Kernel::Triangular => 1.0 - clip(d.abs() / h),
            Kernel::Parabolic => {
                let z = clip(d.abs() / h);
                0.75 * (1.0 - z * z)
            }
            Kernel::Gaussian => {
                let z = d / h;
                (1.0 / (2.0 * PI).sqrt()) * (-0.5 * (z * z)).exp()
            }
            Kernel::Bisquare => {
                let z = clip(d / h);
                let t = 1.0 - z * z;
                (15.0 / 16.0) * (t * t)
            }
            Kernel::Tricube => {
                let z = clip(d / h);
                (70.0 / 81.0) * (1.0 - z.powi(3)).powi(3)
            }
            Kernel::Cosine => (PI / 4.0) * (PI / 2.0 * clip(d / h)).cos(),
            Kernel::Boxcar => f64::from(d < h),
            Kernel::Exponential => (-(d / h)).exp(),
            Kernel::Identity => d,
        }
    }
}

/// Builds the kernel graph of interleaved `x0, y0, x1, y1, ...` points.
///
/// Candidate pairs: with `k`, the KNN edges (see [`super::knn::knn`] for ties and coincident
/// points); without `k`, pairs within a fixed bandwidth when the kernel is compact or `taper`
/// is set, otherwise all ordered pairs (O(n^2)). Self weights are never present. Coincident
/// pairs are kept (libpysal's sparse fast path silently drops them). More than
/// `opts.max_edges` candidate pairs is an error.
pub fn kernel_graph(points: &[f64], opts: &Options) -> Result<Csr, String> {
    let pts = split_points(points)?;
    let metric = opts.metric;
    metric.validate(&pts)?;
    // Jitter only matters with `k`; kernel distances then come from the jittered points.
    let pts = if opts.k.is_some() {
        resolve_coincident(pts, opts.coincident)?
    } else {
        pts
    };
    let dist = |i: u32, j: u32| metric.distance(pts[i as usize], pts[j as usize]);
    let n = pts.len();
    let max_edges = opts.max_edges;
    let edges: Vec<(u32, u32, f64)> = match opts.k {
        Some(k) => {
            let count = limits::checked_product("kernel graph", n, k)?;
            limits::ensure_edges("kernel graph", count, max_edges, "use a smaller k")?;
            knn_edges(&pts, k, opts.coincident, metric)?
                .into_iter()
                .map(|(i, j)| (i, j, dist(i, j)))
                .collect()
        }
        None => match opts.bandwidth {
            Bandwidth::Adaptive => {
                return Err("adaptive bandwidth requires k (the neighbor count)".into())
            }
            Bandwidth::Fixed(h)
                if (opts.kernel.is_compact() || opts.taper) && metric == Metric::Haversine =>
            {
                check_bandwidth(h)?;
                band_edges(&pts, h, metric, max_edges, BANDWIDTH_HINT)?
            }
            Bandwidth::Fixed(h) if opts.kernel.is_compact() || opts.taper => {
                check_bandwidth(h)?;
                let tree = build_tree(&pts);
                let h2 = h * h;
                let mut edges = Vec::new();
                for (i, p) in pts.iter().enumerate() {
                    for item in tree.locate_within_distance(*p, h2) {
                        let j = item.data;
                        if j as usize != i {
                            edges.push((i as u32, j, dist(i as u32, j)));
                            limits::ensure_edges(
                                "kernel graph",
                                edges.len(),
                                max_edges,
                                BANDWIDTH_HINT,
                            )?;
                        }
                    }
                }
                edges
            }
            _ => {
                let pairs = limits::checked_product("kernel graph", n, n.saturating_sub(1))?;
                limits::ensure_edges(
                    "kernel graph over all pairs",
                    pairs,
                    max_edges,
                    "pass k: for nearest neighbors or a compact kernel with bandwidth:",
                )?;
                (0..n as u32)
                    .flat_map(|i| (0..n as u32).filter(move |&j| j != i).map(move |j| (i, j)))
                    .map(|(i, j)| (i, j, dist(i, j)))
                    .collect()
            }
        },
    };
    weigh(
        n,
        edges,
        opts.kernel,
        opts.bandwidth,
        opts.k.is_some(),
        opts.taper,
        opts.decay,
    )
}

/// Kernel weights over the distance-band candidates (`Graph.build_distance_band` with
/// `binary=False, kernel=...`): the default bandwidth is the 25th percentile of the candidate
/// distances.
#[allow(clippy::too_many_arguments)]
pub fn band_kernel_graph(
    points: &[f64],
    threshold: f64,
    kernel: Kernel,
    bandwidth: Option<f64>,
    taper: bool,
    decay: bool,
    metric: Metric,
    max_edges: usize,
) -> Result<Csr, String> {
    let pts = split_points(points)?;
    metric.validate(&pts)?;
    let edges = band_edges(
        &pts,
        threshold,
        metric,
        max_edges,
        "stopped collecting; use a smaller threshold",
    )?;
    let bandwidth = bandwidth.map_or(Bandwidth::Default, Bandwidth::Fixed);
    weigh(pts.len(), edges, kernel, bandwidth, false, taper, decay)
}

fn check_bandwidth(h: f64) -> Result<(), String> {
    if h.is_finite() && h > 0.0 {
        Ok(())
    } else {
        Err(format!(
            "bandwidth must be a positive finite number, got {h}"
        ))
    }
}

/// numpy's `percentile(a, 25)` (linear interpolation) of an ascending, non-empty slice.
pub(super) fn percentile25(sorted: &[f64]) -> f64 {
    let virtual_index = 0.25 * (sorted.len() - 1) as f64;
    let lo = virtual_index.floor();
    let t = virtual_index - lo;
    let a = sorted[lo as usize];
    let b = sorted[(lo as usize + 1).min(sorted.len() - 1)];
    // numpy's lerp, symmetric around t = 0.5.
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

/// Resolves bandwidths and applies the kernel to the `(focal, neighbor, distance)` edges.
fn weigh(
    n: usize,
    edges: Vec<(u32, u32, f64)>,
    kernel: Kernel,
    bandwidth: Bandwidth,
    k_given: bool,
    taper: bool,
    decay: bool,
) -> Result<Csr, String> {
    if let Bandwidth::Fixed(h) = bandwidth {
        check_bandwidth(h)?;
    }
    let (focal, neighbor, weight) =
        weigh_arrays(n, edges, kernel, bandwidth, k_given, taper, decay)?;
    Csr::from_arrays(n, &focal, &neighbor, &weight)
}

/// Parallel `(focal, neighbor, weight)` arrays.
pub(super) type EdgeArrays = (Vec<u32>, Vec<u32>, Vec<f64>);

/// Kernel weights of `(focal, neighbor, distance)` edges as `(focal, neighbor, weight)` arrays.
/// With `taper`, edges whose weight is zero are omitted. A fixed bandwidth may be infinite.
pub(super) fn weigh_arrays(
    n: usize,
    edges: Vec<(u32, u32, f64)>,
    kernel: Kernel,
    bandwidth: Bandwidth,
    k_given: bool,
    taper: bool,
    decay: bool,
) -> Result<EdgeArrays, String> {
    if decay && kernel == Kernel::Identity {
        return Err("decay is undefined for the identity kernel (its value at 0 is 0)".into());
    }
    let h_of: Vec<f64> = match bandwidth {
        Bandwidth::Fixed(h) => {
            if h.is_nan() || h <= 0.0 {
                return Err(format!("bandwidth must be positive, got {h}"));
            }
            vec![h; edges.len()]
        }
        Bandwidth::Adaptive => {
            let mut row_max = vec![0.0f64; n];
            for &(i, _, d) in &edges {
                row_max[i as usize] = row_max[i as usize].max(d);
            }
            edges
                .iter()
                .map(|&(i, _, _)| row_max[i as usize] * 1.0000001)
                .collect()
        }
        Bandwidth::Default if edges.is_empty() => Vec::new(),
        Bandwidth::Default => {
            let h = if k_given {
                edges.iter().fold(0.0f64, |m, e| m.max(e.2))
            } else {
                let mut d: Vec<f64> = edges.iter().map(|e| e.2).collect();
                d.sort_unstable_by(f64::total_cmp);
                percentile25(&d)
            };
            vec![h; edges.len()]
        }
    };
    let (mut focal, mut neighbor, mut weight) = (Vec::new(), Vec::new(), Vec::new());
    for (&(i, j, d), &h) in edges.iter().zip(&h_of) {
        if kernel != Kernel::Identity && (h.is_nan() || h <= 0.0) {
            return Err(format!(
                "bandwidth for node {i} is {h}, which is not positive (coincident points?);                  pass an explicit positive bandwidth"
            ));
        }
        let mut w = kernel.eval(d, h);
        if taper && d > h {
            w = 0.0;
        }
        if decay {
            w /= kernel.eval(0.0, h);
        }
        if taper && w == 0.0 {
            continue;
        }
        focal.push(i);
        neighbor.push(j);
        weight.push(w);
    }
    Ok((focal, neighbor, weight))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const MAX: usize = crate::limits::DEFAULT_MAX_EDGES;
    const LINE: [f64; 10] = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.5, 0.0, 10.0, 0.0];

    fn opts(kernel: Kernel, bandwidth: Bandwidth, k: Option<usize>) -> Options {
        Options {
            kernel,
            bandwidth,
            k,
            taper: true,
            decay: false,
            coincident: Coincident::Raise,
            metric: Metric::Euclidean,
            max_edges: crate::limits::DEFAULT_MAX_EDGES,
        }
    }

    fn row(g: &Csr, i: usize) -> (Vec<u32>, Vec<f64>) {
        let r = g.offsets[i]..g.offsets[i + 1];
        (g.neighbors[r.clone()].to_vec(), g.weights[r].to_vec())
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * b.abs().max(1.0)
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
    fn parse_accepts_libpysal_names_only() {
        for (name, k) in [
            ("triangular", Kernel::Triangular),
            ("parabolic", Kernel::Parabolic),
            ("gaussian", Kernel::Gaussian),
            ("bisquare", Kernel::Bisquare),
            ("tricube", Kernel::Tricube),
            ("cosine", Kernel::Cosine),
            ("boxcar", Kernel::Boxcar),
            ("exponential", Kernel::Exponential),
            ("identity", Kernel::Identity),
        ] {
            assert_eq!(Kernel::parse(name).unwrap(), k);
        }
        assert!(Kernel::parse("epanechnikov")
            .unwrap_err()
            .contains("kernel"));
    }

    #[test]
    fn kernel_formulas_at_half_bandwidth() {
        let (d, h) = (0.5, 1.0);
        let cases = [
            (Kernel::Triangular, 0.5),
            (Kernel::Parabolic, 0.75 * 0.75),
            (Kernel::Gaussian, (-0.125f64).exp() / (2.0 * PI).sqrt()),
            (Kernel::Bisquare, 15.0 / 16.0 * 0.75 * 0.75),
            (Kernel::Tricube, 70.0 / 81.0 * 0.875 * 0.875 * 0.875),
            (Kernel::Cosine, PI / 4.0 * (PI / 4.0).cos()),
            (Kernel::Boxcar, 1.0),
            (Kernel::Exponential, (-0.5f64).exp()),
            (Kernel::Identity, 0.5),
        ];
        for (kernel, expected) in cases {
            assert!(close(kernel.eval(d, h), expected), "{kernel:?}");
        }
    }

    #[test]
    fn boxcar_is_strict_and_cosine_leaves_a_residue_at_the_bandwidth() {
        assert_eq!(Kernel::Boxcar.eval(1.0, 1.0), 0.0);
        assert_eq!(Kernel::Boxcar.eval(0.999, 1.0), 1.0);
        let c = Kernel::Cosine.eval(1.0, 1.0);
        assert!(c > 0.0 && c < 1e-15, "{c}");
    }

    #[test]
    fn percentile_uses_numpy_linear_interpolation() {
        assert_eq!(percentile25(&[3.0]), 3.0);
        assert!(close(percentile25(&[1.0, 2.0, 3.0, 4.0, 5.0]), 2.0));
        assert!(close(percentile25(&[1.0, 2.0, 4.0, 8.0]), 1.75));
    }

    #[test]
    fn fixed_bandwidth_tapers_beyond_h_and_drops_zero_weights() {
        let g = kernel_graph(
            &LINE,
            &opts(Kernel::Triangular, Bandwidth::Fixed(1.5), None),
        )
        .unwrap();
        let (n0, w0) = row(&g, 0);
        assert_eq!(n0, vec![1]);
        assert!(close(w0[0], 1.0 - 1.0 / 1.5));
        // d == h gives triangular weight 0, which taper removes.
        assert_eq!(row(&g, 2).0, vec![1]);
        assert_eq!(g.isolates(), vec![3, 4]);
    }

    #[test]
    fn default_bandwidth_is_the_25th_percentile_of_all_pairwise_distances() {
        // Line example: libpysal probes h = 1.5.
        let default =
            kernel_graph(&LINE, &opts(Kernel::Triangular, Bandwidth::Default, None)).unwrap();
        let fixed = kernel_graph(
            &LINE,
            &opts(Kernel::Triangular, Bandwidth::Fixed(1.5), None),
        )
        .unwrap();
        assert_eq!(default, fixed);
    }

    #[test]
    fn default_bandwidth_with_k_is_the_largest_knn_distance() {
        // k = 1: nearest distances 1, 1, 1, 1.5, 6.5 -> h = 6.5.
        let g = kernel_graph(
            &LINE,
            &opts(Kernel::Triangular, Bandwidth::Default, Some(1)),
        )
        .unwrap();
        let (n, w) = row(&g, 3);
        assert_eq!(n, vec![2]);
        assert!(close(w[0], 1.0 - 1.5 / 6.5));
        // Point 4's only neighbor sits at d == h, where triangular is 0: taper drops it.
        assert_eq!(g.isolates(), vec![4]);
    }

    #[test]
    fn adaptive_bandwidth_keeps_the_kth_neighbor() {
        let pts = lcg_points(60, 7);
        for kernel in [
            Kernel::Triangular,
            Kernel::Bisquare,
            Kernel::Tricube,
            Kernel::Parabolic,
        ] {
            let g = kernel_graph(&pts, &opts(kernel, Bandwidth::Adaptive, Some(4))).unwrap();
            for i in 0..60 {
                let (n, w) = row(&g, i);
                assert_eq!(n.len(), 4, "{kernel:?} row {i}");
                assert!(w.iter().all(|&w| w > 0.0));
            }
        }
    }

    #[test]
    fn adaptive_weight_of_the_kth_neighbor_uses_the_inflated_bandwidth() {
        let g = kernel_graph(
            &LINE,
            &opts(Kernel::Triangular, Bandwidth::Adaptive, Some(2)),
        )
        .unwrap();
        // Row 0: neighbors at 1 and 2; h = 2 * 1.0000001.
        let (n, w) = row(&g, 0);
        assert_eq!(n, vec![1, 2]);
        let h = 2.0 * 1.0000001;
        assert!(close(w[0], 1.0 - 1.0 / h));
        assert!(close(w[1], 1.0 - 2.0 / h));
    }

    #[test]
    fn without_taper_gaussian_keeps_every_pair() {
        let mut o = opts(Kernel::Gaussian, Bandwidth::Fixed(1.0), None);
        o.taper = false;
        let g = kernel_graph(&LINE, &o).unwrap();
        assert!(g.neighbors.len() == 20 && g.isolates().is_empty());
        let (n, w) = row(&g, 0);
        assert_eq!(n, vec![1, 2, 3, 4]);
        assert!(close(w[3], (-50.0f64).exp() / (2.0 * PI).sqrt()));
        o.taper = true;
        let g = kernel_graph(&LINE, &o).unwrap();
        assert_eq!(row(&g, 0).0, vec![1]);
    }

    #[test]
    fn decay_divides_by_the_kernel_at_zero() {
        let mut o = opts(Kernel::Gaussian, Bandwidth::Fixed(2.0), None);
        let plain = kernel_graph(&LINE, &o).unwrap();
        o.decay = true;
        let decayed = kernel_graph(&LINE, &o).unwrap();
        for (a, b) in plain.weights.iter().zip(&decayed.weights) {
            assert!(close(*b, a * (2.0 * PI).sqrt()));
        }
        // Parabolic peaks at 0.75; decay rescales it to a peak of 1.
        o.kernel = Kernel::Parabolic;
        let g = kernel_graph(&[0.0, 0.0, 0.0, 0.0], &o).unwrap();
        assert!(close(g.weights[0], 1.0));
    }

    #[test]
    fn identity_returns_tapered_distances() {
        let g = kernel_graph(&LINE, &opts(Kernel::Identity, Bandwidth::Fixed(2.0), None)).unwrap();
        assert_eq!(row(&g, 0), (vec![1, 2], vec![1.0, 2.0]));
        let mut o = opts(Kernel::Identity, Bandwidth::Fixed(2.0), None);
        o.decay = true;
        assert!(kernel_graph(&LINE, &o).unwrap_err().contains("identity"));
    }

    #[test]
    fn coincident_pairs_are_kept_for_fixed_bandwidth() {
        // libpysal's sparse fast path drops them via eliminate_zeros; Spatex keeps them.
        let pts = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let g = kernel_graph(&pts, &opts(Kernel::Triangular, Bandwidth::Fixed(2.0), None)).unwrap();
        assert_eq!(row(&g, 0), (vec![1, 2], vec![1.0, 0.5]));
    }

    #[test]
    fn coincident_with_k_raises_or_forms_cliques() {
        let pts = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0];
        let o = opts(Kernel::Triangular, Bandwidth::Adaptive, Some(1));
        assert!(kernel_graph(&pts, &o).unwrap_err().contains("coincident"));
        let mut o = o;
        o.coincident = Coincident::Clique;
        let g = kernel_graph(&pts, &o).unwrap();
        // Point 0 links to its twin (distance 0) and the site's neighbor at distance 1.
        let (n, w) = row(&g, 0);
        assert_eq!(n, vec![1, 2]);
        let h = 1.0 * 1.0000001;
        assert!(close(w[0], 1.0) && close(w[1], 1.0 - 1.0 / h));
    }

    #[test]
    fn rejects_bad_options() {
        let bad_bw = [0.0, -1.0, f64::NAN, f64::INFINITY];
        for h in bad_bw {
            let o = opts(Kernel::Gaussian, Bandwidth::Fixed(h), None);
            assert!(kernel_graph(&LINE, &o).unwrap_err().contains("bandwidth"));
        }
        let o = opts(Kernel::Gaussian, Bandwidth::Adaptive, None);
        assert!(kernel_graph(&LINE, &o).unwrap_err().contains("adaptive"));
        let o = opts(Kernel::Gaussian, Bandwidth::Default, Some(5));
        assert!(kernel_graph(&LINE, &o).unwrap_err().contains("k = 5"));
        let o = opts(Kernel::Gaussian, Bandwidth::Default, Some(0));
        assert!(kernel_graph(&LINE, &o).is_err());
        assert!(kernel_graph(
            &[0.0, 0.0, 1.0],
            &opts(Kernel::Gaussian, Bandwidth::Default, None)
        )
        .is_err());
    }

    #[test]
    fn single_point_is_an_island() {
        let g = kernel_graph(
            &[1.0, 1.0],
            &opts(Kernel::Gaussian, Bandwidth::Fixed(1.0), None),
        )
        .unwrap();
        assert_eq!(g.isolates(), vec![0]);
    }

    #[test]
    fn fixed_graph_is_symmetric_and_matches_brute_force() {
        let pts = lcg_points(80, 11);
        let h = 2.0;
        let g = kernel_graph(&pts, &opts(Kernel::Bisquare, Bandwidth::Fixed(h), None)).unwrap();
        assert!(g.is_symmetric(1e-15));
        for i in 0..80 {
            let mut expect = Vec::new();
            for j in 0..80 {
                let (dx, dy) = (pts[2 * i] - pts[2 * j], pts[2 * i + 1] - pts[2 * j + 1]);
                let d = (dx * dx + dy * dy).sqrt();
                let w = Kernel::Bisquare.eval(d, h);
                if i != j && d <= h && w != 0.0 {
                    expect.push((j as u32, w));
                }
            }
            let (n, w) = row(&g, i);
            if expect.is_empty() {
                assert_eq!(n, vec![i as u32]);
                continue;
            }
            assert_eq!(n, expect.iter().map(|e| e.0).collect::<Vec<_>>());
            for (a, e) in w.iter().zip(&expect) {
                assert!(close(*a, e.1));
            }
        }
    }

    #[test]
    fn band_kernel_uses_candidate_distances_and_percentile_default() {
        // Threshold 1.5 candidates: (0,1)=1, (1,2)=1, (2,3)=1.5 and mirrors.
        let g = band_kernel_graph(
            &LINE,
            1.5,
            Kernel::Triangular,
            Some(3.0),
            true,
            false,
            Metric::Euclidean,
            MAX,
        )
        .unwrap();
        let (n, w) = row(&g, 2);
        assert_eq!(n, vec![1, 3]);
        assert!(close(w[0], 1.0 - 1.0 / 3.0) && close(w[1], 0.5));
        // Default bandwidth: 25th percentile of the six candidate distances [1,1,1,1,1.5,1.5] = 1.
        let g = band_kernel_graph(
            &LINE,
            1.5,
            Kernel::Gaussian,
            None,
            true,
            false,
            Metric::Euclidean,
            MAX,
        )
        .unwrap();
        let (n, w) = row(&g, 0);
        assert_eq!(n, vec![1]);
        assert!(close(w[0], (-0.5f64).exp() / (2.0 * PI).sqrt()));
        // The (2, 3) candidate at d = 1.5 lies beyond h = 1 and is tapered away.
        assert_eq!(g.isolates(), vec![3, 4]);
    }

    fn limited(
        kernel: Kernel,
        bandwidth: Bandwidth,
        k: Option<usize>,
        taper: bool,
        max: usize,
    ) -> Options {
        Options {
            taper,
            max_edges: max,
            ..opts(kernel, bandwidth, k)
        }
    }

    #[test]
    fn all_pairs_path_is_guarded_with_a_hint() {
        // Default bandwidth with a non-compact kernel builds every ordered pair: 5 * 4 = 20.
        let o = limited(Kernel::Gaussian, Bandwidth::Default, None, false, 19);
        let err = kernel_graph(&LINE, &o).unwrap_err();
        assert!(
            err.contains("20 edges") && err.contains("max_edges"),
            "{err}"
        );
        assert!(err.contains("k:") && err.contains("bandwidth:"), "{err}");
        let ok = limited(Kernel::Gaussian, Bandwidth::Default, None, false, 20);
        assert!(kernel_graph(&LINE, &ok).is_ok());
    }

    #[test]
    fn non_compact_taper_false_with_fixed_bandwidth_is_guarded() {
        let o = limited(Kernel::Gaussian, Bandwidth::Fixed(2.0), None, false, 10);
        assert!(kernel_graph(&LINE, &o).unwrap_err().contains("max_edges"));
    }

    #[test]
    fn fixed_bandwidth_edge_growth_is_stopped_while_collecting() {
        let o = limited(Kernel::Triangular, Bandwidth::Fixed(100.0), None, true, 6);
        let err = kernel_graph(&LINE, &o).unwrap_err();
        assert!(
            err.contains("max_edges") && err.contains("bandwidth"),
            "{err}"
        );
    }

    #[test]
    fn knn_kernel_edge_budget_is_n_times_k() {
        let o = limited(Kernel::Triangular, Bandwidth::Adaptive, Some(4), true, 10);
        let err = kernel_graph(&LINE, &o).unwrap_err();
        assert!(
            err.contains("20 edges") && err.contains("smaller k"),
            "{err}"
        );
    }

    #[test]
    fn distance_band_kernel_is_guarded() {
        let err = band_kernel_graph(
            &LINE,
            100.0,
            Kernel::Gaussian,
            None,
            true,
            false,
            Metric::Euclidean,
            5,
        )
        .unwrap_err();
        assert!(err.contains("max_edges"), "{err}");
    }

    #[test]
    fn haversine_kernels_use_great_circle_kilometres() {
        use super::super::metric::haversine_km;
        let pts = [-3.70, 40.42, 2.35, 48.86, -0.13, 51.51, 13.40, 52.52];
        let p = |i: usize| [pts[2 * i], pts[2 * i + 1]];
        let o = |bandwidth, k| Options {
            metric: Metric::Haversine,
            ..opts(Kernel::Triangular, bandwidth, k)
        };
        // Fixed bandwidth in km: only Paris - London (about 344 km) is within 500 km.
        let g = kernel_graph(&pts, &o(Bandwidth::Fixed(500.0), None)).unwrap();
        let (n, w) = row(&g, 1);
        assert_eq!(n, vec![2]);
        assert!(close(w[0], 1.0 - haversine_km(p(1), p(2)) / 500.0));
        // Adaptive bandwidth: the inflated k-th great-circle distance.
        let g = kernel_graph(&pts, &o(Bandwidth::Adaptive, Some(2))).unwrap();
        let (n, w) = row(&g, 0);
        assert_eq!(n, vec![1, 2]);
        let (d1, d2) = (haversine_km(p(0), p(1)), haversine_km(p(0), p(2)));
        let h = d2 * 1.0000001;
        assert!(close(w[0], 1.0 - d1 / h) && close(w[1], 1.0 - d2 / h));
        // All-pairs default bandwidth and the band kernel accept the metric too.
        assert!(kernel_graph(&pts, &o(Bandwidth::Default, None)).is_ok());
        let g = band_kernel_graph(
            &pts,
            1200.0,
            Kernel::Triangular,
            Some(1200.0),
            true,
            false,
            Metric::Haversine,
            MAX,
        )
        .unwrap();
        let (n, _) = row(&g, 0);
        assert_eq!(n, vec![1]);
    }

    #[test]
    fn jittered_kernel_uses_the_separated_points() {
        // Two coincident pairs; adaptive k = 1 would have a zero bandwidth without jitter.
        let pts = [0.0, 0.0, 0.0, 0.0, 4.0, 1.0, 4.0, 1.0];
        let o = Options {
            coincident: Coincident::Jitter { seed: 3 },
            ..opts(Kernel::Triangular, Bandwidth::Adaptive, Some(1))
        };
        let g = kernel_graph(&pts, &o).unwrap();
        assert_eq!(g.cardinalities(), vec![1; 4]);
        let (n, w) = row(&g, 0);
        assert_eq!(n, vec![1]);
        assert!(w[0] > 0.0 && w[0] < 1e-6, "{}", w[0]);
        let raise = opts(Kernel::Triangular, Bandwidth::Adaptive, Some(1));
        assert!(kernel_graph(&pts, &raise).is_err());
    }
}
