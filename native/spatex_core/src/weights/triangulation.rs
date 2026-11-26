//! Triangulation-based graphs, mirroring `libpysal.graph.Graph.build_triangulation`:
//! Delaunay, Gabriel, relative neighborhood and Voronoi graphs over a Euclidean point set.
//!
//! The Delaunay triangulation comes from the `spade` crate (exact, robust predicates).
//! For points in general position it is unique, so its edge set equals the one scipy's Qhull
//! returns; for cocircular points (e.g. a square) the diagonal is implementation-defined and
//! the two libraries may disagree. Gabriel and relative neighborhood graphs are filtered
//! Delaunay subgraphs using libpysal's exact rules, including their tie behavior:
//!
//! * Gabriel drops `(i, j)` when some *other Delaunay neighbor `k` of `i`* has
//!   `d_ij^2 > d_ik^2 + d_jk^2` (strict; points on the diametral circle keep the edge);
//! * relative neighborhood drops `(i, j)` when *any* point `k` has `max(d_ik, d_jk) <= d_ij`
//!   (non-strict; a point on the lune boundary prunes).
//!
//! The Voronoi graph is the rook contiguity of the Voronoi cells clipped to a region (libpysal's
//! `voronoi_frames` + vertex rook): Delaunay edge `(i, j)` is kept when its dual Voronoi edge (the
//! segment between the circumcenters of its two triangles, or an outward ray for a hull edge)
//! has positive length inside the region. Zero-length Voronoi edges (cocircular sites) are not
//! shared cell edges, so the result does not depend on the Delaunay diagonal chosen there.
//!
//! Edge weights are `kernel(d_ij, bandwidth)` as in [`super::kernel`]. Unlike libpysal, a node
//! whose edges are all tapered away is kept as an island instead of vanishing from the graph.

use rayon::prelude::*;
use spade::{DelaunayTriangulation, Point2, Triangulation};

use super::kernel::{weigh_arrays, Bandwidth, EdgeArrays, Kernel};
use super::knn::{coincident_groups, split_points, Coincident};
use crate::graph::Csr;

/// Which triangulation-derived graph to build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Delaunay,
    Gabriel,
    RelativeNeighborhood,
    Voronoi,
}

/// The region the Voronoi cells are clipped to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Clip {
    /// The bounding box of the points (libpysal's default `clip="bounding_box"`).
    BoundingBox,
    /// The convex hull of the points (`clip="convex_hull"`).
    ConvexHull,
    /// No clip (`clip=None`): GEOS still bounds the diagram by the bounding box expanded by its
    /// largest side on every side, which is reproduced.
    Extended,
}

impl Clip {
    /// Parses `bounding_box`, `convex_hull` or `none`.
    pub fn parse(name: &str) -> Result<Clip, String> {
        match name {
            "bounding_box" => Ok(Clip::BoundingBox),
            "convex_hull" => Ok(Clip::ConvexHull),
            "none" => Ok(Clip::Extended),
            other => Err(format!("unknown Voronoi clip {other:?}")),
        }
    }
}

impl Method {
    /// Parses `delaunay`, `gabriel`, `relative_neighborhood` or `voronoi`.
    pub fn parse(name: &str) -> Result<Method, String> {
        match name {
            "delaunay" => Ok(Method::Delaunay),
            "gabriel" => Ok(Method::Gabriel),
            "relative_neighborhood" => Ok(Method::RelativeNeighborhood),
            "voronoi" => Ok(Method::Voronoi),
            other => Err(format!("unknown triangulation method {other:?}")),
        }
    }
}

/// Options of [`triangulation_graph`].
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub method: Method,
    pub kernel: Kernel,
    /// Positive; `f64::INFINITY` (libpysal's default) leaves every edge untapered.
    pub bandwidth: f64,
    pub taper: bool,
    pub decay: bool,
    pub coincident: Coincident,
    /// Clip region of [`Method::Voronoi`] (ignored by the other methods).
    pub clip: Clip,
}

type Pt = [f64; 2];

fn dist2(p: Pt, q: Pt) -> f64 {
    let (dx, dy) = (p[0] - q[0], p[1] - q[1]);
    dx * dx + dy * dy
}

/// Undirected Delaunay edges `(a, b)`, `a < b`, over distinct points (indices into `pts`).
fn triangulate(pts: &[Pt]) -> Result<DelaunayTriangulation<Point2<f64>>, String> {
    if pts.len() < 3 {
        return Err(format!(
            "a triangulation needs at least 3 distinct points, got {}",
            pts.len()
        ));
    }
    let mut dt: DelaunayTriangulation<Point2<f64>> = DelaunayTriangulation::new();
    for (i, p) in pts.iter().enumerate() {
        let handle = dt
            .insert(Point2::new(p[0], p[1]))
            .map_err(|e| format!("point {i} cannot be triangulated: {e}"))?;
        debug_assert_eq!(handle.index(), i);
    }
    if dt.num_inner_faces() == 0 {
        return Err("the points are collinear; a triangulation needs a 2-D point set".into());
    }
    Ok(dt)
}

fn delaunay_edges(pts: &[Pt]) -> Result<Vec<(u32, u32)>, String> {
    let dt = triangulate(pts)?;
    let mut edges: Vec<(u32, u32)> = dt
        .undirected_edges()
        .map(|e| {
            let [a, b] = e.vertices().map(|v| v.fix().index() as u32);
            (a.min(b), a.max(b))
        })
        .collect();
    edges.sort_unstable();
    Ok(edges)
}

fn cross(a: Pt, b: Pt) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn sub(a: Pt, b: Pt) -> Pt {
    [a[0] - b[0], a[1] - b[1]]
}

/// Counter-clockwise vertices of the clip region.
fn clip_polygon(dt: &DelaunayTriangulation<Point2<f64>>, pts: &[Pt], clip: Clip) -> Vec<Pt> {
    let (mut x0, mut y0, mut x1, mut y1) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for p in pts {
        (x0, y0, x1, y1) = (x0.min(p[0]), y0.min(p[1]), x1.max(p[0]), y1.max(p[1]));
    }
    match clip {
        Clip::BoundingBox => vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        Clip::Extended => {
            let e = (x1 - x0).max(y1 - y0);
            vec![
                [x0 - e, y0 - e],
                [x1 + e, y0 - e],
                [x1 + e, y1 + e],
                [x0 - e, y1 + e],
            ]
        }
        Clip::ConvexHull => {
            let mut hull: Vec<Pt> = dt
                .convex_hull()
                .map(|e| {
                    let p = e.from().position();
                    [p.x, p.y]
                })
                .collect();
            let area: f64 = (0..hull.len())
                .map(|k| cross(hull[k], hull[(k + 1) % hull.len()]))
                .sum();
            if area < 0.0 {
                hull.reverse();
            }
            hull
        }
    }
}

/// Cyrus-Beck: the parameter interval of `origin + t dir`, `t` in `[t0, t1]`, inside the convex
/// counter-clockwise `polygon`, if any.
fn clip_line(
    origin: Pt,
    dir: Pt,
    (mut t0, mut t1): (f64, f64),
    polygon: &[Pt],
) -> Option<(f64, f64)> {
    for k in 0..polygon.len() {
        let (a, b) = (polygon[k], polygon[(k + 1) % polygon.len()]);
        let edge = sub(b, a);
        let num = cross(edge, sub(origin, a));
        let den = cross(edge, dir);
        if den == 0.0 {
            if num < 0.0 {
                return None;
            }
        } else {
            let t = -num / den;
            if den > 0.0 {
                t0 = t0.max(t);
            } else {
                t1 = t1.min(t);
            }
        }
        if t0 > t1 {
            return None;
        }
    }
    Some((t0, t1))
}

/// Delaunay edges whose dual Voronoi edge has positive length inside the `clip` region.
fn voronoi_edges(pts: &[Pt], clip: Clip) -> Result<Vec<(u32, u32)>, String> {
    let dt = triangulate(pts)?;
    let polygon = clip_polygon(&dt, pts, clip);
    let scale = polygon
        .iter()
        .flat_map(|p| p.iter())
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .max(1.0);
    let tol = 1e-12 * scale;
    let mut edges: Vec<(u32, u32)> = Vec::new();
    for e in dt.undirected_edges() {
        let d = e.as_directed();
        let [a, b] = e.vertices().map(|v| v.fix().index() as u32);
        let center =
            |f: spade::handles::FaceHandle<'_, spade::handles::PossiblyOuterTag, _, _, _, _>| {
                f.as_inner().map(|inner| {
                    let c = inner.circumcenter();
                    [c.x, c.y]
                })
            };
        let (left, right) = (center(d.face()), center(d.rev().face()));
        let (origin, dir, range) = match (left, right) {
            (Some(c1), Some(c2)) => (c1, sub(c2, c1), (0.0, 1.0)),
            (Some(c), None) | (None, Some(c)) => {
                // Hull edge: a ray from the inner circumcenter, away from the inner triangle.
                let inner = if left.is_some() { d } else { d.rev() };
                let (pa, pb) = (inner.from().position(), inner.to().position());
                let opposite = inner.next().to().position();
                let mut n = [-(pb.y - pa.y), pb.x - pa.x];
                if n[0] * (opposite.x - pa.x) + n[1] * (opposite.y - pa.y) > 0.0 {
                    n = [-n[0], -n[1]];
                }
                (c, n, (0.0, f64::INFINITY))
            }
            (None, None) => continue,
        };
        let norm = dir[0].hypot(dir[1]);
        if let Some((t0, t1)) = clip_line(origin, dir, range, &polygon) {
            if (t1 - t0) * norm > tol {
                edges.push((a.min(b), a.max(b)));
            }
        }
    }
    edges.sort_unstable();
    Ok(edges)
}

/// Keeps the Gabriel edges, applying libpysal's neighbor-restricted strict test.
fn gabriel_filter(pts: &[Pt], edges: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    let mut neighbors: Vec<Vec<u32>> = vec![Vec::new(); pts.len()];
    for &(a, b) in &edges {
        neighbors[a as usize].push(b);
        neighbors[b as usize].push(a);
    }
    let violated = |i: u32, j: u32| {
        let (pi, pj) = (pts[i as usize], pts[j as usize]);
        let dij2 = dist2(pi, pj);
        neighbors[i as usize].iter().filter(|&&k| k != j).any(|&k| {
            let pk = pts[k as usize];
            dij2 > dist2(pi, pk) + dist2(pj, pk)
        })
    };
    // libpysal evaluates the rule from both endpoints and drops the edge if either fires.
    edges
        .into_iter()
        .filter(|&(a, b)| !(violated(a, b) || violated(b, a)))
        .collect()
}

/// Keeps the relative-neighborhood edges: no point `k` with `max(d_ik, d_jk) <= d_ij`.
fn relative_neighborhood_filter(pts: &[Pt], edges: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    edges
        .into_par_iter()
        .filter(|&(i, j)| {
            let (pi, pj) = (pts[i as usize], pts[j as usize]);
            let dij = dist2(pi, pj).sqrt();
            !pts.iter().enumerate().any(|(k, &pk)| {
                k as u32 != i
                    && k as u32 != j
                    && dist2(pi, pk).sqrt().max(dist2(pj, pk).sqrt()) <= dij
            })
        })
        .collect()
}

/// Builds the triangulation graph of interleaved `x0, y0, x1, y1, ...` points.
///
/// Errors on fewer than 3 distinct or on collinear points, and on coincident points when
/// `coincident` is [`Coincident::Raise`]. With [`Coincident::Clique`] the triangulation is built
/// over distinct locations; points sharing a site take the site's edges and are mutually linked
/// with weight `kernel(0, bandwidth)`.
pub fn triangulation_graph(points: &[f64], opts: &Options) -> Result<Csr, String> {
    if opts.bandwidth.is_nan() || opts.bandwidth <= 0.0 {
        return Err(format!(
            "bandwidth must be positive (or infinite), got {}",
            opts.bandwidth
        ));
    }
    let pts = split_points(points)?;
    let n = pts.len();
    let groups = coincident_groups(&pts);
    if !groups.is_empty() && opts.coincident == Coincident::Raise {
        return Err(format!(
            "{} points share a location with another point (coincident points); \
             use coincident: :clique to connect them",
            groups.iter().map(Vec::len).sum::<usize>()
        ));
    }

    // Distinct locations: the first point of every site represents it.
    let mut site_of: Vec<usize> = (0..n).collect();
    for g in &groups {
        for &i in g {
            site_of[i] = g[0];
        }
    }
    let reps: Vec<usize> = (0..n).filter(|&i| site_of[i] == i).collect();
    let unique: Vec<Pt> = reps.iter().map(|&i| pts[i]).collect();

    let edges = match opts.method {
        Method::Delaunay => delaunay_edges(&unique)?,
        Method::Gabriel => gabriel_filter(&unique, delaunay_edges(&unique)?),
        Method::RelativeNeighborhood => {
            relative_neighborhood_filter(&unique, delaunay_edges(&unique)?)
        }
        Method::Voronoi => voronoi_edges(&unique, opts.clip)?,
    };

    let directed: Vec<(u32, u32, f64)> = edges
        .iter()
        .flat_map(|&(a, b)| {
            let (i, j) = (reps[a as usize] as u32, reps[b as usize] as u32);
            let d = dist2(pts[i as usize], pts[j as usize]).sqrt();
            [(i, j, d), (j, i, d)]
        })
        .collect();
    let (mut focal, mut neighbor, mut weight) = weigh_arrays(
        n,
        directed,
        opts.kernel,
        Bandwidth::Fixed(opts.bandwidth),
        false,
        opts.taper,
        opts.decay,
    )?;

    if !groups.is_empty() {
        (focal, neighbor, weight) = expand_cliques(
            (focal, neighbor, weight),
            &groups,
            opts.kernel.eval(0.0, opts.bandwidth),
        );
    }
    Csr::from_arrays(n, &focal, &neighbor, &weight)
}

/// Spreads every edge over the points sitting on its two sites and links each site's points.
fn expand_cliques(
    (focal, neighbor, weight): EdgeArrays,
    groups: &[Vec<usize>],
    fill: f64,
) -> EdgeArrays {
    let mut site: std::collections::HashMap<u32, &[usize]> = std::collections::HashMap::new();
    for g in groups {
        site.insert(g[0] as u32, g);
    }
    let members = |i: u32| -> Vec<u32> {
        site.get(&i)
            .map_or_else(|| vec![i], |g| g.iter().map(|&m| m as u32).collect())
    };
    let (mut f, mut t, mut w) = (Vec::new(), Vec::new(), Vec::new());
    for ((&a, &b), &x) in focal.iter().zip(&neighbor).zip(&weight) {
        let to = members(b);
        for from in members(a) {
            for &target in &to {
                f.push(from);
                t.push(target);
                w.push(x);
            }
        }
    }
    for g in groups {
        for &a in g {
            for &b in g {
                if a != b {
                    f.push(a as u32);
                    t.push(b as u32);
                    w.push(fill);
                }
            }
        }
    }
    (f, t, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(method: Method) -> Options {
        Options {
            method,
            kernel: Kernel::Boxcar,
            bandwidth: f64::INFINITY,
            taper: true,
            decay: false,
            coincident: Coincident::Raise,
            clip: Clip::BoundingBox,
        }
    }

    fn pairs(g: &Csr) -> Vec<(u32, u32)> {
        let (f, t, _) = g.to_arrays();
        f.into_iter().zip(t).filter(|(a, b)| a != b).collect()
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

    // Right triangle with legs 3 and 4: sides 3, 4, 5.
    const TRI: [f64; 6] = [0.0, 0.0, 3.0, 0.0, 0.0, 4.0];

    #[test]
    fn single_triangle_delaunay_is_complete() {
        let g = triangulation_graph(&TRI, &opts(Method::Delaunay)).unwrap();
        assert_eq!(
            pairs(&g),
            vec![(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        );
        assert!(g.to_arrays().2.iter().all(|&w| w == 1.0));
    }

    #[test]
    fn gabriel_keeps_a_right_angle_hypotenuse_rng_drops_it() {
        // Vertex 0 sits exactly on the circle with diameter (1, 2): the strict test keeps it.
        let g = triangulation_graph(&TRI, &opts(Method::Gabriel)).unwrap();
        assert_eq!(g.nonzero_count(), 6);
        // RNG: d_01 = 3, d_02 = 4 <= 5 prunes (1, 2).
        let g = triangulation_graph(&TRI, &opts(Method::RelativeNeighborhood)).unwrap();
        assert_eq!(pairs(&g), vec![(0, 1), (0, 2), (1, 0), (2, 0)]);
    }

    #[test]
    fn obtuse_triangle_loses_its_long_side_under_gabriel() {
        let pts = [0.0, 0.0, 4.0, 0.0, 2.0, 0.5];
        let g = triangulation_graph(&pts, &opts(Method::Gabriel)).unwrap();
        assert_eq!(pairs(&g), vec![(0, 2), (1, 2), (2, 0), (2, 1)]);
    }

    #[test]
    fn kernel_weights_and_taper() {
        let o = Options {
            kernel: Kernel::Triangular,
            bandwidth: 5.0,
            ..opts(Method::Delaunay)
        };
        let g = triangulation_graph(&TRI, &o).unwrap();
        // The hypotenuse (d == h) has weight 0 and is tapered away.
        assert_eq!(pairs(&g), vec![(0, 1), (0, 2), (1, 0), (2, 0)]);
        let (_, _, w) = g.to_arrays();
        assert_eq!(w, vec![1.0 - 0.6, 1.0 - 0.8, 1.0 - 0.6, 1.0 - 0.8]);
    }

    #[test]
    fn fully_tapered_nodes_become_islands() {
        let pts = [0.0, 0.0, 1.0, 0.0, 0.4, 0.9, 9.0, 9.5];
        let o = Options {
            bandwidth: 2.0,
            ..opts(Method::Delaunay)
        };
        let g = triangulation_graph(&pts, &o).unwrap();
        assert_eq!(g.n, 4);
        assert_eq!(g.to_arrays().0.iter().filter(|&&f| f == 3).count(), 1);
        assert_eq!(g.neighbors[g.offsets[3]], 3);
    }

    #[test]
    fn gabriel_is_a_subset_of_delaunay_and_rng_of_gabriel() {
        for seed in [1, 2, 3] {
            let pts = lcg_points(150, seed);
            let d = pairs(&triangulation_graph(&pts, &opts(Method::Delaunay)).unwrap());
            let g = pairs(&triangulation_graph(&pts, &opts(Method::Gabriel)).unwrap());
            let r = pairs(&triangulation_graph(&pts, &opts(Method::RelativeNeighborhood)).unwrap());
            assert!(g.iter().all(|e| d.contains(e)));
            assert!(r.iter().all(|e| g.contains(e)));
            assert!(r.len() < g.len() && g.len() < d.len());
            assert!(d.len() <= 2 * (3 * 150 - 6));
        }
    }

    #[test]
    fn delaunay_satisfies_the_empty_circle_property() {
        let pts = lcg_points(40, 11);
        let g = triangulation_graph(&pts, &opts(Method::Delaunay)).unwrap();
        let p = |i: usize| [pts[2 * i], pts[2 * i + 1]];
        let edges: std::collections::HashSet<(u32, u32)> = pairs(&g).into_iter().collect();
        // Every nearest-neighbor pair is a Delaunay edge.
        for i in 0..40usize {
            let j = (0..40usize)
                .filter(|&j| j != i)
                .min_by(|&a, &b| dist2(p(i), p(a)).total_cmp(&dist2(p(i), p(b))))
                .unwrap();
            assert!(edges.contains(&(i as u32, j as u32)));
        }
    }

    #[test]
    fn coincident_points_raise_or_form_cliques() {
        let pts = [0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 3.0, 0.0];
        let err = triangulation_graph(&pts, &opts(Method::Delaunay)).unwrap_err();
        assert!(err.contains("coincident"), "{err}");
        let o = Options {
            coincident: Coincident::Clique,
            clip: Clip::BoundingBox,
            ..opts(Method::Delaunay)
        };
        let g = triangulation_graph(&pts, &o).unwrap();
        let row = |i: usize| g.neighbors[g.offsets[i]..g.offsets[i + 1]].to_vec();
        assert_eq!(row(0), vec![1, 2, 3]);
        assert_eq!(row(1), vec![0, 2, 3]);
        assert_eq!(row(3), vec![0, 1, 2]);
        assert_eq!(row(2), vec![0, 1, 3]);
    }

    #[test]
    fn rejects_degenerate_input() {
        let two = triangulation_graph(&[0.0, 0.0, 1.0, 1.0], &opts(Method::Delaunay));
        assert!(two.unwrap_err().contains("at least 3"));
        let line = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let err = triangulation_graph(&line, &opts(Method::Delaunay)).unwrap_err();
        assert!(err.contains("collinear"), "{err}");
        let bad = Options {
            bandwidth: 0.0,
            ..opts(Method::Delaunay)
        };
        assert!(triangulation_graph(&TRI, &bad).is_err());
    }
}
