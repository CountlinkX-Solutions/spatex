//! Alpha shapes: libpysal's `alpha_shape` and `alpha_shape_auto` (`libpysal.cg.alpha_shapes`).
//!
//! libpysal triangulates the points (Qhull), keeps the triangles whose circumradius is below
//! `1 / alpha`, takes the edges that belong to exactly one kept triangle and hands them to
//! shapely's `polygonize`. Here the triangulation is spade's (exact predicates; identical to
//! Qhull's for points in general position) and the polygons are traced from the boundary edges
//! without GEOS:
//!
//! * every boundary edge is directed with its triangle on the left; at a vertex the walk continues
//!   with the first boundary edge clockwise from the way back, so a walk follows one face of the
//!   region, and walks are split at repeated vertices into simple rings (a bow-tie is two
//!   polygons; a hole touching the shell at a vertex is a shell plus a hole);
//! * counter-clockwise rings are shells, clockwise rings are holes, and a hole belongs to the
//!   smallest shell that contains it. This is the even-odd result of libpysal's `_filter_holes`
//!   (an island inside a hole is its own polygon; the hole's face polygon that `polygonize` also
//!   returns is dropped);
//! * output rings are open, shells clockwise, holes counter-clockwise, starting at the smallest
//!   vertex, polygons (and holes) sorted by their vertices, lexicographically.
//!
//! The automatic variant replicates libpysal's float arithmetic literally
//! (`alpha = 1 / r - EPS`, then `radius < 1 / alpha`), including its sensitivity to the last bit
//! of a radius: see `docs/parity.md`.

use std::collections::{HashMap, HashSet};

use robust::{orient2d, Coord};
use spade::{DelaunayTriangulation, Point2, Triangulation};

use super::Pt;

/// A polygon with open rings: the shell is clockwise (as GEOS's `polygonize` returns it) and the
/// holes counter-clockwise; every ring starts at its lexicographically smallest vertex.
#[derive(Clone, Debug, PartialEq)]
pub struct Polygon {
    pub shell: Vec<Pt>,
    pub holes: Vec<Vec<Pt>>,
}

/// A triangulation with the circumradius of every triangle (vertices counter-clockwise).
#[derive(Clone, Debug)]
pub struct Mesh {
    pub pts: Vec<Pt>,
    pub tris: Vec<[usize; 3]>,
    pub radii: Vec<f64>,
}

fn dist(a: Pt, b: Pt) -> f64 {
    ((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])).sqrt()
}

/// Circumradius as libpysal's `r_circumcircle_triangle_single` computes it (same operation
/// order). A degenerate triangle gives half its longest side; rounding can make the radicand
/// negative, which gives NaN as in numpy.
pub fn circumradius(a: Pt, b: Pt, c: Pt) -> f64 {
    let (ab, bc, ca) = (dist(a, b), dist(b, c), dist(c, a));
    let num = ab * bc * ca;
    let den = ((ab + bc + ca) * (bc + ca - ab) * (ca + ab - bc) * (ab + bc - ca)).sqrt();
    if den == 0.0 {
        ab.max(bc).max(ca) / 2.0
    } else {
        num / den
    }
}

fn check_points(points: &[Pt]) -> Result<(), String> {
    if points
        .iter()
        .any(|p| !(p[0].is_finite() && p[1].is_finite()))
    {
        return Err("points must be finite".into());
    }
    Ok(())
}

/// Distinct points in first-occurrence order (`-0.0` equals `0.0`).
fn distinct(points: &[Pt]) -> Vec<Pt> {
    let mut seen = HashSet::new();
    points
        .iter()
        .filter(|p| seen.insert(((p[0] + 0.0).to_bits(), (p[1] + 0.0).to_bits())))
        .copied()
        .collect()
}

/// Delaunay triangulation of the distinct points with circumradii.
pub fn mesh(points: &[Pt]) -> Result<Mesh, String> {
    check_points(points)?;
    let pts = distinct(points);
    if pts.len() < 3 {
        return Err(format!(
            "an alpha shape needs at least 3 distinct points, got {}",
            pts.len()
        ));
    }
    let mut dt: DelaunayTriangulation<Point2<f64>> = DelaunayTriangulation::new();
    for (i, p) in pts.iter().enumerate() {
        let h = dt
            .insert(Point2::new(p[0], p[1]))
            .map_err(|e| format!("point {i} cannot be triangulated: {e}"))?;
        debug_assert_eq!(h.index(), i);
    }
    if dt.num_inner_faces() == 0 {
        return Err("the points are collinear; an alpha shape needs a 2-D point set".into());
    }
    let tris: Vec<[usize; 3]> = dt
        .inner_faces()
        .map(|f| f.vertices().map(|v| v.fix().index()))
        .collect();
    let radii = tris
        .iter()
        .map(|t| circumradius(pts[t[0]], pts[t[1]], pts[t[2]]))
        .collect();
    Ok(Mesh { pts, tris, radii })
}

fn side(o: Pt, a: Pt, b: Pt) -> f64 {
    orient2d(
        Coord { x: o[0], y: o[1] },
        Coord { x: a[0], y: a[1] },
        Coord { x: b[0], y: b[1] },
    )
}

/// The rings of the region made of the triangles with `keep[i]`, as vertex indices: counter-
/// clockwise shells and clockwise holes (simple loops).
struct Rings {
    shells: Vec<Vec<usize>>,
    holes: Vec<Vec<usize>>,
}

/// Boundary edges of the kept triangles, directed with the triangle on the left.
fn boundary_edges(mesh: &Mesh, keep: &[bool]) -> Vec<(usize, usize)> {
    let mut directed: HashSet<(usize, usize)> = HashSet::new();
    for (t, _) in mesh.tris.iter().zip(keep).filter(|(_, k)| **k) {
        directed.insert((t[0], t[1]));
        directed.insert((t[1], t[2]));
        directed.insert((t[2], t[0]));
    }
    let mut boundary: Vec<(usize, usize)> = directed
        .iter()
        .filter(|(u, v)| !directed.contains(&(*v, *u)))
        .copied()
        .collect();
    boundary.sort_unstable();
    boundary
}

fn trace(mesh: &Mesh, keep: &[bool]) -> Rings {
    trace_edges(&mesh.pts, boundary_edges(mesh, keep))
}

/// Rings of a closed set of directed edges (every vertex has as many edges in as out).
fn trace_edges(pts: &[Pt], boundary: Vec<(usize, usize)>) -> Rings {
    let mut out: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(u, v) in &boundary {
        out.entry(u).or_default().push(v);
    }
    // Continuation of the walk that arrived along `u -> v`: the outgoing boundary edge with the
    // largest counter-clockwise angle from the direction `v -> u` (the first one clockwise).
    let next = |u: usize, v: usize| -> usize {
        let cands = &out[&v];
        let mut best = cands[0];
        for &w in &cands[1..] {
            if ccw_before(pts[v], pts[u], pts[best], pts[w]) {
                best = w;
            }
        }
        best
    };
    let mut visited: HashSet<(usize, usize)> = HashSet::new();
    let mut rings = Rings {
        shells: Vec::new(),
        holes: Vec::new(),
    };
    for &(a, b) in &boundary {
        if visited.contains(&(a, b)) {
            continue;
        }
        let mut walk = vec![a];
        let (mut u, mut v) = (a, b);
        while visited.insert((u, v)) {
            walk.push(v);
            let w = next(u, v);
            u = v;
            v = w;
        }
        walk.pop(); // the walk returned to its first vertex
        split_loops(&walk, pts, &mut rings);
    }
    rings
}

/// Whether direction `o -> p` comes before `o -> q` counter-clockwise from the reference
/// direction `o -> r` (angles in `(0, 2 pi)`).
fn ccw_before(o: Pt, r: Pt, p: Pt, q: Pt) -> bool {
    let class = |x: Pt| {
        let s = side(o, r, x);
        if s > 0.0 {
            0
        } else if s == 0.0 {
            1
        } else {
            2
        }
    };
    let (cp, cq) = (class(p), class(q));
    if cp != cq {
        return cp < cq;
    }
    side(o, p, q) > 0.0
}

/// Splits a closed walk at repeated vertices into simple loops and sorts them by orientation.
fn split_loops(walk: &[usize], pts: &[Pt], rings: &mut Rings) {
    let mut stack: Vec<usize> = Vec::new();
    let mut at: HashMap<usize, usize> = HashMap::new();
    let mut loops: Vec<Vec<usize>> = Vec::new();
    for &v in walk {
        if let Some(&p) = at.get(&v) {
            let lp: Vec<usize> = stack.drain(p..).collect();
            for x in &lp {
                at.remove(x);
            }
            loops.push(lp);
        }
        at.insert(v, stack.len());
        stack.push(v);
    }
    loops.push(stack);
    for lp in loops {
        if signed_area(&lp, pts) > 0.0 {
            rings.shells.push(lp);
        } else {
            rings.holes.push(lp);
        }
    }
}

fn signed_area(ring: &[usize], pts: &[Pt]) -> f64 {
    let o = pts[ring[0]];
    let mut s = 0.0;
    for i in 0..ring.len() {
        let (p, q) = (pts[ring[i]], pts[ring[(i + 1) % ring.len()]]);
        s += (p[0] - o[0]) * (q[1] - o[1]) - (q[0] - o[0]) * (p[1] - o[1]);
    }
    s / 2.0
}

/// Point in a simple ring (any orientation): `true` when inside or on the boundary.
fn in_ring(ring: &[usize], pts: &[Pt], p: Pt) -> bool {
    let mut inside = false;
    for i in 0..ring.len() {
        let (a, b) = (pts[ring[i]], pts[ring[(i + 1) % ring.len()]]);
        if a == p || b == p {
            return true;
        }
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let s = side(a, b, p);
            if s == 0.0 {
                return true;
            }
            if (s > 0.0) == (b[1] > a[1]) {
                inside = !inside;
            }
        } else if a[1] == p[1] && b[1] == p[1] && (a[0] <= p[0]) != (b[0] <= p[0]) {
            return true;
        }
    }
    inside
}

fn normalize(ring: &[usize], pts: &[Pt], clockwise: bool) -> Vec<Pt> {
    let mut v: Vec<Pt> = ring.iter().map(|&i| pts[i]).collect();
    if (signed_area(ring, pts) < 0.0) != clockwise {
        v.reverse();
    }
    let k = (0..v.len())
        .min_by(|&i, &j| {
            v[i][0]
                .total_cmp(&v[j][0])
                .then(v[i][1].total_cmp(&v[j][1]))
        })
        .unwrap_or(0);
    v.rotate_left(k);
    v
}

/// A polygon as vertex-index rings.
struct Raw {
    shell: Vec<usize>,
    holes: Vec<Vec<usize>>,
}

/// Pairs every hole with the smallest shell that contains it (holes outside every shell, the
/// unbounded face's outline, are dropped).
fn assemble(rings: Rings, pts: &[Pt]) -> Vec<Raw> {
    let areas: Vec<f64> = rings.shells.iter().map(|s| signed_area(s, pts)).collect();
    let sets: Vec<HashSet<usize>> = rings
        .shells
        .iter()
        .map(|s| s.iter().copied().collect())
        .collect();
    let mut holes: Vec<Vec<Vec<usize>>> = vec![Vec::new(); rings.shells.len()];
    for h in rings.holes {
        let probe = h
            .iter()
            .find(|i| sets.iter().all(|s| !s.contains(i)))
            .map(|&i| pts[i])
            .unwrap_or_else(|| {
                let (a, b) = (pts[h[0]], pts[h[1]]);
                [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
            });
        let owner = (0..rings.shells.len())
            .filter(|&s| in_ring(&rings.shells[s], pts, probe))
            .min_by(|&i, &j| areas[i].total_cmp(&areas[j]));
        if let Some(s) = owner {
            holes[s].push(h);
        }
    }
    rings
        .shells
        .into_iter()
        .zip(holes)
        .map(|(shell, holes)| Raw { shell, holes })
        .collect()
}

fn bbox(ring: &[usize], pts: &[Pt]) -> [f64; 4] {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for &i in ring {
        b[0] = b[0].min(pts[i][0]);
        b[1] = b[1].min(pts[i][1]);
        b[2] = b[2].max(pts[i][0]);
        b[3] = b[3].max(pts[i][1]);
    }
    b
}

/// Whether the ring `i` lies inside the closed region of the ring `j` (the rings do not cross).
fn ring_within(i: &[usize], j: &[usize], pts: &[Pt]) -> bool {
    let on_j: HashSet<usize> = j.iter().copied().collect();
    if let Some(&v) = i.iter().find(|v| !on_j.contains(v)) {
        return in_ring(j, pts, pts[v]);
    }
    // every vertex of `i` is a vertex of `j`: probe the middle of an edge that is not one of `j`
    let edge = |r: &[usize], k: usize| {
        let (a, b) = (r[k], r[(k + 1) % r.len()]);
        (a.min(b), a.max(b))
    };
    let edges_j: HashSet<(usize, usize)> = (0..j.len()).map(|k| edge(j, k)).collect();
    match (0..i.len()).find(|&k| !edges_j.contains(&edge(i, k))) {
        Some(k) => {
            let (a, b) = (pts[i[k]], pts[i[(k + 1) % i.len()]]);
            in_ring(j, pts, [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0])
        }
        None => true,
    }
}

/// libpysal's `_filter_holes`: when any polygon has a hole, keep the polygons whose shell lies
/// inside an odd number of shells (itself included). `polygonize` also returns the faces of the
/// holes, which sit inside the polygon around them and so are dropped; a hole face bounded by
/// several polygons that only touch at vertices lies inside none of them and is kept, as in
/// libpysal.
fn filter_holes(all: Vec<Raw>, pts: &[Pt]) -> Vec<Raw> {
    if all.iter().all(|p| p.holes.is_empty()) {
        return all;
    }
    let boxes: Vec<[f64; 4]> = all.iter().map(|p| bbox(&p.shell, pts)).collect();
    let inside = |i: usize, j: usize| {
        let (a, b) = (boxes[i], boxes[j]);
        b[0] <= a[0]
            && b[1] <= a[1]
            && a[2] <= b[2]
            && a[3] <= b[3]
            && ring_within(&all[i].shell, &all[j].shell, pts)
    };
    let keep: Vec<bool> = (0..all.len())
        .map(|i| (1 + (0..all.len()).filter(|&j| j != i && inside(i, j)).count()) % 2 == 1)
        .collect();
    all.into_iter()
        .zip(keep)
        .filter_map(|(p, k)| k.then_some(p))
        .collect()
}

fn finish(raw: Vec<Raw>, pts: &[Pt]) -> Vec<Polygon> {
    let mut polys: Vec<Polygon> = raw
        .iter()
        .map(|p| {
            let mut hs: Vec<Vec<Pt>> = p.holes.iter().map(|h| normalize(h, pts, false)).collect();
            hs.sort_by(|a, b| cmp_rings(a, b));
            Polygon {
                shell: normalize(&p.shell, pts, true),
                holes: hs,
            }
        })
        .collect();
    polys.sort_by(|a, b| cmp_rings(&a.shell, &b.shell));
    polys
}

/// Lexicographic order of rings by their vertices (polygons that touch can share a first vertex).
fn cmp_rings(a: &[Pt], b: &[Pt]) -> std::cmp::Ordering {
    for (p, q) in a.iter().zip(b) {
        let o = p[0].total_cmp(&q[0]).then(p[1].total_cmp(&q[1]));
        if o.is_ne() {
            return o;
        }
    }
    a.len().cmp(&b.len())
}

/// Polygons of the triangles with `keep[i]`, as libpysal's `_alpha_geoms` followed by
/// `_filter_holes`.
pub fn polygons(mesh: &Mesh, keep: &[bool]) -> Vec<Polygon> {
    let pts = &mesh.pts;
    let edges = boundary_edges(mesh, keep);
    let reversed = edges.iter().map(|&(u, v)| (v, u)).collect();
    // polygonize returns the faces of the region and the bounded faces of its complement
    let mut all = assemble(trace_edges(pts, edges), pts);
    all.extend(assemble(trace_edges(pts, reversed), pts));
    finish(filter_holes(all, pts), pts)
}

/// Whether the triangles with `keep[i]` make one polygon without holes that has every point
/// (libpysal's `_valid_hull`: exactly one polygon, all points intersecting it). `polygonize`
/// also returns the face of every hole, so a single polygon has neither holes nor hole faces; a
/// point is covered exactly when it is a vertex of a kept triangle, since no point lies inside
/// or on an edge of a Delaunay triangle.
fn is_single_hull(mesh: &Mesh, keep: &[bool]) -> bool {
    let rings = trace(mesh, keep);
    if rings.shells.len() != 1 || !rings.holes.is_empty() {
        return false;
    }
    let mut used = vec![false; mesh.pts.len()];
    for (t, _) in mesh.tris.iter().zip(keep).filter(|(_, k)| **k) {
        for &v in t {
            used[v] = true;
        }
    }
    used.iter().all(|&u| u)
}

fn hull_of_few(points: &[Pt]) -> Vec<Polygon> {
    let d = distinct(points);
    if d.len() == 3 && side(d[0], d[1], d[2]) != 0.0 {
        let mut ring = d;
        if side(ring[0], ring[1], ring[2]) > 0.0 {
            ring.swap(1, 2);
        }
        let k = (0..3)
            .min_by(|&i, &j| {
                ring[i][0]
                    .total_cmp(&ring[j][0])
                    .then(ring[i][1].total_cmp(&ring[j][1]))
            })
            .unwrap_or(0);
        ring.rotate_left(k);
        return vec![Polygon {
            shell: ring,
            holes: Vec::new(),
        }];
    }
    Vec::new()
}

/// Alpha shape of `points` for `alpha >= 0` (`0` keeps every triangle). Fewer than four points
/// give their convex hull whatever `alpha` is, as libpysal does.
pub fn alpha_shape(points: &[Pt], alpha: f64) -> Result<Vec<Polygon>, String> {
    check_points(points)?;
    if !(alpha >= 0.0 && alpha.is_finite()) {
        return Err("alpha must be a finite non-negative number".into());
    }
    if points.len() < 4 {
        return Ok(hull_of_few(points));
    }
    let m = mesh(points)?;
    let threshold = 1.0 / alpha;
    let keep: Vec<bool> = m.radii.iter().map(|&r| r < threshold).collect();
    Ok(polygons(&m, &keep))
}

/// Little-endian WKB of a polygon (rings closed).
pub fn polygon_wkb(p: &Polygon) -> Vec<u8> {
    let mut w = vec![1u8];
    w.extend(3u32.to_le_bytes());
    w.extend(((1 + p.holes.len()) as u32).to_le_bytes());
    for ring in std::iter::once(&p.shell).chain(p.holes.iter()) {
        w.extend(((ring.len() + 1) as u32).to_le_bytes());
        for q in ring.iter().chain(ring.first()) {
            w.extend(q[0].to_le_bytes());
            w.extend(q[1].to_le_bytes());
        }
    }
    w
}

/// Result of the automatic search.
#[derive(Clone, Debug, PartialEq)]
pub struct Auto {
    pub polygon: Polygon,
    /// The circumradius whose threshold produced the polygon.
    pub radius: f64,
}

/// Triangles with `radius < 1 / alpha` for `alpha = 1 / radius - EPS`, exactly as libpysal's loop
/// evaluates it (the last bit of the radius matters).
fn keep_below(radii: &[f64], radius: f64) -> Vec<bool> {
    let alpha = 1.0 / radius - f64::EPSILON;
    let threshold = 1.0 / alpha;
    radii.iter().map(|&r| r < threshold).collect()
}

/// libpysal's `alpha_shape_auto` on a triangulation. Radii that are NaN (degenerate triangles) are
/// 0, so those triangles are always kept. Starting from the largest radius, every `step`-th
/// radius gives a candidate shape; the last candidate that is one polygon containing every point
/// wins and the first one that is not ends the search. (libpysal falls back to the first polygon
/// of the all-triangles shape when even the first candidate fails, and raises
/// `UnboundLocalError` when asked for the radius; this is an error here.)
pub fn auto_on_mesh(mesh: &Mesh, step: usize) -> Result<Auto, String> {
    if step == 0 {
        return Err("step must be a positive integer".into());
    }
    let radii: Vec<f64> = mesh
        .radii
        .iter()
        .map(|&r| if r.is_nan() { 0.0 } else { r })
        .collect();
    let mut sorted = radii.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let mut best: Option<(Vec<bool>, f64)> = None;
    for &radius in sorted.iter().step_by(step) {
        let keep = keep_below(&radii, radius);
        if !is_single_hull(mesh, &keep) {
            break;
        }
        best = Some((keep, radius));
    }
    let (keep, radius) = best.ok_or_else(|| {
        "no valid alpha shape: the search found no single polygon containing every point \
         (libpysal's float comparison `radius < 1 / (1 / radius - EPS)` can drop the largest triangle)"
            .to_string()
    })?;
    let polygon = polygons(mesh, &keep)
        .pop()
        .ok_or("alpha shape lost its polygon")?;
    Ok(Auto { polygon, radius })
}

/// libpysal's `alpha_shape_auto`: the tightest alpha shape that is one polygon and contains every
/// point, found by walking the circumradii from the largest down in jumps of `step`. Needs three
/// distinct, non-collinear points.
pub fn alpha_shape_auto(points: &[Pt], step: usize) -> Result<Auto, String> {
    check_points(points)?;
    if points.len() < 3 {
        return Err("alpha_shape_auto needs at least 3 points".into());
    }
    if step == 0 {
        return Err("step must be a positive integer".into());
    }
    auto_on_mesh(&mesh(points)?, step)
}

/// Centres of libpysal's bounding circles, one per boundary edge of the shell
/// (`construct_bounding_circles`).
pub fn bounding_circle_centers(p: &Polygon, radius: f64) -> Vec<Pt> {
    let n = p.shell.len();
    (0..n)
        .map(|i| circle_center(p.shell[i], p.shell[(i + 1) % n], radius))
        .collect()
}

/// libpysal's `_construct_centers`: of the two circles of radius `radius` through `a` and `b`,
/// the one on the left of `a -> b` (the other when the sign is not exactly positive).
fn circle_center(a: Pt, b: Pt, radius: f64) -> Pt {
    let (mx, my) = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5);
    let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    let rotation = if b[0] - a[0] == 0.0 {
        std::f64::consts::FRAC_PI_2
    } else {
        ((b[1] - a[1]) / (b[0] - a[0])).atan()
    };
    let chord = (0.5 * d / radius).acos().sin() * radius;
    let (dx, dy) = (chord * rotation.sin(), chord * rotation.cos());
    let (up, down) = ([mx - dx, my + dy], [mx + dx, my - dy]);
    let sign = (b[0] - a[0]) * (up[1] - a[1]) - (b[1] - a[1]) * (up[0] - a[0]);
    if sign > 0.0 {
        up
    } else {
        down
    }
}
