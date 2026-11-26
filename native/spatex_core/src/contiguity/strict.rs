//! Geometric contiguity of polygons: libpysal's `strict=True` predicates (shapely `touches` for
//! queen, DE-9IM `F***1****` for rook) and `by_perimeter` weights (the length of
//! `intersection(geom_i, geom_j)`).
//!
//! Every pair of polygonal geometries with overlapping bounding boxes is classified from its
//! boundary segments with exact orientation tests (`robust`): proper crossings, collinear
//! overlaps (with their length and direction) and point contacts. Rings are oriented so that
//! the interior lies to the left of every edge, so a collinear overlap traversed in the same
//! direction by both geometries has both interiors on one side. The interiors intersect when a
//! proper crossing exists, when such a same-direction overlap exists, or when a piece of one
//! boundary lies strictly inside the other geometry (tested on the midpoints of the pieces
//! between contact points). Then:
//!
//! * queen (touches): the boundaries meet and the interiors do not intersect;
//! * rook: additionally the boundaries share a positive length;
//! * perimeter: the shared length (0 for a point contact); overlapping interiors are an error.
//!
//! Cost: `O(m_i m_j)` segment tests per candidate pair of geometries with `m` vertices.

use robust::{orient2d, Coord};
use rstar::primitives::{GeomWithData, Rectangle};
use rstar::{RTree, AABB};
use wkb::reader::read_wkb;

use geo_traits::{
    CoordTrait, GeometryTrait, GeometryType, LineStringTrait, MultiPolygonTrait, PolygonTrait,
};

pub(super) type Pt = [f64; 2];

/// A polygonal geometry: closed rings with the interior on the left of every edge.
pub struct Shape {
    rings: Vec<Vec<Pt>>,
    pub(super) bbox: [f64; 4],
}

impl Shape {
    /// A shape from closed rings with the interior on the left of every edge.
    pub(super) fn from_rings(rings: Vec<Vec<Pt>>) -> Shape {
        let mut bbox = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for p in rings.iter().flatten() {
            bbox = [
                bbox[0].min(p[0]),
                bbox[1].min(p[1]),
                bbox[2].max(p[0]),
                bbox[3].max(p[1]),
            ];
        }
        Shape { rings, bbox }
    }

    /// The first vertex of every ring (exterior rings and holes of all parts).
    pub(super) fn ring_starts(&self) -> impl Iterator<Item = Pt> + '_ {
        self.rings.iter().filter_map(|r| r.first().copied())
    }

    pub(super) fn segments(&self) -> impl Iterator<Item = (Pt, Pt)> + '_ {
        self.rings
            .iter()
            .flat_map(|r| r.windows(2).map(|w| (w[0], w[1])))
            .filter(|(a, b)| a != b)
    }
}

fn signed_area(ring: &[Pt]) -> f64 {
    ring.windows(2)
        .map(|w| w[0][0] * w[1][1] - w[1][0] * w[0][1])
        .sum::<f64>()
        / 2.0
}

fn ring<R: LineStringTrait<T = f64>>(r: &R, geom: usize, hole: bool) -> Result<Vec<Pt>, String> {
    let mut pts: Vec<Pt> = Vec::new();
    for c in r.coords() {
        let (x, y) = (c.x(), c.y());
        if !x.is_finite() || !y.is_finite() {
            return Err(format!("geometry {geom}: non-finite coordinate"));
        }
        pts.push([x, y]);
    }
    if pts.len() > 1 && pts.first() != pts.last() {
        pts.push(pts[0]);
    }
    // Exterior counter-clockwise, holes clockwise: the interior is on the left.
    let ccw = signed_area(&pts) > 0.0;
    if ccw == hole {
        pts.reverse();
    }
    Ok(pts)
}

fn push_polygon<P: PolygonTrait<T = f64>>(
    rings: &mut Vec<Vec<Pt>>,
    p: &P,
    geom: usize,
) -> Result<(), String> {
    if let Some(e) = p.exterior() {
        rings.push(ring(&e, geom, false)?);
    }
    for h in p.interiors() {
        rings.push(ring(&h, geom, true)?);
    }
    Ok(())
}

/// Parses polygon / multipolygon WKBs into shapes.
pub fn shapes(wkbs: &[&[u8]]) -> Result<Vec<Shape>, String> {
    wkbs.iter()
        .enumerate()
        .map(|(i, bytes)| {
            let g = read_wkb(bytes).map_err(|e| format!("geometry {i}: invalid WKB ({e})"))?;
            let mut rings = Vec::new();
            match g.as_type() {
                GeometryType::Polygon(p) => push_polygon(&mut rings, p, i)?,
                GeometryType::MultiPolygon(mp) => {
                    for p in mp.polygons() {
                        push_polygon(&mut rings, &p, i)?;
                    }
                }
                _ => {
                    return Err(format!(
                        "geometry {i}: strict, by_perimeter and fuzzy contiguity need Polygon or \
                         MultiPolygon geometries"
                    ))
                }
            }
            Ok(Shape::from_rings(rings))
        })
        .collect()
}

pub(super) fn orient(a: Pt, b: Pt, c: Pt) -> f64 {
    let p = |q: Pt| Coord { x: q[0], y: q[1] };
    orient2d(p(a), p(b), p(c))
}

pub(super) fn within_box(p: Pt, a: Pt, b: Pt) -> bool {
    p[0] >= a[0].min(b[0])
        && p[0] <= a[0].max(b[0])
        && p[1] >= a[1].min(b[1])
        && p[1] <= a[1].max(b[1])
}

fn boxes_meet(a: (Pt, Pt), b: (Pt, Pt)) -> bool {
    a.0[0].min(a.1[0]) <= b.0[0].max(b.1[0])
        && b.0[0].min(b.1[0]) <= a.0[0].max(a.1[0])
        && a.0[1].min(a.1[1]) <= b.0[1].max(b.1[1])
        && b.0[1].min(b.1[1]) <= a.0[1].max(a.1[1])
}

fn param(p1: Pt, p2: Pt, q: Pt) -> f64 {
    let u = [p2[0] - p1[0], p2[1] - p1[1]];
    ((q[0] - p1[0]) * u[0] + (q[1] - p1[1]) * u[1]) / (u[0] * u[0] + u[1] * u[1])
}

/// How the boundaries of two shapes meet.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Contact {
    /// The boundaries share at least one point.
    pub touch: bool,
    /// Total length of the collinear boundary overlaps.
    pub length: f64,
    /// The interiors intersect.
    pub overlap: bool,
    /// Two edges cross properly (each passes strictly to both sides of the other).
    pub crossing: bool,
}

/// Per-segment contact parameters (`t` in `[0, 1]`) and collinear-overlap intervals.
#[derive(Default)]
struct Marks {
    cuts: Vec<f64>,
    shared: Vec<(f64, f64)>,
}

fn segment_contact(p: (Pt, Pt), q: (Pt, Pt), c: &mut Contact, mp: &mut Marks, mq: &mut Marks) {
    let (p1, p2, q1, q2) = (p.0, p.1, q.0, q.1);
    let (o1, o2) = (orient(p1, p2, q1), orient(p1, p2, q2));
    let (o3, o4) = (orient(q1, q2, p1), orient(q1, q2, p2));
    if o1 == 0.0 && o2 == 0.0 {
        let (t1, t2) = (param(p1, p2, q1), param(p1, p2, q2));
        let (lo, hi) = (t1.min(t2).max(0.0), t1.max(t2).min(1.0));
        if hi < lo {
            return;
        }
        c.touch = true;
        let (s1, s2) = (
            param(q1, q2, p1).clamp(0.0, 1.0),
            param(q1, q2, p2).clamp(0.0, 1.0),
        );
        mp.cuts.extend([lo, hi]);
        mq.cuts.extend([s1.min(s2), s1.max(s2)]);
        if hi > lo {
            let len = (hi - lo) * (p2[0] - p1[0]).hypot(p2[1] - p1[1]);
            c.length += len;
            mp.shared.push((lo, hi));
            mq.shared.push((s1.min(s2), s1.max(s2)));
            let same = (p2[0] - p1[0]) * (q2[0] - q1[0]) + (p2[1] - p1[1]) * (q2[1] - q1[1]) > 0.0;
            if same {
                c.overlap = true;
            }
        }
        return;
    }
    let crosses = |a: f64, b: f64| (a > 0.0 && b < 0.0) || (a < 0.0 && b > 0.0);
    if crosses(o1, o2) && crosses(o3, o4) {
        c.touch = true;
        c.overlap = true;
        c.crossing = true;
        return;
    }
    let mut hit = |on_p: Option<f64>, on_q: Option<f64>| {
        c.touch = true;
        if let Some(t) = on_p {
            mp.cuts.push(t);
        }
        if let Some(t) = on_q {
            mq.cuts.push(t);
        }
    };
    if o1 == 0.0 && within_box(q1, p1, p2) {
        hit(Some(param(p1, p2, q1)), Some(0.0));
    }
    if o2 == 0.0 && within_box(q2, p1, p2) {
        hit(Some(param(p1, p2, q2)), Some(1.0));
    }
    if o3 == 0.0 && within_box(p1, q1, q2) {
        hit(Some(0.0), Some(param(q1, q2, p1)));
    }
    if o4 == 0.0 && within_box(p2, q1, q2) {
        hit(Some(1.0), Some(param(q1, q2, p2)));
    }
}

/// Where a point lies relative to a polygonal shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Loc {
    Inside,
    Boundary,
    Outside,
}

/// Location of `p` with respect to `s`, by even-odd crossings (holes are outside).
pub(super) fn locate(p: Pt, s: &Shape) -> Loc {
    let mut inside = false;
    for (a, b) in s.segments() {
        if orient(a, b, p) == 0.0 && within_box(p, a, b) {
            return Loc::Boundary;
        }
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
            if x > p[0] {
                inside = !inside;
            }
        }
    }
    if inside {
        Loc::Inside
    } else {
        Loc::Outside
    }
}

/// Whether a boundary piece of `segs` (between its contact cuts, outside shared intervals)
/// lies at `want` with respect to `other`.
fn piece_at(segs: &[(Pt, Pt)], marks: &mut [Marks], other: &Shape, want: Loc) -> bool {
    for (&(a, b), m) in segs.iter().zip(marks.iter_mut()) {
        m.cuts.extend([0.0, 1.0]);
        m.cuts.retain(|t| (0.0..=1.0).contains(t));
        m.cuts.sort_by(f64::total_cmp);
        m.cuts.dedup();
        for w in m.cuts.windows(2) {
            let mid = (w[0] + w[1]) / 2.0;
            if w[1] - w[0] <= 0.0 || m.shared.iter().any(|&(lo, hi)| mid >= lo && mid <= hi) {
                continue;
            }
            let p = [a[0] + mid * (b[0] - a[0]), a[1] + mid * (b[1] - a[1])];
            if locate(p, other) == want {
                return true;
            }
        }
    }
    false
}

/// Both shapes' boundary segments with their contact marks, and the segment-level contact.
struct Scan {
    contact: Contact,
    sa: Vec<(Pt, Pt)>,
    sb: Vec<(Pt, Pt)>,
    ma: Vec<Marks>,
    mb: Vec<Marks>,
}

fn scan(a: &Shape, b: &Shape) -> Scan {
    let sa: Vec<(Pt, Pt)> = a.segments().collect();
    let sb: Vec<(Pt, Pt)> = b.segments().collect();
    let mut ma: Vec<Marks> = sa.iter().map(|_| Marks::default()).collect();
    let mut mb: Vec<Marks> = sb.iter().map(|_| Marks::default()).collect();
    let mut contact = Contact::default();
    for (i, &p) in sa.iter().enumerate() {
        for (j, &q) in sb.iter().enumerate() {
            if boxes_meet(p, q) {
                let (left, right) = (&mut ma[i], &mut mb[j]);
                segment_contact(p, q, &mut contact, left, right);
            }
        }
    }
    Scan {
        contact,
        sa,
        sb,
        ma,
        mb,
    }
}

/// Classifies how two shapes meet.
pub fn contact(a: &Shape, b: &Shape) -> Contact {
    let mut sc = scan(a, b);
    if !sc.contact.overlap {
        sc.contact.overlap = piece_at(&sc.sa, &mut sc.ma, b, Loc::Inside)
            || piece_at(&sc.sb, &mut sc.mb, a, Loc::Inside);
    }
    sc.contact
}

/// Topological relation of two polygonal shapes `a` and `b`, enough for the shapely binary
/// predicates on polygons.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Relation {
    /// The boundaries share a point.
    pub touch: bool,
    /// The interiors intersect.
    pub overlap: bool,
    /// Every point of `a` lies in `b` (closed sets).
    pub a_in_b: bool,
    /// Every point of `b` lies in `a`.
    pub b_in_a: bool,
}

impl Relation {
    /// The relation seen from the other shape.
    pub fn swapped(self) -> Relation {
        Relation {
            a_in_b: self.b_in_a,
            b_in_a: self.a_in_b,
            ..self
        }
    }
}

/// `a` lies in `b` when no boundary piece of `a` is outside `b` and no boundary piece of `b`
/// (a hole or a concavity) is strictly inside `a`.
fn covered_by(sc: &mut Scan, a: &Shape, b: &Shape) -> bool {
    !sc.contact.crossing
        && !piece_at(&sc.sa, &mut sc.ma, b, Loc::Outside)
        && !piece_at(&sc.sb, &mut sc.mb, a, Loc::Inside)
}

/// Computes the [`Relation`] of two shapes.
pub fn relate(a: &Shape, b: &Shape) -> Relation {
    let mut sc = scan(a, b);
    let touch = sc.contact.touch;
    let overlap = sc.contact.overlap
        || piece_at(&sc.sa, &mut sc.ma, b, Loc::Inside)
        || piece_at(&sc.sb, &mut sc.mb, a, Loc::Inside);
    if !overlap {
        return Relation {
            touch,
            ..Relation::default()
        };
    }
    let a_in_b = covered_by(&mut sc, a, b);
    // Swap the roles for the converse test.
    let mut rev = Scan {
        contact: sc.contact,
        sa: sc.sb,
        sb: sc.sa,
        ma: sc.mb,
        mb: sc.ma,
    };
    let b_in_a = covered_by(&mut rev, b, a);
    Relation {
        touch,
        overlap,
        a_in_b,
        b_in_a,
    }
}

/// Pairs `i < j` of shapes whose bounding boxes meet (inclusive).
pub fn candidate_pairs(shapes: &[Shape]) -> Vec<(u32, u32)> {
    candidate_pairs_within(shapes, 0.0)
}

/// Pairs `i < j` of shapes whose bounding boxes, one grown by `margin` on every side, meet
/// (inclusive): every pair that can be within `margin` of each other.
pub fn candidate_pairs_within(shapes: &[Shape], margin: f64) -> Vec<(u32, u32)> {
    let tree = RTree::bulk_load(
        shapes
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let r = Rectangle::from_corners([s.bbox[0], s.bbox[1]], [s.bbox[2], s.bbox[3]]);
                GeomWithData::new(r, i as u32)
            })
            .collect(),
    );
    let mut pairs = Vec::new();
    for (i, s) in shapes.iter().enumerate() {
        let env = AABB::from_corners(
            [s.bbox[0] - margin, s.bbox[1] - margin],
            [s.bbox[2] + margin, s.bbox[3] + margin],
        );
        for item in tree.locate_in_envelope_intersecting(env) {
            if item.data as usize > i {
                pairs.push((i as u32, item.data));
            }
        }
    }
    pairs.sort_unstable();
    pairs
}

/// Directed pairs of libpysal's strict contiguity, sorted.
pub fn strict_edges(shapes: &[Shape], rook: bool) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for (i, j) in candidate_pairs(shapes) {
        let c = contact(&shapes[i as usize], &shapes[j as usize]);
        let keep = !c.overlap && if rook { c.length > 0.0 } else { c.touch };
        if keep {
            out.extend([(i, j), (j, i)]);
        }
    }
    out.sort_unstable();
    out
}

/// Shared boundary length of every directed pair; an error when two neighbors overlap.
pub fn perimeter_weights(shapes: &[Shape], pairs: &[(u32, u32)]) -> Result<Vec<f64>, String> {
    pairs
        .iter()
        .map(|&(i, j)| {
            let c = contact(&shapes[i as usize], &shapes[j as usize]);
            if c.overlap {
                Err(format!(
                    "geometries {i} and {j} overlap; by_perimeter needs geometries whose \
                     interiors do not intersect"
                ))
            } else {
                Ok(c.length)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x0: f64, y0: f64, x1: f64, y1: f64) -> Shape {
        let r = vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]];
        Shape {
            rings: vec![r],
            bbox: [x0, y0, x1, y1],
        }
    }

    #[test]
    fn t_junction_corner_and_overlap() {
        let long = square(0.0, 0.0, 2.0, 1.0);
        let c = contact(&long, &square(0.0, 1.0, 1.0, 2.0));
        assert!(c.touch && !c.overlap && (c.length - 1.0).abs() < 1e-15);
        let corner = contact(&square(1.0, 1.0, 2.0, 2.0), &square(2.0, 2.0, 3.0, 3.0));
        assert!(corner.touch && !corner.overlap && corner.length == 0.0);
        let over = contact(&square(0.0, 0.0, 2.0, 2.0), &square(1.0, 1.0, 3.0, 3.0));
        assert!(over.overlap);
        let same = contact(&long, &square(0.0, 0.0, 2.0, 1.0));
        assert!(same.overlap);
        let inside = contact(&square(0.0, 0.0, 4.0, 4.0), &square(1.0, 1.0, 2.0, 2.0));
        assert!(inside.overlap && !inside.touch);
        let apart = contact(&long, &square(5.0, 5.0, 6.0, 6.0));
        assert_eq!(apart, Contact::default());
    }

    #[test]
    fn a_polygon_inside_a_hole_touches_along_the_hole() {
        let mut outer = square(0.0, 0.0, 4.0, 4.0);
        let mut hole = vec![[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0], [1.0, 1.0]];
        hole.reverse(); // clockwise: interior of the outer polygon on the left
        outer.rings.push(hole);
        let c = contact(&outer, &square(1.0, 1.0, 3.0, 3.0));
        assert!(c.touch && !c.overlap && (c.length - 8.0).abs() < 1e-12);
    }
}
