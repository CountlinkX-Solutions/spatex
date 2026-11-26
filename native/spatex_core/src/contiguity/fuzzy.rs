//! Fuzzy contiguity (libpysal's `Graph.build_fuzzy_contiguity`).
//!
//! libpysal optionally replaces every geometry by its GEOS buffer and then keeps the pairs
//! `(i, j)` for which `predicate(geom_i, geom_j)` holds. Spatex has no GEOS, so the round
//! buffer is never built: the buffers of two polygons at distance `d` intersect exactly when
//! `d <= 2 * buffer` (the union of discs of radius `buffer` around each), so `intersects` with
//! a buffer is an exact polygon-polygon distance test. GEOS approximates the arcs by chords
//! (16 per quadrant), which shrinks its buffers by up to 0.12 % of the radius, so a pair whose
//! distance lies within about 0.2 % below `2 * buffer` can be a neighbor here and not in
//! libpysal (see `docs/parity.md`).
//!
//! Predicates without a buffer come from [`strict::relate`]; with a buffer only `intersects`
//! and the bounding-box test are defined.

use super::strict::{
    candidate_pairs_within, contact, locate, orient, relate, shapes, Loc, Pt, Relation, Shape,
};

/// A spatial predicate `predicate(focal, neighbor)` of libpysal's `sindex.query`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Predicate {
    Intersects,
    Touches,
    Overlaps,
    Within,
    Contains,
    Covers,
    CoveredBy,
    ContainsProperly,
    Crosses,
    /// `predicate=None`: the bounding boxes intersect.
    BBox,
}

impl Predicate {
    /// Parses libpysal's predicate name (`"bbox"` stands for `predicate=None`).
    pub fn parse(name: &str) -> Result<Predicate, String> {
        Ok(match name {
            "intersects" => Predicate::Intersects,
            "touches" => Predicate::Touches,
            "overlaps" => Predicate::Overlaps,
            "within" => Predicate::Within,
            "contains" => Predicate::Contains,
            "covers" => Predicate::Covers,
            "covered_by" => Predicate::CoveredBy,
            "contains_properly" => Predicate::ContainsProperly,
            "crosses" => Predicate::Crosses,
            "bbox" => Predicate::BBox,
            other => {
                return Err(format!(
                    "unsupported predicate {other:?}; expected one of intersects, touches, \
                     overlaps, within, contains, covers, covered_by, contains_properly, \
                     crosses (dwithin needs a distance and is not ported) or nil"
                ))
            }
        })
    }

    /// Whether the predicate holds for `focal` (`a`) and `neighbor` (`b`) given their relation.
    /// Polygons have no dimension for `crosses` to hold in, so it never does; the bounding-box
    /// predicate is decided by the candidate search and always holds here.
    pub fn holds(self, rel: &Relation) -> bool {
        match self {
            Predicate::Intersects => rel.touch || rel.overlap,
            Predicate::Touches => rel.touch && !rel.overlap,
            Predicate::Overlaps => rel.overlap && !rel.a_in_b && !rel.b_in_a,
            Predicate::Within | Predicate::CoveredBy => rel.a_in_b,
            Predicate::Contains | Predicate::Covers => rel.b_in_a,
            Predicate::ContainsProperly => rel.b_in_a && !rel.touch,
            Predicate::Crosses => false,
            Predicate::BBox => true,
        }
    }
}

fn point_segment(p: Pt, a: Pt, b: Pt) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    let t = if len2 == 0.0 {
        0.0
    } else {
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
    };
    (p[0] - (a[0] + t * dx)).hypot(p[1] - (a[1] + t * dy))
}

/// Distance between two segments: 0 when they cross properly, otherwise attained at an
/// endpoint.
fn segment_distance(p: (Pt, Pt), q: (Pt, Pt)) -> f64 {
    let opposite = |a: f64, b: f64| (a > 0.0 && b < 0.0) || (a < 0.0 && b > 0.0);
    if opposite(orient(p.0, p.1, q.0), orient(p.0, p.1, q.1))
        && opposite(orient(q.0, q.1, p.0), orient(q.0, q.1, p.1))
    {
        return 0.0;
    }
    point_segment(p.0, q.0, q.1)
        .min(point_segment(p.1, q.0, q.1))
        .min(point_segment(q.0, p.0, p.1))
        .min(point_segment(q.1, p.0, p.1))
}

/// Distance between the bounding boxes of two segments.
fn box_gap(p: (Pt, Pt), q: (Pt, Pt)) -> f64 {
    let gap = |axis: usize| {
        let (plo, phi) = (p.0[axis].min(p.1[axis]), p.0[axis].max(p.1[axis]));
        let (qlo, qhi) = (q.0[axis].min(q.1[axis]), q.0[axis].max(q.1[axis]));
        (qlo - phi).max(plo - qhi).max(0.0)
    };
    gap(0).hypot(gap(1))
}

/// Smallest distance between boundary segments of two shapes, stopping as soon as one is at most `stop`.
fn boundary_distance(a: &Shape, b: &Shape, stop: f64) -> f64 {
    let sb: Vec<(Pt, Pt)> = b.segments().collect();
    let mut best = f64::INFINITY;
    for p in a.segments() {
        for &q in &sb {
            if box_gap(p, q) >= best {
                continue;
            }
            best = best.min(segment_distance(p, q));
            if best <= stop {
                return best;
            }
        }
    }
    best
}

/// Some ring of `a` lies strictly inside `b`.
fn ring_inside(a: &Shape, b: &Shape) -> bool {
    a.ring_starts().any(|p| locate(p, b) == Loc::Inside)
}

/// Minimum distance between two polygonal shapes (0 when they intersect).
pub fn polygon_distance(a: &Shape, b: &Shape) -> f64 {
    let c = contact(a, b);
    if c.touch || c.overlap {
        0.0
    } else {
        boundary_distance(a, b, 0.0)
    }
}

/// Whether two polygonal shapes are at most `limit` apart. Equivalent to
/// `polygon_distance(a, b) <= limit`, but stops at the first close pair of segments.
pub fn within_distance(a: &Shape, b: &Shape, limit: f64) -> bool {
    boundary_distance(a, b, limit) <= limit || ring_inside(a, b) || ring_inside(b, a)
}

fn total_bounds(shapes: &[Shape]) -> [f64; 4] {
    shapes.iter().fold(
        [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ],
        |t, s| {
            [
                t[0].min(s.bbox[0]),
                t[1].min(s.bbox[1]),
                t[2].max(s.bbox[2]),
                t[3].max(s.bbox[3]),
            ]
        },
    )
}

/// The buffer distance (`None` for no buffer, a zero buffer included) from the two options.
fn resolve_buffer(
    shapes: &[Shape],
    tolerance: Option<f64>,
    buffer: Option<f64>,
) -> Result<Option<f64>, String> {
    let buffer = match (tolerance, buffer) {
        (Some(_), Some(_)) => {
            return Err("only one of tolerance and buffer can be given, not both".to_string())
        }
        (Some(t), None) => {
            if !t.is_finite() || t < 0.0 {
                return Err(format!(
                    "tolerance must be finite and not negative, got {t}"
                ));
            }
            let b = total_bounds(shapes);
            Some(t * 0.5 * (b[2] - b[0]).min(b[3] - b[1]).abs())
        }
        (None, b) => b,
    };
    match buffer {
        Some(b) if !b.is_finite() => Err(format!("buffer must be finite, got {b}")),
        Some(b) if b < 0.0 => Err(format!("buffer must not be negative, got {b}")),
        Some(b) if b > 0.0 => Ok(Some(b)),
        _ => Ok(None),
    }
}

/// Directed neighbor pairs of libpysal's fuzzy contiguity, sorted, without self pairs.
///
/// `tolerance` and `buffer` are exclusive; a buffer needs `intersects` or the bounding-box
/// predicate.
pub fn fuzzy_edges(
    wkbs: &[&[u8]],
    tolerance: Option<f64>,
    buffer: Option<f64>,
    predicate: Predicate,
) -> Result<Vec<(u32, u32)>, String> {
    let shapes = shapes(wkbs)?;
    let buffer = resolve_buffer(&shapes, tolerance, buffer)?;
    if buffer.is_some() && !matches!(predicate, Predicate::Intersects | Predicate::BBox) {
        return Err(format!(
            "a buffer (or tolerance) works with the predicates intersects and nil only, not \
             {predicate:?}: the buffered geometries are not built"
        ));
    }
    let reach = buffer.map_or(0.0, |b| 2.0 * b);
    let mut out = Vec::new();
    for (i, j) in candidate_pairs_within(&shapes, reach) {
        let (a, b) = (&shapes[i as usize], &shapes[j as usize]);
        let (forward, backward) = match (predicate, buffer) {
            (Predicate::BBox, _) => (true, true),
            (Predicate::Intersects, Some(_)) => {
                let hit = within_distance(a, b, reach);
                (hit, hit)
            }
            (Predicate::Intersects, None) => {
                let c = contact(a, b);
                let hit = c.touch || c.overlap;
                (hit, hit)
            }
            _ => {
                let rel = relate(a, b);
                (predicate.holds(&rel), predicate.holds(&rel.swapped()))
            }
        };
        if forward {
            out.push((i, j));
        }
        if backward {
            out.push((j, i));
        }
    }
    out.sort_unstable();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wkb_polygon(rings: &[Vec<(f64, f64)>]) -> Vec<u8> {
        let mut out = vec![1u8];
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(rings.len() as u32).to_le_bytes());
        for r in rings {
            out.extend_from_slice(&(r.len() as u32).to_le_bytes());
            for (x, y) in r {
                out.extend_from_slice(&x.to_le_bytes());
                out.extend_from_slice(&y.to_le_bytes());
            }
        }
        out
    }

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<u8> {
        wkb_polygon(&[vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]])
    }

    fn two(a: &[u8], b: &[u8]) -> (Shape, Shape) {
        let mut s = shapes(&[a, b]).unwrap();
        let b = s.pop().unwrap();
        (s.pop().unwrap(), b)
    }

    fn hole_polygon() -> Vec<u8> {
        wkb_polygon(&[
            vec![
                (0.0, 0.0),
                (10.0, 0.0),
                (10.0, 10.0),
                (0.0, 10.0),
                (0.0, 0.0),
            ],
            vec![(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0), (3.0, 3.0)],
        ])
    }

    fn holds(p: Predicate, a: &[u8], b: &[u8]) -> bool {
        let (sa, sb) = two(a, b);
        p.holds(&relate(&sa, &sb))
    }

    #[test]
    fn parses_libpysal_predicate_names() {
        assert_eq!(Predicate::parse("intersects"), Ok(Predicate::Intersects));
        assert_eq!(Predicate::parse("covered_by"), Ok(Predicate::CoveredBy));
        assert_eq!(
            Predicate::parse("contains_properly"),
            Ok(Predicate::ContainsProperly)
        );
        assert_eq!(Predicate::parse("bbox"), Ok(Predicate::BBox));
        assert!(Predicate::parse("dwithin").unwrap_err().contains("dwithin"));
        assert!(Predicate::parse("disjoint").is_err());
    }

    #[test]
    fn distance_between_squares() {
        let (a, b) = two(&rect(0.0, 0.0, 1.0, 1.0), &rect(4.0, 0.0, 5.0, 1.0));
        assert_eq!(polygon_distance(&a, &b), 3.0);
        let (a, b) = two(&rect(0.0, 0.0, 1.0, 1.0), &rect(4.0, 5.0, 5.0, 6.0));
        assert_eq!(polygon_distance(&a, &b), 5.0);
        let (a, b) = two(&rect(0.0, 0.0, 2.0, 2.0), &rect(1.0, 1.0, 3.0, 3.0));
        assert_eq!(polygon_distance(&a, &b), 0.0);
        let (a, b) = two(&rect(0.0, 0.0, 1.0, 1.0), &rect(1.0, 1.0, 2.0, 2.0));
        assert_eq!(polygon_distance(&a, &b), 0.0);
        let (a, b) = two(&rect(0.0, 0.0, 9.0, 9.0), &rect(1.0, 1.0, 2.0, 2.0));
        assert_eq!(polygon_distance(&a, &b), 0.0);
    }

    #[test]
    fn distance_across_a_hole_is_to_the_hole_boundary() {
        let (a, b) = two(&hole_polygon(), &rect(4.5, 4.5, 5.5, 5.5));
        assert_eq!(polygon_distance(&a, &b), 1.5);
    }

    #[test]
    fn crossing_boundaries_are_at_distance_zero() {
        // a plus sign: no vertex of one bar is near the other bar
        let (a, b) = two(&rect(0.0, 4.0, 10.0, 6.0), &rect(4.0, 0.0, 6.0, 10.0));
        assert_eq!(polygon_distance(&a, &b), 0.0);
        assert!(within_distance(&a, &b, 0.0));
        let (a, b) = two(&rect(0.0, 4.0, 10.0, 6.0), &rect(4.0, 20.0, 6.0, 30.0));
        assert_eq!(polygon_distance(&a, &b), 14.0);
    }

    #[test]
    fn within_distance_is_inclusive() {
        let (a, b) = two(&rect(0.0, 0.0, 1.0, 1.0), &rect(3.0, 0.0, 4.0, 1.0));
        assert!(within_distance(&a, &b, 2.0));
        assert!(!within_distance(&a, &b, 1.999));
        assert!(within_distance(&a, &b, 5.0));
    }

    #[test]
    fn binary_predicates_on_polygons() {
        let big = rect(0.0, 0.0, 4.0, 4.0);
        let inner = rect(1.0, 1.0, 2.0, 2.0);
        let flush = rect(0.0, 0.0, 2.0, 2.0);
        let side = rect(4.0, 0.0, 6.0, 4.0);
        let lap = rect(3.0, 3.0, 7.0, 7.0);
        let far = rect(10.0, 10.0, 11.0, 11.0);

        assert!(holds(Predicate::Contains, &big, &inner));
        assert!(holds(Predicate::Within, &inner, &big));
        assert!(!holds(Predicate::Within, &big, &inner));
        assert!(holds(Predicate::ContainsProperly, &big, &inner));
        assert!(holds(Predicate::Covers, &big, &flush));
        assert!(holds(Predicate::CoveredBy, &flush, &big));
        assert!(!holds(Predicate::ContainsProperly, &big, &flush));
        assert!(holds(Predicate::Within, &big, &big));
        assert!(holds(Predicate::Touches, &big, &side));
        assert!(!holds(Predicate::Touches, &big, &lap));
        assert!(!holds(Predicate::Overlaps, &big, &side));
        assert!(holds(Predicate::Overlaps, &big, &lap));
        assert!(!holds(Predicate::Overlaps, &big, &inner));
        assert!(!holds(Predicate::Overlaps, &big, &big));
        assert!(holds(Predicate::Intersects, &big, &side));
        assert!(!holds(Predicate::Intersects, &big, &far));
        assert!(!holds(Predicate::Crosses, &big, &lap));
    }

    #[test]
    fn holes_are_not_covered() {
        let ring = hole_polygon();
        let in_hole = rect(4.0, 4.0, 6.0, 6.0);
        let fills = rect(3.0, 3.0, 7.0, 7.0);
        let over_hole = rect(2.0, 2.0, 8.0, 8.0);
        assert!(!holds(Predicate::Intersects, &ring, &in_hole));
        assert!(holds(Predicate::Touches, &ring, &fills));
        assert!(!holds(Predicate::Contains, &ring, &fills));
        assert!(holds(Predicate::Overlaps, &ring, &over_hole));
        assert!(!holds(Predicate::Contains, &ring, &over_hole));
        assert!(!holds(Predicate::Within, &over_hole, &ring));
    }

    #[test]
    fn fuzzy_edges_without_buffer_are_intersections() {
        let wkbs = [
            rect(0.0, 0.0, 1.0, 1.0),
            rect(1.0, 0.0, 2.0, 1.0),
            rect(2.5, 0.0, 3.5, 1.0),
            rect(9.0, 9.0, 10.0, 10.0),
        ];
        let refs: Vec<&[u8]> = wkbs.iter().map(|w| w.as_slice()).collect();
        let edges = fuzzy_edges(&refs, None, None, Predicate::Intersects).unwrap();
        assert_eq!(edges, vec![(0, 1), (1, 0)]);
        let edges = fuzzy_edges(&refs, None, Some(0.25), Predicate::Intersects).unwrap();
        assert_eq!(edges, vec![(0, 1), (1, 0), (1, 2), (2, 1)]);
        let edges = fuzzy_edges(&refs, None, Some(0.2), Predicate::Intersects).unwrap();
        assert_eq!(edges, vec![(0, 1), (1, 0)]);
    }

    #[test]
    fn tolerance_is_a_share_of_half_the_shorter_side_of_the_total_bounds() {
        let wkbs = [rect(0.0, 0.0, 1.0, 1.0), rect(3.0, 0.0, 100.0, 4.0)];
        let refs: Vec<&[u8]> = wkbs.iter().map(|w| w.as_slice()).collect();
        // total bounds 100 x 4: buffer = tolerance * 0.5 * 4; gap 2 needs buffer >= 1
        assert!(fuzzy_edges(&refs, Some(0.49), None, Predicate::Intersects)
            .unwrap()
            .is_empty());
        assert_eq!(
            fuzzy_edges(&refs, Some(0.5), None, Predicate::Intersects).unwrap(),
            vec![(0, 1), (1, 0)]
        );
    }

    #[test]
    fn directed_predicates_give_directed_pairs() {
        let wkbs = [rect(0.0, 0.0, 4.0, 4.0), rect(1.0, 1.0, 2.0, 2.0)];
        let refs: Vec<&[u8]> = wkbs.iter().map(|w| w.as_slice()).collect();
        assert_eq!(
            fuzzy_edges(&refs, None, None, Predicate::Contains).unwrap(),
            vec![(0, 1)]
        );
        assert_eq!(
            fuzzy_edges(&refs, None, None, Predicate::Within).unwrap(),
            vec![(1, 0)]
        );
    }

    #[test]
    fn bounding_boxes_grow_with_the_buffer() {
        let wkbs = [rect(0.0, 0.0, 1.0, 1.0), rect(2.0, 2.0, 3.0, 3.0)];
        let refs: Vec<&[u8]> = wkbs.iter().map(|w| w.as_slice()).collect();
        assert!(fuzzy_edges(&refs, None, None, Predicate::BBox)
            .unwrap()
            .is_empty());
        assert_eq!(
            fuzzy_edges(&refs, None, Some(0.5), Predicate::BBox).unwrap(),
            vec![(0, 1), (1, 0)]
        );
    }

    #[test]
    fn rejects_unsupported_combinations() {
        let wkbs = [rect(0.0, 0.0, 1.0, 1.0), rect(2.0, 0.0, 3.0, 1.0)];
        let refs: Vec<&[u8]> = wkbs.iter().map(|w| w.as_slice()).collect();
        let both = fuzzy_edges(&refs, Some(0.1), Some(1.0), Predicate::Intersects);
        assert!(both.unwrap_err().contains("tolerance"));
        let negative = fuzzy_edges(&refs, None, Some(-1.0), Predicate::Intersects);
        assert!(negative.unwrap_err().contains("negative"));
        let touches = fuzzy_edges(&refs, None, Some(1.0), Predicate::Touches);
        assert!(touches.unwrap_err().contains("buffer"));
        // a zero buffer is no buffer at all
        assert!(fuzzy_edges(&refs, None, Some(0.0), Predicate::Touches).is_ok());
    }
}
