//! Convex hull, minimum rotated rectangle and minimum bounding circle.

use super::Pt;

fn cross(o: Pt, a: Pt, b: Pt) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Convex hull by Andrew's monotone chain: counter-clockwise, strict turns (collinear boundary
/// points dropped), starting at the lowest-x (then lowest-y) vertex.
pub fn convex_hull(p: &[Pt]) -> Result<Vec<Pt>, String> {
    let mut v: Vec<Pt> = p.to_vec();
    v.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    v.dedup();
    if v.len() < 3 {
        return Err("hull needs at least 3 distinct points".into());
    }
    let mut h: Vec<Pt> = Vec::with_capacity(v.len() + 1);
    for &q in &v {
        while h.len() >= 2 && cross(h[h.len() - 2], h[h.len() - 1], q) <= 0.0 {
            h.pop();
        }
        h.push(q);
    }
    let lower = h.len() + 1;
    for &q in v.iter().rev().skip(1) {
        while h.len() >= lower && cross(h[h.len() - 2], h[h.len() - 1], q) <= 0.0 {
            h.pop();
        }
        h.push(q);
    }
    h.pop();
    if h.len() < 3 {
        return Err("points are collinear: the hull is degenerate".into());
    }
    Ok(h)
}

/// A minimum-area rotated rectangle.
#[derive(Clone, Debug)]
pub struct RotatedRect {
    /// Corners, counter-clockwise.
    pub corners: [Pt; 4],
    /// Direction of a side in degrees, in `[0, 90)`.
    pub angle: f64,
    /// Area.
    pub area: f64,
}

/// Minimum-area rectangle by rotating calipers over the hull edges (ties: first edge).
pub fn minimum_rotated_rectangle(p: &[Pt]) -> Result<RotatedRect, String> {
    let h = convex_hull(p)?;
    let mut best: Option<RotatedRect> = None;
    for i in 0..h.len() {
        let (a, b) = (h[i], h[(i + 1) % h.len()]);
        let len = (b[0] - a[0]).hypot(b[1] - a[1]);
        let u = [(b[0] - a[0]) / len, (b[1] - a[1]) / len];
        let v = [-u[1], u[0]];
        let (mut s0, mut s1, mut t0, mut t1) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for q in &h {
            let s = q[0] * u[0] + q[1] * u[1];
            let t = q[0] * v[0] + q[1] * v[1];
            s0 = s0.min(s);
            s1 = s1.max(s);
            t0 = t0.min(t);
            t1 = t1.max(t);
        }
        let area = (s1 - s0) * (t1 - t0);
        if best.as_ref().is_none_or(|r| area < r.area) {
            let at = |s: f64, t: f64| [s * u[0] + t * v[0], s * u[1] + t * v[1]];
            best = Some(RotatedRect {
                corners: [at(s0, t0), at(s1, t0), at(s1, t1), at(s0, t1)],
                angle: (b[1] - a[1])
                    .atan2(b[0] - a[0])
                    .to_degrees()
                    .rem_euclid(90.0),
                area,
            });
        }
    }
    best.ok_or_else(|| "empty hull".into())
}

type Circle = (Pt, f64);

fn inside(c: &Circle, q: Pt) -> bool {
    let d = ((q[0] - c.0[0]).powi(2) + (q[1] - c.0[1]).powi(2)).sqrt();
    d <= c.1 * (1.0 + 1e-14) + 1e-300
}

fn diameter(a: Pt, b: Pt) -> Circle {
    let c = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
    (c, (a[0] - b[0]).hypot(a[1] - b[1]) / 2.0)
}

fn circumcircle(a: Pt, b: Pt, c: Pt) -> Circle {
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    if d == 0.0 {
        // collinear: the widest pair
        return [diameter(a, b), diameter(a, c), diameter(b, c)]
            .into_iter()
            .max_by(|x, y| x.1.total_cmp(&y.1))
            .unwrap_or((a, 0.0));
    }
    let (a2, b2, c2) = (
        a[0] * a[0] + a[1] * a[1],
        b[0] * b[0] + b[1] * b[1],
        c[0] * c[0] + c[1] * c[1],
    );
    let ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d;
    let uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d;
    ([ux, uy], (a[0] - ux).hypot(a[1] - uy))
}

/// Smallest enclosing circle `((cx, cy), r)`. pointpats runs Skyum's algorithm on the convex
/// hull; the minimum circle is unique, so an incremental construction on the hull gives the same
/// circle. Collinear or tiny inputs fall back to the extreme pair.
pub fn minimum_bounding_circle(p: &[Pt]) -> Result<(Pt, f64), String> {
    if p.is_empty() {
        return Err("points must not be empty".into());
    }
    let pts = match convex_hull(p) {
        Ok(h) => h,
        Err(_) => {
            let mut v = p.to_vec();
            v.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
            vec![v[0], v[v.len() - 1]]
        }
    };
    let mut c: Circle = (pts[0], 0.0);
    for i in 1..pts.len() {
        if inside(&c, pts[i]) {
            continue;
        }
        c = (pts[i], 0.0);
        for j in 0..i {
            if inside(&c, pts[j]) {
                continue;
            }
            c = diameter(pts[i], pts[j]);
            for k in 0..j {
                if !inside(&c, pts[k]) {
                    c = circumcircle(pts[i], pts[j], pts[k]);
                }
            }
        }
    }
    Ok(c)
}
