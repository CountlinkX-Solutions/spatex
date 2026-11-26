//! Point pattern analysis (pointpats 2.6.0): centrography, distance statistics, random point
//! processes and quadrat statistics.
//!
//! Points are `[x, y]` pairs. Every simulation draws from `ChaCha8` streams derived from
//! `(seed, replicate)` ([`crate::perm::replicate_rng`]), so results do not depend on the number
//! of rayon threads. Bit-for-bit equality with numpy's generators is not a goal.

pub mod alpha;
pub mod centrography;
pub mod distance;
pub mod hull;
pub mod quadrat;
pub mod random;

#[cfg(test)]
mod alpha_tests;
#[cfg(test)]
mod tests;

use rand_core::Rng;

/// A point.
pub type Pt = [f64; 2];

/// The study window of a simulation: an axis-aligned box or a convex polygon (CCW vertices).
#[derive(Clone, Debug, PartialEq)]
pub enum Window {
    /// `[xmin, ymin, xmax, ymax]`.
    Bbox([f64; 4]),
    /// Counter-clockwise convex polygon.
    Convex(Vec<Pt>),
}

impl Window {
    /// The bounding box `[xmin, ymin, xmax, ymax]`.
    pub fn bbox(&self) -> [f64; 4] {
        match self {
            Window::Bbox(b) => *b,
            Window::Convex(v) => bounds(v),
        }
    }

    /// Area: `|(xmax - xmin)(ymax - ymin)|` for a box (as `pointpats.geometry.area`), the
    /// shoelace area for a polygon.
    pub fn area(&self) -> f64 {
        match self {
            Window::Bbox(b) => ((b[2] - b[0]) * (b[3] - b[1])).abs(),
            Window::Convex(v) => {
                let mut s = 0.0;
                for i in 0..v.len() {
                    let (p, q) = (v[i], v[(i + 1) % v.len()]);
                    s += p[0] * q[1] - q[0] * p[1];
                }
                s.abs() / 2.0
            }
        }
    }

    /// Whether the point lies inside; the boundary counts as inside.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        match self {
            Window::Bbox(b) => b[0] <= x && x <= b[2] && b[1] <= y && y <= b[3],
            Window::Convex(v) => (0..v.len()).all(|i| {
                let (p, q) = (v[i], v[(i + 1) % v.len()]);
                (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0]) >= 0.0
            }),
        }
    }

    /// Fails for a window without positive area (rejection sampling would never finish).
    pub fn validate(&self) -> Result<(), String> {
        let b = self.bbox();
        if !(b.iter().all(|v| v.is_finite()) && self.area() > 0.0 && b[2] > b[0] && b[3] > b[1]) {
            return Err("window has zero area or non-finite bounds".into());
        }
        Ok(())
    }

    /// One uniform point of the window by rejection from its bounding box.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Pt {
        let b = self.bbox();
        loop {
            let x = b[0] + (b[2] - b[0]) * unit(rng);
            let y = b[1] + (b[3] - b[1]) * unit(rng);
            if self.contains(x, y) {
                return [x, y];
            }
        }
    }
}

/// `[xmin, ymin, xmax, ymax]` of a non-empty point set.
pub fn bounds(p: &[Pt]) -> [f64; 4] {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for q in p {
        b[0] = b[0].min(q[0]);
        b[1] = b[1].min(q[1]);
        b[2] = b[2].max(q[0]);
        b[3] = b[3].max(q[1]);
    }
    b
}

/// Uniform `f64` in `[0, 1)` (53 random bits).
pub fn unit<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// Decodes interleaved `x0, y0, x1, y1, ...` into points.
pub fn pairs(flat: &[f64]) -> Result<Vec<Pt>, String> {
    if !flat.len().is_multiple_of(2) {
        return Err("points binary must hold an even number of values".into());
    }
    Ok(flat.as_chunks::<2>().0.to_vec())
}

/// Flattens points to interleaved `x, y`.
pub fn flatten(p: &[Pt]) -> Vec<f64> {
    p.iter().flat_map(|q| [q[0], q[1]]).collect()
}
