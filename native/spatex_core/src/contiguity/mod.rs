//! Queen and rook contiguity from WKB polygons and lines, and block contiguity.
//!
//! Follows libpysal's `_vertex_set_intersection` (`libpysal/graph/_contiguity.py`):
//!
//! * every coordinate of every geometry (exterior and interior rings, all parts, closing
//!   vertex included) is a vertex owned by its geometry;
//! * queen: geometries sharing an identical vertex are neighbors;
//! * rook: geometries sharing an identical edge are neighbors, where an edge is a pair of
//!   consecutive vertices inside one part (so there is no phantom edge between the parts of a
//!   multipolygon). As in libpysal, rings of the same part are not separated: the step from
//!   the exterior ring's closing vertex to a hole's first vertex counts as an edge, as do
//!   zero-length edges;
//! * equality is exact `f64` equality, with `-0.0` equal to `+0.0` (Python semantics);
//! * lines (`LineString`, `MultiLineString`) contribute their vertices and edges, each line a
//!   part, as in libpysal;
//! * strict (`touches` / DE-9IM) contiguity and `by_perimeter` weights live in [`strict`].
//!
//! Output is a sorted list of directed pairs with no self pairs; nodes without a pair are
//! islands, which the caller turns into zero-weight self-loops.

use geo_traits::{
    CoordTrait, GeometryTrait, GeometryType, LineStringTrait, MultiLineStringTrait,
    MultiPolygonTrait, PolygonTrait,
};
use wkb::reader::read_wkb;

/// Canonical bit pattern of a coordinate: `-0.0` hashes like `+0.0`.
pub fn canonical_bits(v: f64) -> u64 {
    if v == 0.0 {
        0.0f64.to_bits()
    } else {
        v.to_bits()
    }
}

/// Hash key of a vertex (exact coordinate equality).
pub fn vertex_key(x: f64, y: f64) -> [u64; 2] {
    [canonical_bits(x), canonical_bits(y)]
}

/// Orientation-independent hash key of an edge: the endpoints sorted lexicographically by
/// `(x, y)` numerically, like `tuple(sorted([a, b]))` in libpysal.
pub fn edge_key(a: (f64, f64), b: (f64, f64)) -> [u64; 4] {
    let (lo, hi) = if a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)).is_gt() {
        (b, a)
    } else {
        (a, b)
    };
    let [lx, ly] = vertex_key(lo.0, lo.1);
    let [hx, hy] = vertex_key(hi.0, hi.1);
    [lx, ly, hx, hy]
}

/// Vertices of all geometries, in libpysal order.
struct Vertices {
    coords: Vec<(f64, f64)>,
    /// Geometry index of each vertex.
    geom: Vec<u32>,
    /// Index of the single part (polygon of a multipolygon) of each vertex.
    part: Vec<u32>,
}

fn push_ring<R: LineStringTrait<T = f64>>(
    v: &mut Vertices,
    ring: &R,
    geom: u32,
    part: u32,
) -> Result<(), String> {
    for c in ring.coords() {
        let (x, y) = (c.x(), c.y());
        if !x.is_finite() || !y.is_finite() {
            return Err(format!("geometry {geom}: non-finite coordinate"));
        }
        v.coords.push((x, y));
        v.geom.push(geom);
        v.part.push(part);
    }
    Ok(())
}

fn push_polygon<P: PolygonTrait<T = f64>>(
    v: &mut Vertices,
    polygon: &P,
    geom: u32,
    part: u32,
) -> Result<(), String> {
    if let Some(exterior) = polygon.exterior() {
        push_ring(v, &exterior, geom, part)?;
    }
    for interior in polygon.interiors() {
        push_ring(v, &interior, geom, part)?;
    }
    Ok(())
}

fn collect_vertices(wkbs: &[&[u8]]) -> Result<Vertices, String> {
    let mut v = Vertices {
        coords: Vec::new(),
        geom: Vec::new(),
        part: Vec::new(),
    };
    let mut part = 0u32;
    for (i, bytes) in wkbs.iter().enumerate() {
        let geom = u32::try_from(i).map_err(|_| "too many geometries".to_string())?;
        let parsed = read_wkb(bytes).map_err(|e| format!("geometry {i}: invalid WKB ({e})"))?;
        match parsed.as_type() {
            GeometryType::Polygon(p) => {
                push_polygon(&mut v, p, geom, part)?;
                part += 1;
            }
            GeometryType::MultiPolygon(mp) => {
                for p in mp.polygons() {
                    push_polygon(&mut v, &p, geom, part)?;
                    part += 1;
                }
            }
            GeometryType::LineString(l) => {
                push_ring(&mut v, l, geom, part)?;
                part += 1;
            }
            GeometryType::MultiLineString(ml) => {
                for l in ml.line_strings() {
                    push_ring(&mut v, &l, geom, part)?;
                    part += 1;
                }
            }
            other => {
                let name = match other {
                    GeometryType::Point(_) => "Point",
                    GeometryType::MultiPoint(_) => "MultiPoint",
                    GeometryType::GeometryCollection(_) => "GeometryCollection",
                    _ => "unsupported geometry",
                };
                return Err(format!(
                    "geometry {i}: expected Polygon, MultiPolygon, LineString or MultiLineString, got {name}"
                ));
            }
        }
    }
    Ok(v)
}

/// Turns `(key, geometry)` observations into directed neighbor pairs: all geometries that
/// observed the same key become mutual neighbors.
fn pairs_from_keys<K: Ord + Copy>(mut observed: Vec<(K, u32)>) -> Vec<(u32, u32)> {
    observed.sort_unstable();
    observed.dedup();
    let mut pairs = Vec::new();
    let mut start = 0;
    while start < observed.len() {
        let mut end = start + 1;
        while end < observed.len() && observed[end].0 == observed[start].0 {
            end += 1;
        }
        let group = &observed[start..end];
        for &(_, a) in group {
            for &(_, b) in group {
                if a != b {
                    pairs.push((a, b));
                }
            }
        }
        start = end;
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// Directed neighbor pairs `(focal, neighbor)` of the polygon WKBs, sorted ascending, without
/// self pairs. `rook` selects shared edges instead of shared vertices.
pub fn contiguity_edges(wkbs: &[&[u8]], rook: bool) -> Result<Vec<(u32, u32)>, String> {
    let v = collect_vertices(wkbs)?;
    Ok(if rook {
        let observed = (0..v.coords.len().saturating_sub(1))
            .filter(|&i| v.part[i] == v.part[i + 1])
            .map(|i| (edge_key(v.coords[i], v.coords[i + 1]), v.geom[i]))
            .collect();
        pairs_from_keys(observed)
    } else {
        let observed = v
            .coords
            .iter()
            .zip(&v.geom)
            .map(|(&(x, y), &g)| (vertex_key(x, y), g))
            .collect();
        pairs_from_keys(observed)
    })
}

/// Directed pairs `(i, j)` for every two distinct nodes with the same regime code, sorted.
pub fn block_edges(codes: &[u32]) -> Vec<(u32, u32)> {
    let mut members: std::collections::BTreeMap<u32, Vec<u32>> = std::collections::BTreeMap::new();
    for (i, &c) in codes.iter().enumerate() {
        members.entry(c).or_default().push(i as u32);
    }
    let mut pairs: Vec<(u32, u32)> = members
        .values()
        .flat_map(|m| {
            m.iter()
                .flat_map(move |&a| m.iter().filter(move |&&b| b != a).map(move |&b| (a, b)))
        })
        .collect();
    pairs.sort_unstable();
    pairs
}

pub mod fuzzy;
pub mod strict;
#[cfg(test)]
mod tests;
