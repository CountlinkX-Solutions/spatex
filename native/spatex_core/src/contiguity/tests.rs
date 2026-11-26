use super::*;

// -- WKB builders (little endian) -------------------------------------------------------

fn header(out: &mut Vec<u8>, kind: u32) {
    out.push(1);
    out.extend_from_slice(&kind.to_le_bytes());
}

fn ring(out: &mut Vec<u8>, pts: &[(f64, f64)]) {
    out.extend_from_slice(&(pts.len() as u32).to_le_bytes());
    for (x, y) in pts {
        out.extend_from_slice(&x.to_le_bytes());
        out.extend_from_slice(&y.to_le_bytes());
    }
}

fn polygon_body(out: &mut Vec<u8>, rings: &[Vec<(f64, f64)>]) {
    out.extend_from_slice(&(rings.len() as u32).to_le_bytes());
    for r in rings {
        ring(out, r);
    }
}

fn polygon(rings: &[Vec<(f64, f64)>]) -> Vec<u8> {
    let mut out = Vec::new();
    header(&mut out, 3);
    polygon_body(&mut out, rings);
    out
}

fn multipolygon(parts: &[Vec<Vec<(f64, f64)>>]) -> Vec<u8> {
    let mut out = Vec::new();
    header(&mut out, 6);
    out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
    for p in parts {
        header(&mut out, 3);
        polygon_body(&mut out, p);
    }
    out
}

fn square(x: f64, y: f64, w: f64, h: f64) -> Vec<Vec<(f64, f64)>> {
    vec![vec![(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]]
}

fn sq(x: f64, y: f64) -> Vec<u8> {
    polygon(&square(x, y, 1.0, 1.0))
}

fn edges(wkbs: &[Vec<u8>], rook: bool) -> Vec<(u32, u32)> {
    let refs: Vec<&[u8]> = wkbs.iter().map(|w| w.as_slice()).collect();
    contiguity_edges(&refs, rook).unwrap()
}

fn err(wkbs: &[Vec<u8>]) -> String {
    let refs: Vec<&[u8]> = wkbs.iter().map(|w| w.as_slice()).collect();
    contiguity_edges(&refs, false).unwrap_err()
}

// -- hashing ----------------------------------------------------------------------------

#[test]
fn negative_zero_hashes_like_positive_zero() {
    assert_eq!(canonical_bits(-0.0), canonical_bits(0.0));
    assert_eq!(vertex_key(-0.0, 1.0), vertex_key(0.0, 1.0));
    assert_eq!(vertex_key(1.0, -0.0), vertex_key(1.0, 0.0));
    assert_ne!(canonical_bits(0.0), canonical_bits(1.0));
    // exact equality only: neighbouring floats are different vertices
    assert_ne!(vertex_key(1.0, 1.0), vertex_key(1.0 + f64::EPSILON, 1.0));
}

#[test]
fn edge_key_ignores_orientation_and_negative_zero() {
    let a = (0.0, 1.0);
    let b = (2.0, -0.0);
    assert_eq!(edge_key(a, b), edge_key(b, a));
    assert_eq!(edge_key((-0.0, 1.0), b), edge_key(a, (2.0, 0.0)));
    assert_ne!(edge_key(a, b), edge_key(a, (3.0, 0.0)));
}

#[test]
fn edge_key_orders_by_x_then_y_numerically() {
    // Negative numbers sort below positives numerically, whatever their bit patterns.
    assert_eq!(
        edge_key((-1.0, 5.0), (1.0, 0.0)),
        edge_key((1.0, 0.0), (-1.0, 5.0))
    );
    assert_eq!(
        edge_key((1.0, 2.0), (1.0, -3.0)),
        edge_key((1.0, -3.0), (1.0, 2.0))
    );
}

// -- contiguity -------------------------------------------------------------------------

#[test]
fn edge_neighbors_are_rook_and_queen() {
    let g = vec![sq(0.0, 0.0), sq(1.0, 0.0)];
    assert_eq!(edges(&g, true), vec![(0, 1), (1, 0)]);
    assert_eq!(edges(&g, false), vec![(0, 1), (1, 0)]);
}

#[test]
fn corner_neighbors_are_queen_only() {
    let g = vec![sq(0.0, 0.0), sq(1.0, 1.0)];
    assert!(edges(&g, true).is_empty());
    assert_eq!(edges(&g, false), vec![(0, 1), (1, 0)]);
}

#[test]
fn shared_vertex_clique_is_symmetric_and_sorted() {
    // Four squares around the vertex (1, 1).
    let g = vec![sq(0.0, 0.0), sq(1.0, 0.0), sq(0.0, 1.0), sq(1.0, 1.0)];
    let q = edges(&g, false);
    assert_eq!(q.len(), 12);
    let mut sorted = q.clone();
    sorted.sort_unstable();
    assert_eq!(q, sorted);
    assert!(q.iter().all(|(a, b)| a != b));
    assert_eq!(edges(&g, true).len(), 8); // the diagonals drop out
}

#[test]
fn t_junction_is_not_rook_contiguous() {
    let g = vec![
        polygon(&square(0.0, 0.0, 2.0, 1.0)),
        sq(0.0, 1.0),
        sq(1.0, 1.0),
    ];
    assert_eq!(edges(&g, true), vec![(1, 2), (2, 1)]);
    assert_eq!(edges(&g, false).len(), 6);
}

#[test]
fn negative_zero_vertices_match_positive_zero() {
    let a = polygon(&[vec![
        (-1.0, -1.0),
        (0.0, -1.0),
        (0.0, 0.0),
        (-1.0, 0.0),
        (-1.0, -1.0),
    ]]);
    let b = polygon(&[vec![
        (-0.0, -1.0),
        (1.0, -1.0),
        (1.0, -0.0),
        (-0.0, -0.0),
        (-0.0, -1.0),
    ]]);
    assert_eq!(edges(&[a.clone(), b.clone()], true), vec![(0, 1), (1, 0)]);
    assert_eq!(edges(&[a, b], false), vec![(0, 1), (1, 0)]);
}

#[test]
fn polygon_filling_a_hole_is_a_neighbor() {
    let holed = polygon(&[
        square(0.0, 0.0, 3.0, 3.0).remove(0),
        square(1.0, 1.0, 1.0, 1.0).remove(0),
    ]);
    let filler = sq(1.0, 1.0);
    let g = vec![holed, filler];
    assert_eq!(edges(&g, true), vec![(0, 1), (1, 0)]);
    assert_eq!(edges(&g, false), vec![(0, 1), (1, 0)]);
}

#[test]
fn multipolygon_parts_touch_the_same_neighbor_without_phantom_edges() {
    let multi = multipolygon(&[square(0.0, 0.0, 1.0, 1.0), square(2.0, 0.0, 1.0, 1.0)]);
    let middle = sq(1.0, 0.0);
    // The phantom edge between the parts' closing and opening vertices is (0,0)-(2,0); a
    // polygon with exactly that edge must not become a rook neighbor.
    let phantom = polygon(&[vec![
        (0.0, 0.0),
        (2.0, 0.0),
        (2.0, -1.0),
        (0.0, -1.0),
        (0.0, 0.0),
    ]]);
    let g = vec![multi, middle, phantom];
    let rook = edges(&g, true);
    assert!(rook.contains(&(0, 1)) && rook.contains(&(1, 0)));
    assert!(!rook.contains(&(0, 2)) && !rook.contains(&(2, 0)));
}

#[test]
fn disjoint_polygons_are_islands() {
    let g = vec![sq(0.0, 0.0), sq(5.0, 5.0)];
    assert!(edges(&g, true).is_empty());
    assert!(edges(&g, false).is_empty());
}

#[test]
fn duplicate_geometries_are_neighbors_of_each_other_once() {
    let g = vec![sq(0.0, 0.0), sq(0.0, 0.0)];
    assert_eq!(edges(&g, true), vec![(0, 1), (1, 0)]);
}

#[test]
fn z_coordinates_are_dropped() {
    // PolygonZ (type 1003): the Z value must not influence matching.
    let mut z = Vec::new();
    z.push(1u8);
    z.extend_from_slice(&1003u32.to_le_bytes());
    z.extend_from_slice(&1u32.to_le_bytes());
    z.extend_from_slice(&5u32.to_le_bytes());
    for (x, y, h) in [
        (1.0f64, 0.0, 9.0),
        (2.0, 0.0, 8.0),
        (2.0, 1.0, 7.0),
        (1.0, 1.0, 6.0),
        (1.0, 0.0, 9.0),
    ] {
        for v in [x, y, h] {
            z.extend_from_slice(&v.to_le_bytes());
        }
    }
    assert_eq!(edges(&[sq(0.0, 0.0), z], true), vec![(0, 1), (1, 0)]);
}

#[test]
fn rejects_non_polygon_geometry_with_its_index() {
    let mut point = Vec::new();
    header(&mut point, 1);
    point.extend_from_slice(&1.0f64.to_le_bytes());
    point.extend_from_slice(&2.0f64.to_le_bytes());
    let message = err(&[sq(0.0, 0.0), point]);
    assert!(message.contains("geometry 1"), "{message}");
    assert!(message.contains("Point"), "{message}");
}

#[test]
fn rejects_malformed_and_truncated_wkb() {
    let mut cut = sq(0.0, 0.0);
    cut.truncate(20);
    assert!(err(&[cut]).contains("geometry 0"));
    assert!(err(&[vec![]]).contains("geometry 0"));
}

#[test]
fn rejects_non_finite_coordinates() {
    let nan = polygon(&[vec![(0.0, 0.0), (f64::NAN, 0.0), (1.0, 1.0), (0.0, 0.0)]]);
    assert!(err(&[nan]).contains("non-finite"));
    let inf = polygon(&[vec![
        (0.0, 0.0),
        (f64::INFINITY, 0.0),
        (1.0, 1.0),
        (0.0, 0.0),
    ]]);
    assert!(err(&[inf]).contains("non-finite"));
}

#[test]
fn empty_input_gives_no_edges() {
    assert!(contiguity_edges(&[], true).unwrap().is_empty());
}

// -- block contiguity -------------------------------------------------------------------

#[test]
fn block_edges_connect_same_regime_members() {
    // a b a c b
    let e = block_edges(&[0, 1, 0, 2, 1]);
    assert_eq!(e, vec![(0, 2), (1, 4), (2, 0), (4, 1)]);
}

#[test]
fn block_edges_of_singletons_are_empty() {
    assert!(block_edges(&[0, 1, 2]).is_empty());
    assert!(block_edges(&[]).is_empty());
}
