use super::alpha::*;
use super::*;

fn area(ring: &[Pt]) -> f64 {
    let n = ring.len();
    (0..n)
        .map(|i| ring[i][0] * ring[(i + 1) % n][1] - ring[(i + 1) % n][0] * ring[i][1])
        .sum::<f64>()
        / 2.0
}

/// A 5-point set in general position: a quadrilateral with one interior point.
const QUAD: [Pt; 5] = [[0.0, 0.0], [4.0, 0.3], [4.2, 4.0], [0.1, 3.9], [2.1, 1.9]];

#[test]
fn circumradius_of_a_right_triangle_and_a_degenerate_one() {
    let r = circumradius([0.0, 0.0], [3.0, 0.0], [0.0, 4.0]);
    assert!((r - 2.5).abs() < 1e-12, "{r}");
    // collinear: libpysal returns half of the longest side
    assert_eq!(circumradius([0.0, 0.0], [1.0, 0.0], [3.0, 0.0]), 1.5);
}

#[test]
fn a_large_radius_threshold_gives_the_convex_hull_clockwise_from_the_smallest_vertex() {
    let polys = alpha_shape(&QUAD, 1e-9).unwrap();
    assert_eq!(polys.len(), 1);
    let p = &polys[0];
    assert!(p.holes.is_empty());
    assert_eq!(p.shell[0], [0.0, 0.0]);
    assert_eq!(p.shell.len(), 4);
    assert!(area(&p.shell) < 0.0, "shell must be clockwise");
    assert!((area(&p.shell).abs() - polygon_area_of_quad()).abs() < 1e-9);
}

fn polygon_area_of_quad() -> f64 {
    // convex hull of QUAD (the interior point is not a vertex)
    area(&[[0.0, 0.0], [4.0, 0.3], [4.2, 4.0], [0.1, 3.9]]).abs()
}

#[test]
fn alpha_zero_is_the_convex_hull_and_a_huge_alpha_is_empty() {
    let hull = alpha_shape(&QUAD, 0.0).unwrap();
    assert_eq!(hull.len(), 1);
    assert_eq!(alpha_shape(&QUAD, 1e9).unwrap(), Vec::<Polygon>::new());
}

#[test]
fn fewer_than_four_points_give_the_hull_whatever_alpha_is() {
    let tri = [[0.0, 0.0], [4.0, 0.5], [1.0, 3.0]];
    for alpha in [0.01, 50.0] {
        let polys = alpha_shape(&tri, alpha).unwrap();
        assert_eq!(polys.len(), 1);
        assert_eq!(polys[0].shell.len(), 3);
        assert!(area(&polys[0].shell) < 0.0);
    }
    let line = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    assert!(alpha_shape(&line, 1.0).unwrap().is_empty());
    assert!(alpha_shape(&line[..2], 1.0).unwrap().is_empty());
    assert!(alpha_shape(&line[..1], 1.0).unwrap().is_empty());
}

#[test]
fn rejects_non_finite_input_negative_alpha_and_degenerate_sets() {
    let mut bad = QUAD.to_vec();
    bad[1][0] = f64::NAN;
    assert!(alpha_shape(&bad, 1.0).unwrap_err().contains("finite"));
    assert!(alpha_shape(&QUAD, -1.0).unwrap_err().contains("alpha"));
    assert!(alpha_shape(&QUAD, f64::NAN).unwrap_err().contains("alpha"));
    let col = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    assert!(alpha_shape(&col, 1.0).unwrap_err().contains("collinear"));
    let dup = [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]];
    assert!(alpha_shape(&dup, 1.0).unwrap_err().contains("distinct"));
}

#[test]
fn duplicate_points_are_ignored() {
    let mut p = QUAD.to_vec();
    p.push(QUAD[1]);
    p.push(QUAD[4]);
    assert_eq!(
        alpha_shape(&p, 0.0).unwrap(),
        alpha_shape(&QUAD, 0.0).unwrap()
    );
}

#[test]
fn wkb_of_a_polygon_closes_the_rings() {
    let p = Polygon {
        shell: vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        holes: vec![],
    };
    let w = polygon_wkb(&p);
    assert_eq!(w[0], 1);
    assert_eq!(u32::from_le_bytes(w[1..5].try_into().unwrap()), 3);
    assert_eq!(u32::from_le_bytes(w[5..9].try_into().unwrap()), 1);
    assert_eq!(u32::from_le_bytes(w[9..13].try_into().unwrap()), 4);
    assert_eq!(w.len(), 13 + 4 * 16);
    assert_eq!(w[13..29], w[13 + 3 * 16..13 + 4 * 16]);
}

/// A `cells x cells` grid of unit squares, each split into two CCW triangles.
fn grid(cells: usize) -> Mesh {
    let w = cells + 1;
    let mut pts = Vec::new();
    for j in 0..w {
        for i in 0..w {
            pts.push([i as f64, j as f64]);
        }
    }
    let mut tris = Vec::new();
    for j in 0..cells {
        for i in 0..cells {
            let (a, b, c, d) = (
                j * w + i,
                j * w + i + 1,
                (j + 1) * w + i + 1,
                (j + 1) * w + i,
            );
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    let radii = vec![1.0; tris.len()];
    Mesh { pts, tris, radii }
}

fn cell_triangles(cells: usize, i: usize, j: usize) -> [usize; 2] {
    [2 * (j * cells + i), 2 * (j * cells + i) + 1]
}

#[test]
fn a_removed_interior_cell_becomes_a_counter_clockwise_hole() {
    let m = grid(3);
    let mut keep = vec![true; m.tris.len()];
    for t in cell_triangles(3, 1, 1) {
        keep[t] = false;
    }
    let polys = polygons(&m, &keep);
    assert_eq!(polys.len(), 1);
    let p = &polys[0];
    assert_eq!(p.shell.len(), 12);
    assert_eq!(p.shell[0], [0.0, 0.0]);
    assert_eq!(area(&p.shell), -9.0);
    assert_eq!(p.holes.len(), 1);
    assert_eq!(
        p.holes[0],
        vec![[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]
    );
    assert_eq!(area(&p.holes[0]), 1.0);
}

#[test]
fn an_island_inside_a_hole_is_a_separate_polygon() {
    // 5 x 5 cells: drop the 3 x 3 block in the middle, keep its centre cell.
    let m = grid(5);
    let mut keep = vec![true; m.tris.len()];
    for j in 1..4 {
        for i in 1..4 {
            if (i, j) != (2, 2) {
                for t in cell_triangles(5, i, j) {
                    keep[t] = false;
                }
            }
        }
    }
    let polys = polygons(&m, &keep);
    assert_eq!(polys.len(), 2);
    let outer = polys.iter().find(|p| p.shell.len() == 20).unwrap();
    assert_eq!(outer.holes.len(), 1);
    assert_eq!(outer.holes[0].len(), 12);
    let island = polys.iter().find(|p| p.shell.len() == 4).unwrap();
    assert!(island.holes.is_empty());
    assert_eq!(area(&island.shell), -1.0);
}

#[test]
fn polygons_touching_at_a_vertex_stay_two_polygons() {
    // cells (0,0) and (1,1) of a 2 x 2 grid share only the centre vertex.
    let m = grid(2);
    let mut keep = vec![false; m.tris.len()];
    for t in cell_triangles(2, 0, 0)
        .into_iter()
        .chain(cell_triangles(2, 1, 1))
    {
        keep[t] = true;
    }
    let polys = polygons(&m, &keep);
    assert_eq!(polys.len(), 2);
    assert!(polys
        .iter()
        .all(|p| p.shell.len() == 4 && p.holes.is_empty()));
}

#[test]
fn a_hole_touching_the_shell_at_a_vertex_keeps_one_polygon() {
    // 3 x 3 cells without the centre cell (1, 1) and the corner cell (0, 0): the hole and the
    // notch meet at the vertex (1, 1), which both rings visit.
    let m = grid(3);
    let mut keep = vec![true; m.tris.len()];
    for t in cell_triangles(3, 1, 1)
        .into_iter()
        .chain(cell_triangles(3, 0, 0))
    {
        keep[t] = false;
    }
    let polys = polygons(&m, &keep);
    assert_eq!(polys.len(), 1);
    assert_eq!(polys[0].shell.len(), 12);
    assert_eq!(
        polys[0].holes,
        vec![vec![[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]]
    );
}

#[test]
fn a_hole_face_bounded_by_touching_polygons_is_kept_like_libpysal() {
    // four cells around the centre cell of a 3 x 3 grid touch only at its corners; no polygon has
    // a hole, so libpysal's polygonize output (the four cells and the centre face) is not filtered
    let m = grid(3);
    let mut keep = vec![false; m.tris.len()];
    for (i, j) in [(1, 0), (0, 1), (2, 1), (1, 2)] {
        for t in cell_triangles(3, i, j) {
            keep[t] = true;
        }
    }
    let polys = polygons(&m, &keep);
    assert_eq!(polys.len(), 5);
    assert!(polys
        .iter()
        .all(|p| p.shell.len() == 4 && p.holes.is_empty()));
}

#[test]
fn auto_returns_a_single_polygon_with_the_radius_of_its_last_triangle() {
    let a = alpha_shape_auto(&QUAD, 1).unwrap();
    assert!(a.polygon.holes.is_empty());
    assert!(a.polygon.shell.len() >= 4);
    assert!(area(&a.polygon.shell) < 0.0);
    // the radius is one of the circumradii of the triangulation
    let m = mesh(&QUAD).unwrap();
    assert!(m.radii.contains(&a.radius));
    // every point is a vertex of the polygon or inside it: here every point is on the boundary
    // or inside the hull, so the polygon is at most the hull
    let hull = alpha_shape(&QUAD, 0.0).unwrap();
    assert!(area(&a.polygon.shell).abs() <= area(&hull[0].shell).abs() + 1e-12);
}

#[test]
fn auto_with_a_large_step_returns_the_first_valid_hull() {
    let pts: Vec<Pt> = (0..40)
        .map(|i| {
            let t = i as f64;
            [
                (t * 0.7).sin() * 3.0 + t * 0.01,
                (t * 1.3).cos() * 3.0 - t * 0.02,
            ]
        })
        .collect();
    let first = alpha_shape_auto(&pts, 1000).unwrap();
    let m = mesh(&pts).unwrap();
    let rmax = m.radii.iter().cloned().fold(0.0, f64::max);
    assert_eq!(first.radius, rmax);
    assert!(area(&first.polygon.shell).abs() > 0.0);
}

#[test]
fn auto_needs_three_non_collinear_points_and_a_positive_step() {
    assert!(alpha_shape_auto(&QUAD[..2], 1)
        .unwrap_err()
        .contains("at least 3"));
    let col = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    assert!(alpha_shape_auto(&col, 1).unwrap_err().contains("collinear"));
    assert!(alpha_shape_auto(&QUAD, 0).unwrap_err().contains("step"));
}

#[test]
fn auto_of_a_triangle_is_the_triangle_with_its_circumradius() {
    let tri = [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]];
    let a = alpha_shape_auto(&tri, 1).unwrap();
    assert_eq!(a.polygon.shell.len(), 3);
    assert!((a.radius - 2.5).abs() < 1e-12);
}

#[test]
fn auto_reports_a_search_with_no_valid_hull() {
    // The largest triangle is dropped by libpysal's float arithmetic (`r < 1 / (1 / r - EPS)`
    // fails for some radii) and it owns a point, so not even the first step is a valid hull.
    let r = (1..2000)
        .map(|k| 0.05 + k as f64 * 1e-4)
        .find(|&r| r >= 1.0 / (1.0 / r - f64::EPSILON))
        .unwrap();
    let mut m = grid(2);
    // triangle 5 is the upper-left half of cell (0, 1): the corner (0, 2) belongs to it only
    m.radii = vec![0.01, 0.01, 0.01, 0.01, 0.01, r, 0.01, 0.01];
    let err = auto_on_mesh(&m, 1).unwrap_err();
    assert!(err.contains("no valid"), "{err}");
}

#[test]
fn bounding_circle_centres_follow_libpysal() {
    let c = bounding_circle_centers(
        &Polygon {
            shell: vec![[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]],
            holes: vec![],
        },
        2f64.sqrt(),
    );
    assert_eq!(c.len(), 4);
    // edge (0,0) -> (0,2): vertical (rotation pi / 2), the centre is on the left, at (-1, 1)
    assert!(
        (c[0][0] + 1.0).abs() < 1e-12 && (c[0][1] - 1.0).abs() < 1e-12,
        "{:?}",
        c[0]
    );
    // edge (0,2) -> (2,2): the centre is above, at (1, 3)
    assert!(
        (c[1][0] - 1.0).abs() < 1e-12 && (c[1][1] - 3.0).abs() < 1e-12,
        "{:?}",
        c[1]
    );
}
