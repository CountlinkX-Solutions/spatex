//! Quadrat statistics (`pointpats/quadrat_statistics.py`): rectangle and hexagon grids and the
//! Pearson chi-square test of complete spatial randomness.

use super::random::{parse_size_and_intensity, poisson_one};
use super::{bounds, Pt, Window};
use crate::esda::losh::chi2_sf;
use crate::limits;
use crate::perm::replicate_rng;
use rayon::prelude::*;

/// Grid layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    /// Rectangles: a `nx x ny` grid, or cells of `width x height` when both are non-zero.
    Rectangle {
        /// Columns.
        nx: usize,
        /// Rows.
        ny: usize,
        /// Cell width (0 = derive from `nx`).
        width: f64,
        /// Cell height (0 = derive from `ny`).
        height: f64,
    },
    /// Flat-topped hexagons of side `lh`.
    Hexagon {
        /// Side length.
        lh: f64,
    },
}

/// Cell counts of one grid.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    /// Cell ids in pointpats' dictionary order.
    pub cell_ids: Vec<usize>,
    /// Point count per cell, aligned with `cell_ids`.
    pub counts: Vec<usize>,
    /// Columns.
    pub count_column: usize,
    /// Rows (rectangles; 0 for hexagons).
    pub count_row: usize,
    /// Rows of even columns (hexagons; 0 for rectangles).
    pub count_row_even: usize,
    /// Rows of odd columns (hexagons; 0 for rectangles).
    pub count_row_odd: usize,
    /// Cell width (rectangles).
    pub rectangle_width: f64,
    /// Cell height (rectangles).
    pub rectangle_height: f64,
    /// Minimum bounding box `[xmin, ymin, xmax, ymax]`.
    pub mbb: [f64; 4],
}

/// Python's float floor division `a // b` (fmod based, so ties match CPython).
pub(crate) fn py_floordiv(a: f64, b: f64) -> f64 {
    let mut modulo = a % b;
    let mut div = (a - modulo) / b;
    if modulo != 0.0 && ((b < 0.0) != (modulo < 0.0)) {
        modulo += b;
        div -= 1.0;
    }
    let _ = modulo;
    if div != 0.0 {
        let mut fl = div.floor();
        if div - fl > 0.5 {
            fl += 1.0;
        }
        fl
    } else {
        (0.0_f64).copysign(a / b)
    }
}

/// Counts the points per cell. A grid with more than `max_cells` cells is an error raised before
/// anything is allocated.
pub fn grid_counts(p: &[Pt], shape: Shape, max_cells: usize) -> Result<Grid, String> {
    if p.is_empty() {
        return Err("points must not be empty".into());
    }
    let mbb = bounds(p);
    match shape {
        Shape::Rectangle {
            nx,
            ny,
            width,
            height,
        } => rectangle(p, mbb, nx, ny, width, height, max_cells),
        Shape::Hexagon { lh } => hexagon(p, mbb, lh, max_cells),
    }
}

fn rectangle(
    p: &[Pt],
    mbb: [f64; 4],
    nx: usize,
    ny: usize,
    width: f64,
    height: f64,
    max_cells: usize,
) -> Result<Grid, String> {
    let (xr, yr) = (mbb[2] - mbb[0], mbb[3] - mbb[1]);
    if xr <= 0.0 || yr <= 0.0 {
        return Err("points need a positive extent in both x and y".into());
    }
    let (cols, rows, w, h) = if width != 0.0 && height != 0.0 {
        let (cols, rows) = ((xr / width).ceil(), (yr / height).ceil());
        check_float_cells(cols * rows, max_cells)?;
        (cols as usize, rows as usize, width, height)
    } else {
        if nx == 0 || ny == 0 {
            return Err("nx and ny must be positive".into());
        }
        (nx, ny, xr / nx as f64, yr / ny as f64)
    };
    check_cells(
        limits::checked_product("quadrat grid", cols, rows)?,
        max_cells,
    )?;
    let mut counts = vec![0usize; cols * rows];
    for q in p {
        let mut ix = py_floordiv(q[0] - mbb[0], w);
        let mut iy = py_floordiv(q[1] - mbb[1], h);
        if ix == cols as f64 {
            ix -= 1.0;
        }
        if iy == rows as f64 {
            iy -= 1.0;
        }
        if !(0.0..cols as f64).contains(&ix) || !(0.0..rows as f64).contains(&iy) {
            return Err("point fell outside the quadrat grid".into());
        }
        counts[iy as usize * cols + ix as usize] += 1;
    }
    Ok(Grid {
        cell_ids: (0..cols * rows).collect(),
        counts,
        count_column: cols,
        count_row: rows,
        count_row_even: 0,
        count_row_odd: 0,
        rectangle_width: w,
        rectangle_height: h,
        mbb,
    })
}

/// Errors naming the cell count when it exceeds `max_cells`.
fn check_cells(cells: usize, max_cells: usize) -> Result<(), String> {
    if cells > max_cells {
        Err(format!(
            "quadrat grid would have {cells} cells, above max_cells: {max_cells}; \
             use fewer or larger cells, or raise max_cells:"
        ))
    } else {
        Ok(())
    }
}

/// [`check_cells`] for a count still in floating point, so a huge or infinite count is caught
/// before it saturates a cast.
fn check_float_cells(cells: f64, max_cells: usize) -> Result<(), String> {
    if cells.is_finite() && cells <= max_cells as f64 {
        Ok(())
    } else {
        Err(format!(
            "quadrat grid would have about {cells:e} cells, above max_cells: {max_cells}; \
             use fewer or larger cells, or raise max_cells:"
        ))
    }
}

fn hexagon(p: &[Pt], mbb: [f64; 4], lh: f64, max_cells: usize) -> Result<Grid, String> {
    if !(lh.is_finite() && lh > 0.0) {
        return Err("lh must be positive".into());
    }
    let (xmin, ymin) = (mbb[0], mbb[1]);
    let (rx, ry) = (mbb[2] - mbb[0], mbb[3] - mbb[1]);
    let sh = lh * (std::f64::consts::PI / 6.0).cos();
    let semi = lh / 2.0;
    let cols = 1.0
        + if semi < rx {
            ((rx - semi) / (1.5 * lh)).ceil()
        } else {
            0.0
        };
    let even = 1.0
        + if sh < ry {
            ((ry - sh) / (2.0 * sh)).ceil()
        } else {
            0.0
        };
    check_float_cells(cols * even, max_cells)?;
    let (cols, even) = (cols as usize, even as usize);
    let odd = (ry / (2.0 * sh)).ceil() as usize;

    let mut cell_ids = Vec::new();
    let mut slot = vec![usize::MAX; cols * even];
    for i in 0..even {
        for j in 0..cols {
            if even != odd && i == even - 1 && j % 2 == 1 {
                continue;
            }
            slot[j + i * cols] = cell_ids.len();
            cell_ids.push(j + i * cols);
        }
    }
    let mut counts = vec![0usize; cell_ids.len()];

    for q in p {
        let idx = py_floordiv(q[0] - xmin, semi) as i64;
        let yev = ((q[1] + sh - ymin) / (2.0 * sh)) as i64;
        let yod = ((q[1] - ymin) / (2.0 * sh)) as i64;
        let (cxi, cyi);
        if idx.rem_euclid(3) != 1 {
            cxi = (idx + 1).div_euclid(3);
            cyi = if cxi % 2 == 0 { yev } else { yod };
        } else {
            let mut cx_i = idx.div_euclid(3);
            let center_x = cx_i as f64 * semi * 3.0 + xmin;
            let mut cy_i = yod;
            let mut center_y = (cy_i as f64 * 2.0 + 1.0) * sh + ymin;
            if cx_i % 2 == 0 {
                cy_i = yev;
                center_y = cy_i as f64 * 2.0 * sh + ymin;
            }
            let (x0, y0, x1, y1, sign);
            if q[1] > center_y {
                (x0, y0) = (center_x + lh, center_y);
                (x1, y1) = (center_x + semi, center_y + sh);
                sign = -1.0;
            } else {
                (x0, y0) = (center_x + semi, center_y - sh);
                (x1, y1) = (center_x + lh, center_y);
                sign = 1.0;
            }
            let line = (y0 - y1) / (x0 - x1) * q[0] + (x0 * y1 - x1 * y0) / (x0 - x1);
            let indicator = sign * (q[1] - line);
            if indicator <= 0.0 {
                cx_i += 1;
                cy_i = if cx_i % 2 == 0 { yev } else { yod };
            }
            cxi = cx_i;
            cyi = cy_i;
        }
        let key = cxi + cyi * cols as i64;
        let s = usize::try_from(key)
            .ok()
            .and_then(|k| slot.get(k).copied())
            .filter(|s| *s != usize::MAX)
            .ok_or_else(|| "point fell outside the hexagon grid".to_string())?;
        counts[s] += 1;
    }
    Ok(Grid {
        cell_ids,
        counts,
        count_column: cols,
        count_row: 0,
        count_row_even: even,
        count_row_odd: odd,
        rectangle_width: 0.0,
        rectangle_height: 0.0,
        mbb,
    })
}

/// `scipy.stats.chisquare(counts)` against the uniform expectation: `(chi2, p, df, contrib)`.
pub fn chisquare(counts: &[usize]) -> (f64, f64, usize, Vec<f64>) {
    let k = counts.len();
    let mean = counts.iter().sum::<usize>() as f64 / k as f64;
    let contrib: Vec<f64> = counts
        .iter()
        .map(|&c| {
            let d = c as f64 - mean;
            d * d / mean
        })
        .collect();
    let stat: f64 = contrib.iter().sum();
    let df = k - 1;
    (stat, chi2_sf(stat, df as f64), df, contrib)
}

/// Quadrat test result.
#[derive(Clone, Debug)]
pub struct QuadratTest {
    /// The observed grid.
    pub grid: Grid,
    /// Pearson statistic.
    pub chi2: f64,
    /// Analytic p-value.
    pub chi2_pvalue: f64,
    /// Degrees of freedom (`cells - 1`).
    pub df: usize,
    /// Per-cell contributions.
    pub chi2_contrib: Vec<f64>,
    /// Simulated statistics (empty without realizations).
    pub chi2_realizations: Vec<f64>,
    /// `(#{sim >= obs} + 1) / (realizations + 1)` when simulated.
    pub chi2_r_pvalue: Option<f64>,
}

/// Quadrat test with optional CSR realizations in the observed bounding box. Each realization
/// is gridded on its own bounding box with the same `nx`/`ny` (or `lh`); a rectangle's
/// `width`/`height` are not forwarded, as in pointpats.
pub fn quadrat_test(
    p: &[Pt],
    shape: Shape,
    realizations: usize,
    seed: u64,
    max_cells: usize,
) -> Result<QuadratTest, String> {
    let grid = grid_counts(p, shape, max_cells)?;
    let (chi2, chi2_pvalue, df, chi2_contrib) = chisquare(&grid.counts);
    let mut chi2_realizations = Vec::new();
    let mut chi2_r_pvalue = None;
    if realizations > 0 {
        let window = Window::Bbox(grid.mbb);
        window.validate()?;
        let area = window.area();
        let intensity = p.len() as f64 / area;
        let (n_obs, _, _) =
            parse_size_and_intensity(area, Some(intensity), None, Some(realizations))?;
        let sim_shape = match shape {
            Shape::Rectangle { nx, ny, .. } => Shape::Rectangle {
                nx,
                ny,
                width: 0.0,
                height: 0.0,
            },
            h => h,
        };
        chi2_realizations = (0..realizations)
            .into_par_iter()
            .map(|r| {
                let pts = poisson_one(&window, n_obs, &mut replicate_rng(seed, r as u64));
                grid_counts(&pts, sim_shape, max_cells).map(|g| chisquare(&g.counts).0)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ge = chi2_realizations.iter().filter(|s| **s >= chi2).count();
        chi2_r_pvalue = Some((ge as f64 + 1.0) / (realizations as f64 + 1.0));
    }
    Ok(QuadratTest {
        grid,
        chi2,
        chi2_pvalue,
        df,
        chi2_contrib,
        chi2_realizations,
        chi2_r_pvalue,
    })
}
