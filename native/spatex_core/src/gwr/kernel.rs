//! Distances and kernel weights of one focal location (`mgwr.kernels.Kernel`).

use super::Model;

/// Earth radius (km) used by the haversine distance, as in mgwr.
const EARTH_RADIUS_KM: f64 = 6371.0;
/// mgwr widens an adaptive bandwidth by this factor so the m-th neighbor is inside the band.
const ADAPTIVE_EPS: f64 = 1.000_000_1;

/// Supported kernel functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// `(1 - z^2)^2`, zero beyond the bandwidth.
    Bisquare,
    /// `exp(-z^2 / 2)`, never truncated.
    Gaussian,
    /// `exp(-z)`, never truncated.
    Exponential,
}

impl Kernel {
    /// Parses `"bisquare"`, `"gaussian"` or `"exponential"`.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "bisquare" => Ok(Self::Bisquare),
            "gaussian" => Ok(Self::Gaussian),
            "exponential" => Ok(Self::Exponential),
            other => Err(format!(
                "unsupported kernel {other:?}; expected bisquare, gaussian or exponential"
            )),
        }
    }
}

/// Distances from location `i` to every location (self distance is exactly 0).
pub fn distances(m: &Model, i: usize, out: &mut [f64]) {
    distances_from(m, m.coords[2 * i], m.coords[2 * i + 1], out);
}

/// Distances from an arbitrary point `(xi, yi)` to every calibration location.
pub fn distances_from(m: &Model, xi: f64, yi: f64, out: &mut [f64]) {
    if m.spherical {
        // Port of the literal numpy expression order (coords are lon, lat in degrees).
        let lat2 = yi.to_radians();
        for (j, o) in out.iter_mut().enumerate() {
            let (xj, yj) = (m.coords[2 * j], m.coords[2 * j + 1]);
            let d_lat = (yj - yi).to_radians();
            let d_lon = (xj - xi).to_radians();
            let lat1 = yj.to_radians();
            let s_lat = (d_lat / 2.0).sin();
            let s_lon = (d_lon / 2.0).sin();
            let a = s_lat * s_lat + lat1.cos() * lat2.cos() * (s_lon * s_lon);
            *o = EARTH_RADIUS_KM * (2.0 * a.sqrt().asin());
        }
    } else {
        for (j, o) in out.iter_mut().enumerate() {
            let dx = xi - m.coords[2 * j];
            let dy = yi - m.coords[2 * j + 1];
            *o = (dx * dx + dy * dy).sqrt();
        }
    }
}

/// Reusable buffers for [`weights`], one set per worker thread.
pub struct Workspace {
    pub(crate) dist: Vec<f64>,
    pub(crate) scratch: Vec<f64>,
    pub(crate) w: Vec<f64>,
}

impl Workspace {
    pub fn new(n: usize) -> Self {
        Self {
            dist: vec![0.0; n],
            scratch: vec![0.0; n],
            w: vec![0.0; n],
        }
    }
}

/// Validates a bandwidth for the model: finite and positive when fixed, an integer count in
/// `2..=n` (after truncation toward zero, as mgwr's `int(bw)`) when adaptive. `+inf` always
/// means the global model.
pub fn check_bandwidth(m: &Model, bw: f64) -> Result<(), String> {
    if bw.is_nan() {
        return Err("bandwidth is NaN".to_string());
    }
    if bw == f64::INFINITY {
        return Ok(());
    }
    if m.fixed {
        if bw > 0.0 && bw.is_finite() {
            return Ok(());
        }
        return Err(format!("fixed bandwidth must be positive, got {bw}"));
    }
    let count = bw.trunc();
    if count < 2.0 || count > m.n as f64 {
        return Err(format!(
            "adaptive bandwidth must be between 2 and n = {} neighbors, got {bw}",
            m.n
        ));
    }
    Ok(())
}

/// Kernel weights of location `i` for bandwidth `bw` into `ws.w`.
///
/// Fixed bandwidths are distances; adaptive ones are neighbor counts (including the focal
/// point), widened by `1.0000001` exactly like mgwr. `bw = +inf` gives all-ones weights.
pub fn weights(m: &Model, i: usize, bw: f64, ws: &mut Workspace) -> Result<(), String> {
    weights_from(m, m.coords[2 * i], m.coords[2 * i + 1], bw, ws)
        .map_err(|e| format!("{e} at location {i}"))
}

/// Kernel weights of the point `(xi, yi)` (not necessarily a calibration location) into
/// `ws.w`, as `mgwr.kernels.Kernel` does when `points` is given. The adaptive bandwidth is the
/// distance to the `bw`-th nearest calibration location.
pub fn weights_from(
    m: &Model,
    xi: f64,
    yi: f64,
    bw: f64,
    ws: &mut Workspace,
) -> Result<(), String> {
    if bw == f64::INFINITY {
        ws.w.fill(1.0);
        return Ok(());
    }
    distances_from(m, xi, yi, &mut ws.dist);
    let h = if m.fixed {
        bw
    } else {
        let count = bw.trunc() as usize;
        ws.scratch.copy_from_slice(&ws.dist);
        let (_, kth, _) = ws.scratch.select_nth_unstable_by(count - 1, f64::total_cmp);
        *kth * ADAPTIVE_EPS
    };
    if h <= 0.0 {
        return Err(
            "bandwidth is zero (adaptive bandwidth not larger than the number of \
             coincident points)"
                .to_string(),
        );
    }
    for (w, &d) in ws.w.iter_mut().zip(&ws.dist) {
        let z = d / h;
        *w = match m.kernel {
            Kernel::Bisquare => {
                if d >= h {
                    0.0
                } else {
                    let t = 1.0 - z * z;
                    t * t
                }
            }
            Kernel::Gaussian => (-0.5 * (z * z)).exp(),
            Kernel::Exponential => (-z).exp(),
        };
    }
    Ok(())
}
