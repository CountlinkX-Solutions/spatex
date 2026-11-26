//! Distance metrics of the point-based builders: Euclidean on planar `x, y`, or great-circle
//! (haversine) on `lon, lat` degrees, in kilometres.
//!
//! Haversine neighbor searches run on unit-sphere vectors: the chord between two of them is
//! monotone in the great-circle distance, so the Euclidean R-tree over the vectors finds the
//! same neighbors. Reported distances use the haversine formula (scikit-learn's).

/// Earth radius of the haversine metric, in kilometres (libpysal's intended unit).
pub const EARTH_RADIUS_KM: f64 = 6371.0;

/// Distance metric of the KNN, distance-band and kernel builders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    Euclidean,
    /// Great-circle distance in kilometres between `lon, lat` points in degrees.
    Haversine,
}

impl Metric {
    pub fn parse(name: &str) -> Result<Metric, String> {
        match name {
            "euclidean" => Ok(Metric::Euclidean),
            "haversine" => Ok(Metric::Haversine),
            other => Err(format!(
                "unknown metric {other:?}; expected \"euclidean\" or \"haversine\""
            )),
        }
    }

    /// libpysal's range check for haversine: `-180 < lon < 180` and `-90 < lat < 90`.
    pub fn validate(self, pts: &[[f64; 2]]) -> Result<(), String> {
        if self == Metric::Haversine {
            if let Some(i) = pts
                .iter()
                .position(|p| !(p[0] > -180.0 && p[0] < 180.0 && p[1] > -90.0 && p[1] < 90.0))
            {
                return Err(format!(
                    "point {i} ({}, {}) is outside the haversine range: longitude must be in \
                     (-180, 180) and latitude in (-90, 90) degrees",
                    pts[i][0], pts[i][1]
                ));
            }
        }
        Ok(())
    }

    /// Distance between two points under this metric.
    pub fn distance(self, a: [f64; 2], b: [f64; 2]) -> f64 {
        match self {
            Metric::Euclidean => {
                let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
                (dx * dx + dy * dy).sqrt()
            }
            Metric::Haversine => haversine_km(a, b),
        }
    }
}

/// Great-circle distance in kilometres between `lon, lat` points in degrees.
pub fn haversine_km(a: [f64; 2], b: [f64; 2]) -> f64 {
    let (lon1, lat1, lon2, lat2) = (
        a[0].to_radians(),
        a[1].to_radians(),
        b[0].to_radians(),
        b[1].to_radians(),
    );
    let s_lat = ((lat2 - lat1) / 2.0).sin();
    let s_lon = ((lon2 - lon1) / 2.0).sin();
    let h = s_lat * s_lat + lat1.cos() * lat2.cos() * s_lon * s_lon;
    2.0 * EARTH_RADIUS_KM * h.sqrt().min(1.0).asin()
}

/// Unit-sphere vectors of `lon, lat` points in degrees.
pub fn unit_vectors(pts: &[[f64; 2]]) -> Vec<[f64; 3]> {
    pts.iter()
        .map(|p| {
            let (lon, lat) = (p[0].to_radians(), p[1].to_radians());
            [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
        })
        .collect()
}

/// Chord length on the unit sphere covering a great-circle distance of `km`, slightly enlarged
/// so that a search radius never misses a pair at exactly `km` through rounding.
pub fn search_chord(km: f64) -> f64 {
    let angle = (km / EARTH_RADIUS_KM).min(std::f64::consts::PI);
    (2.0 * (angle / 2.0).sin()) * (1.0 + 1e-9) + 1e-12
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haversine_matches_known_distances() {
        // One degree of longitude on the equator: R * pi / 180.
        let d = haversine_km([0.0, 0.0], [1.0, 0.0]);
        assert!(
            (d - EARTH_RADIUS_KM * std::f64::consts::PI / 180.0).abs() < 1e-9,
            "{d}"
        );
        // Symmetric and zero on the diagonal.
        let (a, b) = ([-3.7, 40.4], [2.35, 48.86]);
        assert_eq!(haversine_km(a, b), haversine_km(b, a));
        assert_eq!(haversine_km(a, a), 0.0);
        // Madrid - Paris is about 1053 km.
        assert!((haversine_km(a, b) - 1053.0).abs() < 5.0);
    }

    #[test]
    fn chord_is_monotone_in_great_circle_distance() {
        let pts = [[0.0, 0.0], [10.0, 5.0], [-20.0, 30.0], [100.0, -45.0]];
        let v = unit_vectors(&pts);
        let chord = |a: [f64; 3], b: [f64; 3]| {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        };
        let mut pairs: Vec<(f64, f64)> = (1..4)
            .map(|j| (haversine_km(pts[0], pts[j]), chord(v[0], v[j])))
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert!(pairs.windows(2).all(|w| w[0].1 < w[1].1));
        assert!(search_chord(haversine_km(pts[0], pts[1])) >= chord(v[0], v[1]));
    }

    #[test]
    fn validate_rejects_out_of_range_coordinates() {
        assert!(Metric::Haversine.validate(&[[179.9, 89.9]]).is_ok());
        for bad in [[180.0, 0.0], [0.0, -90.0], [-200.0, 10.0]] {
            let e = Metric::Haversine.validate(&[[0.0, 0.0], bad]).unwrap_err();
            assert!(
                e.contains("point 1") && e.contains("haversine range"),
                "{e}"
            );
        }
        assert!(Metric::Euclidean.validate(&[[500.0, 500.0]]).is_ok());
        assert!(Metric::parse("manhattan").is_err());
    }
}
