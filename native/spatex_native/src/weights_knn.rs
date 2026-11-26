//! NIFs for the KNN and distance-band graph builders.

use rustler::{Binary, ResourceArc};
use spatex_core::weights::distance_band::distance_band_limited;
use spatex_core::weights::knn::{knn_limited, Coincident};
use spatex_core::weights::metric::Metric;

use crate::{Graph, GraphResource};

#[rustler::nif(schedule = "DirtyCpu")]
fn weights_knn(
    points: Binary,
    k: usize,
    coincident: (&str, u64),
    metric: &str,
    max_edges: usize,
) -> Result<Graph, String> {
    let points = spatex_core::bytes_to_f64s(points.as_slice())?;
    let coincident = match coincident {
        ("raise", _) => Coincident::Raise,
        ("clique", _) => Coincident::Clique,
        ("jitter", seed) => Coincident::Jitter { seed },
        (other, _) => return Err(format!("unknown coincident mode {other:?}")),
    };
    knn_limited(&points, k, coincident, Metric::parse(metric)?, max_edges)
        .map(|g| ResourceArc::new(GraphResource(g)))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn weights_distance_band(
    points: Binary,
    threshold: f64,
    binary: bool,
    alpha: f64,
    metric: &str,
    max_edges: usize,
) -> Result<Graph, String> {
    let points = spatex_core::bytes_to_f64s(points.as_slice())?;
    let metric = Metric::parse(metric)?;
    distance_band_limited(&points, threshold, binary, alpha, metric, max_edges)
        .map(|g| ResourceArc::new(GraphResource(g)))
}
