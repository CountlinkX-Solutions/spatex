//! NIFs for kernel-weighted graphs.

use rustler::{Binary, ResourceArc};
use spatex_core::weights::kernel::{band_kernel_graph, kernel_graph, Bandwidth, Kernel, Options};
use spatex_core::weights::knn::Coincident;
use spatex_core::weights::metric::Metric;

use crate::{Graph, GraphResource};

fn bandwidth(mode: &str, value: f64) -> Result<Bandwidth, String> {
    match mode {
        "default" => Ok(Bandwidth::Default),
        "fixed" => Ok(Bandwidth::Fixed(value)),
        "adaptive" => Ok(Bandwidth::Adaptive),
        other => Err(format!("unknown bandwidth mode {other:?}")),
    }
}

#[allow(clippy::too_many_arguments)]
#[rustler::nif(schedule = "DirtyCpu")]
fn weights_kernel(
    points: Binary,
    kernel: &str,
    bandwidth_mode: (&str, f64),
    k: Option<usize>,
    taper_decay: (bool, bool),
    coincident: (&str, u64),
    metric: &str,
    max_edges: usize,
) -> Result<Graph, String> {
    let points = spatex_core::bytes_to_f64s(points.as_slice())?;
    let (taper, decay) = taper_decay;
    let coincident = match coincident {
        ("raise", _) => Coincident::Raise,
        ("clique", _) => Coincident::Clique,
        ("jitter", seed) => Coincident::Jitter { seed },
        (other, _) => return Err(format!("unknown coincident mode {other:?}")),
    };
    let opts = Options {
        kernel: Kernel::parse(kernel)?,
        bandwidth: bandwidth(bandwidth_mode.0, bandwidth_mode.1)?,
        k,
        taper,
        decay,
        coincident,
        metric: Metric::parse(metric)?,
        max_edges,
    };
    kernel_graph(&points, &opts).map(|g| ResourceArc::new(GraphResource(g)))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn weights_distance_band_kernel(
    points: Binary,
    threshold: f64,
    kernel: &str,
    bandwidth: Option<f64>,
    taper_decay: (bool, bool),
    metric: &str,
    max_edges: usize,
) -> Result<Graph, String> {
    let points = spatex_core::bytes_to_f64s(points.as_slice())?;
    let (taper, decay) = taper_decay;
    let kernel = Kernel::parse(kernel)?;
    let metric = Metric::parse(metric)?;
    band_kernel_graph(
        &points, threshold, kernel, bandwidth, taper, decay, metric, max_edges,
    )
    .map(|g| ResourceArc::new(GraphResource(g)))
}
