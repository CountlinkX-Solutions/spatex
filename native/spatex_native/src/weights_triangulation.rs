//! NIF for triangulation-based graphs.

use rustler::Binary;
use spatex_core::weights::kernel::Kernel;
use spatex_core::weights::knn::Coincident;
use spatex_core::weights::triangulation::{triangulation_graph, Clip, Method, Options};

use crate::{Graph, GraphResource};

#[allow(clippy::too_many_arguments)]
#[rustler::nif(schedule = "DirtyCpu")]
fn weights_triangulation(
    points: Binary,
    method: &str,
    kernel: &str,
    bandwidth: Option<f64>,
    taper: bool,
    decay: bool,
    coincident: &str,
    clip: &str,
) -> Result<Graph, String> {
    let points = spatex_core::bytes_to_f64s(points.as_slice())?;
    let coincident = match coincident {
        "raise" => Coincident::Raise,
        "clique" => Coincident::Clique,
        other => return Err(format!("unknown coincident mode {other:?}")),
    };
    let opts = Options {
        method: Method::parse(method)?,
        kernel: Kernel::parse(kernel)?,
        bandwidth: bandwidth.unwrap_or(f64::INFINITY),
        taper,
        decay,
        coincident,
        clip: Clip::parse(clip)?,
    };
    triangulation_graph(&points, &opts).map(|g| rustler::ResourceArc::new(GraphResource(g)))
}
