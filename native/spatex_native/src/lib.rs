//! Thin Rustler layer over `spatex_core`. NIFs return `Result<_, String>`; heavy ones run on
//! the dirty CPU scheduler.

mod classify;
mod contiguity;
mod dynamics;
mod esda_global;
mod esda_local;
mod esda_local_ext;
mod gwr;
mod inequality;
mod pointpats;
mod region;
mod regression;

use rustler::{Binary, Env, OwnedBinary, ResourceArc};
use spatex_core::graph::{Csr, Transform};

mod weights_kernel;
mod weights_knn;
mod weights_triangulation;

#[rustler::nif]
fn nif_version() -> u32 {
    spatex_core::abi_version()
}

#[rustler::nif(schedule = "DirtyCpu")]
fn sum_f64(binary: Binary) -> Result<f64, String> {
    let values = spatex_core::bytes_to_f64s(binary.as_slice())?;
    Ok(spatex_core::sum_f64(&values))
}

/// Immutable CSR graph held in Rust memory.
pub struct GraphResource(pub(crate) Csr);

#[rustler::resource_impl]
impl rustler::Resource for GraphResource {}

pub(crate) type Graph = ResourceArc<GraphResource>;

fn u32_binary<'a>(env: Env<'a>, values: &[u32]) -> Binary<'a> {
    let mut owned = OwnedBinary::new(values.len() * 4);
    let (slots, _) = owned.as_mut_slice().as_chunks_mut::<4>();
    for (slot, value) in slots.iter_mut().zip(values) {
        *slot = value.to_ne_bytes();
    }
    Binary::from_owned(owned, env)
}

pub(crate) fn f64_binary<'a>(env: Env<'a>, values: &[f64]) -> Binary<'a> {
    let mut owned = OwnedBinary::new(values.len() * 8);
    let (slots, _) = owned.as_mut_slice().as_chunks_mut::<8>();
    for (slot, value) in slots.iter_mut().zip(values) {
        *slot = value.to_ne_bytes();
    }
    Binary::from_owned(owned, env)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn graph_from_arrays(
    n: usize,
    focal: Binary,
    neighbor: Binary,
    weight: Binary,
) -> Result<Graph, String> {
    let focal = spatex_core::bytes_to_u32s(focal.as_slice())?;
    let neighbor = spatex_core::bytes_to_u32s(neighbor.as_slice())?;
    let weight = spatex_core::bytes_to_f64s(weight.as_slice())?;
    Csr::from_arrays(n, &focal, &neighbor, &weight).map(|g| ResourceArc::new(GraphResource(g)))
}

#[rustler::nif]
fn graph_nnz(graph: Graph) -> usize {
    graph.0.nonzero_count()
}

#[rustler::nif(schedule = "DirtyCpu")]
fn graph_to_arrays<'a>(env: Env<'a>, graph: Graph) -> (Binary<'a>, Binary<'a>, Binary<'a>) {
    let (focal, neighbor, weight) = graph.0.to_arrays();
    (
        u32_binary(env, &focal),
        u32_binary(env, &neighbor),
        f64_binary(env, &weight),
    )
}

/// CSR parts `(indptr, indices, data)`: `indptr` as native-endian i64, `indices` as u32,
/// `data` as f64.
#[rustler::nif(schedule = "DirtyCpu")]
fn graph_to_csr<'a>(env: Env<'a>, graph: Graph) -> (Binary<'a>, Binary<'a>, Binary<'a>) {
    let indptr = graph.0.indptr();
    let mut owned = OwnedBinary::new(indptr.len() * 8);
    let (slots, _) = owned.as_mut_slice().as_chunks_mut::<8>();
    for (slot, value) in slots.iter_mut().zip(&indptr) {
        *slot = value.to_ne_bytes();
    }
    (
        Binary::from_owned(owned, env),
        u32_binary(env, &graph.0.neighbors),
        f64_binary(env, &graph.0.weights),
    )
}

#[rustler::nif(schedule = "DirtyCpu")]
fn graph_transform(graph: Graph, kind: &str) -> Result<Graph, String> {
    let kind = Transform::parse(kind)?;
    Ok(ResourceArc::new(GraphResource(graph.0.transform(kind))))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn graph_lag<'a>(env: Env<'a>, graph: Graph, y: Binary) -> Result<Binary<'a>, String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    graph.0.lag(&y).map(|lag| f64_binary(env, &lag))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn graph_cardinalities<'a>(env: Env<'a>, graph: Graph) -> Binary<'a> {
    u32_binary(env, &graph.0.cardinalities())
}

#[rustler::nif(schedule = "DirtyCpu")]
fn graph_isolates<'a>(env: Env<'a>, graph: Graph) -> Binary<'a> {
    u32_binary(env, &graph.0.isolates())
}

#[rustler::nif(schedule = "DirtyCpu")]
fn graph_stats(graph: Graph) -> (f64, f64, f64) {
    (graph.0.s0(), graph.0.s1(), graph.0.s2())
}

#[rustler::nif(schedule = "DirtyCpu")]
fn graph_symmetric(graph: Graph, tol: f64) -> bool {
    graph.0.is_symmetric(tol)
}

rustler::init!("Elixir.Spatex.Native");
