//! NIFs for queen/rook and block contiguity. Edge lists cross the boundary as a pair of
//! native-endian `u32` binaries (focal positions, neighbor positions).

use rustler::{Binary, Env, OwnedBinary};
use spatex_core::contiguity;

fn pair_binaries<'a>(env: Env<'a>, pairs: &[(u32, u32)]) -> (Binary<'a>, Binary<'a>) {
    let mut focal = OwnedBinary::new(pairs.len() * 4);
    let mut neighbor = OwnedBinary::new(pairs.len() * 4);
    let (f_slots, _) = focal.as_mut_slice().as_chunks_mut::<4>();
    let (n_slots, _) = neighbor.as_mut_slice().as_chunks_mut::<4>();
    for ((f, n), &(a, b)) in f_slots.iter_mut().zip(n_slots.iter_mut()).zip(pairs) {
        *f = a.to_ne_bytes();
        *n = b.to_ne_bytes();
    }
    (
        Binary::from_owned(focal, env),
        Binary::from_owned(neighbor, env),
    )
}

/// `{focal, neighbor, weights}`; `weights` (f64) is empty unless `by_perimeter`, which gives
/// the shared boundary length of every pair.
#[rustler::nif(schedule = "DirtyCpu")]
fn contiguity_edges<'a>(
    env: Env<'a>,
    wkbs: Vec<Binary<'a>>,
    rook: bool,
    strict: bool,
    by_perimeter: bool,
) -> Result<(Binary<'a>, Binary<'a>, Binary<'a>), String> {
    let slices: Vec<&[u8]> = wkbs.iter().map(|b| b.as_slice()).collect();
    let shapes = if strict || by_perimeter {
        Some(contiguity::strict::shapes(&slices)?)
    } else {
        None
    };
    let pairs = match &shapes {
        Some(s) if strict => contiguity::strict::strict_edges(s, rook),
        _ => contiguity::contiguity_edges(&slices, rook)?,
    };
    let weights = match &shapes {
        Some(s) if by_perimeter => contiguity::strict::perimeter_weights(s, &pairs)?,
        _ => Vec::new(),
    };
    let (focal, neighbor) = pair_binaries(env, &pairs);
    Ok((focal, neighbor, crate::f64_binary(env, &weights)))
}

/// `{focal, neighbor}` of libpysal's fuzzy contiguity: `buffer` (or `tolerance`, a share of
/// half the shorter side of the total bounds) grows every polygon, `predicate` (`None` for the
/// bounding-box test) selects the pairs.
#[rustler::nif(schedule = "DirtyCpu")]
fn fuzzy_contiguity_edges<'a>(
    env: Env<'a>,
    wkbs: Vec<Binary<'a>>,
    tolerance: Option<f64>,
    buffer: Option<f64>,
    predicate: Option<String>,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let slices: Vec<&[u8]> = wkbs.iter().map(|b| b.as_slice()).collect();
    let predicate = match predicate {
        Some(name) => contiguity::fuzzy::Predicate::parse(&name)?,
        None => contiguity::fuzzy::Predicate::BBox,
    };
    let pairs = contiguity::fuzzy::fuzzy_edges(&slices, tolerance, buffer, predicate)?;
    Ok(pair_binaries(env, &pairs))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn block_edges<'a>(env: Env<'a>, codes: Binary<'a>) -> Result<(Binary<'a>, Binary<'a>), String> {
    let codes = spatex_core::bytes_to_u32s(codes.as_slice())?;
    Ok(pair_binaries(env, &contiguity::block_edges(&codes)))
}
