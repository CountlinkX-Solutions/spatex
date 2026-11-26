//! NIFs for `Spatex.Classify`.

use rustler::{Binary, Env, OwnedBinary};
use spatex_core::classify::{bins, classify, Classification, Method};

/// Builds a core [`Method`] from the flat NIF arguments: `scalar` carries `k` or `hinge`,
/// `list` the multiples, percentiles, user bins or (natural breaks) `[initial]`, (sampled
/// classifiers) `[pct]`, `flag` the StdMean anchor or the FisherJenksSampled `truncate`,
/// `lowest` the optional user lower bound and `seed` the random seed.
fn parse_method(
    name: &str,
    scalar: f64,
    list: Vec<f64>,
    flag: bool,
    lowest: Option<f64>,
    seed: u64,
) -> Result<Method, String> {
    let k = scalar as usize;
    match name {
        "equal_interval" => Ok(Method::EqualInterval { k }),
        "quantiles" => Ok(Method::Quantiles { k }),
        "box_plot" => Ok(Method::BoxPlot { hinge: scalar }),
        "std_mean" => Ok(Method::StdMean {
            multiples: list,
            anchor: flag,
        }),
        "percentiles" => Ok(Method::Percentiles { pct: list }),
        "maximum_breaks" => Ok(Method::MaximumBreaks { k }),
        "head_tail_breaks" => Ok(Method::HeadTailBreaks),
        "user_defined" => Ok(Method::UserDefined { bins: list, lowest }),
        "fisher_jenks" => Ok(Method::FisherJenks { k }),
        "fisher_jenks_sampled" => Ok(Method::FisherJenksSampled {
            k,
            pct: list.first().copied().unwrap_or(0.1),
            truncate: flag,
            seed,
        }),
        "jenks_caspall" => Ok(Method::JenksCaspall { k }),
        "jenks_caspall_sampled" => Ok(Method::JenksCaspallSampled {
            k,
            pct: list.first().copied().unwrap_or(0.1),
            seed,
        }),
        "jenks_caspall_forced" => Ok(Method::JenksCaspallForced { k }),
        "natural_breaks" => Ok(Method::NaturalBreaks {
            k,
            initial: list.first().map_or(10, |v| *v as usize),
            seed,
        }),
        "pretty_breaks" => Ok(Method::PrettyBreaks { k }),
        "max_p" => match list.as_slice() {
            [initial, seed1, seed2] => Ok(Method::MaxP {
                k,
                initial: *initial as usize,
                seed1: *seed1 as u32,
                seed2: *seed2 as u32,
            }),
            _ => Err("max_p expects [initial, seed1, seed2]".into()),
        },
        other => Err(format!("unknown classifier {other:?}")),
    }
}

fn u32_binary<'a>(env: Env<'a>, values: &[u32]) -> Binary<'a> {
    let mut owned = OwnedBinary::new(values.len() * 4);
    let (slots, _) = owned.as_mut_slice().as_chunks_mut::<4>();
    for (slot, value) in slots.iter_mut().zip(values) {
        *slot = value.to_ne_bytes();
    }
    Binary::from_owned(owned, env)
}

type Output<'a> = (
    Vec<f64>,
    usize,
    Binary<'a>,
    Vec<u32>,
    (f64, f64, f64),
    (f64, Option<f64>, Vec<String>, Option<usize>),
);

fn output<'a>(env: Env<'a>, c: Classification) -> Output<'a> {
    (
        c.bins,
        c.k,
        u32_binary(env, &c.yb),
        c.counts,
        (c.tss, c.adcm, c.gadf),
        (c.lowest, c.iqr, c.warnings, c.iterations),
    )
}

#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::too_many_arguments)]
pub fn classify_run<'a>(
    env: Env<'a>,
    method: &str,
    y: Binary,
    scalar: f64,
    list: Vec<f64>,
    flag: bool,
    lowest: Option<f64>,
    seed: u64,
) -> Result<Output<'a>, String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let method = parse_method(method, scalar, list, flag, lowest, seed)?;
    classify(&y, &method).map(|c| output(env, c))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn classify_find_bin<'a>(
    env: Env<'a>,
    bins_: Vec<f64>,
    x: Binary,
) -> Result<Binary<'a>, String> {
    let x = spatex_core::bytes_to_f64s(x.as_slice())?;
    Ok(u32_binary(env, &bins::find_bin(&bins_, &x)))
}
