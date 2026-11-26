//! NIFs for `Spatex.Region`. Attribute matrices cross as column-major f64 binaries (`p`
//! columns of `n` values); label vectors come back as lists.

use rustler::Binary;
use spatex_core::region::{
    azp::{self, AzpOpts},
    kmeans,
    maxp::{self, MaxPOpts, Policy},
    objective,
    random::{self, RandomOpts},
    sa3::{self, Extraction},
    skater,
    skater::SkaterOpts,
    spenc::{self, Assign, Kernel, KernelParams, SpencOpts},
    util::{self, Mat},
    ward, Adj, Metric,
};

use crate::{f64_binary, Graph};

fn matrix(x: Binary, p: usize, n: usize) -> Result<Vec<f64>, String> {
    let cols = spatex_core::bytes_to_f64s(x.as_slice())?;
    util::columns_to_rows(&cols, n, p)
}

fn mat<'a>(rows: &'a [f64], n: usize, p: usize) -> Result<Mat<'a>, String> {
    Mat::new(rows, n, p)
}

type Trace = Vec<(Vec<u32>, Option<u32>, Option<u32>, Option<f64>)>;
type SkaterReply = (Vec<u32>, Vec<(u32, u32)>, f64, Trace, Vec<String>);

#[rustler::nif(schedule = "DirtyCpu")]
pub fn region_skater(
    graph: Graph,
    x: Binary,
    p: usize,
    n_clusters: usize,
    floor: f64,
    metric: &str,
    flags: Vec<bool>,
) -> Result<SkaterReply, String> {
    let [ignore_islands, trace, drop_zero, symmetrize] = flags[..] else {
        return Err("skater expects four flags".into());
    };
    let n = graph.0.n;
    let rows = matrix(x, p, n)?;
    let x = mat(&rows, n, p)?;
    let metric = Metric::parse(metric)?;
    if metric == Metric::SqEuclidean {
        return Err("skater supports the manhattan and euclidean dissimilarities".into());
    }
    let adj = Adj::from_csr(&graph.0, symmetrize)?;
    let out = skater::skater(
        &adj,
        &x,
        &SkaterOpts {
            n_clusters,
            floor,
            ignore_islands,
            trace,
            metric,
            drop_zero,
        },
    )?;
    let trace = out
        .trace
        .into_iter()
        .map(|s| {
            let (a, b) = s.edge.map_or((None, None), |(a, b)| (Some(a), Some(b)));
            (s.labels, a, b, Some(s.score).filter(|v| v.is_finite()))
        })
        .collect();
    Ok((out.labels, out.msf, out.score, trace, out.warnings))
}

type WardReply = (Vec<u32>, Vec<(u32, u32)>, Vec<f64>, usize, Vec<String>);

#[rustler::nif(schedule = "DirtyCpu")]
pub fn region_ward(
    graph: Graph,
    x: Binary,
    p: usize,
    n_clusters: usize,
) -> Result<WardReply, String> {
    let n = graph.0.n;
    let rows = matrix(x, p, n)?;
    let x = mat(&rows, n, p)?;
    let adj = Adj::from_csr(&graph.0, true)?;
    let out = ward::ward(&adj, &x, n_clusters)?;
    Ok((
        out.labels,
        out.children,
        out.distances,
        out.n_connected_components,
        out.warnings,
    ))
}

type Sa3Tree = (Vec<u32>, Vec<(u32, u32, f64, u32)>);
type Sa3Reply = (Vec<i64>, usize, Vec<Sa3Tree>);

#[rustler::nif(schedule = "DirtyCpu")]
pub fn region_sa3(
    graph: Graph,
    x: Binary,
    p: usize,
    min_cluster_size: usize,
    extraction: &str,
) -> Result<Sa3Reply, String> {
    let n = graph.0.n;
    let rows = matrix(x, p, n)?;
    let x = mat(&rows, n, p)?;
    let adj = Adj::from_csr(&graph.0, true)?;
    let extraction = match extraction {
        "eom" => Extraction::Eom,
        "leaf" => Extraction::Leaf,
        other => return Err(format!("unknown extraction {other:?}")),
    };
    let out = sa3::sa3(&adj, &x, min_cluster_size, extraction)?;
    Ok((out.labels, out.n_components, out.trees))
}

type SpencReply<'a> = (
    Vec<u32>,
    Binary<'a>,
    Binary<'a>,
    usize,
    Vec<f64>,
    i64,
    Vec<String>,
);

/// `modes` is `{affinity, assign_labels, floor_weights}`; `params` is `{gamma, degree, coef0, n_neighbors, floor, grid_resolution}`, `flags`
/// `{keep_affinity, keep_embedding}`; `n_clusters` 0 means infinity (hierarchical only) and an
/// empty `floor_weights` means all ones. Unrequested matrices come back as empty binaries.
#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::too_many_arguments)]
pub fn region_spenc<'a>(
    env: rustler::Env<'a>,
    graph: Graph,
    x: Binary,
    p: usize,
    n_clusters: usize,
    modes: (String, String, Vec<f64>),
    params: (f64, f64, f64, usize, f64, usize),
    seed: u64,
    flags: (bool, bool),
) -> Result<SpencReply<'a>, String> {
    let n = graph.0.n;
    let rows = matrix(x, p, n)?;
    let x = mat(&rows, n, p)?;
    let (kernel, assign, floor_weights) = modes;
    let (gamma, degree, coef0, n_neighbors, floor, grid_resolution) = params;
    let opts = SpencOpts {
        n_clusters: (n_clusters != 0).then_some(n_clusters),
        kernel: Kernel::parse(&kernel)?,
        params: KernelParams {
            gamma,
            degree,
            coef0,
            n_neighbors,
        },
        assign: Assign::parse(&assign)?,
        floor,
        floor_weights: (!floor_weights.is_empty()).then_some(floor_weights),
        grid_resolution,
        seed,
    };
    let out = spenc::spenc(&graph.0, &x, &opts)?;
    let (keep_affinity, keep_embedding) = flags;
    let affinity = if keep_affinity {
        &out.affinity[..]
    } else {
        &[]
    };
    let (vectors, eigenvalues, cols) = match &out.embedding {
        Some(e) => (
            if keep_embedding { &e.vectors[..] } else { &[] },
            e.eigenvalues.clone(),
            e.eigenvalues.len(),
        ),
        None => (&[][..], Vec::new(), 0),
    };
    Ok((
        out.labels,
        f64_binary(env, affinity),
        f64_binary(env, vectors),
        cols,
        eigenvalues,
        out.start_row.map_or(-1, |r| r as i64),
        out.warnings,
    ))
}

type KMeansReply<'a> = (Vec<u32>, Binary<'a>, usize, Vec<u32>, Vec<String>);

#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::too_many_arguments)]
pub fn region_kmeans<'a>(
    env: rustler::Env<'a>,
    graph: Graph,
    x: Binary,
    p: usize,
    n_clusters: usize,
    seeds: Vec<u32>,
    seed: u64,
    max_iterations: usize,
    symmetrize: bool,
) -> Result<KMeansReply<'a>, String> {
    let n = graph.0.n;
    let rows = matrix(x, p, n)?;
    let x = mat(&rows, n, p)?;
    let adj = Adj::from_csr(&graph.0, symmetrize)?;
    let seeds = if seeds.is_empty() {
        kmeans::draw_seeds(&adj, n_clusters, seed)?
    } else if seeds.len() != n_clusters {
        return Err(format!(
            "seeds has {} entries but n_clusters is {n_clusters}",
            seeds.len()
        ));
    } else {
        seeds
    };
    let out = kmeans::region_kmeans(&adj, &x, &seeds, max_iterations)?;
    Ok((
        out.labels,
        f64_binary(env, &out.centroids),
        out.iters,
        out.seeds,
        out.warnings,
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn region_objective(
    x: Binary,
    p: usize,
    labels: Vec<u32>,
    kind: &str,
    metric: &str,
) -> Result<f64, String> {
    let n = labels.len();
    let rows = matrix(x, p, n)?;
    let x = mat(&rows, n, p)?;
    let metric = Metric::parse(metric)?;
    match kind {
        "pairwise" => objective::pairwise(&x, &labels, metric),
        "center" => objective::center(&x, &labels, metric),
        other => Err(format!("unknown objective kind {other:?}")),
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn region_non_contiguous(graph: Graph, labels: Vec<u32>) -> Result<Vec<u32>, String> {
    if labels.len() != graph.0.n {
        return Err(format!(
            "labels has {} entries but the graph has {} nodes",
            labels.len(),
            graph.0.n
        ));
    }
    let adj = Adj::from_csr(&graph.0, true)?;
    Ok(util::non_contiguous_regions(&adj, &labels))
}

type AzpReply = (Vec<u32>, f64, Vec<u32>, usize);

fn strategy(method: &str, params: &[f64]) -> Result<azp::Strategy, String> {
    match (method, params) {
        ("basic", []) => Ok(azp::Strategy::Basic),
        ("tabu", [tabu_length, repetitions]) => Ok(azp::Strategy::Tabu {
            tabu_length: *tabu_length as usize,
            repetitions: *repetitions as usize,
        }),
        ("annealing", [t, cooling, term, repetitions, nonmoving]) => Ok(azp::Strategy::Annealing {
            init_temperature: *t,
            cooling_factor: *cooling,
            sa_moves_term: *term as usize,
            repetitions: *repetitions as usize,
            nonmoving_steps: *nonmoving as usize,
        }),
        (other, _) => Err(format!("unknown AZP method {other:?} or bad parameters")),
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::too_many_arguments)]
pub fn region_azp(
    graph: Graph,
    x: Binary,
    p: usize,
    (n_clusters, seed, max_moves): (usize, u64, usize),
    initial: Vec<u32>,
    metric: &str,
    symmetrize: bool,
    (method, params): (&str, Vec<f64>),
) -> Result<AzpReply, String> {
    let n = graph.0.n;
    let rows = matrix(x, p, n)?;
    let x = mat(&rows, n, p)?;
    let adj = Adj::from_csr(&graph.0, symmetrize)?;
    let out = azp::azp(
        &adj,
        &x,
        &AzpOpts {
            n_clusters,
            metric: Metric::parse(metric)?,
            seed,
            initial: if initial.is_empty() {
                None
            } else {
                Some(initial)
            },
            max_moves,
            strategy: strategy(method, &params)?,
        },
    )?;
    Ok((out.labels, out.objective, out.initial_labels, out.moves))
}

type MaxPReply = (Vec<i64>, usize, f64, usize);
type MaxPSettings = (f64, usize, usize, usize, u64, bool);

#[rustler::nif(schedule = "DirtyCpu")]
pub fn region_max_p(
    graph: Graph,
    x: Binary,
    p: usize,
    t: Binary,
    (threshold, top_n, iterations_construction, iterations_sa, seed, symmetrize): MaxPSettings,
    policy: &str,
    coords: Binary,
) -> Result<MaxPReply, String> {
    let n = graph.0.n;
    let rows = matrix(x, p, n)?;
    let t = spatex_core::bytes_to_f64s(t.as_slice())?;
    if t.len() != n {
        return Err(format!(
            "threshold_variable has {} values for {n} areas",
            t.len()
        ));
    }
    let policy = Policy::parse(policy)?;
    let coords = spatex_core::bytes_to_f64s(coords.as_slice())?;
    let coords: Option<Vec<[f64; 2]>> = if coords.is_empty() {
        None
    } else if coords.len() == 2 * n {
        Some(coords.as_chunks::<2>().0.to_vec())
    } else {
        return Err("coords must hold one {x, y} pair per area".into());
    };
    let adj = Adj::from_csr(&graph.0, symmetrize)?;
    let (adj, keep) = maxp::modify_components(&adj, &t, threshold, coords.as_deref(), policy)?;
    let kept: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
    let sub_rows: Vec<f64> = kept
        .iter()
        .flat_map(|&i| rows[i * p..(i + 1) * p].iter().copied())
        .collect();
    let sub_t: Vec<f64> = kept.iter().map(|&i| t[i]).collect();
    let xm = mat(&sub_rows, kept.len(), p)?;
    let (labels, p_found, obj, cands) = maxp::max_p(
        &adj,
        &xm,
        &sub_t,
        &MaxPOpts {
            threshold,
            top_n,
            max_iterations_construction: iterations_construction,
            max_iterations_sa: iterations_sa,
            seed,
            policy,
        },
    )?;
    let mut out = vec![-1i64; n];
    for (pos, &i) in kept.iter().enumerate() {
        out[i] = i64::from(labels[pos]);
    }
    Ok((out, p_found, obj, cands))
}

#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::too_many_arguments)]
pub fn region_max_p_construct(
    graph: Graph,
    x: Binary,
    p: usize,
    t: Binary,
    threshold: f64,
    order: Vec<u32>,
    top_n: usize,
    seed: u64,
) -> Result<(Vec<u32>, usize), String> {
    let n = graph.0.n;
    let rows = matrix(x, p, n)?;
    let x = mat(&rows, n, p)?;
    let t = spatex_core::bytes_to_f64s(t.as_slice())?;
    let adj = Adj::from_csr(&graph.0, true)?;
    maxp::construct_with_order(&adj, &x, &t, threshold, &order, top_n, seed)
}

type RandomReply = Vec<(Vec<Vec<u32>>, bool, usize)>;

/// `permutations` random partitions (stream `i` of `seed` for permutation `i`). `graph` is
/// `nil` to ignore contiguity; `num_regions` is `nil` when not given.
#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::too_many_arguments)]
pub fn region_random(
    graph: Option<Graph>,
    n: usize,
    num_regions: Option<usize>,
    cardinality: Vec<usize>,
    (maxiter, max_swaps, compact, symmetrize): (usize, usize, bool, bool),
    seed: u64,
    permutations: usize,
) -> Result<RandomReply, String> {
    let adj = match &graph {
        Some(g) => Some(Adj::from_csr(&g.0, symmetrize)?),
        None => None,
    };
    let out = random::random_regions(
        &RandomOpts {
            n,
            adj: adj.as_ref(),
            num_regions,
            cardinality: if cardinality.is_empty() {
                None
            } else {
                Some(cardinality)
            },
            maxiter,
            compact,
            max_swaps,
        },
        permutations,
        seed,
    )?;
    Ok(out
        .into_iter()
        .map(|o| (o.regions, o.feasible, o.attempts))
        .collect())
}
