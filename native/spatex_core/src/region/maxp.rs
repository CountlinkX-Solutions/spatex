//! Max-p-regions (Duque, Anselin and Rey 2012) after `spopt`'s `MaxPHeuristic`: a randomized
//! construction phase finds the largest number of contiguous regions whose extensive attribute
//! meets a threshold, enclaves are attached by similarity, and a simulated-annealing search
//! with a tabu list lowers the within-region dissimilarity (cityblock) of the best partitions.
//!
//! Iteration `i` of the construction and every annealing run draw from their own `ChaCha8`
//! stream derived from `(seed, stream)`, so results do not depend on the thread count.

use super::util::{Adj, Mat, Metric};
use crate::perm::{replicate_rng, shuffle, uniform_below};
use rand_chacha::ChaCha8Rng;
use rand_core::Rng;
use rayon::prelude::*;

const FROZEN: u32 = u32::MAX;
const ALPHA: f64 = 0.998;
const TABU_LENGTH: usize = 10;

/// What to do with components whose threshold total is too small.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Join non-largest components to the largest by one link between the nearest areas.
    Single,
    /// Join every area of a non-largest component to its nearest area in the largest one.
    Multiple,
    /// Leave the graph as is (an infeasible component is an error).
    Keep,
    /// Drop the infeasible components from the problem.
    Drop,
}

impl Policy {
    /// Parses `single`, `multiple`, `keep` or `drop`.
    pub fn parse(s: &str) -> Result<Policy, String> {
        match s {
            "single" => Ok(Policy::Single),
            "multiple" => Ok(Policy::Multiple),
            "keep" => Ok(Policy::Keep),
            "drop" => Ok(Policy::Drop),
            other => Err(format!("unknown policy {other:?}")),
        }
    }
}

/// Options of [`max_p`].
#[derive(Clone, Debug)]
pub struct MaxPOpts {
    /// Minimum total of the threshold variable per region.
    pub threshold: f64,
    /// Number of nearest neighboring areas considered for each enclave.
    pub top_n: usize,
    /// Construction iterations.
    pub max_iterations_construction: usize,
    /// Independent annealing runs per candidate partition.
    pub max_iterations_sa: usize,
    /// RNG seed.
    pub seed: u64,
    /// Infeasible component policy.
    pub policy: Policy,
}

/// Result of [`max_p`].
#[derive(Clone, Debug)]
pub struct MaxPOut {
    /// Region of each area (`None` for areas dropped with `Policy::Drop`), 0-based.
    pub labels: Vec<Option<u32>>,
    /// Number of regions.
    pub p: usize,
    /// Within-region pairwise cityblock objective of the returned partition.
    pub objective: f64,
    /// Candidate partitions that reached the maximum `p`.
    pub n_candidates: usize,
}

/// A feasible partition: `labels` hold region ids `1..=p`.
#[derive(Clone)]
struct Partition {
    labels: Vec<u32>,
    regions: Vec<Vec<u32>>,
    attr: Vec<f64>,
}

/// Applies the infeasible-component policy. Returns the adjusted graph and a keep mask.
pub fn modify_components(
    adj: &Adj,
    t: &[f64],
    threshold: f64,
    coords: Option<&[[f64; 2]]>,
    policy: Policy,
) -> Result<(Adj, Vec<bool>), String> {
    let n = adj.n();
    let (count, comp) = adj.components();
    let mut total = vec![0.0; count];
    let mut size = vec![0usize; count];
    for i in 0..n {
        total[comp[i] as usize] += t[i];
        size[comp[i] as usize] += 1;
    }
    let infeasible: Vec<usize> = (0..count).filter(|&c| total[c] < threshold).collect();
    if infeasible.len() == count {
        return Err("No feasible components found in input.".into());
    }
    let all = vec![true; n];
    if infeasible.is_empty() {
        return Ok((adj.clone(), all));
    }
    match policy {
        Policy::Keep => Err(
            "the graph has infeasible components (their threshold total is below the threshold); \
             use policy: :single, :multiple or :drop"
                .into(),
        ),
        Policy::Drop => {
            let keep: Vec<bool> = comp
                .iter()
                .map(|&c| !infeasible.contains(&(c as usize)))
                .collect();
            let mut new_id = vec![u32::MAX; n];
            let mut next = 0u32;
            for (id, &kept) in new_id.iter_mut().zip(&keep) {
                if kept {
                    *id = next;
                    next += 1;
                }
            }
            let nb: Vec<Vec<u32>> = (0..n)
                .filter(|&i| keep[i])
                .map(|i| adj.nb[i].iter().map(|&j| new_id[j as usize]).collect())
                .collect();
            Ok((Adj { nb }, keep))
        }
        Policy::Single | Policy::Multiple => {
            let coords = coords.ok_or_else(|| {
                "the graph has infeasible components; coords: is required to join them \
                 (policy :single or :multiple)"
                    .to_string()
            })?;
            let largest = (0..count).fold(0, |b, c| if size[c] > size[b] { c } else { b });
            let big: Vec<u32> = (0..n as u32)
                .filter(|&i| comp[i as usize] as usize == largest)
                .collect();
            let d2 = |a: usize, b: usize| {
                let dx = coords[a][0] - coords[b][0];
                let dy = coords[a][1] - coords[b][1];
                dx * dx + dy * dy
            };
            let mut adj = adj.clone();
            let mut links: Vec<(u32, u32)> = Vec::new();
            for c in (0..count).filter(|&c| c != largest) {
                let small: Vec<u32> = (0..n as u32)
                    .filter(|&i| comp[i as usize] as usize == c)
                    .collect();
                let nearest = |a: u32| {
                    big.iter().fold((f64::INFINITY, 0u32), |b, &g| {
                        let d = d2(a as usize, g as usize);
                        if d < b.0 {
                            (d, g)
                        } else {
                            b
                        }
                    })
                };
                if policy == Policy::Single {
                    let mut best = (f64::INFINITY, 0u32, 0u32);
                    for &a in &small {
                        let (d, g) = nearest(a);
                        if d < best.0 {
                            best = (d, a, g);
                        }
                    }
                    links.push((best.1, best.2));
                } else {
                    for &a in &small {
                        links.push((a, nearest(a).1));
                    }
                }
            }
            for (a, b) in links {
                adj.nb[a as usize].push(b);
                adj.nb[b as usize].push(a);
            }
            for row in &mut adj.nb {
                row.sort_unstable();
                row.dedup();
            }
            Ok((adj, all))
        }
    }
}

fn d1(x: &Mat, a: u32, b: u32) -> f64 {
    Metric::Manhattan.dist(x.row(a as usize), x.row(b as usize))
}

fn region_pairwise(x: &Mat, members: &[u32]) -> f64 {
    super::objective::region_pairwise(x, members, Metric::Manhattan)
}

/// `_grow_cluster_for_poly`: grow region `c` from `p` breadth-first until the threshold is met.
fn grow(
    adj: &Adj,
    t: &[f64],
    labels: &mut [u32],
    p: u32,
    c: u32,
    threshold: f64,
) -> (Vec<u32>, f64) {
    labels[p as usize] = c;
    let mut members = vec![p];
    let mut total = t[p as usize];
    let mut queue: Vec<u32> = adj.nb[p as usize].clone();
    let mut queued = vec![false; adj.n()];
    queue.iter().for_each(|&q| queued[q as usize] = true);
    let mut i = 0;
    while i < queue.len() {
        if total >= threshold {
            break;
        }
        let q = queue[i];
        if labels[q as usize] == 0 {
            labels[q as usize] = c;
            members.push(q);
            total += t[q as usize];
            if total < threshold {
                for &w in &adj.nb[q as usize] {
                    if !queued[w as usize] {
                        queued[w as usize] = true;
                        queue.push(w);
                    }
                }
            }
        }
        i += 1;
    }
    (members, total)
}

struct Grown {
    labels: Vec<u32>,
    regions: Vec<Vec<u32>>,
    attr: Vec<f64>,
    enclave: Vec<u32>,
}

/// Construction for a given visiting `order`: returns the grown regions and the enclaves.
fn construct(adj: &Adj, t: &[f64], threshold: f64, order: &[u32]) -> Grown {
    let n = adj.n();
    let mut labels = vec![0u32; n];
    let mut regions: Vec<Vec<u32>> = Vec::new();
    let mut attr: Vec<f64> = Vec::new();
    let mut enclave: Vec<u32> = Vec::new();
    for &p in order {
        if labels[p as usize] != 0 {
            continue;
        }
        let c = regions.len() as u32 + 1;
        let (members, total) = grow(adj, t, &mut labels, p, c, threshold);
        if total < threshold {
            members.iter().for_each(|&m| labels[m as usize] = FROZEN);
            enclave.extend(members);
        } else {
            regions.push(members);
            attr.push(total);
        }
    }
    Grown {
        labels,
        regions,
        attr,
        enclave,
    }
}

/// `_assign_enclave`: attach each enclave area to one of the `top_n` most similar neighboring
/// areas' regions. `None` when an enclave cannot be attached to any region.
fn assign_enclaves(
    adj: &Adj,
    x: &Mat,
    t: &[f64],
    mut g: Grown,
    top_n: usize,
    rng: &mut ChaCha8Rng,
) -> Option<Partition> {
    let mut in_enclave = vec![false; adj.n()];
    g.enclave
        .iter()
        .for_each(|&e| in_enclave[e as usize] = true);
    let mut idx = 0usize;
    while !g.enclave.is_empty() {
        if idx >= g.enclave.len() {
            return None;
        }
        let e = g.enclave[idx];
        let mut cands: Vec<(u32, f64)> = Vec::new();
        for &q in &adj.nb[e as usize] {
            if in_enclave[q as usize] {
                continue;
            }
            let r = g.labels[q as usize] as usize - 1;
            let cost: f64 = g.regions[r].iter().map(|&j| d1(x, e, j)).sum();
            cands.push((q, cost));
        }
        cands.sort_by(|a, b| a.1.total_cmp(&b.1));
        let top = cands.len().min(top_n);
        if top > 0 {
            let pick = uniform_below(rng, top as u64) as usize;
            let region = g.labels[cands[pick].0 as usize];
            g.labels[e as usize] = region;
            g.regions[region as usize - 1].push(e);
            g.attr[region as usize - 1] += t[e as usize];
            in_enclave[e as usize] = false;
            g.enclave.remove(idx);
            idx = 0;
        } else {
            idx += 1;
        }
    }
    Some(Partition {
        labels: g.labels,
        regions: g.regions,
        attr: g.attr,
    })
}

/// Deterministic construction for a fixed visiting `order` (test hook). Returns 0-based labels
/// and `p`; `top_n = 1` makes the enclave assignment deterministic too.
pub fn construct_with_order(
    adj: &Adj,
    x: &Mat,
    t: &[f64],
    threshold: f64,
    order: &[u32],
    top_n: usize,
    seed: u64,
) -> Result<(Vec<u32>, usize), String> {
    if order.len() != adj.n() {
        return Err("order must list every area once".into());
    }
    let g = construct(adj, t, threshold, order);
    let p = g.regions.len();
    let mut rng = replicate_rng(seed, 0);
    let part = assign_enclaves(adj, x, t, g, top_n.max(1), &mut rng)
        .ok_or_else(|| "an enclave could not be attached to any region".to_string())?;
    Ok((part.labels.iter().map(|l| l - 1).collect(), p))
}

/// Areas that can leave their region: removing them keeps the threshold and the region
/// connected, and they touch another region.
fn movable(adj: &Adj, t: &[f64], threshold: f64, labels: &[u32], part: &Partition) -> Vec<u32> {
    let mut out = Vec::new();
    let mut member = vec![false; adj.n()];
    for (r, members) in part.regions.iter().enumerate() {
        if members.len() < 2 {
            continue;
        }
        members.iter().for_each(|&v| member[v as usize] = true);
        let cut = articulation_points(adj, members, &member);
        let mut sorted = members.clone();
        sorted.sort_unstable();
        for &a in &sorted {
            if part.attr[r] - t[a as usize] < threshold || cut.contains(&a) {
                continue;
            }
            if adj.nb[a as usize]
                .iter()
                .any(|&q| labels[q as usize] != labels[a as usize])
            {
                out.push(a);
            }
        }
        members.iter().for_each(|&v| member[v as usize] = false);
    }
    out
}

/// Articulation points of the subgraph induced by `members` (assumed connected).
fn articulation_points(
    adj: &Adj,
    members: &[u32],
    member: &[bool],
) -> std::collections::HashSet<u32> {
    use std::collections::HashMap;
    let mut cut = std::collections::HashSet::new();
    let mut disc: HashMap<u32, u32> = HashMap::new();
    let mut low: HashMap<u32, u32> = HashMap::new();
    let root = members[0];
    let mut timer = 0u32;
    let mut root_children = 0;
    disc.insert(root, 0);
    low.insert(root, 0);
    let mut stack: Vec<(u32, u32, usize)> = vec![(root, u32::MAX, 0)];
    while let Some(&mut (v, parent, ref mut next)) = stack.last_mut() {
        let row = &adj.nb[v as usize];
        if *next < row.len() {
            let w = row[*next];
            *next += 1;
            if !member[w as usize] || w == parent {
                continue;
            }
            if let Some(&dw) = disc.get(&w) {
                let lv = low[&v].min(dw);
                low.insert(v, lv);
            } else {
                timer += 1;
                disc.insert(w, timer);
                low.insert(w, timer);
                if v == root {
                    root_children += 1;
                }
                stack.push((w, v, 0));
            }
        } else {
            stack.pop();
            if parent != u32::MAX {
                let lv = low[&v];
                let lp = low[&parent].min(lv);
                low.insert(parent, lp);
                if parent != root && lv >= disc[&parent] {
                    cut.insert(parent);
                }
            }
        }
    }
    if root_children > 1 {
        cut.insert(root);
    }
    cut
}

/// One simulated-annealing run (`_perform_sa`).
fn anneal(
    adj: &Adj,
    x: &Mat,
    t: &[f64],
    threshold: f64,
    start: &Partition,
    rng: &mut ChaCha8Rng,
) -> Partition {
    let n = adj.n();
    let mut part = start.clone();
    let mut temp = 1.0f64;
    let mut no_move = 0usize;
    let mut tabu: Vec<(u32, u32, u32)> = Vec::new();
    let mut pool: Vec<u32> = Vec::new();
    while no_move <= n {
        if pool.is_empty() {
            pool = movable(adj, t, threshold, &part.labels.clone(), &part);
        }
        if pool.is_empty() {
            break;
        }
        let a = pool[uniform_below(rng, pool.len() as u64) as usize];
        let donor = part.labels[a as usize];
        let lost: f64 = part.regions[donor as usize - 1]
            .iter()
            .map(|&j| d1(x, a, j))
            .sum();
        let mut best: Option<(u32, f64)> = None;
        for &q in &adj.nb[a as usize] {
            let r = part.labels[q as usize];
            if r != donor {
                let added: f64 = part.regions[r as usize - 1]
                    .iter()
                    .map(|&j| d1(x, a, j))
                    .sum();
                if best.is_none_or(|(_, b)| added < b) {
                    best = Some((r, added));
                }
            }
        }
        let Some((recipient, added)) = best else {
            pool.retain(|&v| v != a);
            continue;
        };
        let diff = lost - added;
        let go = if diff > 0.0 {
            if !tabu.contains(&(a, recipient, donor)) {
                if tabu.len() == TABU_LENGTH {
                    tabu.remove(0);
                }
                tabu.push((a, recipient, donor));
            }
            no_move = 0;
            true
        } else {
            no_move += 1;
            let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            (diff / temp).exp() > u && !tabu.contains(&(a, donor, recipient))
        };
        pool.retain(|&v| v != a);
        if go {
            part.labels[a as usize] = recipient;
            let from = &mut part.regions[donor as usize - 1];
            if let Some(pos) = from.iter().position(|&v| v == a) {
                from.swap_remove(pos);
            }
            part.regions[recipient as usize - 1].push(a);
            part.attr[donor as usize - 1] -= t[a as usize];
            part.attr[recipient as usize - 1] += t[a as usize];
            pool.retain(|&v| {
                let l = part.labels[v as usize];
                l != recipient && l != donor
            });
        }
        temp *= ALPHA;
    }
    part
}

fn objective_of(x: &Mat, part: &Partition) -> f64 {
    part.regions.iter().map(|r| region_pairwise(x, r)).sum()
}

/// Runs Max-p on a graph whose components are all feasible (see [`modify_components`]).
pub fn max_p(
    adj: &Adj,
    x: &Mat,
    t: &[f64],
    opts: &MaxPOpts,
) -> Result<(Vec<u32>, usize, f64, usize), String> {
    let n = adj.n();
    if opts.top_n == 0 {
        return Err("top_n must be at least 1".into());
    }
    if opts.max_iterations_construction == 0 || opts.max_iterations_sa == 0 {
        return Err("iteration counts must be at least 1".into());
    }
    // Construction phase: iteration i uses stream i for the order and the enclave draws.
    let grown: Vec<(usize, Grown, ChaCha8Rng)> = (0..opts.max_iterations_construction)
        .into_par_iter()
        .map(|it| {
            let mut rng = replicate_rng(opts.seed, it as u64);
            let mut order: Vec<u32> = (0..n as u32).collect();
            shuffle(&mut order, &mut rng);
            let g = construct(adj, t, opts.threshold, &order);
            (g.regions.len(), g, rng)
        })
        .collect();
    let max_p = grown.iter().map(|g| g.0).max().unwrap_or(0);
    if max_p == 0 {
        return Err("no feasible region could be constructed".into());
    }
    let candidates: Vec<Partition> = grown
        .into_iter()
        .filter(|g| g.0 == max_p)
        .filter_map(|(_, g, mut rng)| assign_enclaves(adj, x, t, g, opts.top_n, &mut rng))
        .collect();
    if candidates.is_empty() {
        return Err("no construction could attach all enclaves".into());
    }
    // Annealing: every (candidate, run) pair is independent.
    let base = opts.max_iterations_construction as u64;
    let runs = opts.max_iterations_sa;
    let results: Vec<(f64, Partition)> = (0..candidates.len() * runs)
        .into_par_iter()
        .map(|job| {
            let cand = &candidates[job / runs];
            let mut rng = replicate_rng(opts.seed, base + job as u64);
            let part = anneal(adj, x, t, opts.threshold, cand, &mut rng);
            (objective_of(x, &part), part)
        })
        .collect();
    let mut best: Option<&(f64, Partition)> = None;
    for r in &results {
        if best.is_none_or(|b| r.0 < b.0) {
            best = Some(r);
        }
    }
    let (obj, part) = best.ok_or_else(|| "no solution".to_string())?;
    Ok((
        part.labels.iter().map(|l| l - 1).collect(),
        max_p,
        *obj,
        candidates.len(),
    ))
}

#[cfg(test)]
mod articulation_tests {
    use super::*;

    fn brute_force_cuts(adj: &Adj, members: &[u32]) -> std::collections::HashSet<u32> {
        let mut member = vec![false; adj.n()];
        members.iter().for_each(|&v| member[v as usize] = true);
        members
            .iter()
            .copied()
            .filter(|&a| !adj.connected_subset(members, &member, Some(a)) && members.len() > 1)
            .collect()
    }

    #[test]
    fn articulation_points_match_brute_force_on_random_graphs() {
        let mut rng = replicate_rng(42, 0);
        for _ in 0..200 {
            let n = 3 + uniform_below(&mut rng, 8) as usize;
            let mut edges = Vec::new();
            for a in 0..n as u32 {
                for b in (a + 1)..n as u32 {
                    if uniform_below(&mut rng, 100) < 35 {
                        edges.push((a, b));
                    }
                }
            }
            let adj = Adj::from_edges(n, &edges);
            let (count, _) = adj.components();
            if count != 1 {
                continue;
            }
            let members: Vec<u32> = (0..n as u32).collect();
            let member = vec![true; n];
            let fast = articulation_points(&adj, &members, &member);
            assert_eq!(fast, brute_force_cuts(&adj, &members), "{edges:?}");
        }
    }
}
