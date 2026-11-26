//! Region-K-Means (Duque, Church and Middleton 2011) as in `spopt`'s
//! `RegionKMeansHeuristic`: regions grow simultaneously from seed areas by attribute
//! similarity to the region centroid, then areas move to the nearest centroid while the move
//! keeps both regions contiguous.

use super::util::{sq_dist, Adj, Mat};
use crate::perm::{replicate_rng, shuffle, uniform_below};

/// Result of [`region_kmeans`].
#[derive(Clone, Debug)]
pub struct KMeansOut {
    /// Region of each area.
    pub labels: Vec<u32>,
    /// Region means, row-major `k x p`.
    pub centroids: Vec<f64>,
    /// Improvement-loop iterations (spopt starts counting at 1).
    pub iters: usize,
    /// Seed areas, in region order.
    pub seeds: Vec<u32>,
    /// Warnings (iteration cap).
    pub warnings: Vec<String>,
}

/// Draws `k` distinct seeds, at least one in every component, from the stream of `seed`.
pub fn draw_seeds(adj: &Adj, k: usize, seed: u64) -> Result<Vec<u32>, String> {
    let n = adj.n();
    let (count, labels) = adj.components();
    if k < count {
        return Err(format!(
            "n_clusters ({k}) is smaller than the number of connected components ({count}); \
             every component needs a seed"
        ));
    }
    let mut rng = replicate_rng(seed, 0);
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); count];
    for (i, &l) in labels.iter().enumerate() {
        members[l as usize].push(i as u32);
    }
    let mut chosen = vec![false; n];
    let mut seeds = Vec::with_capacity(k);
    for m in &members {
        let s = m[uniform_below(&mut rng, m.len() as u64) as usize];
        chosen[s as usize] = true;
        seeds.push(s);
    }
    let mut rest: Vec<u32> = (0..n as u32).filter(|&i| !chosen[i as usize]).collect();
    shuffle(&mut rest, &mut rng);
    seeds.extend(rest.into_iter().take(k - count));
    shuffle(&mut seeds, &mut rng);
    Ok(seeds)
}

/// Runs Region-K-Means from explicit `seeds` (one region per seed, in order).
pub fn region_kmeans(
    adj: &Adj,
    x: &Mat,
    seeds: &[u32],
    max_iterations: usize,
) -> Result<KMeansOut, String> {
    let n = adj.n();
    let p = x.p;
    let k = seeds.len();
    if k == 0 || k > n {
        return Err(format!("n_clusters must be between 1 and {n}, got {k}"));
    }
    let mut seen = vec![false; n];
    for &s in seeds {
        if s as usize >= n {
            return Err(format!("seeds: index {s} is out of range for {n} areas"));
        }
        if std::mem::replace(&mut seen[s as usize], true) {
            return Err("seeds must be distinct areas".into());
        }
    }
    let (count, comp) = adj.components();
    let mut seeded = vec![false; count];
    seeds
        .iter()
        .for_each(|&s| seeded[comp[s as usize] as usize] = true);
    if let Some(c) = seeded.iter().position(|&s| !s) {
        let first = comp.iter().position(|&l| l as usize == c).unwrap_or(0);
        return Err(format!(
            "the connected component containing area {first} has no seed; every component \
             needs at least one seed"
        ));
    }

    const NONE: u32 = u32::MAX;
    let mut labels = vec![NONE; n];
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); k];
    for (r, &s) in seeds.iter().enumerate() {
        labels[s as usize] = r as u32;
        members[r].push(s);
    }
    let mut unassigned = n - k;
    while unassigned > 0 {
        // claims: area -> (region, distance) with the smallest distance, lowest region on ties
        let mut claims: Vec<(u32, u32, f64)> = Vec::new();
        for (rid, reg) in members.iter().enumerate() {
            let centroid = x.mean_of(reg.iter().copied());
            let mut best: Option<(u32, f64)> = None;
            for &i in reg {
                for &j in &adj.nb[i as usize] {
                    if labels[j as usize] == NONE {
                        let d = sq_dist(x.row(j as usize), &centroid);
                        if best.is_none_or(|(_, bd)| d < bd) {
                            best = Some((j, d));
                        }
                    }
                }
            }
            if let Some((area, d)) = best {
                claims.push((area, rid as u32, d));
            }
        }
        if claims.is_empty() {
            return Err("region growth stalled; the graph has unreachable areas".into());
        }
        claims.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut idx = 0;
        while idx < claims.len() {
            let area = claims[idx].0;
            let mut win = claims[idx];
            let mut j = idx;
            while j < claims.len() && claims[j].0 == area {
                if claims[j].2 < win.2 {
                    win = claims[j];
                }
                j += 1;
            }
            labels[area as usize] = win.1;
            members[win.1 as usize].push(area);
            unassigned -= 1;
            idx = j;
        }
        members.iter_mut().for_each(|m| m.sort_unstable());
    }

    // Improvement phase.
    let centroids_of = |labels: &[u32]| -> Vec<f64> {
        let mut by: Vec<Vec<u32>> = vec![Vec::new(); k];
        for (i, &l) in labels.iter().enumerate() {
            by[l as usize].push(i as u32);
        }
        let mut out = Vec::with_capacity(k * p);
        for m in &by {
            out.extend(x.mean_of(m.iter().copied()));
        }
        out
    };
    let closest_of = |cen: &[f64]| -> Vec<u32> {
        (0..n)
            .map(|i| {
                let row = x.row(i);
                let mut best = (f64::INFINITY, 0u32);
                for r in 0..k {
                    let d = sq_dist(row, &cen[r * p..(r + 1) * p]);
                    if d < best.0 {
                        best = (d, r as u32);
                    }
                }
                best.1
            })
            .collect()
    };
    let sizes_of = |labels: &[u32]| -> Vec<usize> {
        let mut s = vec![0usize; k];
        labels.iter().for_each(|&l| s[l as usize] += 1);
        s
    };
    let move_ok = |a: usize, labels: &[u32], cl: &[u32], sizes: &[usize]| -> bool {
        let (src, dest) = (labels[a], cl[a]);
        if !adj.nb[a].iter().any(|&m| labels[m as usize] == dest) {
            return false;
        }
        if sizes[src as usize] < 2 {
            return false;
        }
        // BFS over the source region without `a`.
        let Some(&start) = adj.nb[a].iter().find(|&&m| labels[m as usize] == src) else {
            return false;
        };
        let mut seen = vec![false; n];
        seen[start as usize] = true;
        let mut stack = vec![start];
        let mut reached = 1;
        while let Some(v) = stack.pop() {
            for &w in &adj.nb[v as usize] {
                let wi = w as usize;
                if wi != a && labels[wi] == src && !seen[wi] {
                    seen[wi] = true;
                    reached += 1;
                    stack.push(w);
                }
            }
        }
        reached == sizes[src as usize] - 1
    };
    let candidates_of = |labels: &[u32], cl: &[u32]| -> Vec<u32> {
        let sizes = sizes_of(labels);
        (0..n)
            .filter(|&a| cl[a] != labels[a] && move_ok(a, labels, cl, &sizes))
            .map(|a| a as u32)
            .collect()
    };

    let mut cen = centroids_of(&labels);
    let mut cl = closest_of(&cen);
    let mut cand = candidates_of(&labels, &cl);
    let mut iters = 1usize;
    let mut warnings = Vec::new();
    while let Some(a) = cand.pop() {
        if iters >= max_iterations {
            warnings.push(format!(
                "the improvement loop stopped after {max_iterations} iterations"
            ));
            break;
        }
        let sizes = sizes_of(&labels);
        if move_ok(a as usize, &labels, &cl, &sizes) {
            labels[a as usize] = cl[a as usize];
            cen = centroids_of(&labels);
            cl = closest_of(&cen);
            cand = candidates_of(&labels, &cl);
        }
        iters += 1;
    }
    Ok(KMeansOut {
        labels,
        centroids: cen,
        iters,
        seeds: seeds.to_vec(),
        warnings,
    })
}
