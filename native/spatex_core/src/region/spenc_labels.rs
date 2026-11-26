//! Label assignment of Spenc: scikit-learn's `discretize` (Yu and Shi's iterative rotation,
//! spopt's default) and a k-means (k-means++ seeding, Lloyd iterations) like
//! `sklearn.cluster.KMeans(n_clusters)` as spopt calls it.
//!
//! `discretize` is a pure function of the one row it draws with `random_state.randint(n)`
//! ([`discretize_from_row`]); only that draw differs from numpy's generator (here `ChaCha8`).

use faer::Mat;
use rand_chacha::ChaCha8Rng;
use rand_core::Rng;

use crate::perm::uniform_below;

const MAX_SVD_RESTARTS: usize = 30;
const N_ITER_MAX: usize = 20;

/// `vectors` (`n x k`, row-major) normalized the way `discretize` does: columns scaled to the
/// norm of a vector of ones and oriented by their first element, then rows to unit length.
fn normalized(vectors: &[f64], n: usize, k: usize) -> Vec<f64> {
    let mut v = vectors.to_vec();
    let scale = (n as f64).sqrt();
    for c in 0..k {
        let norm = (0..n).map(|i| v[i * k + c].powi(2)).sum::<f64>().sqrt();
        for i in 0..n {
            v[i * k + c] = v[i * k + c] / norm * scale;
        }
        let first = v[c];
        if first != 0.0 {
            let sign = first.signum();
            for i in 0..n {
                v[i * k + c] *= -sign;
            }
        }
    }
    for i in 0..n {
        let norm = (0..k).map(|c| v[i * k + c].powi(2)).sum::<f64>().sqrt();
        if norm > 0.0 {
            (0..k).for_each(|c| v[i * k + c] /= norm);
        }
    }
    v
}

/// `discretize` started from embedding row `row` (the row `random_state.randint(n)` picks).
pub fn discretize_from_row(
    vectors: &[f64],
    n: usize,
    k: usize,
    row: usize,
) -> Result<Vec<u32>, String> {
    if vectors.len() != n * k || row >= n || k == 0 {
        return Err("discretize: inconsistent embedding".into());
    }
    let v = normalized(vectors, n, k);
    let mut rotation = vec![0.0; k * k];
    for c in 0..k {
        rotation[c * k] = v[row * k + c];
    }
    // The remaining columns: rows as orthogonal as possible to those already picked.
    let mut acc = vec![0.0; n];
    for j in 1..k {
        for (i, a) in acc.iter_mut().enumerate() {
            let dot: f64 = (0..k).map(|c| v[i * k + c] * rotation[c * k + j - 1]).sum();
            *a += dot.abs();
        }
        let pick = (0..n).fold(0, |best, i| if acc[i] < acc[best] { i } else { best });
        for c in 0..k {
            rotation[c * k + j] = v[pick * k + c];
        }
    }
    let eps = f64::EPSILON;
    let mut last = 0.0;
    let mut n_iter = 0;
    loop {
        n_iter += 1;
        let mut labels = vec![0u32; n];
        for (i, label) in labels.iter_mut().enumerate() {
            let mut best = 0;
            let mut best_v = f64::NEG_INFINITY;
            for j in 0..k {
                let t: f64 = (0..k).map(|c| v[i * k + c] * rotation[c * k + j]).sum();
                if t > best_v {
                    best_v = t;
                    best = j;
                }
            }
            *label = best as u32;
        }
        // t_svd = D' V with D the n x k indicator matrix of `labels`.
        let mut t = vec![0.0; k * k];
        for (i, &l) in labels.iter().enumerate() {
            for c in 0..k {
                t[l as usize * k + c] += v[i * k + c];
            }
        }
        let m = Mat::from_fn(k, k, |i, j| t[i * k + j]);
        let svd = m
            .svd()
            .map_err(|e| format!("SVD did not converge: {e:?}"))?;
        let s_sum: f64 = (0..k).map(|i| svd.S().column_vector()[i]).sum();
        let ncut = 2.0 * (n as f64 - s_sum);
        if (ncut - last).abs() < eps || n_iter > N_ITER_MAX {
            return Ok(labels);
        }
        last = ncut;
        // rotation = Vh' U' = V U'.
        let (u, vv) = (svd.U(), svd.V());
        for i in 0..k {
            for j in 0..k {
                rotation[i * k + j] = (0..k).map(|l| vv[(i, l)] * u[(j, l)]).sum();
            }
        }
    }
}

/// `discretize` with the start row drawn from `rng`; returns the labels and that row. A failed
/// SVD redraws, up to scikit-learn's 30 restarts.
pub fn discretize(
    vectors: &[f64],
    n: usize,
    k: usize,
    rng: &mut ChaCha8Rng,
) -> Result<(Vec<u32>, usize), String> {
    let mut last = String::new();
    for _ in 0..MAX_SVD_RESTARTS {
        let row = uniform_below(rng, n as u64) as usize;
        match discretize_from_row(vectors, n, k, row) {
            Ok(labels) => return Ok((labels, row)),
            Err(e) => last = e,
        }
    }
    Err(last)
}

fn unit(rng: &mut ChaCha8Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

fn sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// k-means++ with scikit-learn's greedy variant (`2 + ln k` candidates per step).
fn plus_plus(x: &[f64], n: usize, dim: usize, k: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
    let row = |i: usize| &x[i * dim..(i + 1) * dim];
    let mut centers = vec![0.0; k * dim];
    let first = uniform_below(rng, n as u64) as usize;
    centers[..dim].copy_from_slice(row(first));
    let mut closest: Vec<f64> = (0..n).map(|i| sq(row(i), row(first))).collect();
    let trials = 2 + (k as f64).ln() as usize;
    for c in 1..k {
        let pot: f64 = closest.iter().sum();
        let mut cum = Vec::with_capacity(n);
        let mut run = 0.0;
        for d in &closest {
            run += d;
            cum.push(run);
        }
        let mut best: Option<(usize, f64, Vec<f64>)> = None;
        for _ in 0..trials {
            let r = unit(rng) * pot;
            let id = cum.partition_point(|&v| v < r).min(n - 1);
            let dist: Vec<f64> = (0..n)
                .map(|i| closest[i].min(sq(row(i), row(id))))
                .collect();
            let total: f64 = dist.iter().sum();
            if best.as_ref().is_none_or(|b| total < b.1) {
                best = Some((id, total, dist));
            }
        }
        let (id, _, dist) = best.expect("at least one trial");
        centers[c * dim..(c + 1) * dim].copy_from_slice(row(id));
        closest = dist;
    }
    centers
}

/// k-means on the rows of `x` (`n x dim`, row-major): k-means++ seeding, then Lloyd iterations
/// (at most 300, scikit-learn's `tol = 1e-4` times the mean feature variance), a single run.
pub fn kmeans(x: &[f64], n: usize, dim: usize, k: usize, rng: &mut ChaCha8Rng) -> Vec<u32> {
    let row = |i: usize| &x[i * dim..(i + 1) * dim];
    let mut centers = plus_plus(x, n, dim, k, rng);
    let var = (0..dim)
        .map(|c| {
            let mean = (0..n).map(|i| x[i * dim + c]).sum::<f64>() / n as f64;
            (0..n).map(|i| (x[i * dim + c] - mean).powi(2)).sum::<f64>() / n as f64
        })
        .sum::<f64>()
        / dim as f64;
    let tol = 1e-4 * var;
    let assign = |centers: &[f64]| -> (Vec<u32>, Vec<f64>) {
        let mut labels = vec![0u32; n];
        let mut dists = vec![0.0; n];
        for i in 0..n {
            let (mut best, mut best_d) = (0, f64::INFINITY);
            for c in 0..k {
                let d = sq(row(i), &centers[c * dim..(c + 1) * dim]);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            labels[i] = best as u32;
            dists[i] = best_d;
        }
        (labels, dists)
    };
    let (mut labels, mut dists) = assign(&centers);
    for _ in 0..300 {
        let mut sums = vec![0.0; k * dim];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = labels[i] as usize;
            counts[c] += 1;
            (0..dim).for_each(|d| sums[c * dim + d] += x[i * dim + d]);
        }
        // Empty clusters take the points farthest from their centers.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| dists[b].total_cmp(&dists[a]));
        let mut far = order.into_iter();
        let mut next = centers.clone();
        for c in 0..k {
            if counts[c] > 0 {
                (0..dim).for_each(|d| next[c * dim + d] = sums[c * dim + d] / counts[c] as f64);
            } else if let Some(i) = far.next() {
                next[c * dim..(c + 1) * dim].copy_from_slice(row(i));
            }
        }
        let shift: f64 = sq(&centers, &next);
        centers = next;
        let (new_labels, new_dists) = assign(&centers);
        let same = new_labels == labels;
        labels = new_labels;
        dists = new_dists;
        if same || shift <= tol {
            break;
        }
    }
    labels
}
