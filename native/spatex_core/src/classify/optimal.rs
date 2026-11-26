//! Optimal and iterative classifiers: Fisher-Jenks, Jenks-Caspall (plain, forced, sampled) and
//! natural breaks (seeded 1-D k-means).

use super::{bins, stats};
use rand_chacha::rand_core::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// `_fisher_jenks_means`: exact dynamic program over the sorted values.
///
/// mapclassify runs the recursion in `float32` (numba work arrays), and the tie-breaking
/// comparison is non-strict, so the arithmetic here is `f32` with the same operation order to
/// reproduce its breaks bit for bit. The returned bins are actual (`f64`) data values.
pub fn fisher_jenks_bins(x: &[f64], classes: usize) -> Vec<f64> {
    let n = x.len();
    let w1 = classes + 1;
    let mut mat1 = vec![0i32; (n + 1) * w1];
    let mut mat2 = vec![0f32; (n + 1) * w1];
    for j in 1..=classes {
        mat1[w1 + j] = 1;
    }
    for l in 2..=n {
        for j in 1..=classes {
            mat2[l * w1 + j] = f32::INFINITY;
        }
    }
    let mut v = 0f32;
    for l in 2..=n {
        let (mut s1, mut s2, mut w) = (0f32, 0f32, 0f32);
        for m in 1..=l {
            let i3 = l - m + 1;
            let val = x[i3 - 1] as f32;
            s2 += val * val;
            s1 += val;
            w += 1.0;
            v = s2 - (s1 * s1) / w;
            let i4 = i3 - 1;
            if i4 != 0 {
                for j in 2..=classes {
                    let cand = v + mat2[i4 * w1 + j - 1];
                    if mat2[l * w1 + j] >= cand {
                        mat1[l * w1 + j] = i3 as i32;
                        mat2[l * w1 + j] = cand;
                    }
                }
            }
        }
        mat1[l * w1 + 1] = 1;
        mat2[l * w1 + 1] = v;
    }
    let mut kclass = vec![0.0; classes + 1];
    kclass[classes] = x[n - 1];
    kclass[0] = x[0];
    let mut kk = n;
    for c in (2..=classes).rev() {
        let pivot = mat1[kk * w1 + c] as isize;
        // numpy indexing: a pivot of 1 wraps around to the last element.
        let id = (pivot - 2).rem_euclid(n as isize) as usize;
        kclass[c - 1] = x[id];
        kk = (pivot - 1) as usize;
    }
    kclass.split_off(1)
}

/// `JenksCaspall._set_bins`: returns `(bins, iterations)`.
///
/// Deviation: mapclassify takes the median of an emptied class as NaN and then misassigns every
/// observation; here the iteration stops and the non-empty classes are returned.
pub fn jenks_caspall_bins(x: &[f64], k: usize) -> (Vec<f64>, usize) {
    let (q, _) = bins::quantile_bins(x, k);
    let (mut xb0, _) = bins::bin1d(x, &q);
    let mut ids: Vec<u32> = xb0.clone();
    ids.sort_unstable();
    ids.dedup();
    let mut centers: Vec<f64> = ids.iter().map(|c| class_median(x, &xb0, *c)).collect();
    let m = centers.len();
    let mut iterations = 0;
    let xb = loop {
        let xb: Vec<u32> = x
            .iter()
            .map(|v| {
                let mut best = 0;
                let mut bd = f64::INFINITY;
                for (i, c) in centers.iter().enumerate() {
                    let d = (v - c).abs();
                    if d < bd {
                        bd = d;
                        best = i;
                    }
                }
                best as u32
            })
            .collect();
        let stop = xb == xb0;
        xb0.clone_from(&xb);
        iterations += 1;
        let next: Vec<Option<f64>> = (0..m as u32)
            .map(|c| xb.contains(&c).then(|| class_median(x, &xb, c)))
            .collect();
        if stop || next.iter().any(Option::is_none) {
            break xb;
        }
        centers = next.into_iter().flatten().collect();
    };
    (class_maxima(x, &xb), iterations)
}

fn class_median(x: &[f64], xb: &[u32], c: u32) -> f64 {
    let v: Vec<f64> = x
        .iter()
        .zip(xb)
        .filter(|(_, b)| **b == c)
        .map(|(v, _)| *v)
        .collect();
    stats::median(&v)
}

/// `max(x[xb == c])` for every class id present, in ascending class order.
fn class_maxima(x: &[f64], xb: &[u32]) -> Vec<f64> {
    let mut out: Vec<(u32, f64)> = Vec::new();
    for (v, b) in x.iter().zip(xb) {
        match out.iter_mut().find(|(c, _)| c == b) {
            Some((_, m)) => *m = m.max(*v),
            None => out.push((*b, *v)),
        }
    }
    out.sort_by_key(|(c, _)| *c);
    out.into_iter().map(|(_, m)| m).collect()
}

/// Within-class sum of squares as `JenksCaspallForced` evaluates it: class means by pairwise
/// summation, deviations accumulated sequentially (Python's `sum`).
fn within_ss(x: &[f64], xb: &[u32], nclass: usize) -> f64 {
    let mut groups: Vec<Vec<f64>> = vec![Vec::new(); nclass];
    for (v, b) in x.iter().zip(xb) {
        groups[*b as usize].push(*v);
    }
    let means: Vec<f64> = groups
        .iter()
        .map(|g| if g.is_empty() { 0.0 } else { stats::mean(g) })
        .collect();
    let mut ss = 0.0;
    for (v, b) in x.iter().zip(xb) {
        let d = v - means[*b as usize];
        ss += d * d;
    }
    ss
}

/// `JenksCaspallForced._set_bins`: returns `(bins, iterations)`.
///
/// Starts from the quantile classes and moves, one observation at a time, the highest-index
/// member of a class up or the lowest-index member down while the within-class sum of squares
/// decreases. "Highest/lowest index" refers to the position in `y`, as in mapclassify.
///
/// Because of that, classes are not necessarily contiguous in value. mapclassify returns the raw
/// class maxima (possibly unsorted, and the resulting `yb`/`counts` may contain an empty class);
/// this implementation returns them sorted so downstream binning stays well defined.
pub fn jenks_caspall_forced_bins(x: &[f64], k: usize) -> Result<(Vec<f64>, usize), String> {
    let (q, _) = bins::quantile_bins(x, k);
    let (mut xb, _) = bins::bin1d(x, &q);
    let nclass = q.len();
    let present = |xb: &[u32]| -> Vec<u32> {
        let mut ids = xb.to_vec();
        ids.sort_unstable();
        ids.dedup();
        ids
    };
    if present(&xb) != (0..nclass as u32).collect::<Vec<_>>() {
        return Err("quantile classes are not contiguous; cannot run JenksCaspallForced".into());
    }
    let mut ss = within_ss(x, &xb, nclass);
    // One accepted move per call: the first class (scanned in order) whose edge member can move
    // to the neighbouring class with a lower within-class sum of squares.
    let mut step = |xb: &mut Vec<u32>, up: bool| -> bool {
        let ids = present(xb);
        let nk: Vec<usize> = ids
            .iter()
            .map(|c| xb.iter().filter(|b| *b == c).count())
            .collect();
        let range = if up { 0..nk.len() - 1 } else { 1..nk.len() };
        for i in range {
            if nk[i] > 1 {
                let mover = if up {
                    xb.iter().rposition(|b| *b == ids[i])
                } else {
                    xb.iter().position(|b| *b == ids[i])
                }
                .expect("class is non-empty");
                let mut tmp = xb.clone();
                if up {
                    tmp[mover] += 1;
                } else {
                    tmp[mover] -= 1;
                }
                let tss = within_ss(x, &tmp, nclass);
                if tss < ss {
                    *xb = tmp;
                    ss = tss;
                    return true;
                }
            }
        }
        false
    };
    // mapclassify's outer `while solving` loop always ends after one round: both phases run until
    // their last pass makes no move, so "no up and no down moves" holds on the first check.
    while step(&mut xb, true) {}
    while step(&mut xb, false) {}
    let iterations = 1;
    let mut cuts = class_maxima(x, &xb);
    // Deviation from mapclassify (see docs/parity.md): moves are by position in `y`, so the class
    // maxima can come out of value order. Sorting keeps `bins` a valid ascending break vector.
    cuts.sort_by(f64::total_cmp);
    if cuts.len() != k {
        // mapclassify reshapes the cuts to `(k,)` and fails when ties left fewer classes.
        return Err(format!(
            "cannot form {k} classes: only {} distinct quantile classes",
            cuts.len()
        ));
    }
    Ok((cuts, iterations))
}

/// A seeded `ChaCha8` stream; `stream` separates independent consumers of one `seed`.
fn rng(seed: u64, stream: u64) -> ChaCha8Rng {
    let mut r = ChaCha8Rng::seed_from_u64(seed);
    r.set_stream(stream);
    r
}

/// Uniform `f64` in `[0, 1)`.
fn unit(r: &mut ChaCha8Rng) -> f64 {
    (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

/// Uniform index in `0..n`.
fn below(r: &mut ChaCha8Rng, n: usize) -> usize {
    ((u128::from(r.next_u64()) * n as u128) >> 64) as usize
}

/// k-means++ seeding on sorted values (D^2 sampling).
fn kmeans_pp(xs: &[f64], k: usize, r: &mut ChaCha8Rng) -> Vec<f64> {
    let mut centers = vec![xs[below(r, xs.len())]];
    let mut d2: Vec<f64> = xs.iter().map(|v| (v - centers[0]).powi(2)).collect();
    while centers.len() < k {
        let total: f64 = d2.iter().sum();
        let target = unit(r) * total;
        let mut acc = 0.0;
        let mut pick = d2.iter().rposition(|d| *d > 0.0).unwrap_or(0);
        for (i, d) in d2.iter().enumerate() {
            acc += d;
            if acc > target && *d > 0.0 {
                pick = i;
                break;
            }
        }
        let c = xs[pick];
        centers.push(c);
        for (d, v) in d2.iter_mut().zip(xs) {
            *d = d.min((v - c).powi(2));
        }
    }
    centers.sort_by(|a, b| a.total_cmp(b));
    centers
}

/// One Lloyd run to convergence; returns `(inertia, labels)` over sorted values.
fn lloyd(xs: &[f64], mut centers: Vec<f64>) -> (f64, Vec<usize>) {
    let k = centers.len();
    let mut labels = vec![usize::MAX; xs.len()];
    for _ in 0..300 {
        let bounds: Vec<f64> = centers.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();
        // Ties at a boundary go to the lower center, like argmin.
        let new: Vec<usize> = xs
            .iter()
            .map(|v| bounds.partition_point(|b| b < v))
            .collect();
        if new == labels {
            break;
        }
        labels = new;
        let mut sums = vec![0.0; k];
        let mut cnt = vec![0usize; k];
        for (v, l) in xs.iter().zip(&labels) {
            sums[*l] += v;
            cnt[*l] += 1;
        }
        let mut far = (0.0, 0);
        for (i, (v, l)) in xs.iter().zip(&labels).enumerate() {
            let d = (v - centers[*l]).powi(2);
            if d > far.0 {
                far = (d, i);
            }
        }
        for c in 0..k {
            if cnt[c] > 0 {
                centers[c] = sums[c] / cnt[c] as f64;
            } else {
                // Relocate an emptied center onto the worst-fitted observation.
                centers[c] = xs[far.1];
            }
        }
        centers.sort_by(|a, b| a.total_cmp(b));
    }
    let mut sums = vec![0.0; k];
    let mut cnt = vec![0usize; k];
    for (v, l) in xs.iter().zip(&labels) {
        sums[*l] += v;
        cnt[*l] += 1;
    }
    let inertia = xs
        .iter()
        .zip(&labels)
        .map(|(v, l)| (v - sums[*l] / cnt[*l] as f64).powi(2))
        .sum();
    (inertia, labels)
}

/// `NaturalBreaks._set_bins`: 1-D k-means with seeded k-means++ restarts; returns
/// `(bins, fewer_unique_than_k)`.
///
/// The `initial` restarts use independent ChaCha streams of `seed` and the lowest inertia wins,
/// so the result is a pure function of `(y, k, initial, seed)`. Lloyd iterations run until the
/// labels stop changing.
pub fn natural_breaks_bins(y: &[f64], k: usize, initial: usize, seed: u64) -> (Vec<f64>, bool) {
    let xs = stats::sorted(y);
    let mut uv = xs.clone();
    uv.dedup();
    if uv.len() < k {
        return (uv, true);
    }
    let mut best: Option<(f64, Vec<usize>)> = None;
    for init in 0..initial.max(1) {
        let mut r = rng(seed, init as u64);
        let (inertia, labels) = lloyd(&xs, kmeans_pp(&xs, k, &mut r));
        if best.as_ref().is_none_or(|(b, _)| inertia < *b) {
            best = Some((inertia, labels));
        }
    }
    let (_, labels) = best.expect("at least one restart");
    let labels: Vec<u32> = labels.iter().map(|l| *l as u32).collect();
    (class_maxima(&xs, &labels), false)
}

/// The random subsample of `FisherJenksSampled` / `JenksCaspallSampled`: `int(n * pct)` draws
/// with replacement. Returns the sample, in draw order.
pub fn sample(y: &[f64], pct: f64, truncate: bool, seed: u64) -> Result<Vec<f64>, String> {
    if !(pct > 0.0 && pct <= 1.0) {
        return Err("pct must be in (0, 1]".into());
    }
    let n = y.len();
    let pct = if truncate && pct * n as f64 > 1000.0 {
        1000.0 / n as f64
    } else {
        pct
    };
    let m = (n as f64 * pct) as usize;
    if m == 0 {
        return Err("the sample is empty; increase pct".into());
    }
    let mut r = rng(seed, 0);
    Ok((0..m).map(|_| y[below(&mut r, n)]).collect())
}
