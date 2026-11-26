//! `mapclassify.MaxP`: grow `k` contiguous classes of the sorted values from quantile seeds, keep
//! the best of `initial` growths by total sum of squares, then one round of swaps of border
//! values between neighboring classes that do not raise the sum of squares.
//!
//! mapclassify reseeds numpy's legacy generator with `seed1` before every growth and with `seed2`
//! before every swap round; [`Mt19937`] reproduces those permutations. mapclassify computes the
//! class positions on the sorted copy but evaluates sums of squares and bins on the unsorted
//! input; Spatex uses the sorted values throughout, which is identical for sorted input (see
//! `docs/parity.md`).

use std::collections::HashMap;

use super::bins::quantile_bins;
use super::mt19937::Mt19937;
use super::stats;

/// Sum of squares of `x[class]` around its mean: numpy's pairwise `yc.mean()`, then Python's
/// sequential `sum` of the squared deviations, in class order.
fn ss(x: &[f64], class: &[usize]) -> f64 {
    let yc: Vec<f64> = class.iter().map(|&i| x[i]).collect();
    let m = stats::mean(&yc);
    yc.iter()
        .map(|v| (v - m) * (v - m))
        .fold(0.0, |acc, d| acc + d)
}

/// `_swap`: moving `a` from `from` to `to` is accepted unless it raises the sum of squares.
fn swap_helps(x: &[f64], from: &[usize], to: &[usize], a: usize) -> bool {
    let before = ss(x, from) + ss(x, to);
    let from_after: Vec<usize> = from.iter().copied().filter(|&i| i != a).collect();
    let mut to_after = to.to_vec();
    to_after.push(a);
    let after = ss(x, &from_after) + ss(x, &to_after);
    before >= after
}

/// One growth from the quantile seeds; classes are ascending runs of sorted positions.
fn grow(seeds: &[usize], n: usize, seed1: u32) -> Vec<Vec<usize>> {
    let mut remaining = vec![true; n];
    for &s in seeds {
        remaining[s] = false;
    }
    let mut classes: Vec<Vec<usize>> = seeds.iter().map(|&s| vec![s]).collect();
    let mut order = Mt19937::new(seed1).permutation(seeds.len());
    while let Some(id) = order.pop() {
        loop {
            let (low, high) = (classes[id][0], *classes[id].last().unwrap());
            let mut moved = false;
            if low > 0 && remaining[low - 1] {
                classes[id].insert(0, low - 1);
                remaining[low - 1] = false;
                moved = true;
            }
            if high + 1 < n && remaining[high + 1] {
                classes[id].push(high + 1);
                remaining[high + 1] = false;
                moved = true;
            }
            if !moved {
                break;
            }
        }
    }
    classes
}

/// Upper bounds of the MaxP classes of `y` (`k` classes; errors as mapclassify does when the
/// quantile seeds collapse).
pub fn max_p_bins(
    y: &[f64],
    k: usize,
    initial: usize,
    seed1: u32,
    seed2: u32,
) -> Result<Vec<f64>, String> {
    let mut x = y.to_vec();
    x.sort_by(f64::total_cmp);
    let n = x.len();
    let (q, _) = quantile_bins(&x, k);
    if q.len() < k {
        return Err(format!(
            "Not enough unique values in array to form {k} classes (MaxP needs {k} distinct \
             quantiles, got {})",
            q.len()
        ));
    }
    // First position nearest to each quantile (np.nonzero(di == min(di))[0][0]).
    let seeds: Vec<usize> = q
        .iter()
        .map(|&qi| {
            let mut best = 0;
            for i in 1..n {
                if (x[i] - qi).abs() < (x[best] - qi).abs() {
                    best = i;
                }
            }
            best
        })
        .collect();
    let mut sorted_seeds = seeds.clone();
    sorted_seeds.sort_unstable();
    sorted_seeds.dedup();
    if sorted_seeds.len() < seeds.len() {
        return Err("MaxP quantile seeds coincide; use fewer classes".into());
    }
    // Every growth replays the same seed1 permutation, so all `initial` solutions coincide;
    // the loop is kept for its acceptance rule (strictly below the total variance).
    let all = (0..n).collect::<Vec<usize>>();
    let mut best_tss = ss(&x, &all);
    let mut best: Option<Vec<Vec<usize>>> = None;
    for _ in 0..initial {
        let classes = grow(&seeds, n, seed1);
        let tss: f64 = classes.iter().map(|c| ss(&x, c)).sum();
        if tss < best_tss {
            best_tss = tss;
            best = Some(classes);
        }
    }
    let mut classes = best.ok_or("MaxP found no solution below the total sum of squares")?;
    let mut a2c: HashMap<usize, usize> = HashMap::new();
    for (r, cl) in classes.iter().enumerate() {
        for &a in cl {
            a2c.insert(a, r);
        }
    }
    // mapclassify adds `n_moves` to `total_moves` only after a class stops moving (when it is
    // 0), so its `while swapping` loop always ends after one round; one round is kept.
    let mut order = Mt19937::new(seed2).permutation(k);
    while let Some(id) = order.pop() {
        loop {
            let left = classes[id][0].checked_sub(1);
            let right = *classes[id].last().unwrap() + 1;
            let mut n_moves = 0;
            if let Some(&lc) = left.and_then(|l| a2c.get(&l)) {
                if classes[lc].len() > 1 {
                    let a = *classes[lc].last().unwrap();
                    if swap_helps(&x, &classes[lc], &classes[id], a) {
                        classes[id].insert(0, a);
                        classes[lc].retain(|&i| i != a);
                        a2c.insert(a, id);
                        n_moves += 1;
                    }
                }
            }
            if let Some(&rc) = a2c.get(&right) {
                if classes[rc].len() > 1 {
                    let a = classes[rc][0];
                    if swap_helps(&x, &classes[rc], &classes[id], a) {
                        classes[id].push(a);
                        classes[rc].retain(|&i| i != a);
                        a2c.insert(a, id);
                        n_moves += 1;
                    }
                }
            }
            if n_moves == 0 {
                break;
            }
        }
    }
    Ok(classes.iter().map(|c| x[*c.last().unwrap()]).collect())
}
