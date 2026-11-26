//! Bin computation and assignment, following `mapclassify.classifiers`.

use super::stats;

/// `mapclassify.classifiers.quantile`: percentile cut points at `100/k` steps, deduplicated
/// (`numpy.unique`). The second value is `true` when fewer than `k` unique cuts remain.
pub fn quantile_bins(y: &[f64], k: usize) -> (Vec<f64>, bool) {
    let w = 100.0 / k as f64;
    let mut p = stats::arange(w, 100.0 + w, w);
    if let Some(last) = p.last_mut() {
        if *last > 100.0 {
            *last = 100.0;
        }
    }
    let s = stats::sorted(y);
    let mut q: Vec<f64> = p
        .iter()
        .map(|pc| stats::percentile_sorted(&s, *pc))
        .collect();
    // Deviation from mapclassify: its last percentile can drift just below 100 (k = 23 gives
    // 99.99999999999999), which would leave max(y) in class 0. The top bin is always max(y).
    if let (Some(last), Some(max)) = (q.last_mut(), s.last()) {
        *last = *max;
    }
    q.sort_by(|a, b| a.total_cmp(b));
    q.dedup();
    let short = q.len() < k;
    (q, short)
}

/// Equal-width upper bounds (`EqualInterval._set_bins`).
pub fn equal_interval_bins(y: &[f64], k: usize) -> Vec<f64> {
    let max_y = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_y = y.iter().copied().fold(f64::INFINITY, f64::min);
    let width = (max_y - min_y) * 1.0 / k as f64;
    let mut cuts = stats::arange(min_y + width, max_y + width, width);
    if cuts.len() > k {
        cuts.truncate(k);
    }
    if let Some(last) = cuts.last_mut() {
        *last = max_y;
    }
    cuts
}

/// `BoxPlot._set_bins`; returns `(bins, iqr)`.
pub fn box_plot_bins(y: &[f64], hinge: f64) -> (Vec<f64>, f64) {
    let s = stats::sorted(y);
    let mut bins: Vec<f64> = [25.0, 50.0, 75.0, 100.0]
        .iter()
        .map(|p| stats::percentile_sorted(&s, *p))
        .collect();
    let iqr = bins[2] - bins[0];
    let pivot = hinge * iqr;
    let left_fence = bins[0] - pivot;
    let right_fence = bins[2] + pivot;
    if right_fence < bins[3] {
        bins.insert(3, right_fence);
    } else {
        bins[3] = right_fence;
    }
    bins.insert(0, left_fence);
    (bins, iqr)
}

/// `StdMean._set_bins`.
pub fn std_mean_bins(y: &[f64], multiples: &[f64], anchor: bool) -> Result<Vec<f64>, String> {
    if y.len() < 2 {
        return Err("std_mean needs at least two values".into());
    }
    let s = stats::std_ddof(y, 1);
    let m = stats::mean(y);
    let y_min = y.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let multiples: Vec<f64> = if anchor {
        if s == 0.0 {
            return Err("anchor: true needs a non-zero standard deviation".into());
        }
        let min_z = ((y_min - m) / s).trunc() as i64;
        let max_z = ((y_max - m) / s).trunc() as i64 + 1;
        (min_z..max_z).map(|z| z as f64).collect()
    } else {
        multiples.to_vec()
    };
    if multiples.is_empty() {
        return Err("multiples must not be empty".into());
    }
    let mut cuts: Vec<f64> = multiples.iter().map(|w| m + s * w).collect();
    if cuts[cuts.len() - 1] < y_max {
        cuts.push(y_max);
    }
    Ok(cuts)
}

/// `Percentiles._set_bins`.
pub fn percentile_bins(y: &[f64], pct: &[f64]) -> Result<Vec<f64>, String> {
    if pct.is_empty() {
        return Err("pct must not be empty".into());
    }
    if let Some(p) = pct.iter().find(|p| !(0.0..=100.0).contains(*p)) {
        return Err(format!("percentile {p} must be in the range [0, 100]"));
    }
    let s = stats::sorted(y);
    Ok(pct
        .iter()
        .map(|p| stats::percentile_sorted(&s, *p))
        .collect())
}

/// `MaximumBreaks._set_bins`; returns `(bins, insufficient_unique_diffs)`.
///
/// `numpy.argsort` is not stable, so the order among equal gaps is unspecified in
/// mapclassify; this implementation breaks ties by position (later gap ranks larger).
pub fn maximum_breaks_bins(y: &[f64], k: usize) -> Result<(Vec<f64>, bool), String> {
    if k < 2 {
        return Err("k must be at least 2".into());
    }
    let xs = stats::sorted(y);
    if xs.len() < k {
        return Err(format!("need at least {k} values to form {k} classes"));
    }
    let diffs: Vec<f64> = xs.windows(2).map(|w| w[1] - w[0]).collect();
    let mut idxs: Vec<usize> = (0..diffs.len()).collect();
    idxs.sort_by(|a, b| diffs[*a].total_cmp(&diffs[*b]).then(a.cmp(b)));
    let mut ud = diffs.clone();
    ud.sort_by(|a, b| a.total_cmp(b));
    ud.dedup();
    let insufficient = ud.len() < k - 1;
    let mut mp: Vec<f64> = (1..k)
        .map(|c| {
            let idx = idxs[idxs.len() - c];
            (xs[idx] + xs[idx + 1]) / 2.0
        })
        .collect();
    mp.push(xs[xs.len() - 1]);
    mp.sort_by(|a, b| a.total_cmp(b));
    Ok((mp, insufficient))
}

/// `head_tail_breaks`: successive means of the head (values above the mean).
pub fn head_tail_bins(y: &[f64]) -> Vec<f64> {
    let mut cuts: Vec<f64> = Vec::new();
    let mut values = y.to_vec();
    loop {
        if values.is_empty() {
            return cuts;
        }
        let m = stats::mean(&values);
        if cuts.last() == Some(&m) {
            return cuts;
        }
        cuts.push(m);
        let first = values[0];
        if values.iter().all(|v| *v == first) {
            return cuts;
        }
        values.retain(|v| *v > m);
    }
}

/// `bin1d`: class ids (`x` in `(bins[i-1], bins[i]]`, first class open below) and counts.
///
/// Values above the last bin belong to no interval and keep class 0, as in mapclassify.
/// Unsorted bins make intervals overlap; ids then add up exactly as mapclassify's loop does.
pub fn bin1d(x: &[f64], bins: &[f64]) -> (Vec<u32>, Vec<u32>) {
    let sorted = bins.windows(2).all(|w| w[0] <= w[1]);
    let yb: Vec<u32> = x
        .iter()
        .map(|v| {
            if sorted {
                let i = bins.partition_point(|b| b < v);
                if i < bins.len() {
                    i as u32
                } else {
                    0
                }
            } else {
                let mut id = 0u32;
                for (i, r) in bins.iter().enumerate() {
                    let l = if i == 0 {
                        f64::NEG_INFINITY
                    } else {
                        bins[i - 1]
                    };
                    if *v > l && *v <= *r {
                        id += i as u32;
                    }
                }
                id
            }
        })
        .collect();
    let mut counts = vec![0u32; bins.len()];
    for id in &yb {
        let i = *id as usize;
        if i >= counts.len() {
            counts.resize(i + 1, 0);
        }
        counts[i] += 1;
    }
    (yb, counts)
}

/// `MapClassifier.find_bin`: `numpy.digitize(x, bins, right=True)` clamped to the last class.
pub fn find_bin(bins: &[f64], x: &[f64]) -> Vec<u32> {
    let last = bins.len().saturating_sub(1) as u32;
    x.iter()
        .map(|v| (bins.partition_point(|b| b < v) as u32).min(last))
        .collect()
}
