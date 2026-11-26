//! False discovery rate cut-off (`esda.fdr`).

/// The Benjamini-Hochberg style p-value cut-off of `esda.fdr`.
///
/// The p-values are sorted in descending order and compared with `index * alpha / n`
/// (`index = n, n - 1, ..., 1`); the cut-off is the first threshold strictly above its p-value.
/// When no p-value qualifies the conservative Bonferroni bound `alpha / n` is returned. This is
/// a threshold, not a vector of adjusted p-values: reject the sites with `p < cutoff`.
///
/// The threshold is computed as `index * alpha / n` (multiply, then divide) for bitwise parity
/// with esda. Errors on an empty slice, NaN or negative p-values, and `alpha` outside `(0, 1]`.
pub fn fdr(pvalues: &[f64], alpha: f64) -> Result<f64, String> {
    if pvalues.is_empty() {
        return Err("at least one p-value is required".into());
    }
    if !(alpha > 0.0 && alpha <= 1.0) {
        return Err(format!("alpha must be in (0, 1], got {alpha}"));
    }
    if let Some(i) = pvalues.iter().position(|p| p.is_nan() || *p < 0.0) {
        return Err(format!(
            "p-value at index {i} is {}: p-values must be non-negative numbers (NaN is not allowed)",
            pvalues[i]
        ));
    }
    let n = pvalues.len();
    let mut sorted = pvalues.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    for (pos, p) in sorted.iter().enumerate() {
        let threshold = (n - pos) as f64 * alpha / n as f64;
        if *p < threshold {
            return Ok(threshold);
        }
    }
    Ok(alpha / n as f64)
}
