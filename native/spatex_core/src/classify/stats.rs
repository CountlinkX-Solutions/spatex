//! Numeric helpers replicating numpy/scipy summation and percentile semantics.
//!
//! The summation order matters for bit-level parity: numpy's `sum`/`mean` use pairwise
//! summation with eight accumulators for blocks up to 128 elements.

/// Sums like `numpy.add.reduce` on a contiguous `f64` array (pairwise summation).
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for v in a {
            res += *v;
        }
        // numpy starts from -0.0 so an empty sum is -0.0 there; 0.0 is fine for our use.
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i + 8 <= n {
            for (j, slot) in r.iter_mut().enumerate() {
                *slot += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// Arithmetic mean, `numpy.mean`.
pub fn mean(a: &[f64]) -> f64 {
    pairwise_sum(a) / a.len() as f64
}

/// Standard deviation with `ddof` delta degrees of freedom, `numpy.std`.
pub fn std_ddof(a: &[f64], ddof: usize) -> f64 {
    let m = mean(a);
    let sq: Vec<f64> = a
        .iter()
        .map(|v| {
            let d = *v - m;
            d * d
        })
        .collect();
    (pairwise_sum(&sq) / (a.len() - ddof) as f64).sqrt()
}

/// Ascending copy of `a` (inputs are finite; ties keep their value).
pub fn sorted(a: &[f64]) -> Vec<f64> {
    let mut v = a.to_vec();
    v.sort_by(|x, y| x.total_cmp(y));
    v
}

/// `scipy.stats.scoreatpercentile(a, per)` on an already sorted slice (linear interpolation).
pub fn percentile_sorted(s: &[f64], per: f64) -> f64 {
    let idx = per / 100.0 * (s.len() as f64 - 1.0);
    let i = idx as usize;
    if i as f64 == idx {
        s[i]
    } else {
        let w0 = (i + 1) as f64 - idx;
        let w1 = idx - i as f64;
        (s[i] * w0 + s[i + 1] * w1) / (w0 + w1)
    }
}

/// `numpy.median`: the mean of the two middle values for even lengths.
pub fn median(a: &[f64]) -> f64 {
    let s = sorted(a);
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// `numpy.arange(start, stop, step)` for positive `step`: length `ceil((stop-start)/step)`,
/// values `start + i * delta` with `delta = (start + step) - start`, as numpy fills them.
pub fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let len = ((stop - start) / step).ceil();
    if len.is_nan() || len <= 0.0 {
        return Vec::new();
    }
    let len = len as usize;
    let delta = (start + step) - start;
    (0..len)
        .map(|i| {
            if i == 0 {
                start
            } else if i == 1 {
                start + step
            } else {
                start + i as f64 * delta
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_sum_matches_naive_for_exact_inputs() {
        let v: Vec<f64> = (0..300).map(|i| i as f64).collect();
        assert_eq!(pairwise_sum(&v), 44850.0);
        assert_eq!(pairwise_sum(&[]), 0.0);
    }

    #[test]
    fn mean_and_std_ddof1() {
        let y = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert_eq!(mean(&y), 5.0);
        let s = std_ddof(&y, 1);
        assert!((s - (32.0_f64 / 7.0).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn percentile_interpolates_like_scipy() {
        let mut x: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        x.reverse();
        let s = sorted(&x);
        assert_eq!(percentile_sorted(&s, 25.0), 249.75);
        assert_eq!(percentile_sorted(&s, 50.0), 499.5);
        assert_eq!(percentile_sorted(&s, 100.0), 999.0);
        assert_eq!(percentile_sorted(&s, 0.0), 0.0);
    }

    #[test]
    fn median_even_and_odd() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
    }

    #[test]
    fn arange_len_and_values_follow_numpy() {
        // np.arange(0.1, 1.0, 0.3) -> [0.1, 0.4, 0.7]
        let a = arange(0.1, 1.0, 0.3);
        assert_eq!(a.len(), 3);
        assert!((a[2] - 0.7).abs() < 1e-15);
        assert_eq!(arange(5.0, 5.0, 1.0).len(), 0);
    }
}
