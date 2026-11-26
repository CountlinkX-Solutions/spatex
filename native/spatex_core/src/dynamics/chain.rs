//! Ergodic summaries of a transition matrix (`giddy.ergodic`, quantecon structure).
//!
//! Matrices are row-major `k x k` slices.

/// Sets the diagonal to 1 on every all-zero row (`util.fill_empty_diagonals`).
pub fn fill_empty_diagonals(p: &mut [f64], k: usize) {
    for i in 0..k {
        if p[i * k..(i + 1) * k].iter().sum::<f64>() == 0.0 {
            p[i * k + i] = 1.0;
        }
    }
}

fn filled(p: &[f64], k: usize) -> Vec<f64> {
    let mut q = p.to_vec();
    fill_empty_diagonals(&mut q, k);
    q
}

/// Communication structure of a chain.
#[derive(Clone, Debug, PartialEq)]
pub struct Structure {
    /// Communication classes (strongly connected components), each and all sorted ascending.
    pub communication: Vec<Vec<usize>>,
    /// Closed communication classes.
    pub recurrent: Vec<Vec<usize>>,
    /// Communication classes that are not closed.
    pub transient: Vec<Vec<usize>>,
    /// States with `p[i][i] == 1`.
    pub absorbing: Vec<usize>,
}

/// Reachability (transitive closure, reflexive) of the positive-probability graph.
fn reachability(p: &[f64], k: usize) -> Vec<bool> {
    let mut r = vec![false; k * k];
    for i in 0..k {
        r[i * k + i] = true;
        for j in 0..k {
            if p[i * k + j] > 0.0 {
                r[i * k + j] = true;
            }
        }
    }
    for m in 0..k {
        for i in 0..k {
            if r[i * k + m] {
                for j in 0..k {
                    if r[m * k + j] {
                        r[i * k + j] = true;
                    }
                }
            }
        }
    }
    r
}

/// Communication structure of `p` (empty rows are filled with absorbing states first).
pub fn structure(p: &[f64], k: usize) -> Structure {
    let p = filled(p, k);
    let reach = reachability(&p, k);
    let mut seen = vec![false; k];
    let mut communication = Vec::new();
    for i in 0..k {
        if seen[i] {
            continue;
        }
        let class: Vec<usize> = (0..k)
            .filter(|&j| reach[i * k + j] && reach[j * k + i])
            .collect();
        for &j in &class {
            seen[j] = true;
        }
        communication.push(class);
    }
    let mut recurrent = Vec::new();
    let mut transient = Vec::new();
    for class in &communication {
        let closed = class
            .iter()
            .all(|&i| (0..k).all(|j| p[i * k + j] <= 0.0 || class.contains(&j)));
        if closed {
            recurrent.push(class.clone());
        } else {
            transient.push(class.clone());
        }
    }
    let absorbing = (0..k).filter(|&i| p[i * k + i] == 1.0).collect();
    Structure {
        communication,
        recurrent,
        transient,
        absorbing,
    }
}

/// Stationary distribution of a closed class by the GTH algorithm.
fn gth(p: &[f64], k: usize, class: &[usize]) -> Vec<f64> {
    let n = class.len();
    let mut a: Vec<f64> = (0..n * n)
        .map(|x| p[class[x / n] * k + class[x % n]])
        .collect();
    for l in (1..n).rev() {
        let s: f64 = (0..l).map(|j| a[l * n + j]).sum();
        for i in 0..l {
            a[i * n + l] /= s;
        }
        for i in 0..l {
            for j in 0..l {
                a[i * n + j] += a[i * n + l] * a[l * n + j];
            }
        }
    }
    let mut x = vec![0.0; n];
    x[0] = 1.0;
    for l in 1..n {
        x[l] = (0..l).map(|i| x[i] * a[i * n + l]).sum();
    }
    let total: f64 = x.iter().sum();
    x.iter().map(|v| v / total).collect()
}

/// Stationary distributions, one row per recurrent class (zero outside the class).
///
/// A single row means the answer is unique (an irreducible chain, or one recurrent class).
/// Empty rows of `p` must be filled by the caller; they error here.
pub fn steady_state(p: &[f64], k: usize) -> Result<Vec<Vec<f64>>, String> {
    check_stochastic(p, k)?;
    let s = structure(p, k);
    Ok(s.recurrent
        .iter()
        .map(|class| {
            let pi = gth(p, k, class);
            let mut row = vec![0.0; k];
            for (&state, v) in class.iter().zip(pi) {
                row[state] = v;
            }
            row
        })
        .collect())
}

fn check_stochastic(p: &[f64], k: usize) -> Result<(), String> {
    if p.len() != k * k {
        return Err(format!(
            "expected a {k} x {k} matrix, got {} values",
            p.len()
        ));
    }
    let empty = (0..k)
        .filter(|&i| p[i * k..(i + 1) * k].iter().sum::<f64>() == 0.0)
        .count();
    if empty > 0 {
        return Err(format!(
            "Input transition probability matrix has {empty} rows full of 0s. Please set \
             fill_empty_classes=true to set diagonal elements for these rows to be 1 to make \
             sure the matrix is stochastic."
        ));
    }
    Ok(())
}

/// Solves `a x = b` by Gaussian elimination with partial pivoting (`a` is `n x n`).
fn solve(a: &mut [f64], b: &mut [f64], n: usize) -> Result<(), String> {
    for c in 0..n {
        let pivot = (c..n)
            .max_by(|&i, &j| a[i * n + c].abs().total_cmp(&a[j * n + c].abs()))
            .unwrap_or(c);
        if a[pivot * n + c].abs() < 1e-14 {
            return Err("singular matrix".to_string());
        }
        if pivot != c {
            for j in 0..n {
                a.swap(c * n + j, pivot * n + j);
            }
            b.swap(c, pivot);
        }
        for i in c + 1..n {
            let f = a[i * n + c] / a[c * n + c];
            if f != 0.0 {
                for j in c..n {
                    a[i * n + j] -= f * a[c * n + j];
                }
                b[i] -= f * b[c];
            }
        }
    }
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|j| a[i * n + j] * b[j]).sum();
        b[i] = (b[i] - s) / a[i * n + i];
    }
    Ok(())
}

/// Inverse by solving against each unit vector.
fn inverse(a: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let mut inv = vec![0.0; n * n];
    for col in 0..n {
        let mut m = a.to_vec();
        let mut e = vec![0.0; n];
        e[col] = 1.0;
        solve(&mut m, &mut e, n)?;
        for row in 0..n {
            inv[row * n + col] = e[row];
        }
    }
    Ok(inv)
}

/// Mean first passage times; `m[i][j]` is the expected steps from `i` to first reach `j`
/// (the recurrence time on the diagonal). `inf` where passage is not certain.
///
/// Irreducible chains use Kemeny's fundamental matrix; reducible ones solve one linear system
/// per destination over the states that reach it with probability one. Empty rows must be
/// filled by the caller.
pub fn mfpt(p: &[f64], k: usize) -> Result<Vec<f64>, String> {
    check_stochastic(p, k)?;
    let s = structure(p, k);
    if s.communication.len() == 1 {
        let class: Vec<usize> = (0..k).collect();
        let pi = gth(p, k, &class);
        // Z = inv(I - P + A) with A[i][j] = pi_j
        let mut a = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                a[i * k + j] = f64::from(u8::from(i == j)) - p[i * k + j] + pi[j];
            }
        }
        let z = inverse(&a, k)?;
        let mut m = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                let d = if pi[j] == 0.0 { 1.0 } else { 1.0 / pi[j] };
                let delta = f64::from(u8::from(i == j));
                m[i * k + j] = (delta - z[i * k + j] + z[j * k + j]) * d;
            }
        }
        return Ok(m);
    }
    let mut out = vec![f64::INFINITY; k * k];
    for d in 0..k {
        // Passage to `d` stops there, so make `d` absorbing and keep the states from which every
        // reachable state still reaches `d`.
        let mut stopped = p.to_vec();
        for j in 0..k {
            stopped[d * k + j] = f64::from(u8::from(j == d));
        }
        let reach = reachability(&stopped, k);
        let sure: Vec<usize> = (0..k)
            .filter(|&i| i != d && (0..k).all(|j| !reach[i * k + j] || reach[j * k + d]))
            .collect();
        let n = sure.len();
        let mut m = vec![f64::INFINITY; k];
        if n > 0 {
            let mut a = vec![0.0; n * n];
            for (r, &i) in sure.iter().enumerate() {
                for (c, &j) in sure.iter().enumerate() {
                    a[r * n + c] = f64::from(u8::from(r == c)) - p[i * k + j];
                }
            }
            let mut b = vec![1.0; n];
            solve(&mut a, &mut b, n)?;
            for (r, &i) in sure.iter().enumerate() {
                m[i] = b[r];
            }
        }
        let mut recurrence = 1.0;
        for j in (0..k).filter(|&j| j != d) {
            if p[d * k + j] > 0.0 {
                recurrence += p[d * k + j] * m[j];
            }
        }
        m[d] = recurrence;
        for i in 0..k {
            out[i * k + d] = m[i];
        }
    }
    Ok(out)
}

/// Expected consecutive periods spent in a state, `1 / (1 - p_ii)` (`inf` when absorbing).
pub fn sojourn_time(p: &[f64], k: usize) -> Vec<f64> {
    let p = filled(p, k);
    (0..k)
        .map(|i| {
            let pii = p[i * k + i];
            if pii == 1.0 {
                f64::INFINITY
            } else {
                1.0 / (1.0 - pii)
            }
        })
        .collect()
}
