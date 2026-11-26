//! Spatial graph in compressed sparse row (CSR) form, mirroring `libpysal.graph.Graph`.
//!
//! Nodes are positions `0..n`. Rows are focal nodes; within a row, neighbors are strictly
//! increasing. A node without any outgoing edge is an island and is stored as a zero-weight
//! self-loop `(i, i, 0.0)`, exactly like libpysal's adjacency table. Weights of exactly zero
//! never count as neighbors (cardinalities, isolates, symmetry), as in libpysal.

/// Weight transformation, as in `libpysal.graph.Graph.transform`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transform {
    /// Original weights.
    O,
    /// Row-standardised: each non-empty row sums to one.
    R,
    /// Binary: non-zero weights become one.
    B,
    /// Double-standardised: all weights sum to one.
    D,
    /// Variance stabilising.
    V,
}

impl Transform {
    /// Parses `"O" | "R" | "B" | "D" | "V"`, case-insensitively.
    pub fn parse(s: &str) -> Result<Transform, String> {
        match s.to_ascii_uppercase().as_str() {
            "O" => Ok(Transform::O),
            "R" => Ok(Transform::R),
            "B" => Ok(Transform::B),
            "D" => Ok(Transform::D),
            "V" => Ok(Transform::V),
            _ => Err(format!(
                "unknown transform {s:?}, expected one of O, R, B, D, V"
            )),
        }
    }
}

/// Immutable CSR adjacency.
#[derive(Clone, Debug, PartialEq)]
pub struct Csr {
    /// Number of nodes.
    pub n: usize,
    /// Row pointers, length `n + 1`.
    pub offsets: Vec<usize>,
    /// Neighbor positions, sorted within each row.
    pub neighbors: Vec<u32>,
    /// Weights aligned with `neighbors`.
    pub weights: Vec<f64>,
}

impl Csr {
    /// Builds a graph over `n` nodes from `(focal, neighbor, weight)` triplets.
    ///
    /// Rows and neighbors are sorted into canonical order. Errors on mismatched lengths,
    /// out-of-range indices, non-finite weights and duplicate `(focal, neighbor)` pairs.
    /// Nodes with no outgoing edge become zero-weight self-loop islands.
    pub fn from_arrays(
        n: usize,
        focal: &[u32],
        neighbor: &[u32],
        weight: &[f64],
    ) -> Result<Csr, String> {
        if u32::try_from(n).is_err() {
            return Err(format!("n = {n} exceeds the u32 index range"));
        }
        if focal.len() != neighbor.len() || focal.len() != weight.len() {
            return Err(format!(
                "focal, neighbor and weight must have equal lengths, got {}, {} and {}",
                focal.len(),
                neighbor.len(),
                weight.len()
            ));
        }
        for (k, ((&f, &t), &w)) in focal.iter().zip(neighbor).zip(weight).enumerate() {
            if f as usize >= n || t as usize >= n {
                return Err(format!(
                    "edge {k}: index ({f}, {t}) out of bounds for {n} nodes"
                ));
            }
            if !w.is_finite() {
                return Err(format!("edge {k}: weight {w} is not finite"));
            }
        }
        let mut order: Vec<usize> = (0..focal.len()).collect();
        order.sort_unstable_by_key(|&k| (focal[k], neighbor[k]));
        if let Some(pair) = order
            .windows(2)
            .find(|p| (focal[p[0]], neighbor[p[0]]) == (focal[p[1]], neighbor[p[1]]))
        {
            let k = pair[0];
            return Err(format!("duplicate edge ({}, {})", focal[k], neighbor[k]));
        }
        let mut offsets = Vec::with_capacity(n + 1);
        let mut neighbors = Vec::with_capacity(order.len() + n);
        let mut weights = Vec::with_capacity(order.len() + n);
        let mut cursor = 0;
        offsets.push(0);
        for i in 0..n {
            let start = cursor;
            while cursor < order.len() && focal[order[cursor]] as usize == i {
                neighbors.push(neighbor[order[cursor]]);
                weights.push(weight[order[cursor]]);
                cursor += 1;
            }
            if cursor == start {
                neighbors.push(i as u32);
                weights.push(0.0);
            }
            offsets.push(neighbors.len());
        }
        Ok(Csr {
            n,
            offsets,
            neighbors,
            weights,
        })
    }

    /// Number of stored entries, including zero-weight island self-loops.
    pub fn nnz(&self) -> usize {
        self.neighbors.len()
    }

    /// Number of edges with a non-zero weight (island self-loops excluded).
    pub fn nonzero_count(&self) -> usize {
        self.weights.iter().filter(|&&w| w != 0.0).count()
    }

    /// Returns `(focal, neighbor, weight)` arrays in canonical order.
    pub fn to_arrays(&self) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
        let mut focal = Vec::with_capacity(self.nnz());
        for i in 0..self.n {
            focal.extend(std::iter::repeat_n(
                i as u32,
                self.offsets[i + 1] - self.offsets[i],
            ));
        }
        (focal, self.neighbors.clone(), self.weights.clone())
    }

    /// Row pointers (length `n + 1`) as `i64`, ready to export as a CSR `indptr`.
    ///
    /// Every stored entry is exported, island self-loops (explicit zeros) included, which
    /// is what libpysal's `Graph.sparse` holds.
    pub fn indptr(&self) -> Vec<i64> {
        self.offsets.iter().map(|&o| o as i64).collect()
    }

    pub(crate) fn row(&self, i: usize) -> std::ops::Range<usize> {
        self.offsets[i]..self.offsets[i + 1]
    }

    /// Weight of edge `(i, j)`, or `None` if it is not stored.
    fn get(&self, i: usize, j: u32) -> Option<f64> {
        let r = self.row(i);
        self.neighbors[r.clone()]
            .binary_search(&j)
            .ok()
            .map(|p| self.weights[r.start + p])
    }

    /// Applies a transformation to the current weights (libpysal semantics: rows or
    /// totals that sum to zero yield zero weights).
    pub fn transform(&self, t: Transform) -> Csr {
        let mut out = self.clone();
        match t {
            Transform::O => {}
            Transform::B => {
                for w in &mut out.weights {
                    *w = if *w != 0.0 { 1.0 } else { 0.0 };
                }
            }
            Transform::R => {
                for i in 0..self.n {
                    let r = self.row(i);
                    let sum: f64 = self.weights[r.clone()].iter().sum();
                    for w in &mut out.weights[r] {
                        *w = if sum != 0.0 { *w / sum } else { 0.0 };
                    }
                }
            }
            Transform::D => {
                let total: f64 = self.weights.iter().sum();
                for w in &mut out.weights {
                    *w = if total != 0.0 { *w / total } else { 0.0 };
                }
            }
            Transform::V => {
                for i in 0..self.n {
                    let r = self.row(i);
                    let norm = self.weights[r.clone()]
                        .iter()
                        .map(|w| w * w)
                        .sum::<f64>()
                        .sqrt();
                    for w in &mut out.weights[r] {
                        *w = if norm != 0.0 { *w / norm } else { 0.0 };
                    }
                }
                let total: f64 = out.weights.iter().sum();
                let scale = if total != 0.0 {
                    self.n as f64 / total
                } else {
                    0.0
                };
                for w in &mut out.weights {
                    *w *= scale;
                }
            }
        }
        out
    }

    /// Spatial lag `(Wy)_i = sum_j w_ij y_j`.
    pub fn lag(&self, y: &[f64]) -> Result<Vec<f64>, String> {
        if y.len() != self.n {
            return Err(format!(
                "y has length {} but the graph has {} nodes",
                y.len(),
                self.n
            ));
        }
        Ok((0..self.n)
            .map(|i| {
                self.row(i)
                    .map(|k| self.weights[k] * y[self.neighbors[k] as usize])
                    .sum()
            })
            .collect())
    }

    /// Number of non-zero-weight neighbors per node.
    pub fn cardinalities(&self) -> Vec<u32> {
        (0..self.n)
            .map(|i| {
                self.weights[self.row(i)]
                    .iter()
                    .filter(|&&w| w != 0.0)
                    .count() as u32
            })
            .collect()
    }

    /// Positions of nodes with no non-zero-weight neighbor, ascending.
    pub fn isolates(&self) -> Vec<u32> {
        self.cardinalities()
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == 0)
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// `S0 = sum_ij w_ij`.
    pub fn s0(&self) -> f64 {
        self.weights.iter().sum()
    }

    /// `S1 = 1/2 sum_ij (w_ij + w_ji)^2`, O(nnz log d).
    pub fn s1(&self) -> f64 {
        let mut acc = 0.0;
        for i in 0..self.n {
            for k in self.row(i) {
                let w = self.weights[k];
                acc += match self.get(self.neighbors[k] as usize, i as u32) {
                    Some(v) => (w + v) * (w + v),
                    // Position (j, i) is absent, so it contributes w_ij^2 on its own.
                    None => 2.0 * w * w,
                };
            }
        }
        0.5 * acc
    }

    /// `S2 = sum_i (sum_j w_ij + sum_j w_ji)^2`.
    pub fn s2(&self) -> f64 {
        let mut total = vec![0.0; self.n];
        for i in 0..self.n {
            for k in self.row(i) {
                total[i] += self.weights[k];
                total[self.neighbors[k] as usize] += self.weights[k];
            }
        }
        total.iter().map(|t| t * t).sum()
    }

    /// True when `w_ij == w_ji` for every pair, up to `tol * max(1, |w_ij|, |w_ji|)`.
    pub fn is_symmetric(&self, tol: f64) -> bool {
        (0..self.n).all(|i| {
            self.row(i).all(|k| {
                let w = self.weights[k];
                if w == 0.0 {
                    return true;
                }
                let v = self
                    .get(self.neighbors[k] as usize, i as u32)
                    .unwrap_or(0.0);
                (w - v).abs() <= tol * 1.0_f64.max(w.abs()).max(v.abs())
            })
        })
    }
}

#[cfg(test)]
mod tests;
