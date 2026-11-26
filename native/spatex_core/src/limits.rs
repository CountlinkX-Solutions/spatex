//! Allocation guards: sizes a caller can drive (grid cells, edge lists, class counts, ...) are
//! checked against a limit, with overflow-safe arithmetic, *before* anything is allocated. A
//! failed allocation aborts the whole BEAM, so every user-sized buffer goes through here.

/// Default `max_cells` of the quadrat grid.
pub const DEFAULT_MAX_CELLS: usize = 1_000_000;
/// Default `max_edges` of the weights builders (about 50 bytes per edge at peak).
pub const DEFAULT_MAX_EDGES: usize = 50_000_000;
/// Largest number of distinct classes a Markov chain or spatial Markov accepts.
pub const MAX_CLASSES: usize = 1_000;
/// Largest `m * k * k` of the conditional transition counts of a spatial Markov.
pub const MAX_CONDITIONAL_CELLS: usize = 25_000_000;
/// Largest `permutations * cardinality` index matrix shared by conditional randomization.
pub const MAX_PERMUTATION_ENTRIES: usize = 250_000_000;
/// Largest total number of simulated points (`n_obs * n_sims`).
pub const MAX_SIMULATED_POINTS: usize = 50_000_000;
/// Largest dense `n x n` affinity matrix of Spenc (its eigendecomposition is cubic).
pub const MAX_SPECTRAL_CELLS: usize = 25_000_000;
/// Largest `n * k` dynamic-programming table of the Fisher-Jenks classifier.
pub const MAX_DP_CELLS: usize = 50_000_000;

/// `a * b`, or an error naming `what` when the product overflows `usize`.
pub fn checked_product(what: &str, a: usize, b: usize) -> Result<usize, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("{what}: {a} x {b} overflows the addressable size"))
}

/// Fails with `hint` appended when `count` exceeds `limit`.
pub fn ensure_within(what: &str, count: usize, limit: usize, hint: &str) -> Result<(), String> {
    if count > limit {
        Err(format!(
            "{what}: {count} exceeds the limit of {limit}; {hint}"
        ))
    } else {
        Ok(())
    }
}

/// Fails when a graph builder would produce more than `max_edges` edges, naming the count
/// (`at least` while edges are still being collected) and how to shrink it.
pub fn ensure_edges(what: &str, edges: usize, max_edges: usize, hint: &str) -> Result<(), String> {
    if edges > max_edges {
        Err(format!(
            "{what} would have {edges} edges, above max_edges: {max_edges}; {hint}, \
             or raise max_edges:"
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_product_reports_overflow() {
        assert_eq!(checked_product("cells", 3, 4), Ok(12));
        let err = checked_product("cells", usize::MAX, 2).unwrap_err();
        assert!(err.contains("cells") && err.contains("overflows"), "{err}");
    }

    #[test]
    fn ensure_within_names_count_limit_and_hint() {
        assert!(ensure_within("edges", 5, 5, "h").is_ok());
        let err = ensure_within("edges", 6, 5, "use k:").unwrap_err();
        assert!(err.contains("edges") && err.contains('6') && err.contains('5'));
        assert!(err.contains("use k:"));
    }
}
