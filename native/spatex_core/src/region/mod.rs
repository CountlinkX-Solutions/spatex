//! Regionalization after `spopt` 0.8.0: SKATER, spatially constrained Ward, SA3, Region-K-Means,
//! basic AZP and Max-p, plus the objective helpers they share.
//!
//! SKATER, Ward and Region-K-Means are deterministic ports that reproduce spopt's labels
//! exactly. AZP and Max-p are randomized heuristics: they draw from `ChaCha8` streams derived
//! from `(seed, stream)`, so a seed gives identical results on every run and thread count, but
//! the streams are not numpy's; compare them with spopt by validity and objective only.
//!
//! No dense `n x n` matrix is ever built. Attribute distances are computed on the fly.

pub mod azp;
pub mod kmeans;
pub mod maxp;
pub mod objective;
pub mod random;
pub mod sa3;
pub mod skater;
pub mod spenc;
pub mod spenc_labels;
pub mod util;
pub mod ward;

pub use util::{Adj, Metric};

#[cfg(test)]
mod spenc_tests;
#[cfg(test)]
mod tests;
