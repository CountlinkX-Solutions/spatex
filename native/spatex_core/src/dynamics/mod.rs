//! Spatial dynamics (giddy): Markov chains, spatial Markov and rank mobility.
//!
//! Deterministic quantities follow giddy 2.3.9 to rounding;
//! permutation tests draw from `perm::replicate_rng` and are thread-count independent.

pub mod chain;
pub mod local_tau;
pub mod markov;
pub mod rank;
pub mod spatial;
pub mod special;

#[cfg(test)]
mod tests;
