//! Random regions after `spopt` 0.8.0 `RandomRegion` / `RandomRegions`: random partitions of
//! `n` areas into regions, optionally with a number of regions, region sizes (cardinalities)
//! and spatial contiguity.
//!
//! The algorithm is spopt's. Without a graph the areas are shuffled and cut at random breaks.
//! With a graph, regions are grown one at a time from the next candidate area (largest
//! cardinality first); when no region can be completed from the remaining candidates, an area
//! on the border of an existing region is swapped out of it (keeping that region connected)
//! and offered as a candidate. A run that cannot complete is retried with a fresh shuffle up to
//! `maxiter` times; failure is reported with `feasible == false`, as in spopt.
//!
//! Randomness comes from `ChaCha8` streams, so a seed is reproducible on every platform and
//! thread count but does not reproduce numpy's generator. Differences from spopt: the swap loop
//! is bounded by `max_swaps` (spopt can loop forever when an area has no usable neighbor), and
//! the candidate list and membership tests use flags instead of linear list scans.

use std::collections::VecDeque;

use super::util::Adj;
use crate::perm::{replicate_rng, uniform_below};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Options of [`random_region`].
#[derive(Clone, Debug)]
pub struct RandomOpts<'a> {
    /// Number of areas.
    pub n: usize,
    /// Contiguity graph; `None` ignores contiguity.
    pub adj: Option<&'a Adj>,
    /// Number of regions; drawn when `None` and no cardinality is given.
    pub num_regions: Option<usize>,
    /// Region sizes; they must add up to `n`.
    pub cardinality: Option<Vec<usize>>,
    /// Attempts at finding a contiguous solution.
    pub maxiter: usize,
    /// Grow compact regions (contiguity only).
    pub compact: bool,
    /// Cap on swaps per attempt (contiguity only).
    pub max_swaps: usize,
}

/// One random partition.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomOut {
    /// Members of each region (area positions), in construction order; empty when infeasible.
    pub regions: Vec<Vec<u32>>,
    /// Whether a solution was found.
    pub feasible: bool,
    /// Attempts used (1 without a graph).
    pub attempts: usize,
}

fn shuffle<T>(items: &mut [T], rng: &mut ChaCha8Rng) {
    for i in (1..items.len()).rev() {
        let j = uniform_below(rng, i as u64 + 1) as usize;
        items.swap(i, j);
    }
}

/// spopt's `get_region_breaks`: `k - 1` distinct breaks drawn from `1..n-2`, sorted.
fn region_breaks(n: usize, k: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
    let mut pool: Vec<usize> = (1..n.saturating_sub(1)).collect();
    let need = k - 1;
    for i in 0..need {
        let j = i + uniform_below(rng, (pool.len() - i) as u64) as usize;
        pool.swap(i, j);
    }
    let mut breaks = pool[..need].to_vec();
    breaks.sort_unstable();
    breaks
}

fn cards_from_breaks(n: usize, breaks: &[usize]) -> Vec<usize> {
    let mut cards = Vec::with_capacity(breaks.len() + 1);
    let mut start = 0;
    for &b in breaks {
        cards.push(b - start);
        start = b;
    }
    cards.push(n - start);
    cards
}

fn validate(opts: &RandomOpts) -> Result<(), String> {
    let n = opts.n;
    if n == 0 {
        return Err("at least one area is required".into());
    }
    if let Some(adj) = opts.adj {
        if adj.n() != n {
            return Err(format!("the graph has {} nodes for {n} areas", adj.n()));
        }
    }
    if let Some(cards) = &opts.cardinality {
        if cards.contains(&0) {
            return Err("every cardinality must be at least 1".into());
        }
        let total: usize = cards.iter().sum();
        if total != n {
            return Err(format!(
                "number of areas ({n}) does not match cardinality ({total})"
            ));
        }
        if let Some(k) = opts.num_regions {
            if k != cards.len() {
                return Err(format!(
                    "num_regions ({k}) does not match the number of cardinalities ({})",
                    cards.len()
                ));
            }
        }
    } else if let Some(k) = opts.num_regions {
        if k == 0 || k > n {
            return Err(format!("num_regions must be between 1 and {n}, got {k}"));
        }
        if k > 1 && k - 1 > n.saturating_sub(2) {
            return Err(format!(
                "num_regions {k} is too large for random cardinalities: at most {} for {n} areas",
                n.saturating_sub(1)
            ));
        }
    } else if n < 3 {
        return Err("a random number of regions needs at least 3 areas".into());
    }
    if opts.maxiter == 0 {
        return Err("maxiter must be at least 1".into());
    }
    Ok(())
}

/// One random partition drawn from `rng`.
pub fn random_region(opts: &RandomOpts, rng: &mut ChaCha8Rng) -> Result<RandomOut, String> {
    validate(opts)?;
    let n = opts.n;
    let mut ids: Vec<u32> = (0..n as u32).collect();
    shuffle(&mut ids, rng);

    let cards: Vec<usize> = match (&opts.cardinality, opts.num_regions) {
        (Some(c), _) => c.clone(),
        (None, k) => {
            let k = k.unwrap_or_else(|| 2 + uniform_below(rng, (n - 2) as u64) as usize);
            cards_from_breaks(n, &region_breaks(n, k, rng))
        }
    };
    let Some(adj) = opts.adj else {
        let mut regions = Vec::with_capacity(cards.len());
        let mut start = 0;
        for c in cards {
            regions.push(ids[start..start + c].to_vec());
            start += c;
        }
        return Ok(RandomOut {
            regions,
            feasible: true,
            attempts: 1,
        });
    };
    for attempt in 1..=opts.maxiter {
        if let Some(regions) = Builder::new(adj, &ids, &cards, opts).build(rng) {
            return Ok(RandomOut {
                regions,
                feasible: true,
                attempts: attempt,
            });
        }
        shuffle(&mut ids, rng);
    }
    Ok(RandomOut {
        regions: Vec::new(),
        feasible: false,
        attempts: opts.maxiter,
    })
}

/// `permutations` independent partitions; permutation `i` uses ChaCha8 stream `i` of `seed`, so
/// the result does not depend on the thread count and permutation 0 equals
/// `random_region` with stream 0.
pub fn random_regions(
    opts: &RandomOpts,
    permutations: usize,
    seed: u64,
) -> Result<Vec<RandomOut>, String> {
    validate(opts)?;
    (0..permutations as u64)
        .into_par_iter()
        .map(|i| random_region(opts, &mut replicate_rng(seed, i)))
        .collect()
}

/// State of one contiguous construction attempt (spopt's `build_contig_regions` body).
struct Builder<'a> {
    adj: &'a Adj,
    cards: Vec<usize>,
    compact: bool,
    max_swaps: usize,
    num_regions: usize,
    /// Candidate queue with lazy deletion: an entry is live when its stamp is current.
    queue: VecDeque<(u32, u32)>,
    stamp: Vec<u32>,
    is_cand: Vec<bool>,
    n_cand: usize,
    in_region: Vec<bool>,
    in_potential: Vec<bool>,
    regions: Vec<Vec<u32>>,
    area2region: Vec<u32>,
}

impl<'a> Builder<'a> {
    fn new(adj: &'a Adj, ids: &[u32], cards: &[usize], opts: &RandomOpts) -> Self {
        let n = adj.n();
        let mut sorted = cards.to_vec();
        sorted.sort_unstable(); // largest regions are built first (pop from the end)
        let mut b = Builder {
            adj,
            cards: sorted,
            compact: opts.compact,
            max_swaps: opts.max_swaps,
            num_regions: cards.len(),
            queue: VecDeque::with_capacity(n),
            stamp: vec![0; n],
            is_cand: vec![false; n],
            n_cand: 0,
            in_region: vec![false; n],
            in_potential: vec![false; n],
            regions: Vec::new(),
            area2region: vec![u32::MAX; n],
        };
        for &a in ids {
            b.push_candidate(a);
        }
        b
    }

    fn push_candidate(&mut self, a: u32) {
        self.stamp[a as usize] += 1;
        self.is_cand[a as usize] = true;
        self.n_cand += 1;
        self.queue.push_back((a, self.stamp[a as usize]));
    }

    fn remove_candidate(&mut self, a: u32) {
        self.is_cand[a as usize] = false;
        self.n_cand -= 1;
    }

    fn pop_candidate(&mut self) -> Option<u32> {
        while let Some((a, s)) = self.queue.pop_front() {
            if self.is_cand[a as usize] && self.stamp[a as usize] == s {
                self.is_cand[a as usize] = false;
                self.n_cand -= 1;
                return Some(a);
            }
        }
        None
    }

    fn build(mut self, rng: &mut ChaCha8Rng) -> Option<Vec<Vec<u32>>> {
        let mut size_pre = 0usize;
        let mut counter: i64 = -1;
        let mut swap_count = 0usize;
        while self.n_cand > 0 && swap_count < self.max_swaps {
            if size_pre == self.regions.len() {
                counter += 1;
            } else {
                counter = 0;
                size_pre = self.regions.len();
            }
            if counter == self.n_cand as i64 {
                if !self.swap(rng, &mut swap_count) {
                    return None;
                }
                counter = 0;
            }
            let seed = self.pop_candidate()?;
            let test_card = self.cards.pop()?;
            let mut region = vec![seed];
            self.in_region[seed as usize] = true;
            let mut potential: Vec<u32> = Vec::new();
            for &j in &self.adj.nb[seed as usize] {
                if self.is_cand[j as usize] {
                    self.in_potential[j as usize] = true;
                    potential.push(j);
                }
            }
            let mut building = true;
            while building && region.len() < test_card {
                if potential.is_empty() {
                    building = false;
                    self.cards.push(test_card);
                    if let Some(pos) = self.cards.iter().position(|&c| c == region.len()) {
                        self.cards.remove(pos);
                    } else {
                        for &a in &region {
                            self.in_region[a as usize] = false;
                            self.push_candidate(a);
                        }
                        region.clear();
                    }
                } else if self.compact {
                    self.grow_compact(test_card, &mut region, &mut potential, rng);
                } else {
                    self.grow_free(&mut region, &mut potential, rng);
                }
            }
            for &p in &potential {
                self.in_potential[p as usize] = false;
            }
            for &a in &region {
                self.in_region[a as usize] = false;
            }
            if !region.is_empty() {
                let idx = self.regions.len() as u32;
                for &a in &region {
                    self.area2region[a as usize] = idx;
                }
                self.regions.push(region);
            }
        }
        (self.n_cand == 0 && self.regions.len() >= self.num_regions).then_some(self.regions)
    }

    /// Adds neighbors of `area` that are free candidates to `potential`.
    fn extend_potential(&mut self, area: u32, potential: &mut Vec<u32>) {
        for &j in &self.adj.nb[area as usize] {
            let j = j as usize;
            if !self.in_region[j] && !self.in_potential[j] && self.is_cand[j] {
                self.in_potential[j] = true;
                potential.push(j as u32);
            }
        }
    }

    fn take_potential(
        &mut self,
        region: &mut Vec<u32>,
        potential: &mut Vec<u32>,
        at: usize,
    ) -> u32 {
        let a = potential.swap_remove(at);
        self.in_potential[a as usize] = false;
        self.in_region[a as usize] = true;
        region.push(a);
        self.remove_candidate(a);
        a
    }

    /// One area at a time; the frontier is updated after every addition.
    fn grow_free(&mut self, region: &mut Vec<u32>, potential: &mut Vec<u32>, rng: &mut ChaCha8Rng) {
        let at = uniform_below(rng, potential.len() as u64) as usize;
        let a = self.take_potential(region, potential, at);
        self.extend_potential(a, potential);
    }

    /// Exhausts the current frontier before opening a new one.
    fn grow_compact(
        &mut self,
        test_card: usize,
        region: &mut Vec<u32>,
        potential: &mut Vec<u32>,
        rng: &mut ChaCha8Rng,
    ) {
        let mut added = Vec::new();
        while !potential.is_empty() && region.len() < test_card {
            let at = uniform_below(rng, potential.len() as u64) as usize;
            added.push(self.take_potential(region, potential, at));
        }
        for a in added {
            self.extend_potential(a, potential);
        }
    }

    /// Frees an area of an existing region next to a candidate: the candidate takes its place
    /// in that region, so the freed area becomes a candidate. Returns false when the swap budget
    /// is exhausted.
    fn swap(&mut self, rng: &mut ChaCha8Rng, swap_count: &mut usize) -> bool {
        loop {
            *swap_count += 1;
            if *swap_count > self.max_swaps {
                return false;
            }
            let Some(swap_out) = self.pop_candidate() else {
                return false;
            };
            let mut neighs = self.adj.nb[swap_out as usize].clone();
            shuffle(&mut neighs, rng);
            let mut found: Option<(usize, u32)> = None;
            for &join in &neighs {
                if self.is_cand[join as usize] {
                    continue;
                }
                let idx = self.area2region[join as usize] as usize;
                let mut members = self.regions[idx].clone();
                shuffle(&mut members, rng);
                if let Some(&j) = members
                    .iter()
                    .find(|&&j| self.stays_connected(&members, j, swap_out))
                {
                    found = Some((idx, j));
                    break;
                }
            }
            match found {
                Some((idx, swap_in)) => {
                    let region = &mut self.regions[idx];
                    let pos = region.iter().position(|&a| a == swap_in).expect("member");
                    region.remove(pos);
                    region.push(swap_out);
                    self.area2region[swap_in as usize] = u32::MAX;
                    self.area2region[swap_out as usize] = idx as u32;
                    self.push_candidate(swap_in);
                    return true;
                }
                None => self.push_candidate(swap_out),
            }
        }
    }

    /// Whether `members` without `leaver` plus `extra` form one connected set.
    fn stays_connected(&mut self, members: &[u32], leaver: u32, extra: u32) -> bool {
        let set: Vec<u32> = members
            .iter()
            .copied()
            .filter(|&a| a != leaver)
            .chain(std::iter::once(extra))
            .collect();
        // reuse `in_region` as the membership mark; it is all false between regions
        for &a in &set {
            self.in_region[a as usize] = true;
        }
        let mut seen = 1usize;
        let mut stack = vec![set[0]];
        let mut visited = vec![set[0]];
        self.in_region[set[0] as usize] = false;
        while let Some(a) = stack.pop() {
            for &b in &self.adj.nb[a as usize] {
                if self.in_region[b as usize] {
                    self.in_region[b as usize] = false;
                    visited.push(b);
                    stack.push(b);
                    seen += 1;
                }
            }
        }
        for &a in &set {
            self.in_region[a as usize] = false;
        }
        seen == set.len()
    }
}
