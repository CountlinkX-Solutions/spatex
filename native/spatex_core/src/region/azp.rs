//! AZP (Openshaw 1977; Openshaw and Rao 1995). The basic variant follows `spopt`'s `AZPOrig`
//! with the `AllowMoveAZP` rule: boundary areas move into a randomly chosen region when the
//! pairwise objective does not get worse, until a full pass brings no improvement.
//!
//! The tabu and simulated-annealing variants crash in spopt 0.8.0, so they are defined here from
//! the algorithm description and the intent of spopt's `AZPBasicTabu` and
//! `AZPSimulatedAnnealing`: tabu AZP makes the best move that is not tabu in every iteration
//! (aspiration admits improving tabu moves) and stops once a partition has been visited
//! `repetitions` times; simulated annealing runs basic AZP that also accepts a worse move with
//! probability `exp(-diff / T)` and cools `T` geometrically. Both return the best partition they
//! visit.
//!
//! Beyond spopt, a final deterministic pass guarantees that no single feasible move improves
//! the objective (spopt's loop only approximates this).

use std::collections::{HashMap, VecDeque};

use super::util::{non_contiguous_regions, Adj, Mat, Metric};
use crate::perm::{replicate_rng, uniform_below};
use crate::pointpats::unit;
use rand_chacha::ChaCha8Rng;

/// Options of [`azp`].
#[derive(Clone, Debug)]
pub struct AzpOpts {
    /// Number of regions.
    pub n_clusters: usize,
    /// Pairwise objective metric.
    pub metric: Metric,
    /// RNG seed.
    pub seed: u64,
    /// Starting solution (any labels; renumbered by first appearance).
    pub initial: Option<Vec<u32>>,
    /// Cap on accepted moves (tabu: on iterations).
    pub max_moves: usize,
    /// Local search variant.
    pub strategy: Strategy,
}

/// The local search of [`azp`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Strategy {
    /// spopt's `AZPOrig` with `AllowMoveAZP`.
    Basic,
    /// Best non-tabu move per iteration, FIFO list of the `tabu_length` latest reverse moves,
    /// aspiration for improving tabu moves; stops when a partition is revisited `repetitions`
    /// times or no move is left.
    Tabu {
        tabu_length: usize,
        repetitions: usize,
    },
    /// Basic AZP that also accepts a worse move with probability `exp(-diff / T)`; a run ends
    /// after `sa_moves_term` such moves. `T` starts at `init_temperature` and is multiplied by
    /// `cooling_factor` after every round; the search stops when a partition is revisited
    /// `repetitions` times or after `nonmoving_steps` rounds without a move.
    Annealing {
        init_temperature: f64,
        cooling_factor: f64,
        sa_moves_term: usize,
        repetitions: usize,
        nonmoving_steps: usize,
    },
}

impl Strategy {
    fn validate(&self) -> Result<(), String> {
        match *self {
            Strategy::Basic => Ok(()),
            Strategy::Tabu {
                tabu_length,
                repetitions,
            } => {
                if tabu_length == 0 || repetitions == 0 {
                    Err("tabu_length and repetitions must be at least 1".into())
                } else {
                    Ok(())
                }
            }
            Strategy::Annealing {
                init_temperature,
                cooling_factor,
                sa_moves_term,
                repetitions,
                nonmoving_steps,
            } => {
                if !(init_temperature.is_finite() && init_temperature > 0.0) {
                    Err(format!(
                        "init_temperature must be a positive number, got {init_temperature}"
                    ))
                } else if !(cooling_factor > 0.0 && cooling_factor < 1.0) {
                    Err(format!(
                        "cooling_factor must be in (0, 1), got {cooling_factor}"
                    ))
                } else if sa_moves_term == 0 || repetitions == 0 || nonmoving_steps == 0 {
                    Err("sa_moves_term, repetitions and nonmoving_steps must be at least 1".into())
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// Result of [`azp`].
#[derive(Clone, Debug)]
pub struct AzpOut {
    /// Region of each area, numbered by first appearance.
    pub labels: Vec<u32>,
    /// Final pairwise objective.
    pub objective: f64,
    /// Starting solution.
    pub initial_labels: Vec<u32>,
    /// Number of accepted moves.
    pub moves: usize,
}

/// Runs AZP.
pub fn azp(adj: &Adj, x: &Mat, opts: &AzpOpts) -> Result<AzpOut, String> {
    let n = adj.n();
    let k = opts.n_clusters;
    if n == 0 {
        return Err("the graph is empty".into());
    }
    if k == 0 || k > n {
        return Err(format!("n_clusters must be between 1 and {n}, got {k}"));
    }
    opts.strategy.validate()?;
    let (ncomp, comp) = adj.components();
    if ncomp > k {
        return Err(format!(
            "n_clusters ({k}) is smaller than the number of connected components ({ncomp})"
        ));
    }
    let mut rng = replicate_rng(opts.seed, 0);
    let initial = match &opts.initial {
        Some(given) => validate_initial(adj, given, k)?,
        None => initial_solution(adj, ncomp, &comp, k, &mut rng),
    };
    let mut labels = initial.clone();
    let mut state = State::new(adj, x, opts.metric, &labels);
    let mut moves = 0usize;
    if k > 1 {
        match opts.strategy {
            Strategy::Basic => {
                search(
                    &mut state,
                    &mut labels,
                    &mut rng,
                    &mut moves,
                    opts.max_moves,
                    &mut Rule::Basic,
                );
            }
            Strategy::Tabu {
                tabu_length,
                repetitions,
            } => {
                labels = tabu(
                    &mut state,
                    labels,
                    &mut rng,
                    &mut moves,
                    opts.max_moves,
                    tabu_length,
                    repetitions,
                );
                state = State::new(adj, x, opts.metric, &labels);
            }
            Strategy::Annealing {
                init_temperature,
                cooling_factor,
                sa_moves_term,
                repetitions,
                nonmoving_steps,
            } => {
                labels = annealing(
                    &mut state,
                    labels,
                    &mut rng,
                    &mut moves,
                    opts.max_moves,
                    (init_temperature, cooling_factor),
                    (sa_moves_term, repetitions, nonmoving_steps),
                );
                state = State::new(adj, x, opts.metric, &labels);
            }
        }
        polish(&mut state, &mut labels, &mut moves, opts.max_moves);
    }
    let objective = (0..k)
        .map(|r| super::objective::region_pairwise(x, &state.members[r], opts.metric))
        .sum();
    Ok(AzpOut {
        labels,
        objective,
        initial_labels: initial,
        moves,
    })
}

fn validate_initial(adj: &Adj, given: &[u32], k: usize) -> Result<Vec<u32>, String> {
    if given.len() != adj.n() {
        return Err(format!(
            "initial has {} labels for {} areas",
            given.len(),
            adj.n()
        ));
    }
    let labels = super::util::relabel(given);
    let regions = labels.iter().max().map_or(0, |m| *m as usize + 1);
    if regions != k {
        return Err(format!(
            "initial must contain exactly {k} distinct regions, found {regions}"
        ));
    }
    if !non_contiguous_regions(adj, &labels).is_empty() {
        return Err("every region of initial must be contiguous".into());
    }
    Ok(labels)
}

/// Splits the graph into `k` contiguous regions: regions are distributed over components
/// proportionally to size, then each component's spanning tree is cut `m - 1` times, each
/// time removing the best of five random edges (the one leaving the smallest largest piece).
fn initial_solution(
    adj: &Adj,
    ncomp: usize,
    comp: &[u32],
    k: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<u32> {
    let n = adj.n();
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); ncomp];
    for (i, &c) in comp.iter().enumerate() {
        members[c as usize].push(i as u32);
    }
    // Every component gets one region; the rest go to the component of a random unpicked area.
    let mut regions = vec![1usize; ncomp];
    let mut pool: Vec<u32> = (0..n as u32).collect();
    let mut remaining = k - ncomp;
    while remaining > 0 {
        let at = uniform_below(rng, pool.len() as u64) as usize;
        let area = pool.swap_remove(at);
        let c = comp[area as usize] as usize;
        if regions[c] < members[c].len() {
            regions[c] += 1;
            remaining -= 1;
        }
    }
    let mut labels = vec![0u32; n];
    let mut offset = 0u32;
    for (c, nodes) in members.iter().enumerate() {
        let m = regions[c];
        let mut tree = spanning_tree(adj, nodes);
        for _ in 1..m {
            let mut best: Option<(usize, usize)> = None;
            for _ in 0..5 {
                let e = uniform_below(rng, tree.len() as u64) as usize;
                let largest = largest_piece(n, nodes, &tree, e);
                if best.is_none_or(|(_, b)| largest < b) {
                    best = Some((e, largest));
                }
            }
            let (e, _) = best.unwrap_or((0, 0));
            tree.swap_remove(e);
        }
        let (_, local) = super::util::edge_components(n, &tree);
        let mut ids: std::collections::BTreeMap<u32, u32> = Default::default();
        for &v in nodes {
            let next = ids.len() as u32;
            let id = *ids.entry(local[v as usize]).or_insert(next);
            labels[v as usize] = offset + id;
        }
        offset += ids.len() as u32;
    }
    labels
}

fn spanning_tree(adj: &Adj, nodes: &[u32]) -> Vec<(u32, u32)> {
    let mut dsu = super::util::Dsu::new(adj.n());
    let mut tree = Vec::with_capacity(nodes.len().saturating_sub(1));
    for &i in nodes {
        for &j in &adj.nb[i as usize] {
            if j > i && dsu.union(i, j) {
                tree.push((i, j));
            }
        }
    }
    tree
}

fn largest_piece(n: usize, nodes: &[u32], tree: &[(u32, u32)], skip: usize) -> usize {
    let mut dsu = super::util::Dsu::new(n);
    for (e, &(a, b)) in tree.iter().enumerate() {
        if e != skip {
            dsu.union(a, b);
        }
    }
    let mut count: std::collections::HashMap<u32, usize> = Default::default();
    for &v in nodes {
        *count.entry(dsu.find(v)).or_insert(0) += 1;
    }
    count.values().copied().max().unwrap_or(0)
}

/// Region membership plus the pairwise-objective bookkeeping.
struct State<'a> {
    adj: &'a Adj,
    x: &'a Mat<'a>,
    metric: Metric,
    members: Vec<Vec<u32>>,
}

impl<'a> State<'a> {
    fn new(adj: &'a Adj, x: &'a Mat<'a>, metric: Metric, labels: &[u32]) -> State<'a> {
        let k = labels.iter().max().map_or(0, |m| *m as usize + 1);
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); k];
        for (i, &l) in labels.iter().enumerate() {
            members[l as usize].push(i as u32);
        }
        State {
            adj,
            x,
            metric,
            members,
        }
    }

    /// Sum of distances from `a` to every area of `region`.
    fn cost_to(&self, a: u32, region: usize) -> f64 {
        let ra = self.x.row(a as usize);
        self.members[region]
            .iter()
            .map(|&j| self.metric.dist(ra, self.x.row(j as usize)))
            .sum()
    }

    /// Objective change of moving `a` from its region into `recipient`.
    fn diff(&self, a: u32, donor: usize, recipient: usize) -> f64 {
        self.cost_to(a, recipient) - self.cost_to(a, donor)
    }

    /// Areas outside `region` adjacent to it, ascending.
    fn boundary(&self, region: usize, labels: &[u32]) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for &i in &self.members[region] {
            for &j in &self.adj.nb[i as usize] {
                if labels[j as usize] as usize != region {
                    out.push(j);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// True when `a` can leave its region: it is not the last area and the rest stays connected.
    fn can_leave(&self, a: u32, labels: &[u32]) -> bool {
        let donor = labels[a as usize] as usize;
        let nodes = &self.members[donor];
        if nodes.len() < 2 {
            return false;
        }
        let Some(&start) = self.adj.nb[a as usize]
            .iter()
            .find(|&&m| labels[m as usize] as usize == donor)
        else {
            return false;
        };
        let mut seen = std::collections::HashSet::new();
        seen.insert(start);
        let mut stack = vec![start];
        while let Some(v) = stack.pop() {
            for &w in &self.adj.nb[v as usize] {
                if w != a && labels[w as usize] as usize == donor && seen.insert(w) {
                    stack.push(w);
                }
            }
        }
        seen.len() == nodes.len() - 1
    }

    fn apply(&mut self, a: u32, to: usize, labels: &mut [u32]) {
        let from = labels[a as usize] as usize;
        if let Some(pos) = self.members[from].iter().position(|&v| v == a) {
            self.members[from].swap_remove(pos);
        }
        self.members[to].push(a);
        labels[a as usize] = to as u32;
    }
}

fn relative_eps(scale: f64) -> f64 {
    1e-12 * scale.abs().max(1.0)
}

/// The acceptance rule of [`search`].
enum Rule<'r> {
    /// `diff <= 0` (spopt's `AllowMoveAZP`, ties accepted).
    Basic,
    /// Also a worse move with probability `exp(-diff / temperature)`, counted in `count`; the
    /// run ends once `count` reaches `cap`.
    Annealing {
        temperature: f64,
        cap: usize,
        count: &'r mut usize,
    },
}

impl Rule<'_> {
    fn accept(&mut self, diff: f64, rng: &mut ChaCha8Rng) -> bool {
        if diff <= 0.0 {
            return true;
        }
        match self {
            Rule::Basic => false,
            Rule::Annealing {
                temperature, count, ..
            } => {
                let take = unit(rng) < (-diff / *temperature).exp();
                if take {
                    **count += 1;
                }
                take
            }
        }
    }

    fn exhausted(&self) -> bool {
        match self {
            Rule::Basic => false,
            Rule::Annealing { cap, count, .. } => **count >= *cap,
        }
    }
}

fn total(st: &State) -> f64 {
    (0..st.members.len())
        .map(|r| super::objective::region_pairwise(st.x, &st.members[r], st.metric))
        .sum()
}

/// The randomized AZP passes of spopt under `rule`. Returns whether any move was made.
fn search(
    st: &mut State,
    labels: &mut [u32],
    rng: &mut ChaCha8Rng,
    moves: &mut usize,
    max_moves: usize,
    rule: &mut Rule,
) -> bool {
    let k = st.members.len();
    let first = *moves;
    let mut obj = total(st);
    loop {
        let start = obj;
        let mut pending: Vec<usize> = (0..k).collect();
        while !pending.is_empty() {
            let at = uniform_below(rng, pending.len() as u64) as usize;
            let recipient = pending.swap_remove(at);
            loop {
                let mut cands = st.boundary(recipient, labels);
                let mut moved = false;
                while !cands.is_empty() {
                    let at = uniform_below(rng, cands.len() as u64) as usize;
                    let a = cands.swap_remove(at);
                    if !st.can_leave(a, labels) {
                        continue;
                    }
                    let donor = labels[a as usize] as usize;
                    let diff = st.diff(a, donor, recipient);
                    if rule.accept(diff, rng) {
                        obj += diff;
                        st.apply(a, recipient, labels);
                        *moves += 1;
                        moved = true;
                        break;
                    }
                }
                if !moved || *moves >= max_moves || rule.exhausted() {
                    break;
                }
            }
            if *moves >= max_moves || rule.exhausted() {
                return *moves > first;
            }
        }
        if start - obj <= relative_eps(start) {
            break;
        }
    }
    *moves > first
}

/// Regions adjacent to `a` other than its own, in neighbor order.
fn neighbor_regions(st: &State, a: u32, labels: &[u32]) -> Vec<usize> {
    let own = labels[a as usize] as usize;
    let mut out: Vec<usize> = Vec::new();
    for &j in &st.adj.nb[a as usize] {
        let r = labels[j as usize] as usize;
        if r != own && !out.contains(&r) {
            out.push(r);
        }
    }
    out
}

/// Tabu AZP (`AZPBasicTabu`). Returns the best partition visited.
fn tabu(
    st: &mut State,
    mut labels: Vec<u32>,
    rng: &mut ChaCha8Rng,
    moves: &mut usize,
    max_iterations: usize,
    tabu_length: usize,
    repetitions: usize,
) -> Vec<u32> {
    let n = labels.len() as u32;
    let mut obj = total(st);
    let (mut best, mut best_obj) = (labels.clone(), obj);
    // Reverse moves `(area, from, to)`: moving `area` from `from` to `to` is tabu.
    let mut list: VecDeque<(u32, usize, usize)> = VecDeque::new();
    let mut visited: HashMap<Vec<u32>, usize> = HashMap::new();
    for _ in 0..max_iterations {
        let seen = visited.entry(labels.clone()).or_insert(0);
        *seen += 1;
        let stop = *seen >= repetitions;
        let mut candidate: Option<(f64, u32, usize)> = None;
        for a in 0..n {
            if !st.can_leave(a, &labels) {
                continue;
            }
            let from = labels[a as usize] as usize;
            for to in neighbor_regions(st, a, &labels) {
                if list.contains(&(a, from, to)) {
                    continue;
                }
                let d = st.diff(a, from, to);
                if candidate.is_none_or(|(best_d, _, _)| d < best_d) {
                    candidate = Some((d, a, to));
                }
            }
        }
        let chosen = match candidate {
            Some((d, a, to)) if d <= 0.0 => Some((a, to, d)),
            _ => {
                // Aspiration: an improving tabu move whose area still sits in its old region.
                let aspiring: Vec<(u32, usize, f64)> = list
                    .iter()
                    .filter(|&&(a, from, to)| {
                        labels[a as usize] as usize == from
                            && st.can_leave(a, &labels)
                            && neighbor_regions(st, a, &labels).contains(&to)
                    })
                    .map(|&(a, from, to)| (a, to, st.diff(a, from, to)))
                    .filter(|&(_, _, d)| d < 0.0)
                    .collect();
                if !aspiring.is_empty() {
                    Some(aspiring[uniform_below(rng, aspiring.len() as u64) as usize])
                } else if stop {
                    None
                } else {
                    candidate.map(|(d, a, to)| (a, to, d))
                }
            }
        };
        let Some((a, to, d)) = chosen else {
            break;
        };
        let from = labels[a as usize] as usize;
        st.apply(a, to, &mut labels);
        list.push_back((a, to, from));
        if list.len() > tabu_length {
            list.pop_front();
        }
        *moves += 1;
        obj += d;
        if obj < best_obj - relative_eps(best_obj) {
            (best, best_obj) = (labels.clone(), obj);
        }
    }
    best
}

/// Simulated-annealing AZP (`AZPSimulatedAnnealing`). Returns the best partition visited.
fn annealing(
    st: &mut State,
    mut labels: Vec<u32>,
    rng: &mut ChaCha8Rng,
    moves: &mut usize,
    max_moves: usize,
    (init_temperature, cooling_factor): (f64, f64),
    (sa_moves_term, repetitions, nonmoving_steps): (usize, usize, usize),
) -> Vec<u32> {
    let mut temperature = init_temperature;
    let mut best = labels.clone();
    let mut best_obj = total(st);
    let mut visited: HashMap<Vec<u32>, usize> = HashMap::new();
    let mut nonmoving = 0;
    loop {
        let mut moved = false;
        loop {
            let mut count = 0;
            let mut rule = Rule::Annealing {
                temperature,
                cap: sa_moves_term,
                count: &mut count,
            };
            let changed = search(st, &mut labels, rng, moves, max_moves, &mut rule);
            let exhausted = rule.exhausted();
            moved |= changed;
            let obj = total(st);
            if obj < best_obj - relative_eps(best_obj) {
                (best, best_obj) = (labels.clone(), obj);
            }
            if !changed || exhausted || *moves >= max_moves {
                break;
            }
        }
        let seen = visited.entry(labels.clone()).or_insert(0);
        *seen += 1;
        if *seen >= repetitions || *moves >= max_moves {
            break;
        }
        if !moved {
            nonmoving += 1;
            if nonmoving >= nonmoving_steps {
                break;
            }
        }
        temperature *= cooling_factor;
    }
    best
}

/// Deterministic scan: apply improving single moves until none is left.
fn polish(st: &mut State, labels: &mut [u32], moves: &mut usize, max_moves: usize) {
    let k = st.members.len();
    'again: loop {
        if *moves >= max_moves {
            return;
        }
        let scale: f64 = (0..k)
            .map(|r| super::objective::region_pairwise(st.x, &st.members[r], st.metric))
            .sum();
        let eps = relative_eps(scale);
        for recipient in 0..k {
            for a in st.boundary(recipient, labels) {
                if !st.can_leave(a, labels) {
                    continue;
                }
                let donor = labels[a as usize] as usize;
                if st.diff(a, donor, recipient) < -eps {
                    st.apply(a, recipient, labels);
                    *moves += 1;
                    continue 'again;
                }
            }
        }
        return;
    }
}
