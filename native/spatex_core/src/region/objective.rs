//! Objective functions: spopt's pairwise (`ObjectiveFunctionPairwise`) and center
//! (`ObjectiveFunctionCenter`) heterogeneity measures.

use super::util::{Mat, Metric};
use std::collections::BTreeMap;

fn groups(labels: &[u32]) -> BTreeMap<u32, Vec<u32>> {
    let mut by: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (i, &l) in labels.iter().enumerate() {
        by.entry(l).or_default().push(i as u32);
    }
    by
}

/// `sum_r sum_{i<j in r} d(x_i, x_j)`.
pub fn pairwise(x: &Mat, labels: &[u32], metric: Metric) -> Result<f64, String> {
    check(x, labels)?;
    let mut total = 0.0;
    for members in groups(labels).values() {
        total += region_pairwise(x, members, metric);
    }
    Ok(total)
}

/// Pairwise cost inside one region.
pub fn region_pairwise(x: &Mat, members: &[u32], metric: Metric) -> f64 {
    let mut total = 0.0;
    for (a, &i) in members.iter().enumerate() {
        let ri = x.row(i as usize);
        for &j in &members[a + 1..] {
            total += metric.dist(ri, x.row(j as usize));
        }
    }
    total
}

/// `sum_r sum_{i in r} d(x_i, mean(X_r))`.
pub fn center(x: &Mat, labels: &[u32], metric: Metric) -> Result<f64, String> {
    check(x, labels)?;
    let mut total = 0.0;
    for members in groups(labels).values() {
        total += region_center(x, members, metric);
    }
    Ok(total)
}

/// Center cost of one region.
pub fn region_center(x: &Mat, members: &[u32], metric: Metric) -> f64 {
    let mean = x.mean_of(members.iter().copied());
    members
        .iter()
        .map(|&i| metric.dist(x.row(i as usize), &mean))
        .sum()
}

fn check(x: &Mat, labels: &[u32]) -> Result<(), String> {
    if labels.len() != x.n {
        return Err(format!(
            "labels has {} entries but the attributes have {} rows",
            labels.len(),
            x.n
        ));
    }
    Ok(())
}
