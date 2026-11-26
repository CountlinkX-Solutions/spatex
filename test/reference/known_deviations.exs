# Known, accepted differences between Spatex and the PySAL oracle fixtures.
#
# Machine-readable list consumed by the oracle tests. Spatex's deviations from libpysal 4.15.1 are
# listed below. Each entry is a map:
#
#   * `:fixture` - fixture path relative to test/reference/fixtures (e.g. "graph/columbus_queen.json");
#   * `:path`    - dotted path of the differing value inside the fixture (e.g. "transforms.V.lag");
#   * `:tol`     - the looser tolerance that is accepted, or `:skip` when it intentionally differs;
#   * `:reason`  - why Spatex differs (algorithmic, floating-point, upstream bug);
#   * `:doc`     - the docs/parity.md section that explains it.
#
# Differences not listed here fail the suite. Remove an entry once the difference is fixed.
[
  %{
    fixture: "weights/block_contiguity.json",
    path: "pysal_unique_ids",
    tol: :skip,
    reason:
      "libpysal's build_block_contiguity orders nodes regime by regime (unique_ids of regimes " <>
        "[a, b, a, c, b] are [0, 2, 1, 4, 3]); Spatex keeps the input order. Neighbor sets are identical " <>
        "and are compared by id (see Spatex.GraphContiguityOracleTest).",
    doc: "docs/parity.md#block-contiguity"
  },
  %{
    fixture: "weights/contiguity_fuzzy.json",
    path: "near_threshold.pairs (buffer within about 0.15 % below half the pair distance)",
    tol: :skip,
    reason:
      "libpysal buffers every geometry with GEOS, which draws arcs with 16 chords per quadrant, so its " <>
        "buffers are up to 0.12 % smaller than round buffers; Spatex tests the exact distance " <>
        "(distance <= 2 * buffer, equal to the intersection of exact round buffers) and never builds " <>
        "buffers. A pair whose distance is between about 0.9985 and 1 times 2 * buffer can be a " <>
        "neighbor in Spatex and not in libpysal (never the reverse); the bounding box of the buffer of " <>
        "a non-axis-aligned shape is similarly smaller in GEOS. Axis-aligned edges agree exactly " <>
        "(see Spatex.GraphFuzzyContiguityOracleTest, 'near_threshold')",
    doc: "docs/parity.md#fuzzy-contiguity"
  },
  %{
    fixture: "weights/knn_grid_ties.json",
    path: "cases",
    tol: :skip,
    reason:
      "equidistant neighbors: libpysal uses scikit-learn KD-tree order, Spatex lowest index; " <>
        "the oracle test compares per-row neighbor distances instead of ids",
    doc: "docs/parity.md#deviations-knn-and-distance-band"
  },
  %{
    fixture: "weights/kernel_coincident.json",
    path: "cases[quirk=fast_path_drops_coincident]",
    tol: :skip,
    reason:
      "libpysal's fixed-bandwidth sparse fast path calls eliminate_zeros(), which silently drops " <>
        "coincident (distance 0) pairs; Spatex keeps them uniformly, as libpysal does on its dense " <>
        "default-bandwidth and clique paths (see Spatex.GraphKernelOracleTest)",
    doc: "docs/parity.md#deviations-kernel-weights"
  },
  %{
    fixture: "weights/distance_band_coincident.json",
    path: "cases",
    tol: :skip,
    reason:
      "inverse-distance weights of coincident points are Inf in libpysal; Spatex raises " <>
        "ArgumentError for binary: false with negative alpha (binary bands match the oracle)",
    doc: "docs/parity.md#deviations-knn-and-distance-band"
  },
  %{
    fixture: "weights/triangulation_dropped_node.json",
    path: "cases[quirk=drops_tapered_node]",
    tol: :skip,
    reason:
      "with a finite bandwidth libpysal removes a node whose edges are all tapered away from the " <>
        "graph (n shrinks); Spatex keeps it as an island (see Spatex.GraphTriangulationOracleTest)",
    doc: "docs/parity.md#deviations-triangulation"
  },
  %{
    fixture: "classify/*.json",
    path: "cases[method=quantiles].k",
    tol: :skip,
    reason:
      "mapclassify keeps the requested k although ties leave fewer bins; Spatex reports length(bins)",
    doc: "docs/parity.md (mapclassify notes)"
  },
  %{
    fixture: "classify/quantiles_drift.json",
    path: "cases[method=quantiles,drift=true].{bins[-1],yb,counts}",
    tol: :skip,
    reason:
      "mapclassify's last percentile drifts below 100 (k = 23, 36 on 1..46), so max(y) lands in class 0; Spatex forces the top bin to max(y)",
    doc: "docs/parity.md (mapclassify notes)"
  },
  %{
    fixture: "classify/*.json",
    path: "cases[method=natural_breaks].{bins,yb,counts}",
    tol: :skip,
    reason:
      "mapclassify runs sklearn KMeans without a random_state, so its result is not reproducible; Spatex uses seeded k-means++ restarts. Compared statistically: tss <= mapclassify's, and bins identical on well separated clusters (well_separated.json, k = 3)",
    doc: "docs/parity.md (mapclassify notes)"
  },
  %{
    fixture: "classify/*.json",
    path: "cases[method=jenks_caspall|jenks_caspall_forced].warnings",
    tol: :skip,
    reason:
      "mapclassify emits a varying number of internal quantile and numpy warnings; Spatex reports one quantile warning",
    doc: "docs/parity.md (mapclassify notes)"
  },
  %{
    fixture: "classify/*.json",
    path: "skipped[method=jenks_caspall]",
    tol: :skip,
    reason:
      "when a class empties, mapclassify takes the NaN median of an empty class and either exhausts memory or raises SystemError (probed on edge_dups, edge_small, edge_two); the generator records these as skipped. Spatex stops the iteration and returns the non-empty classes",
    doc: "docs/parity.md (mapclassify notes)"
  },
  %{
    fixture: "esda/moran_constant.json",
    path: "pysal.{i,z_norm,p_sim}",
    tol: :skip,
    reason:
      "PySAL returns NaN statistics (and p_sim = 1/(P+1)) for a constant y; Spatex raises ArgumentError",
    doc: "docs/parity.md (esda notes)"
  },
  %{
    fixture: "esda/join_counts_*.json",
    path: "cases[*].perm.sim_autocorr_chi2",
    tol: :skip,
    reason:
      "esda divides both squared deviations of sim_autocorr_chi2 by mean(sim_autocurr_pos)^2, which is not a " <>
        "chi-square statistic; Spatex reports the Pearson statistic with each cell's own mean " <>
        "(see Spatex.ESDAGearyGJoinsOracleTest, 'known deviation')",
    doc: "docs/parity.md (esda notes)"
  },
  %{
    fixture: "esda/geary_islands.json",
    path: "graph_crash",
    tol: :skip,
    reason:
      "esda 2.10.0 Geary raises ValueError (shape mismatch) for a libpysal Graph with islands because the " <>
        "zero-weight self-loop is kept in .data but dropped by nonzero(); the oracle values come from the W " <>
        "object, and Spatex computes islands directly",
    doc: "docs/parity.md (esda notes)"
  },
  %{
    fixture: "esda/g_complete.json",
    path: "pysal.{vg,z_norm,p_norm}",
    tol: :skip,
    reason:
      "a complete graph gives a variance of rounding noise (-2e-16 in esda); Spatex treats a variance at or " <>
        "below 4 eps EG^2 as non-positive and reports NaN z_norm and p_norm, as esda does for that input",
    doc: "docs/parity.md (esda notes)"
  },
  %{
    fixture: "(no fixture)",
    path: "G y < 0; Join_Counts y not in {0, 1}; constant or all-zero y",
    tol: :skip,
    reason:
      "esda does not validate these inputs and returns meaningless or NaN statistics; Spatex raises " <>
        "ArgumentError (see Spatex.ESDAGearyGJoinsTest 'validates its input')",
    doc: "docs/parity.md (esda notes)"
  },
  %{
    fixture: "esda/moran_*.json",
    path: "cases[*].perm",
    tol: :skip,
    reason:
      "permutation draws come from ChaCha8 streams, not numpy's global RNG; only the distribution matches, so the oracle tests use binomial tolerance bands",
    doc: "docs/parity.md (esda notes)"
  },
  %{
    fixture: "esda/local_moran_constant.json",
    path: "pysal.{is,z}",
    tol: :skip,
    reason: "PySAL returns NaN local statistics for a constant y; Spatex raises ArgumentError",
    doc: "docs/parity.md (local esda notes)"
  },
  %{
    fixture: "esda/local_moran_*.json",
    path: "cases[*].perm",
    tol: :skip,
    reason:
      "conditional randomization draws come from ChaCha8 streams, not numba's Mersenne Twister, and p_sim is f64 where esda stores float32; only the distribution matches, so the oracle tests use binomial bands. On graphs with a handful of nodes (perm_comparable: false) the reference distribution is a few discrete values and PySAL's p_sim depends on floating-point ties",
    doc: "docs/parity.md (local esda notes)"
  },
  %{
    fixture: "esda/local_geary_islands.json",
    path: "pysal_default",
    tol: :skip,
    reason:
      "Geary_Local with islands and drop_islands=True returns localG of length n - #islands next to a p_sim of length n; Spatex returns n entries with island local_g 0",
    doc: "docs/parity.md (local Geary notes)"
  },
  %{
    fixture: "esda/local_joins_islands.json",
    path: "pysal_graph_crash, pysal_island_selfpair, pysal_constructor_permutations",
    tol: :skip,
    reason:
      "Join_Counts_Local raises IndexError on a Graph with islands, counts an island with y = 1 as its own join on a W, and fit() ignores the constructor's permutations; Spatex gives islands LJC 0 and has a single permutations option",
    doc: "docs/parity.md (local Geary notes)"
  },
  %{
    fixture: "esda/losh_islands.json",
    path: "pysal_islands",
    tol: :skip,
    reason:
      "LOSH turns every Hi NaN as soon as one site has no neighbors; Spatex computes over the sites with neighbors (islands NaN)",
    doc: "docs/parity.md (local Geary notes)"
  },
  %{
    fixture: "esda/{local_geary,local_joins}_*.json",
    path: "cases[*].p_sim",
    tol: :skip,
    reason:
      "conditional randomization draws come from ChaCha8 streams, not numba's Mersenne Twister, and p_sim is f64 where esda stores float32; only the distribution matches (binomial bands, wider for two-sided)",
    doc: "docs/parity.md (local Geary notes)"
  },
  %{
    fixture: "esda/g_local_*.json",
    path: "cases[*].perm",
    tol: :skip,
    reason:
      "conditional randomization draws come from ChaCha8 streams, not numba's Mersenne Twister, and p_sim is f64 where esda stores float32; only the distribution matches, so the oracle tests use binomial bands (an absolute bound for two-sided and folded p-values, which hinge on a sampled median/mean)",
    doc: "docs/parity.md (local Getis-Ord notes)"
  },
  %{
    fixture: "(no fixture)",
    path: "G_Local y < 0 or sum(y) == 0; fdr with NaN, negative p-values or alpha outside (0, 1]",
    tol: :skip,
    reason:
      "esda does not validate these inputs and returns meaningless or NaN values; Spatex raises ArgumentError (see Spatex.ESDAGLocalTest)",
    doc: "docs/parity.md (local Getis-Ord notes)"
  },
  %{
    fixture: "dynamics/markov.json",
    path: "cases[label=reducible].out.mfpt",
    tol: :skip,
    reason:
      "giddy.mfpt raises LinAlgError on a reducible chain with a recurrent cycle that does not contain the destination; Spatex returns the exact passage time where it is certain and infinity elsewhere (see Spatex.DynamicsOracleTest)",
    doc: "docs/parity.md#deviations-dynamics"
  },
  %{
    fixture: "dynamics/spatial_markov.json",
    path: "cases[*].out.f_cond",
    tol: :skip,
    reason:
      "for a reducible conditional matrix whose linear system is singular but not rejected, giddy returns meaningless first passage times (for example -5.3e15) where Spatex returns infinity; the oracle test replaces such expected values by infinity",
    doc: "docs/parity.md#deviations-dynamics"
  },
  %{
    fixture: "dynamics/spatial_markov.json",
    path: "cases[permutations>0].out.lclass_ids",
    tol: :skip,
    reason:
      "giddy's permutation loop overwrites self.lclass_ids, leaving the lag classes of the last permuted replicate; Spatex keeps the observed ones. Also, with discrete=True giddy shuffles y but lags the unshuffled classes (every replicate equals the observation); Spatex shuffles the class rows",
    doc: "docs/parity.md#deviations-dynamics"
  },
  %{
    fixture: "dynamics/spatial_markov.json",
    path: "cases[label=synthetic discrete].out.lclass_ids",
    tol: :skip,
    reason:
      "libpysal's lag_categorical breaks ties that survive the focal-label retry at random; Spatex picks the lowest class and the generator patches libpysal to the same rule",
    doc: "docs/parity.md#deviations-dynamics"
  },
  %{
    fixture: "dynamics/lisa_markov.json",
    path: "cases[*].out.chi_2, cases[*].out.expected_t",
    tol: :skip,
    reason:
      "giddy builds the marginal below-mean chains from a (t, n) array, so they count moves between neighboring location columns; Spatex uses the (n, t) orientation. The fixture stores the corrected statistic as chi_2_intended / expected_t_intended and giddy's own value as chi_2 / expected_t",
    doc: "docs/parity.md#deviations-dynamics"
  },
  %{
    fixture: "pointpats/centrography.json",
    path: "cases[*].euclidean_median",
    tol: :skip,
    reason:
      "pointpats minimizes dtot with scipy BFGS (gtol 1e-5), which stops short of the optimum (about 1e-4 relative); Spatex runs Weiszfeld with the Vardi-Zhang step to 1e-12. The oracle test asserts dtot(spatex) <= dtot(pysal) + 1e-9 and coordinates within 1e-3 of the standard distance (see Spatex.PointPatsCentrographyTest)",
    doc: "docs/parity.md (pointpats notes)"
  },
  %{
    fixture: "pointpats/centrography.json",
    path: "cases[c200,diag,r500].mbc",
    tol: :skip,
    reason:
      "pointpats' Skyum loop returns a circle that is not minimal on some inputs (c200: radius 47.256 vs 47.081; diag: 19.85 vs 2.9) and raises ZeroDivisionError on r500; Spatex returns the true minimum circle (verified by brute force over hull vertices) and equals pointpats wherever pointpats is correct",
    doc: "docs/parity.md (pointpats notes)"
  },
  %{
    fixture: "pointpats/centrography.json",
    path: "cases[cross].mrr.{angle,corners}",
    tol: :skip,
    reason:
      "the rotated rectangle of a rhombus has four equal-area solutions; GEOS and Spatex may report different ones (the area matches). Corner order and start corner differ from GEOS in general, so corners are compared as sets",
    doc: "docs/parity.md (pointpats notes)"
  },
  %{
    fixture: "pointpats/simulation.json",
    path: "cases[*].{f,j_untruncated}.statistic and every simulation array",
    tol: :skip,
    reason:
      "pointpats draws unseeded numpy randomness (empty-space points, CSR patterns); Spatex uses ChaCha8 streams per (seed, simulation). Observed G/K/L equal pointpats exactly; F/J observed values and all simulations are compared statistically (see Spatex.PointPatsDistanceTest)",
    doc: "docs/parity.md (pointpats notes)"
  },
  %{
    fixture: "pointpats/processes.json",
    path: "normal[*] and cluster_normal[*] moments and nearest-neighbour summaries",
    tol: :skip,
    reason:
      "pointpats draws unseeded numpy randomness; Spatex uses ChaCha8 streams per (seed, replication). Sample moments and nearest-neighbour distances are compared statistically (see Spatex.PointPatsProcessesTest)",
    doc: "docs/parity.md (pointpats notes)"
  },
  %{
    fixture: "(no fixture)",
    path: "normal with a matrix cov of unequal variances",
    tol: :skip,
    reason:
      "pointpats builds the correlation matrix as (1/sd) * cov * (1/sd) with numpy broadcasting, which divides the off-diagonal entries by the wrong standard deviations; it is correct only for equal variances. Spatex samples the exact covariance (Cholesky factor)",
    doc: "docs/parity.md (pointpats notes)"
  },
  %{
    fixture: "(no fixture)",
    path: "cluster_normal with cov nil, replications after the first",
    tol: :skip,
    reason:
      "pointpats assigns the default covariance (squared mean distance between the centres) to `cov` on the first replication and reuses it for all later ones; Spatex recomputes it per replication from that replication's centres, so replications stay independent",
    doc: "docs/parity.md (pointpats notes)"
  },
  %{
    fixture: "pointpats/processes.json",
    path: "strauss[*] close-pair and nearest-neighbour summaries",
    tol: :skip,
    reason:
      "pointpats' birth-death sampler uses unseeded numpy randomness; Spatex uses ChaCha8 streams per (seed, replication) and counts the pairs a moved point forms instead of rebuilding two KD-trees per step. Summaries are compared statistically (see Spatex.PointPatsProcessesTest)",
    doc: "docs/parity.md (pointpats notes)"
  },
  %{
    fixture: "(no fixture)",
    path: "strauss initial pattern size on windows whose area is not 1",
    tol: :skip,
    reason:
      "pointpats starts from poisson(hull, intensity=int(1.5 n)), which draws int(area * 1.5 n) points; Spatex starts from 1.5 n points whatever the area. The stationary distribution is the same; an impossible request raises ArgumentError where pointpats raises RuntimeError",
    doc: "docs/parity.md (pointpats notes)"
  },
  %{
    fixture: "(no fixture)",
    path: "j_test with truncate: true and a finite J",
    tol: :skip,
    reason:
      "pointpats' j_test falls through and returns None when nothing is truncated; Spatex returns the result. With supports ending at the maximum nearest-neighbour distance this cannot occur (J is 0/0 at the last distance)",
    doc: "docs/parity.md (pointpats notes)"
  },
  %{
    fixture: "(no fixture)",
    path:
      "alpha_shape with alpha = 0 or a negative alpha; collinear or fewer than three distinct points",
    tol: :skip,
    reason:
      "libpysal divides by alpha (ZeroDivisionError for a Python float 0.0, an empty GeoSeries for a negative alpha) and lets Qhull raise QhullError on collinear input; Spatex treats alpha = 0 as the convex hull and raises ArgumentError for a negative or non-finite alpha and for degenerate point sets (see Spatex.PointPatsAlphaShapeTest)",
    doc: "docs/parity.md#alpha-shape"
  },
  %{
    fixture: "(no fixture)",
    path:
      "alpha_shape_auto with fewer than three or collinear points; no valid first candidate; step < 1",
    tol: :skip,
    reason:
      "libpysal returns an empty polygon for one or two points, a LineString for three collinear points, lets Qhull raise on four or more collinear ones, falls back to the first polygon of the all-triangles shape (UnboundLocalError for the radius) when `radius < 1 / (1 / radius - EPS)` drops the largest triangle and leaves a point uncovered, and accepts any step; Spatex raises ArgumentError for all of these (see Spatex.PointPatsAlphaShapeAutoTest, spatex_core alpha_tests)",
    doc: "docs/parity.md#alpha-shape"
  },
  %{
    fixture: "pointpats/alpha_shape_auto.json",
    path: "cases[*].variants",
    tol: :skip,
    reason:
      "the automatic search compares radius < 1 / (1 / radius - EPS), which depends on the last bit of the radius, and Qhull's vertex order inside a triangle changes that bit; the fixture stores libpysal's result for all six vertex orders and Spatex must equal one of them (usually there is only one)",
    doc: "docs/parity.md#alpha-shape"
  },
  %{
    fixture: "pointpats/quadrat.json",
    path: "cases[*].pysal_num",
    tol: :skip,
    reason:
      "HexagonM.num swaps the odd and even row counts and can disagree with the real number of cells (lh = 35 on R30: num 7, 8 cells); Spatex exposes num_cells only (see Spatex.PointPatsQuadratTest)",
    doc: "docs/parity.md (pointpats notes)"
  },
  %{
    fixture: "pointpats/quadrat.json",
    path: "realizations[*]",
    tol: :skip,
    reason:
      "quadrat realizations are CSR draws from ChaCha8 streams, not numpy; the Monte Carlo statistic distribution and p-value are compared within sampling bands",
    doc: "docs/parity.md (pointpats notes)"
  },
  %{
    fixture: "(no fixture)",
    path:
      "poisson / cluster_poisson shape squeeze; std_distance with NaN weights; n < 3 ellipse; collinear hull",
    tol: :skip,
    reason:
      "pointpats squeezes a one-point pattern to shape (2,), performs no input checks (NaN, zero weight sums, ellipse with n = 2 returns inf) and Qhull raises on collinear input; Spatex always returns a list of patterns and raises ArgumentError for invalid input (see Spatex.PointPatsRandomTest, Spatex.PointPatsCentrographyTest)",
    doc: "docs/parity.md (pointpats notes)"
  },
  %{
    fixture: "region/skater.json",
    path: "duplicates.case",
    tol: :skip,
    reason:
      "spopt removes links between areas with identical attribute vectors (eliminate_zeros on the dissimilarity), splitting the graph into false islands and silently raising n_clusters (3x3 lattice with duplicate values: 5 regions for k = 3). Spatex keeps those links by default; zero_dissimilarity: :drop reproduces spopt (see Spatex.RegionOracleTest)",
    doc: "docs/parity.md#spopt-notes"
  },
  %{
    fixture: "region/skater.json",
    path: "cases[dissimilarity=euclidean].msf_edges",
    tol: :skip,
    reason:
      "sklearn's euclidean_distances uses the x^2 + y^2 - 2xy expansion whose rounding noise can reorder near-equal links in the spanning forest; labels and score still match",
    doc: "docs/parity.md#spopt-notes"
  },
  %{
    fixture: "(no fixture)",
    path: "region_kmeans without a seed in every connected component",
    tol: :skip,
    reason:
      "spopt's growth loop never terminates when a component has no seed; Spatex raises ArgumentError. A would-be-empty source region is treated as a non-move where spopt's networkx call raises",
    doc: "docs/parity.md#spopt-notes"
  },
  %{
    fixture: "(no fixture)",
    path:
      "Sel_BW adaptive search with n < 40 + 2 * (columns + 1); bw < 2; bw > n; unsupported kernel",
    tol: :skip,
    reason:
      "mgwr crashes inside numpy (kth out of bounds at the first evaluation, silent wrap-around for bw <= 0, NaN weights for bw = 1) or prints 'Unsupported kernel function' and fails later; Spatex raises ArgumentError up front (see Spatex.GWRTest)",
    doc: "docs/parity.md#notes-gwr"
  },
  %{
    fixture: "(no fixture)",
    path: "Sel_BW search(bw_max=np.inf)",
    tol: :skip,
    reason:
      "mgwr's bw_max=np.inf (global model check) is spelled include_global: true in Spatex and cannot be combined with a numeric bw_max; the search history matches mgwr (see Spatex.GWROracleTest)",
    doc: "docs/parity.md#notes-gwr"
  },
  %{
    fixture: "regression/gwr_search.json",
    path: "scipy.* (Sel_BW.search(search_method='scipy') with AICc, AIC or BIC)",
    tol: :skip,
    reason:
      "with NumPy 2.5 mgwr 2.2.1's scipy search crashes (TypeError: only 0-dimensional arrays can be converted to Python scalars, from int(bw) on the shape-(1,) array its AICc/AIC/BIC criteria return); only CV works. The fixture generator wraps the criteria to return plain floats, which restores the behavior of older NumPy, so Spatex is compared with mgwr's algorithm and not with a crash. A fixed bandwidth found by this search agrees to 1e-6 relative (the criterion is flat to rounding there, so SciPy's parabolic/golden branches can flip and the evaluation count differs by a few calls); adaptive ones agree to 1e-9 with the same count (see Spatex.GWRSearchOracleTest)",
    doc: "docs/parity.md#notes-gwr"
  },
  %{
    fixture: "(no fixture)",
    path: "Sel_BW.search(search_method='scipy', bw_max=np.inf); equal bounds",
    tol: :skip,
    reason:
      "mgwr silently ignores bw_max=np.inf in the scipy search; Spatex has no include_global there and raises ArgumentError for include_global: true. Equal lower and upper bounds raise in both (see Spatex.GWRHatSearchTest)",
    doc: "docs/parity.md#notes-gwr"
  },
  %{
    fixture: "regression/gwr_glm.json",
    path: "fits.poisson_zeros.{deviance,resid_deviance,p_dev}",
    tol: :skip,
    reason:
      "mgwr evaluates 0 * ln(0) for a zero Poisson count in the local deviance and returns NaN for deviance and pDev (the fixture stores NaN); Spatex uses the convention 0 ln 0 = 0 and returns finite values (see Spatex.GWRGLMOracleTest)",
    doc: "docs/parity.md#notes-glm-gwr-and-mgwr"
  },
  %{
    fixture: "(no fixture)",
    path: "MGWR fields influ, std_res, cooks_d, cv, local R2, MGWR.exact_fit",
    tol: :skip,
    reason:
      "mgwr raises TypeError/NotImplementedError for these on MGWRResults (and MGWR.exact_fit ignores the kernel and fixed settings); Spatex returns nil for the fields and does not port exact_fit. MGWR bandwidth bounds are per column (multi_bw_min/max); the outer bw_min/bw_max do not exist",
    doc: "docs/parity.md#notes-glm-gwr-and-mgwr"
  },
  %{
    fixture: "(no fixture)",
    path: "GWR.predict with more prediction points than calibration points; predy, influ, resid",
    tol: :skip,
    reason:
      "mgwr's local fit reads the calibration row X[i] and y[i] at prediction point i, so it raises IndexError for more prediction points than observations and its predy/influ/residuals at prediction points mix rows. Spatex predicts at any number of points and exposes only params, bse, tvalues, predictions and local_r2 (see Spatex.GWRPredictTest)",
    doc: "docs/parity.md#notes-prediction-and-local-collinearity"
  },
  %{
    fixture: "(no fixture)",
    path:
      "local_collinearity with fewer than two explanatory variables or a singular local correlation matrix",
    tol: :skip,
    reason:
      "mgwr crashes inside np.cov/corr (ValueError: Input must be 1- or 2-d) with one explanatory variable and raises LinAlgError for an exactly singular local correlation matrix; Spatex raises ArgumentError (see Spatex.GWRPredictTest)",
    doc: "docs/parity.md#notes-prediction-and-local-collinearity"
  },
  %{
    fixture: "(no fixture)",
    path: "local condition-number warning",
    tol: :skip,
    reason:
      "mgwr 2.2.1 has no condition-number warning (warnings.warn exists only for the deprecated pool argument and the summary never reports it); the local condition number is only an output of local_collinearity(), so Spatex adds none",
    doc: "docs/parity.md#notes-gwr"
  },
  %{
    fixture: "regression/ml_ord_deviation.json",
    path: "cases[method=ord].{rho,lam,logll,betas,vm,vm1,std_err,z_stat,...}",
    tol: :skip,
    reason:
      "spreg's ML_Lag/ML_Error method='ord' takes the eigenvalues of symmetrize(w), the row-standardized BINARY pattern, whenever the binary pattern of W is symmetric: wrong for non-binary weights (inverse distance: rho 0.47861 vs the correct 0.47705). Spatex uses the eigenvalues of W itself (symmetric trick only for 1/card weights); see Spatex.RegressionMLOracleTest 'ord deviation'",
    doc: "docs/parity.md#spreg-ml-notes"
  },
  %{
    fixture: "regression/ml_{columbus,synthetic}.json",
    path: "cases[method=ord,pysal_symmetrized=true].{vm,vm1,std_err,z_stat}",
    tol: :skip,
    reason:
      "spreg's ord path also substitutes the symmetrized matrix for W in the asymptotic variance, so its standard errors differ from its own full method even where rho is right (queen ML lag: std_err[rho] 0.1712 vs 0.1669). Spatex's variance always uses the real W; the oracle test compares Spatex's ord to spreg's full for these cases",
    doc: "docs/parity.md#spreg-ml-notes"
  },
  %{
    fixture: "regression/ml_*.json",
    path: "cases[*].{rho,lam} (method=lu; spatial error fits)",
    tol: 1.0e-7,
    reason:
      "the bounded Brent search ends within xatol = 1e-7 of spreg's value when a one-ulp difference in a function value (BLAS/LAPACK order of operations) flips a late comparison; spreg's own full/lu/ord methods differ by the same amount. Derived quantities are verified at 1e-9 at spreg's own rho (at-parameter tests); logll agrees at 1e-9",
    doc: "docs/parity.md#spreg-ml-notes"
  },
  %{
    fixture: "(no fixture)",
    path:
      "ML_Lag/ML_Error with n so large that numpy's det over/underflows; unsupported method names",
    tol: :skip,
    reason:
      "numpy computes det as a product, so ln(det) is inf or -inf for large n; Spatex sums the logarithms. spreg prints a message and returns a broken object (ML_Lag) or raises a bare Exception (ML_Error) for an unknown method; Spatex raises ArgumentError (see Spatex.RegressionMLTest). `method: :lu` is evaluated with the dense LU",
    doc: "docs/parity.md#spreg-ml-notes"
  },
  %{
    fixture: "regression/gm_{columbus,synthetic,clamp}.json",
    path: "cases[GMM models].{lam, betas, vm, std_err, z_stat, u, predy, ...}",
    tol: 5.0e-6,
    reason:
      "spreg minimizes the GMM moment objective with scipy's L-BFGS-B (forward-difference gradient, factr 1e7), which stops early: lambda is up to ~5e-6 from the exact minimizer (KP98 columbus 5.7e-7). Spatex finds the exact minimizer on [-0.99, 0.99] (1-D grid walk from 0 plus bounded Brent, sigma^2 profiled out for KP98). Derived quantities move with lambda (betas up to ~100x the gap, compared at 5e-4); every formula is verified at 1e-9 by feeding spreg's own lambdas (Spatex.Regression.gm_error_at/5, gm_combo_at/5)",
    doc: "docs/parity.md#spreg-gm-notes"
  },
  %{
    fixture: "regression/gm_optimizer.json",
    path: "cases[gm_error rook_binary, gm_endog_error_het, gm_error_hom rook_binary].lambdas",
    tol: :skip,
    reason:
      "spreg's L-BFGS-B misses the minimizer of its own objective: with binary rook weights the KP98 objective is flat (lambda 0.106667 vs the exact 0.106268, objective 1.858e-6 vs 1.856e-6) and the Hom weighted step stops 1.6e-5 early; GM_Endog_Error_Het's first step ends at 0.8227 where the objective is increasing (minimum at 0.3665, objective 0.028 vs 1.29: abnormal line-search exit). Spatex returns the exact minimizer; at spreg's lambdas everything agrees at 1e-9",
    doc: "docs/parity.md#spreg-gm-notes"
  },
  %{
    fixture: "regression/gm_clamp.json",
    path: "cases[gm_endog_error_het/step1c3].{betas, vm, u, lambdas}",
    tol: :skip,
    reason:
      "spreg's GM_Endog_Error_Het clamps tsls.betas[-1] to +-0.9 in the first stage and every iteration. The clamp is meant for rho (the last column of Z in GM_Combo_Het) but applies to whichever endogenous regressor comes last: HOVAL (coefficient -1.35 unclamped) is pinned at -0.9. Spatex clamps only in the combo models (replicated and tested on synthetic200/gm_combo_het/rho_clamp); the '/unclamped' twin is spreg run with the clamp disabled and matches Spatex at 1e-9",
    doc: "docs/parity.md#spreg-gm-notes"
  },
  %{
    fixture: "(no fixture)",
    path: "optim_moments with a zero moment scale; GMM_Error dispatcher; slx_lags, regimes",
    tol: :skip,
    reason:
      "when min(G, g) is exactly 0 spreg divides by it (NaN objective, L-BFGS-B returns its start 0.0 silently); Spatex returns lambda = 0 and a warning. spreg's GMM_Error dispatcher is replaced by the estimator: option of gm_error/4 and gm_combo/4; inv_method 'true_inv', slx_lags and regimes are not ported",
    doc: "docs/parity.md#spreg-gm-notes"
  },
  %{
    fixture: "regression/tsls_regimes.json",
    path: "cases[chow_layout_bug=true].{chow_joint, chow_regi}",
    tol: :skip,
    reason:
      "spreg's Chow restrictions (buildR1var) put the global columns last, but TSLS_Regimes orders the coefficients x regime blocks, x globals, yend regime blocks, yend globals. With global exogenous columns and regime-specific endogenous ones, spreg's endogenous and joint Chow statistics compare other coefficients. Spatex builds the restrictions on the real layout by default (values from spreg's wald_test on a corrected R, stored as chow_*) and reproduces spreg with spreg_compatible: true (spreg_chow_*); both are tested in Spatex.RegressionTSLSRegimesOracleTest",
    doc: "docs/parity.md#spreg-regimes-notes"
  },
  %{
    fixture: "regression/gm_lag_regimes.json",
    path: "cases[chow_layout_bug=true].{chow_joint, chow_regi}",
    tol: :skip,
    reason:
      "GM_Lag_Regimes shares TSLS_Regimes's Chow restrictions (buildR1var), which put the global columns last although the coefficients are x regime blocks, x globals, yend regime blocks, yend globals. With global exogenous columns and a regime-specific W y or yend (regime_lag_sep with constant_regi 'one' or mixed cols2regi, or regime-specific yend with constant_regi 'one') spreg's endogenous and joint Chow statistics compare other coefficients. Spatex builds the restrictions on the real layout by default (values from spreg's wald_test on a corrected R, stored as chow_*) and reproduces spreg with spreg_compatible: true (spreg_chow_*); both are tested in Spatex.RegressionGMLagRegimesOracleTest",
    doc: "docs/parity.md#spreg-regimes-notes"
  },
  %{
    fixture: "regression/gm_lag_regimes.json",
    path: "(not ported)",
    tol: :skip,
    reason:
      "GM_Lag_Regimes slx_lags / slx_vars (WX computed on the complete W), robust='ogmm', cores, vm and the text summary are not ported, and GM_Lag_Endog_Regimes (endogenous regimes) is out of scope; the options raise ArgumentError (see Spatex.RegressionGMLagRegimesTest).",
    doc: "docs/parity.md#spreg-regimes-notes"
  },
  %{
    fixture: "regression/tsls_regimes.json",
    path: "(not ported)",
    tol: :skip,
    reason:
      "spreg's TSLS_Regimes spat_diag (Anselin-Kelejian), robust='ogmm', slx_lags, cores and the text summary are not ported, and TSLS_Endog_Regimes (endogenous regimes) is out of scope; the options raise ArgumentError (see Spatex.RegressionTSLSRegimesTest).",
    doc: "docs/parity.md#spreg-regimes-notes"
  },
  %{
    fixture: "region/azp.json",
    path: "(heuristic)",
    tol: :skip,
    reason:
      "AZP draws Python random numbers; Spatex uses ChaCha8. Compared by validity, brute-force optimum and objective distribution only. Spatex adds a final strict local-optimality pass, n_clusters = 1 returns one region (spopt crashes), labels are 0-based integers (spopt: floats), and the tabu / simulated-annealing / reactive variants are not provided (they raise TypeError in spopt 0.8.0)",
    doc: "docs/parity.md#spopt-notes"
  },
  %{
    fixture: "region/maxp.json",
    path: "(heuristic)",
    tol: :skip,
    reason:
      "Max-p draws numpy legacy randomness; Spatex uses ChaCha8 streams per construction iteration and annealing run. Compared by validity, p and brute-force bounds. Labels are 0-based (spopt: 1-based); policy: :keep with an infeasible component raises ArgumentError upfront (spopt: IndexError); the construction phase for a fixed order and top_n = 1 matches exactly",
    doc: "docs/parity.md#spopt-notes"
  },
  %{
    fixture: "(no fixture)",
    path: "atkinson / wolfson / Schutz with empty input; Schutz with values summing to zero",
    tol: :skip,
    reason:
      "inequality does not validate these inputs: numpy turns empty vectors into NaN and Schutz fails with IndexError on an all-zero vector; Spatex raises ArgumentError (see Spatex.InequalityIndicesTest)",
    doc: "docs/parity.md#inequality"
  },
  %{
    fixture: "dynamics/kullback.json",
    path: "cases[*].statistic",
    tol: :skip,
    reason:
      "giddy's kullback sums the stratum term over axis 1 (the column totals f_s.j) where Kullback's statistic needs the row totals f_si.; the two agree only when every stratum has equal row and column totals, otherwise giddy's value is wrong (negative for identical strata). Spatex computes the row-total statistic (stored as statistic_intended) and giddy_compatible: true reproduces giddy's (see Spatex.DynamicsKullbackOracleTest)",
    doc: "docs/parity.md#deviations-dynamics"
  },
  %{
    fixture: "dynamics/polarization.json",
    path: "cases[*].{p_value,sim_mean,sim_std}",
    tol: :skip,
    reason:
      "PySAL spawns numpy SeedSequence children per permutation; Spatex draws from ChaCha8 streams, so the simulated indices are not bit equal. The statistic, component counts and labels are exact; p-values and the null's mean and spread are compared with binomial bands (see Spatex.InequalityPolarizationOracleTest). keep_simulations defaults to true (PySAL keep_sim is False); x is aligned with the graph's ids by position",
    doc: "docs/parity.md#inequality"
  },
  %{
    fixture: "regression/gwr_bws.json",
    path: "intervals / mgwr_intervals (fixed bandwidths)",
    tol: 0.005,
    reason:
      "a fixed golden-section search records both the unrounded probe and its two-decimal rounding with the same score, so AICc ties decide which of the pair bounds the interval; pandas sorts ties with numpy's unstable quicksort, Spatex keeps search order. Bounds therefore agree to the rounding step (0.005); adaptive intervals agree exactly (see Spatex.GWRBwsOracleTest)",
    doc: "docs/parity.md#mgwr"
  },
  %{
    fixture: "regression/gwr_bws.json",
    path: "variability[*].{p_values,sds}",
    tol: :skip,
    reason:
      "mgwr draws numpy's global generator (np.random.permutation); Spatex draws ChaCha8 streams from a recorded seed, so permutations differ. The observed statistic and bandwidth are exact; p-values and the null's mean are compared within Monte Carlo bands (see Spatex.GWRBwsOracleTest). mgwr's default seed is the fixed 5536, Spatex's nil draws a seed and records it. MGWR.spatial_variability is not provided: it re-runs a multi-bandwidth backfitting search per draw",
    doc: "docs/parity.md#mgwr"
  },
  %{
    fixture: "region/random_region.json",
    path: "(random partitions)",
    tol: :skip,
    reason:
      "RandomRegion(s) draw numpy's global generator; Spatex uses ChaCha8 streams, so partitions are never compared. Summaries of many runs (number of regions, sizes by construction order, feasible share, compactness) agree within sampling error (see Spatex.RegionRandomOracleTest). Area ids are positions; infeasible runs return feasible: false with labels: nil; spopt hangs for num_regions > n - 1 with random cardinalities (ArgumentError) and when no swap exists (bounded by max_swaps); non-positive cardinalities, num_regions and maxiter raise",
    doc: "docs/parity.md#spopt-notes"
  },
  %{
    fixture: "regression/gm_error_regimes.json",
    path: "cases[chow_layout_bug=true].{chow_joint, chow_regi}",
    tol: :skip,
    reason:
      "GM_Endog_Error_Regimes and GM_Combo_Regimes share TSLS_Regimes's Chow restrictions (buildR1var), which put the global columns last although the coefficients are x regime blocks, x globals, yend regime blocks, yend globals (lambda is trimmed from the betas first). With global exogenous columns and a regime-specific yend (constant_regi 'one' or mixed cols2regi) spreg's endogenous and joint Chow statistics compare other coefficients; GM_Error_Regimes and the separate regressions cannot hit it. Spatex builds the restrictions on the real layout by default (values from spreg's wald_test on a corrected R, stored as chow_*) and reproduces spreg with spreg_compatible: true (spreg_chow_*); both are tested in Spatex.RegressionGMErrorRegimesOracleTest",
    doc: "docs/parity.md#spreg-regimes-notes"
  },
  %{
    fixture: "regression/gm_error_regimes.json",
    path: "cases[optim: |lam - lam_exact| > 5e-6]",
    tol: :skip,
    reason:
      "spreg's L-BFGS-B ends away from the minimizer of its own KP98 moment objective when the objective is flat (binary rook weights on the regime subgraphs: the pooled lambdas differ by up to 4e-5, a separate regime's by 0.04 and 0.4) and, on three queen fits, 1e-5 to 2e-5 early. Spatex returns the minimizer; the fixture stores spreg's lambda next to the exact minimizer and the objective at both (optim), and for these seven cases the tests check lambda against the minimizer (5e-6), that spreg's objective is not lower, and the layout and names, not the quantities derived from spreg's lambda (see Spatex.RegressionGMErrorRegimesOracleTest)",
    doc: "docs/parity.md#spreg-regimes-notes"
  },
  %{
    fixture: "(no fixture)",
    path:
      "GM_Error_Regimes / GM_Endog_Error_Regimes / GM_Combo_Regimes with regime_err_sep and a column constant within a regime",
    tol: :skip,
    reason:
      "spreg's check_const_regi drops the column but kr (and the preallocated vm / betas) keep the stale count, so the per-regime results do not fit and numpy raises ValueError (could not broadcast input array). Spatex drops the column with spreg's warning and fits the remaining coefficients (see Spatex.RegressionGMErrorRegimesTest)",
    doc: "docs/parity.md#spreg-regimes-notes"
  },
  %{
    fixture: "regression/gm_error_regimes.json",
    path: "(not ported)",
    tol: :skip,
    reason:
      "slx_lags / slx_vars, hard_bound, cores, vm, the text summary and the GMM_Error_Regimes dispatcher are not ported for GM_Error_Regimes, GM_Endog_Error_Regimes and GM_Combo_Regimes (see Spatex.RegressionGMErrorRegimesTest).",
    doc: "docs/parity.md#spreg-regimes-notes"
  },
  %{
    fixture: "regression/gm_error_het_hom_regimes.json",
    path: "cases[clamped=true]",
    tol: :skip,
    reason:
      "The separate GM_Endog_Error_Het fits of GM_Endog_Error_Het_Regimes (regime_err_sep) are the base class, which clamps the coefficient of its last endogenous column (HOVAL) to +-0.9 in the first stage and each iteration; that is a defect when the column is not W y. Spatex does not clamp; each clamped case has a twin '<label>/unclamped' run with the clamp disabled, which Spatex matches (the combos' clamp of rho is replicated). See Spatex.RegressionGMErrorHetHomRegimesOracleTest",
    doc: "docs/parity.md#spreg-regimes-notes"
  },
  %{
    fixture: "regression/gm_error_het_hom_regimes.json",
    path: "cases[separate het endog / combo].name_z",
    tol: :skip,
    reason:
      "GM_Endog_Error_Het_Regimes and GM_Combo_Het_Regimes name the lambda of each separate regime '<r>lambda' (no underscore) in name_z; Spatex names it '<r>_lambda' like every other coefficient (the tests normalize spreg's names)",
    doc: "docs/parity.md#spreg-regimes-notes"
  },
  %{
    fixture: "regression/gm_error_het_hom_regimes.json",
    path: "cases[chow_layout_bug=true].{chow_joint, chow_regi}",
    tol: :skip,
    reason:
      "The Het and Hom classes with endogenous columns share TSLS_Regimes's Chow restrictions (buildR1var), which put the global columns last although the coefficients are x regime blocks, x globals, yend regime blocks, yend globals (lambda is the last global). With global exogenous columns and a regime-specific yend (constant_regi 'one' or mixed cols2regi) spreg's endogenous and joint Chow statistics compare other coefficients; the pooled GM_Error_*_Regimes and the separate regressions cannot hit it. Spatex builds the restrictions on the real layout by default (stored as chow_*) and reproduces spreg with spreg_compatible: true (spreg_chow_*); both are tested in Spatex.RegressionGMErrorHetHomRegimesOracleTest",
    doc: "docs/parity.md#spreg-regimes-notes"
  },
  %{
    fixture: "regression/gm_error_het_hom_regimes.json",
    path: "cases[optim: |lam - lam_exact| > 5e-6]",
    tol: :skip,
    reason:
      "spreg's L-BFGS-B ends away from the minimizer of its own weighted moment objective on seven non-clamped fits (binary rook weights on the full graph or the regime subgraphs, and the four-regime queen combo with regime_lag_sep: lambda 5.1e-6 to 6.6e-5 off). Spatex returns the minimizer; the fixture stores spreg's lambda, the exact minimizer and both objectives (optim). The quantities derived from spreg's lambda are still verified at 1e-8 by feeding spreg's lambdas through the internal gm_error_regimes_at / gm_combo_regimes_at hooks; the optimizer-driven fit of those cases is not compared",
    doc: "docs/parity.md#spreg-regimes-notes"
  },
  %{
    fixture: "(no fixture)",
    path:
      "GM_*_Het_Regimes / GM_*_Hom_Regimes: slx_lags, slx_vars, hard_bound, inv_method, cores, vm",
    tol: :skip,
    reason:
      "Not ported for the Het and Hom regimes classes: slx_lags / slx_vars (spreg computes WX on the complete W), hard_bound, inv_method (Spatex always uses the power expansion), cores, vm and the text summary. gm_error_regimes/5 and gm_combo_regimes/5 raise ArgumentError for unknown options (see Spatex.RegressionGMErrorHetHomRegimesTest)",
    doc: "docs/parity.md#spreg-regimes-notes"
  },
  %{
    fixture: "region/sa3.json",
    path: "min_cluster_size < 2",
    tol: :skip,
    reason:
      "spopt's SA3 accepts min_cluster_size 0 and 1 and then returns clusters that are not contiguous (on a path of six areas, min_cluster_size=1 gives [1, -1, 1, 0, -1, 0]: the singletons the condensed tree keeps are labeled as one cluster); Spatex requires an integer >= 2 and raises ArgumentError (see Spatex.RegionSa3Test). Every min_cluster_size >= 2 reproduces spopt's labels and linkage exactly",
    doc: "docs/parity.md#spopt-notes"
  },
  %{
    fixture: "(no fixture)",
    path: "ML_Lag_Regimes with regime_lag_sep and a column constant within a regime",
    tol: :skip,
    reason:
      "spreg's check_const_regi drops the column but kr (and the preallocated vm / betas) keep the stale count, so the per-regime results do not fit and numpy raises ValueError (could not broadcast input array). Spatex drops the column with spreg's warning and fits the remaining coefficients (see Spatex.RegressionMLLagRegimesTest)",
    doc: "docs/parity.md#spreg-regimes-notes"
  },
  %{
    fixture: "regression/ml_lag_regimes.json",
    path: "cases[method=ord].{vm,vm1,std_err,z_stat,chow_*}",
    tol: 1.0e-5,
    reason:
      "spreg's ord path substitutes the symmetrized matrix for W in the variance (see the ML ord deviation), for the pooled fit and for every regime's fit; the fixture stores the same case run with method='full' as full_ref and the oracle test compares Spatex's ord covariance, standard errors, z statistics and Chow tests with it (see Spatex.RegressionMLLagRegimesOracleTest)",
    doc: "docs/parity.md#spreg-regimes-notes"
  },
  %{
    fixture: "(no fixture)",
    path: "ML_Lag_Regimes: regime_err_sep, slx_lags, slx_vars, spat_diag, cores, vm",
    tol: :skip,
    reason:
      "regime_err_sep is not a parameter of the class (spreg raises TypeError; ml_lag_regimes/5 raises ArgumentError); slx_lags / slx_vars (WX on the complete W) with the common-factor spat_diag, cores, vm and the text summary are not ported. regime_lag_sep requires every column by regime and constant_regi :many in both (see Spatex.RegressionMLLagRegimesTest)",
    doc: "docs/parity.md#spreg-regimes-notes"
  },
  %{
    fixture: "(no fixture)",
    path: "ML_Error_Regimes with regime_err_sep and a column constant within a regime",
    tol: :skip,
    reason:
      "spreg's check_const_regi drops the column but kr (and the preallocated vm / betas) keep the stale count, so the per-regime results do not fit and numpy raises ValueError (could not broadcast input array; the fixture stores the message as separate_constant_column_error). Spatex drops the column with spreg's warning and fits the remaining coefficients (see Spatex.RegressionMLErrorRegimesTest)",
    doc: "docs/parity.md#spreg-regimes-notes"
  },
  %{
    fixture: "regression/ml_error_regimes.json",
    path: "cases[method=ord].{vm,vm1,std_err,z_stat,chow_*}",
    tol: 1.0e-5,
    reason:
      "spreg's ord path substitutes the symmetrized matrix for W in the variance (see the ML ord deviation), for the pooled fit and for every regime's fit; the fixture stores the same case run with method='full' as full_ref and the oracle test compares Spatex's ord covariance, standard errors, z statistics and Chow tests with it (see Spatex.RegressionMLErrorRegimesOracleTest)",
    doc: "docs/parity.md#spreg-regimes-notes"
  },
  %{
    fixture: "(no fixture)",
    path: "ML_Error_Regimes: slx_lags, slx_vars, cores, vm, regime_lag_sep",
    tol: :skip,
    reason:
      "slx_lags / slx_vars (WX on the complete W), cores, vm and the text summary are not ported; regime_lag_sep is accepted and ignored by spreg and is an unknown option of ml_error_regimes/5. regime_err_sep requires every column by regime and constant_regi :many in both (see Spatex.RegressionMLErrorRegimesTest)",
    doc: "docs/parity.md#spreg-regimes-notes"
  },
  %{
    fixture: "region/spenc.json",
    path: "cases.hierarchical_*",
    tol: :skip,
    reason:
      "spopt 0.8.0 cannot run Spenc with assign_labels=\"hierarchical\" (or n_clusters=inf): _spectral_bipartition passes floor= to _make_hierarchical_cut, which does not accept it, so every call raises TypeError. Spatex implements what that code intends (the fixtures come from spopt with that one keyword accepted and ignored) and its labels equal that oracle's; the cases where spopt then fails (a cut leaves a part too small for eigsh, or no cut is accepted) raise ArgumentError in Spatex too (see Spatex.RegionSpencOracleTest)",
    doc: "docs/parity.md#spopt-notes"
  },
  %{
    fixture: "region/spenc.json",
    path: "cases.*.embedding, eigenvalues",
    tol: :skip,
    reason:
      "spopt finds the k eigenpairs with ARPACK (eigsh, shift-invert, tol 1e-9); Spatex uses a dense symmetric eigensolver, so the eigenpairs agree to the solver tolerance (eigenvalues 1e-8, embedding up to the sign of each vector 1e-6 in the tests) and n is limited to about 5000 areas. A non-symmetric affinity (for example row-standardized weights) raises ArgumentError; spopt feeds it to eigsh and returns meaningless eigenvectors. The random stages use ChaCha8, not numpy's generator: discretize labels equal sklearn's for the same start row and are tested statistically (see Spatex.RegionSpencOracleTest)",
    doc: "docs/parity.md#spopt-notes"
  },
  %{
    fixture: "(no fixture)",
    path: "ThreeSLS: regimes, df / dict input forms, default variable names, nonspat_diag",
    tol: :skip,
    reason:
      "regimes are not ported (three_sls/5 raises ArgumentError for any :regimes option); only lists of per-equation inputs are accepted; unnamed columns are named var_<i> as in sur/3 where spreg uses var_<i>_<eq>, end_<i>_<eq> and inst_<i>_<eq>; spreg's nonspat_diag only affects the text summary of ThreeSLS, so there is no option (see Spatex.RegressionThreeSlsTest)",
    doc: "docs/parity.md#spreg-spatial-sur-notes"
  },
  %{
    fixture: "regression/sur_error_gm.json",
    path:
      "cases.*.lamsur (and everything derived from it) for two_rook_binary, three_equal_rook_binary",
    tol: :skip,
    reason:
      "spreg minimizes the KP98 moment objective of each equation with L-BFGS-B, which stops up to ~5e-6 from the minimizer (up to 5e-4 with binary weights, where the objective is flat); Spatex returns the exact minimizer. The fixture stores spreg's lambda, the exact minimizer and both objective values (`optim`); the other cases agree at 5e-6 (lambda) and 5e-4 (derived quantities), every formula at 1e-9 given spreg's lambdas (see Spatex.RegressionSurErrorGmOracleTest)",
    doc: "docs/parity.md#spreg-spatial-sur-notes"
  },
  %{
    fixture: "(no fixture)",
    path: "SURerrorGM: regimes, df / dict input forms, spat_diag, vm, nonspat_diag",
    tol: :skip,
    reason:
      "regimes are not ported (sur_error_gm/4 raises ArgumentError for any :regimes option); only lists of per-equation inputs are accepted; spreg implements no spat_diag or vm for SURerrorGM and nonspat_diag only affects its text summary, so there are no such options (see Spatex.RegressionSurErrorGmTest)",
    doc: "docs/parity.md#spreg-spatial-sur-notes"
  },
  %{
    fixture: "regression/sur_error_ml.json",
    path: "cases.*.lamsur (and everything derived from it)",
    tol: :skip,
    reason:
      "spreg alternates the SUR GLS step with an L-BFGS-B search of all lambdas (finite-difference gradient, factr 1e7), which stops up to ~1e-5 from the optimum of each search and ends the loop on |dF| <= epsilon; Spatex minimizes the same concentrated likelihood with analytic gradients (projected BFGS, gradient 1e-10), so its lambdas are the tighter optimum. Both loops take the same number of searches on every fixture case; lambda is within 5e-5 of spreg's and of an epsilon = 1e-10 refit, coefficients, covariances and tests within 5e-4. The fixture stores spreg's lambda path, so every formula matches at 1e-9 through the `sur_error_ml_at/5` hook (see Spatex.RegressionSurErrorMlOracleTest)",
    doc: "docs/parity.md#spreg-spatial-sur-notes"
  },
  %{
    fixture: "regression/sur_error_ml.json",
    path: "cases.*.lamols",
    tol: 2.0e-7,
    reason:
      "the starting lambdas are bounded Brent searches with xatol = 1e-7 (a literal port of SciPy's); spreg evaluates the likelihood with a sparse LU and a normal-equations solve, Spatex with a dense LU, and function values that differ in the last digits send the parabolic steps to a point up to ~5e-8 away. Tested at 2e-7; the hook takes spreg's starting lambdas",
    doc: "docs/parity.md#spreg-spatial-sur-notes"
  },
  %{
    fixture: "(no fixture)",
    path: "SURerrorML: binary (non row-standardized) weights",
    tol: :skip,
    reason:
      "with binary rook weights the Jacobian is singular inside [-1, 1] and spreg stops with `RuntimeError: Factor is exactly singular` (so no fixture exists); Spatex uses ln|det| from a dense LU (infinite at the singular point), so the search returns a local optimum instead of raising (see Spatex.RegressionSurErrorMlTest)",
    doc: "docs/parity.md#spreg-spatial-sur-notes"
  },
  %{
    fixture: "(no fixture)",
    path: "SURerrorML: regimes, df / dict input forms, text summary, sparse log-Jacobian",
    tol: :skip,
    reason:
      "regimes are not ported (sur_error_ml/4 raises ArgumentError for any :regimes option); only lists of per-equation inputs are accepted; the Jacobian is a dense LU (n is capped by :max_dense_n, default 5000) where spreg uses a sparse LU (see Spatex.RegressionSurErrorMlTest)",
    doc: "docs/parity.md#spreg-spatial-sur-notes"
  },
  %{
    fixture: "(no fixture)",
    path:
      "SURlagIV: regimes, regime_lag_sep, df / dict input forms, default names, nonspat_diag, vm",
    tol: :skip,
    reason:
      "regimes are not ported (sur_lag_iv/4 raises ArgumentError for any :regimes option); only lists of per-equation inputs are accepted and :yends / :qs are given for every equation or none; unnamed user columns are named var_<i> where spreg uses end_<i>_<eq> and inst_<i>_<eq>; spreg implements no spatial impacts or further diagnostics and its nonspat_diag and vm only affect the text summary, so there are no such options (see Spatex.RegressionSurLagIvTest)",
    doc: "docs/parity.md#spreg-spatial-sur-notes"
  },
  %{
    fixture: "regression/ml_error_re.json",
    path: "rook5x5 (cases[0].{lam, phi, logll, aic, schwarz})",
    tol: :skip,
    reason:
      "spreg's numpy.linalg.eig returns a non-orthogonal basis inside a degenerate eigenspace (a square rook lattice has repeated eigenvalues even though W is symmetric), so its concentrated objective is not the likelihood and its optimum depends on LAPACK. Spatex uses the orthonormal symmetric eigenvectors (the exact likelihood). The fixture stores spreg's values and the exact optimum of the eigh objective; the coefficients and variances at spreg's own (lam, phi) still match (see Spatex.RegressionPanelREErrorOracleTest)",
    doc: "docs/parity.md#spreg-ml_errorre-notes"
  },
  %{
    fixture: "(no fixture)",
    path:
      "ML_ErrorRE with a non-symmetric W that has repeated or complex eigenvalues; LinAlgError on a singular information matrix",
    tol: :skip,
    reason:
      "spreg's objective then depends on LAPACK's eigenvector basis (repeated eigenvalues) or runs in complex arithmetic (complex eigenvalues) and is not a likelihood; spreg can also raise LinAlgError: Singular matrix in its information matrix. Spatex raises ArgumentError instead of fitting (see Spatex.RegressionPanelREErrorTest)",
    doc: "docs/parity.md#spreg-ml_errorre-notes"
  },
  %{
    fixture: "(no fixture)",
    path: "Panel_RE_Error (spreg/panel_re.py)",
    tol: :skip,
    reason:
      "deprecated in favor of ML_ErrorRE (panel.py); its symmetric branch calls symmetrize(w), which replaces the weights by the row-standardized similar matrix of the binary pattern while the data are lagged with the original matrix, so it is not a consistent model. Not ported",
    doc: "docs/parity.md#spreg-ml_errorre-notes"
  },
  %{
    fixture: "regression/gm_kkp.json",
    path: "cases[*].{betas, vm, std_err, z_stat, lam, sig2_v, sig2_1, u, e_filtered}",
    tol: 1.0e-5,
    reason:
      "spreg minimizes both GM_ErrorRE moment objectives with L-BFGS-B, which stops early: lambda is within 1e-5, the variances within 5e-4 relative and the coefficients within 2e-3 of the exact minimum (worst on the binary rook lattice). Spatex finds the exact minimizer (variances profiled out by non-negative least squares, downhill grid walk and bounded Brent on lambda from spreg's start). The fixture stores spreg's two optimizer results, and the oracle tests check every formula at them to 1e-9 (gm_kkp_at/5), the fit against spreg run with its optimizer results replaced by the exact minimum (1e-6), and the fit against spreg's own result at the tolerances above (see Spatex.RegressionGMKKPOracleTest)",
    doc: "docs/parity.md#spreg-gm_kkp-notes"
  },
  %{
    fixture: "(no fixture)",
    path: "GM_ErrorRE / GM_KKP: regimes, slx_lags, degenerate moment scale",
    tol: :skip,
    reason:
      "regimes are not ported (GM_ErrorRE raises NotImplementedError; the deprecated GM_KKP accepts them) and raise ArgumentError, slx_lags / slx_vars are not ported, and an exactly zero moment scale, which makes spreg's objective NaN and L-BFGS-B return its start silently, raises ArgumentError (see Spatex.RegressionGMKKPTest)",
    doc: "docs/parity.md#spreg-gm_kkp-notes"
  }
]
