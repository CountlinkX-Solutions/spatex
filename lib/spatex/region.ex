defmodule Spatex.Region do
  @moduledoc """
  Regionalization after `spopt` 0.8.0: group areas into contiguous regions.

  | Function | Method | Parity with spopt |
  |---|---|---|
  | `skater/3` | SKATER minimum-spanning-forest pruning | exact labels |
  | `ward_spatial/3` | Ward clustering constrained by the graph | exact labels and tree |
  | `sa3/3` | SA3: spatially constrained Ward tree, HDBSCAN-style (EOM or leaf) extraction, with noise | exact labels and linkage |
  | `spenc/3` | Spenc: spatially encouraged spectral clustering | exact affinity, embedding and `:hierarchical` labels; randomized labels statistically |
  | `region_kmeans/3` | Region-K-Means (seeded growth, then centroid moves) | exact labels for the same seeds |
  | `azp/3` | basic AZP local search | validity and objective, not labels |
  | `max_p/3` | Max-p-regions (construction + annealing) | validity and objective, not labels |
  | `random_region/2`, `random_regions/2` | random (contiguous) partitions | distributions, not labels |

  Helpers: `objective/3`, `contiguous?/2`, `relabel/1`, `standardize/1`.

  ## Inputs

  `attrs` is the attribute table of the areas, in graph order: a list of columns (each a list,
  `Nx` vector or `Explorer.Series`), a single column, a map `name => column`, an `{n, m}`
  `Nx.Tensor` or an `Explorer.DataFrame` (all its columns). **Attributes are never scaled
  implicitly**, exactly like spopt: columns on different scales weigh differently. Call
  `standardize/1` first when that is not what you want.

  `graph` is a `Spatex.Graph`. Only its pattern of non-self links with non-zero weight matters;
  weights are ignored (`spenc/3` is the exception: it multiplies the affinity by them). The pattern must be symmetric, or pass `symmetrize: true` to use
  `A or A'` (`ward_spatial/3` always symmetrizes, like scikit-learn). Islands are allowed:
  every method defines what it does with several connected components.

  Labels are 0-based integers. spopt returns floats for AZP and 1-based labels for Max-p.

  ## Determinism

  `skater/3`, `ward_spatial/3`, `sa3/3` and `region_kmeans/3` (given `seeds:`) are deterministic.
  The randomized parts (`azp/3`, `max_p/3`, `random_region/2`, `random_regions/2`,
  `region_kmeans/3` with `seed:`, and the label assignment of `spenc/3`) use ChaCha8
  streams derived from `seed:`: a seed gives the same result on every run and thread count, but
  does not reproduce numpy's generator, so results differ from spopt's for the same seed.

  ## Memory

  No `n x n` matrix is built (spopt builds dense distance matrices for SKATER and Max-p).
  """

  alias Spatex.Graph
  alias Spatex.Input
  alias Spatex.Native
  alias Spatex.Region.AZP
  alias Spatex.Region.MaxP
  alias Spatex.Region.RandomRegion
  alias Spatex.Region.RandomRegions
  alias Spatex.Region.RegionKMeans
  alias Spatex.Region.Sa3
  alias Spatex.Region.Skater
  alias Spatex.Region.Spenc
  alias Spatex.Region.Ward

  @type attrs :: term()
  @type labels :: [non_neg_integer()]

  # -- SKATER ------------------------------------------------------------------------------

  @doc """
  SKATER: spatial `k`luster analysis by tree edge removal.

  Builds a minimum spanning forest over the graph, weighting each link by the dissimilarity of
  its two areas, then repeatedly deletes the forest edge whose removal gives the lowest
  whole-map score (the sum over regions of the distance of every area to its region mean) until
  `n_clusters` regions remain.

  ## Options

    * `:n_clusters` - required, between 1 and the number of areas.
    * `:floor` - minimum number of **areas** per region (default `0.0`: no floor). A cut leaving
      any region below the floor is not allowed; when no cut is left the result has fewer
      regions and `warnings` says so.
    * `:islands` - `:increase` (default) adds the number of connected components to
      `n_clusters`, so `n_clusters: 5` on a graph with two components yields seven regions
      (spopt's behavior); `:ignore` counts the components towards `n_clusters`.
    * `:trace` - keep the labeling after every cut in `trace` (default `false`).
    * `:dissimilarity` - `:manhattan` (default, spopt's) or `:euclidean`; used for the
      spanning forest and the score.
    * `:zero_dissimilarity` - `:keep` (default) or `:drop`. spopt silently removes links
      between areas with identical attributes, which splits the graph into false islands and
      raises `n_clusters`; Spatex keeps them. `:drop` reproduces spopt.
    * `:symmetrize` - use `A or A'` for an asymmetric graph.

  The edge to cut is the first, in row-major order, with the strictly lowest score. Scores that
  differ by less than about `1.0e-12` (relative) may be ranked differently from spopt because
  floating-point summation order differs.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])
      iex> r = Spatex.Region.skater([[1.0, 1.1, 9.0, 9.2]], g, n_clusters: 2)
      iex> r.labels
      [0, 0, 1, 1]
  """
  @spec skater(attrs(), Graph.t(), keyword()) :: Skater.t()
  def skater(attrs, %Graph{} = graph, opts \\ []) do
    allowed!(opts, [
      :dissimilarity,
      :floor,
      :islands,
      :n_clusters,
      :symmetrize,
      :trace,
      :zero_dissimilarity
    ])

    {bin, p} = attrs!(attrs, graph.n)
    k = n_clusters!(opts, graph.n)
    floor = number_opt!(opts, :floor, 0.0)
    islands = enum_opt!(opts, :islands, :increase, [:increase, :ignore])
    dissimilarity = enum_opt!(opts, :dissimilarity, :manhattan, [:manhattan, :euclidean])
    zero = enum_opt!(opts, :zero_dissimilarity, :keep, [:keep, :drop])

    {labels, edges, score, trace, warnings} =
      Native.region_skater(
        graph.ref,
        bin,
        p,
        k,
        floor * 1.0,
        Atom.to_string(dissimilarity),
        [
          islands == :ignore,
          Keyword.get(opts, :trace, false),
          zero == :drop,
          Keyword.get(opts, :symmetrize, false)
        ]
      )
      |> unwrap()

    %Skater{
      labels: labels,
      n_regions: labels |> Enum.uniq() |> length(),
      msf_edges: edges,
      score: score,
      trace: Enum.map(trace, &trace_step/1),
      warnings: warnings
    }
  end

  defp trace_step({labels, i, o, score}),
    do: %{labels: labels, in_node: i, out_node: o, score: score || :infinity}

  # -- Ward --------------------------------------------------------------------------------

  @doc """
  Spatially constrained Ward clustering: agglomerative clustering that may only merge
  neighboring clusters (scikit-learn's `AgglomerativeClustering` with `linkage="ward"` and the
  graph as connectivity).

  When the graph has several connected components it is completed first: for every pair of
  components the two most similar areas **in attribute space** (not geography) are linked, as
  scikit-learn does, and `n_connected_components` and `warnings` report it.

  Cluster numbers follow scikit-learn: they come from the cut of the merge tree, or, when
  `n_clusters >= max(100, 0.02 * n)` and the tree is stopped early, from the rank of each
  cluster's head node. Use `relabel/1` for first-appearance numbering.

  ## Options

    * `:n_clusters` - required.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])
      iex> r = Spatex.Region.ward_spatial([[1.0, 1.1, 9.0, 9.2]], g, n_clusters: 2)
      iex> Spatex.Region.relabel(r.labels)
      [0, 0, 1, 1]
  """
  @spec ward_spatial(attrs(), Graph.t(), keyword()) :: Ward.t()
  def ward_spatial(attrs, %Graph{} = graph, opts \\ []) do
    allowed!(opts, [:n_clusters])
    {bin, p} = attrs!(attrs, graph.n)
    k = n_clusters!(opts, graph.n)

    {labels, children, distances, components, warnings} =
      graph.ref |> Native.region_ward(bin, p, k) |> unwrap()

    %Ward{
      labels: labels,
      children: children,
      distances: distances,
      n_connected_components: components,
      n_leaves: graph.n,
      warnings: warnings
    }
  end

  # -- SA3 --------------------------------------------------------------------------------

  @doc """
  SA3 (spatial adaptive agglomerative aggregation): spatially constrained Ward clustering
  whose merge tree is cut with HDBSCAN's cluster extraction, giving multiscale contiguous
  clusters and noise.

  Every connected component of the graph with more than `min_cluster_size` areas gets its own
  Ward tree (as in `ward_spatial/3`); if a merge distance drops below the previous one, spopt
  sets it to the previous plus `0.01` (repeated until the tree is monotone, and Spatex does the
  same). The tree is condensed with `min_cluster_size` (lambda is the inverse of the distance)
  and the clusters are the excess-of-mass selection (`:eom`) or the leaves of the condensed
  tree (`:leaf`). Areas in no cluster, and whole components with at most `min_cluster_size`
  areas, are noise: label `-1`. Cluster numbers continue across components. Labels and the
  linkage match spopt exactly.

  ## Options

    * `:min_cluster_size` - minimum number of areas of a cluster, an integer `>= 2` (default
      `15`, spopt's). spopt also accepts smaller values; see the known deviations.
    * `:extraction` - `:eom` (default) or `:leaf`.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2, 3, 4, 4, 5, 2, 3], [1, 0, 2, 1, 4, 3, 5, 4, 3, 2])
      iex> r = Spatex.Region.sa3([[0.0, 0.1, 0.2, 10.0, 10.1, 10.2]], g, min_cluster_size: 3)
      iex> {r.labels, r.n_clusters, r.n_noise}
      {[1, 1, 1, 0, 0, 0], 2, 0}
  """
  @spec sa3(attrs(), Graph.t(), keyword()) :: Sa3.t()
  def sa3(attrs, %Graph{} = graph, opts \\ []) do
    allowed!(opts, [:min_cluster_size, :extraction])
    {bin, p} = attrs!(attrs, graph.n)
    min_size = min_cluster_size!(Keyword.get(opts, :min_cluster_size, 15))
    extraction = enum_opt!(opts, :extraction, :eom, [:eom, :leaf])

    {labels, components, trees} =
      graph.ref |> Native.region_sa3(bin, p, min_size, Atom.to_string(extraction)) |> unwrap()

    n_noise = Enum.count(labels, &(&1 == -1))

    %Sa3{
      labels: labels,
      n_clusters: labels |> Enum.reject(&(&1 == -1)) |> Enum.uniq() |> length(),
      n_noise: n_noise,
      n_connected_components: components,
      components: Enum.map(trees, fn {nodes, linkage} -> %{nodes: nodes, linkage: linkage} end)
    }
  end

  defp min_cluster_size!(m) when is_integer(m) and m >= 2, do: m

  defp min_cluster_size!(other),
    do: raise(ArgumentError, "min_cluster_size must be an integer >= 2, got #{inspect(other)}")

  # -- Spenc ------------------------------------------------------------------------------

  @affinities [
    :rbf,
    :laplacian,
    :linear,
    :polynomial,
    :poly,
    :sigmoid,
    :cosine,
    :nearest_neighbors,
    :precomputed
  ]

  @doc """
  Spenc: spatially encouraged spectral clustering (Wolf 2018), spopt's `Spenc`.

  The attribute affinity (`:affinity` kernel of the attributes) is multiplied elementwise by the
  graph's **weights**, so only neighboring areas are similar; the affinity's normalized Laplacian
  is embedded with its `n_clusters` smallest eigenvectors, divided by the square root of the
  affinity degrees, and the areas are clustered in that embedding. Unlike the other methods in
  this module, the graph weights matter (use a binary graph for spopt's default `W`), and the
  affinity must be symmetric.

  The affinity matrix, the normalized Laplacian, its eigenvalues and the embedding (up to the
  sign of each eigenvector, which is normalized as spopt does) are deterministic and match
  spopt. spopt finds the eigenpairs with ARPACK; Spatex uses a dense symmetric eigensolver, which
  gives the same eigenpairs but needs an `n x n` matrix (n up to 5000). Label assignment is
  randomized as in spopt, with ChaCha8 instead of numpy's generator, so labels match spopt's
  statistically and `:hierarchical` exactly. **Clusters are not guaranteed to be contiguous.**

  ## Options

    * `:n_clusters` - number of clusters, `1 <= k < n` (default `5`, spopt's), or `:infinity`,
      which, as in spopt, selects `assign_labels: :hierarchical` and keeps cutting while cuts
      satisfy `:floor`.
    * `:affinity` - `:rbf` (default), `:laplacian`, `:linear`, `:polynomial` (`:poly`),
      `:sigmoid`, `:cosine`, `:nearest_neighbors` or `:precomputed`. These are the
      `sklearn.metrics.pairwise_kernels` names; `:nearest_neighbors` (a symmetrized `n_neighbors`
      connectivity graph that includes each area itself) and `:precomputed` (`attrs` is the
      `n x n` affinity) ignore the graph weights, as in spopt. Similarity kernels must give
      non-negative values; this is not checked, as in spopt.
    * `:gamma` (default `1.0`), `:degree` (`3`), `:coef0` (`1`), `:n_neighbors` (`10`) - kernel
      parameters as in scikit-learn.
    * `:assign_labels` - `:discretize` (default; Yu and Shi's rotation, random start row),
      `:kmeans` (k-means++, one run, like spopt's `KMeans(n_clusters)`; `n_init` is not used by
      spopt either) or `:hierarchical` (Shi and Malik's recursive bipartition with the Fiedler
      vector cut by grid search; deterministic). spopt 0.8.0 cannot run `:hierarchical` (it
      raises `TypeError`); Spatex implements what that code intends. Like spopt it raises when
      a cut leaves fewer than three areas to split, or when no cut is accepted.
    * `:floor`, `:floor_weights`, `:grid_resolution` - `:hierarchical` only: lower limit on the
      (weighted) size of every part (default `0`, weights all `1`) and grid points of the cut
      search (`100`).
    * `:eigen_solver`, `:eigen_tol`, `:n_init` - accepted for spopt compatibility and ignored,
      as spopt ignores them (`nil`, `:arpack` and `:lobpcg` are accepted; `:amg` raises).
    * `:seed` - non-negative integer for the random stages; `nil` (default) draws one, recorded
      in the result.
    * `:keep_embedding` (default `true`) and `:keep_affinity` (default `false`) - keep the
      matrices in the result: `true` (lists of rows), `false` or `:tensor` (`Nx`).

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
      ...>   [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6])
      iex> x = [[0.0, 0.1, 0.2, 0.3, 3.0, 3.1, 3.2, 3.3]]
      iex> r = Spatex.Region.spenc(x, g, n_clusters: 2, gamma: 0.05, seed: 1)
      iex> Spatex.Region.relabel(r.labels)
      [0, 0, 0, 0, 1, 1, 1, 1]
  """
  @spec spenc(attrs(), Graph.t(), keyword()) :: Spenc.t()
  def spenc(attrs, %Graph{} = graph, opts \\ []) do
    allowed!(opts, [
      :n_clusters,
      :affinity,
      :gamma,
      :degree,
      :coef0,
      :n_neighbors,
      :assign_labels,
      :floor,
      :floor_weights,
      :grid_resolution,
      :eigen_solver,
      :eigen_tol,
      :n_init,
      :seed,
      :keep_embedding,
      :keep_affinity
    ])

    {bin, p} = attrs!(attrs, graph.n)
    n = graph.n
    affinity = enum_opt!(opts, :affinity, :rbf, @affinities)
    spenc_solver!(Keyword.get(opts, :eigen_solver))
    {k, assign} = spenc_clusters!(opts, n)
    n_neighbors = spenc_pos_int!(opts, :n_neighbors, 10)

    if affinity == :nearest_neighbors and n_neighbors > n,
      do: spenc_bad!(:n_neighbors, "at most n")

    grid = spenc_pos_int!(opts, :grid_resolution, 100)
    floor = spenc_number!(opts, :floor, 0)
    weights = spenc_weights!(opts, n)
    keep_embedding = Input.keep_simulations!(Keyword.get(opts, :keep_embedding, true))
    keep_affinity = Input.keep_simulations!(Keyword.get(opts, :keep_affinity, false))
    seed = seed!(opts)

    params = {
      spenc_number!(opts, :gamma, 1.0) * 1.0,
      spenc_number!(opts, :degree, 3) * 1.0,
      spenc_number!(opts, :coef0, 1) * 1.0,
      n_neighbors,
      floor * 1.0,
      grid
    }

    {labels, affinity_bin, embedding_bin, cols, eigenvalues, start_row, warnings} =
      graph.ref
      |> Native.region_spenc(
        bin,
        p,
        k,
        {Atom.to_string(affinity), Atom.to_string(assign), weights},
        params,
        seed,
        {Input.keep_flag(keep_affinity), Input.keep_flag(keep_embedding)}
      )
      |> unwrap()

    %Spenc{
      labels: labels,
      n_clusters: labels |> Enum.uniq() |> length(),
      assign_labels: assign,
      seed: if(assign == :hierarchical, do: nil, else: seed),
      start_row: if(start_row >= 0, do: start_row),
      eigenvalues: eigenvalues,
      embedding: spenc_matrix(embedding_bin, keep_embedding, cols > 0, {n, cols}),
      affinity_matrix: spenc_matrix(affinity_bin, keep_affinity, true, {n, n}),
      warnings: warnings
    }
  end

  defp spenc_matrix(_bin, false, _present, _shape), do: nil
  defp spenc_matrix(_bin, _keep, false, _shape), do: nil
  defp spenc_matrix(bin, keep, true, shape), do: Input.simulations(bin, keep, shape)

  defp spenc_clusters!(opts, n) do
    assign = enum_opt!(opts, :assign_labels, :discretize, [:discretize, :kmeans, :hierarchical])

    case Keyword.get(opts, :n_clusters, 5) do
      :infinity ->
        {0, :hierarchical}

      k when is_integer(k) and k >= 1 and k < n ->
        {k, assign}

      other ->
        raise ArgumentError,
              "n_clusters must be an integer between 1 and #{n - 1} (fewer than the number of areas) or :infinity, got #{inspect(other)}"
    end
  end

  defp spenc_solver!(solver) when solver in [nil, :arpack, :lobpcg], do: :ok

  defp spenc_solver!(other),
    do:
      raise(
        ArgumentError,
        "eigen_solver must be nil, :arpack or :lobpcg (ignored: Spatex uses a dense symmetric eigensolver), got #{inspect(other)}"
      )

  defp spenc_pos_int!(opts, key, default) do
    case Keyword.get(opts, key, default) do
      v when is_integer(v) and v >= 1 -> v
      other -> spenc_bad!(key, "a positive integer, got #{inspect(other)}")
    end
  end

  defp spenc_number!(opts, key, default) do
    case Keyword.get(opts, key, default) do
      v when is_number(v) -> v
      other -> spenc_bad!(key, "a number, got #{inspect(other)}")
    end
  end

  defp spenc_weights!(opts, n) do
    case Keyword.get(opts, :floor_weights) do
      nil ->
        []

      w when is_list(w) and length(w) == n ->
        if Enum.all?(w, &is_number/1),
          do: Enum.map(w, &(&1 * 1.0)),
          else: spenc_bad!(:floor_weights, "a list of numbers")

      other ->
        spenc_bad!(:floor_weights, "a list with one number per area, got #{inspect(other)}")
    end
  end

  defp spenc_bad!(key, what), do: raise(ArgumentError, "#{key} must be #{what}")

  # -- Region-K-Means ----------------------------------------------------------------------

  @doc """
  Region-K-Means: regions grow simultaneously from `n_clusters` seed areas, each step adding to
  every region the neighboring area closest to its centroid; then areas move to the nearest
  centroid while both regions stay contiguous.

  ## Options

    * `:seeds` - the seed areas (0-based indices, distinct); region `i` starts at `seeds[i]`.
      With the seeds spopt drew, the labels, centroids and `iters` are identical to its.
    * `:seed` - when `:seeds` is absent, draw the seeds from this integer (default `nil`: a
      random seed, recorded in `seed`), one
      per connected component first. These are not numpy's draws.
    * `:n_clusters` - required unless `:seeds` is given.
    * `:max_iterations` - cap on the improvement loop (default `10 * n * n_clusters`); a
      warning is set when it is hit. spopt has no cap and is not proven to terminate.
    * `:symmetrize` - use `A or A'` for an asymmetric graph.

  Raises `ArgumentError` when a connected component contains no seed (spopt loops forever).

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])
      iex> r = Spatex.Region.region_kmeans([[1.0, 1.1, 9.0, 9.2]], g, seeds: [0, 3])
      iex> r.labels
      [0, 0, 1, 1]
  """
  @spec region_kmeans(attrs(), Graph.t(), keyword()) :: RegionKMeans.t()
  def region_kmeans(attrs, %Graph{} = graph, opts \\ []) do
    allowed!(opts, [:max_iterations, :n_clusters, :seed, :seeds, :symmetrize])
    {bin, p} = attrs!(attrs, graph.n)
    seeds = Keyword.get(opts, :seeds)

    k =
      case {Keyword.get(opts, :n_clusters), seeds} do
        {nil, seeds} when is_list(seeds) and seeds != [] -> length(seeds)
        _ -> n_clusters!(opts, graph.n)
      end

    seeds = seeds!(seeds)
    seed = seed!(opts)
    max_iterations = Keyword.get(opts, :max_iterations, 10 * graph.n * k)

    {labels, centroids, iters, used, warnings} =
      Native.region_kmeans(
        graph.ref,
        bin,
        p,
        k,
        seeds,
        seed,
        max_iterations,
        Keyword.get(opts, :symmetrize, false)
      )
      |> unwrap()

    %RegionKMeans{
      labels: labels,
      centroids: centroids |> Input.from_f64_binary() |> Enum.chunk_every(p),
      iters: iters,
      seeds: used,
      warnings: warnings
    }
  end

  defp seeds!(nil), do: []

  defp seeds!([_ | _] = seeds) do
    if Enum.all?(seeds, &(is_integer(&1) and &1 >= 0)) do
      seeds
    else
      raise ArgumentError, "seeds must be a list of non-negative area indices"
    end
  end

  defp seeds!(other),
    do:
      raise(
        ArgumentError,
        "seeds must be a non-empty list of area indices, got #{inspect(other)}"
      )

  # -- AZP ---------------------------------------------------------------------------------

  @doc """
  AZP (automatic zoning procedure): starting from `initial` or a random contiguous partition,
  boundary areas move between neighboring regions to lower the pairwise within-region
  dissimilarity.

  `:method` selects the local search:

    * `:basic` (default, spopt's `AZP`) - boundary areas move to a randomly chosen neighboring
      region whenever the objective does not increase, until a full pass brings no improvement.
    * `:tabu` (`AZPBasicTabu`) - every iteration makes the best move that is not tabu (even a
      worse one), keeps the last `:tabu_length` (default `10`) reverse moves tabu unless they
      improve the objective (aspiration), and stops once a partition has been visited
      `:repetitions` (default `5`) times; `:max_iterations` defaults to `10 * n` here.
    * `:annealing` (`AZPSimulatedAnnealing`) - basic AZP that also accepts a worse move with
      probability `exp(-diff / T)`; a run ends after `:sa_moves_term` (default `10`) such
      moves. `T` starts at `:init_temperature` (required) and is multiplied by
      `:cooling_factor` (default `0.85`, in `(0, 1)`) each round; the search stops when a
      partition is visited `:repetitions` (default `5`) times or after `:nonmoving_steps`
      (default `3`) rounds without a move.

  spopt 0.8.0 crashes on the tabu and annealing classes; they follow the algorithm description
  and spopt's intent as summarized above, and return the best partition they visit. For every method
  a final deterministic pass makes the result a strict local optimum: no single feasible move
  lowers the objective (spopt's loop only approximates this). `n_clusters: 1` returns one
  region where spopt crashes.

  ## Options

    * `:n_clusters` - required; must be at least the number of connected components.
    * `:seed` - integer seed (default `nil`: a random seed, recorded in `seed`).
    * `:initial` - starting labels (any labels; exactly `n_clusters` distinct, each region
      contiguous).
    * `:metric` - `:euclidean` (default, spopt's), `:manhattan` or `:sqeuclidean`.
    * `:max_iterations` - cap on accepted moves (default `1000 * n`; tabu: on iterations,
      default `10 * n`).
    * `:symmetrize` - use `A or A'` for an asymmetric graph.
    * `:method` and its options, see above. Reactive tabu is not provided.
  """
  @spec azp(attrs(), Graph.t(), keyword()) :: AZP.t()
  def azp(attrs, %Graph{} = graph, opts \\ []) do
    allowed!(opts, [
      :cooling_factor,
      :init_temperature,
      :initial,
      :max_iterations,
      :method,
      :metric,
      :n_clusters,
      :nonmoving_steps,
      :repetitions,
      :sa_moves_term,
      :seed,
      :symmetrize,
      :tabu_length
    ])

    {bin, p} = attrs!(attrs, graph.n)
    k = n_clusters!(opts, graph.n)
    seed = seed!(opts)
    metric = enum_opt!(opts, :metric, :euclidean, [:euclidean, :manhattan, :sqeuclidean])
    initial = initial!(Keyword.get(opts, :initial), graph.n)
    method = azp_method!(opts)
    default_cap = if elem(method, 0) == "tabu", do: 10 * graph.n, else: 1000 * graph.n
    max_moves = Keyword.get(opts, :max_iterations, default_cap)

    {labels, objective, initial_labels, moves} =
      Native.region_azp(
        graph.ref,
        bin,
        p,
        {k, seed, max_moves},
        initial,
        Atom.to_string(metric),
        Keyword.get(opts, :symmetrize, false),
        method
      )
      |> unwrap()

    %AZP{
      labels: labels,
      n_regions: k,
      objective: objective,
      initial_labels: initial_labels,
      iterations: moves,
      seed: seed,
      method: Keyword.get(opts, :method, :basic)
    }
  end

  defp azp_method!(opts) do
    case Keyword.get(opts, :method, :basic) do
      :basic ->
        {"basic", []}

      :tabu ->
        {"tabu", [pos_opt!(opts, :tabu_length, 10), pos_opt!(opts, :repetitions, 5)]}

      :annealing ->
        {"annealing", annealing_params!(opts)}

      other ->
        raise ArgumentError,
              ":method must be :basic, :tabu or :annealing, got #{inspect(other)}"
    end
  end

  defp annealing_params!(opts) do
    t = Keyword.get(opts, :init_temperature)

    unless is_number(t) and t > 0,
      do: raise(ArgumentError, ":init_temperature must be a positive number, got #{inspect(t)}")

    c = Keyword.get(opts, :cooling_factor, 0.85)

    unless is_number(c) and c > 0 and c < 1,
      do: raise(ArgumentError, ":cooling_factor must be in (0, 1), got #{inspect(c)}")

    [
      t * 1.0,
      c * 1.0,
      pos_opt!(opts, :sa_moves_term, 10),
      pos_opt!(opts, :repetitions, 5),
      pos_opt!(opts, :nonmoving_steps, 3)
    ]
  end

  defp pos_opt!(opts, key, default) do
    case Keyword.get(opts, key, default) do
      v when is_integer(v) and v > 0 ->
        v * 1.0

      other ->
        raise ArgumentError, "#{inspect(key)} must be a positive integer, got #{inspect(other)}"
    end
  end

  # Unknown option keys raise (as in the other modules); boolean flags must be booleans.
  defp allowed!(opts, allowed) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    case Keyword.keys(opts) -- allowed do
      [] ->
        :ok

      bad ->
        raise ArgumentError, "unknown option(s) #{inspect(bad)}; expected #{inspect(allowed)}"
    end

    for key <- [:symmetrize, :trace],
        Keyword.has_key?(opts, key),
        not is_boolean(opts[key]),
        do: raise(ArgumentError, "#{inspect(key)} must be a boolean, got #{inspect(opts[key])}")

    :ok
  end

  defp initial!(nil, _n), do: []

  defp initial!(labels, n) when is_list(labels) do
    if length(labels) != n do
      raise ArgumentError, "initial has #{length(labels)} labels for #{n} areas"
    end

    relabel(labels)
  end

  defp initial!(other, _n),
    do: raise(ArgumentError, "initial must be a list of labels, got #{inspect(other, limit: 3)}")

  # -- Max-p -------------------------------------------------------------------------------

  @doc """
  Max-p-regions: the largest number `p` of contiguous regions whose total of a spatially
  extensive attribute reaches a threshold, then the within-region dissimilarity (cityblock on
  `attrs`) is lowered by simulated annealing with a tabu list.

  The construction phase grows regions from randomly ordered areas until they reach the
  threshold; areas left over are attached by similarity to one of the `top_n` nearest
  neighboring areas' regions. The partitions reaching the highest `p` are each annealed
  `max_iterations_sa` times and the best result is returned.

  ## Options

    * `:threshold_variable` - required, the extensive attribute of every area (a column).
    * `:threshold` - required, the minimum total per region.
    * `:top_n` - areas considered for each enclave (default `2`; counts neighboring *areas*,
      so two neighbors in one region take two slots, as in spopt).
    * `:max_iterations_construction` - construction iterations (default `99`).
    * `:max_iterations_sa` - independent annealing runs per candidate (default `10`).
    * `:seed` - integer seed (default `nil`: a random seed, recorded in `seed`).
    * `:policy` - what to do with connected components whose total is below the threshold:
      `:single` (default) links each other component to the largest by one edge between their
      nearest areas, `:multiple` links every area of it to its nearest area in the largest,
      both needing `:coords`; `:drop` removes them (their labels are `nil`); `:keep` raises.
      Joins apply to every non-largest component once any component is infeasible.
    * `:coords` - one `{x, y}` per area, used by `:single` and `:multiple`.
    * `:symmetrize` - use `A or A'` for an asymmetric graph.

  Raises when no component reaches the threshold. The annealing parameters (`alpha = 0.998`,
  tabu length 10, `n` non-improving moves) are spopt's fixed constants.
  """
  @spec max_p(attrs(), Graph.t(), keyword()) :: MaxP.t()
  def max_p(attrs, %Graph{} = graph, opts \\ []) do
    allowed!(opts, [
      :coords,
      :max_iterations_construction,
      :max_iterations_sa,
      :policy,
      :seed,
      :symmetrize,
      :threshold,
      :threshold_variable,
      :top_n
    ])

    {bin, p} = attrs!(attrs, graph.n)
    n = graph.n

    {t_bin, ^n} =
      opts
      |> required!(:threshold_variable)
      |> Input.to_f64_binary(name: "threshold_variable", length: n)

    threshold = opts |> required!(:threshold) |> number!(:threshold)
    seed = seed!(opts)
    policy = enum_opt!(opts, :policy, :single, [:single, :multiple, :keep, :drop])

    {labels, p_found, objective, candidates} =
      Native.region_max_p(
        graph.ref,
        bin,
        p,
        t_bin,
        {threshold * 1.0, Keyword.get(opts, :top_n, 2),
         Keyword.get(opts, :max_iterations_construction, 99),
         Keyword.get(opts, :max_iterations_sa, 10), seed, Keyword.get(opts, :symmetrize, false)},
        Atom.to_string(policy),
        coords!(Keyword.get(opts, :coords), n)
      )
      |> unwrap()

    %MaxP{
      labels: Enum.map(labels, fn l -> if l < 0, do: nil, else: l end),
      p: p_found,
      objective: objective,
      n_candidates: candidates,
      seed: seed
    }
  end

  defp coords!(nil, _n), do: <<>>

  defp coords!(coords, n) do
    case Input.to_points_binary(coords) do
      {bin, ^n} -> bin
      _ -> raise ArgumentError, "coords must hold one {x, y} pair per area (#{n})"
    end
  end

  @doc false
  # Test hook: the Max-p construction phase for a given visiting `order`.
  @spec maxp_construct(
          attrs(),
          Graph.t(),
          [number()],
          number(),
          [non_neg_integer()],
          pos_integer(),
          integer()
        ) ::
          {labels(), pos_integer()}
  def maxp_construct(attrs, %Graph{} = graph, t, threshold, order, top_n, seed) do
    {bin, p} = attrs!(attrs, graph.n)
    {t_bin, _} = Input.to_f64_binary(t, name: "threshold_variable", length: graph.n)

    graph.ref
    |> Native.region_max_p_construct(bin, p, t_bin, threshold * 1.0, order, top_n, seed)
    |> unwrap()
  end

  # -- helpers -----------------------------------------------------------------------------

  @doc """
  Within-region heterogeneity of a partition.

  ## Options

    * `:kind` - `:pairwise` (default): the sum over regions of the distance between every pair
      of areas (spopt's `ObjectiveFunctionPairwise`, used by AZP and Max-p); `:center`: the sum
      of the distances of every area to its region's mean (`ObjectiveFunctionCenter`, SKATER's
      score).
    * `:metric` - `:manhattan` (default), `:euclidean` or `:sqeuclidean`. Use
      `kind: :center, metric: :sqeuclidean` for the within-region sum of squared deviations.

  `labels` may use any terms as region ids.

  ## Examples

      iex> Spatex.Region.objective([[0.0, 1.0, 5.0, 7.0]], [:a, :a, :b, :b])
      3.0
  """
  @spec objective(attrs(), [term()], keyword()) :: float()
  def objective(attrs, labels, opts \\ []) when is_list(labels) do
    allowed!(opts, [:kind, :metric])
    {bin, p} = attrs!(attrs, length(labels))
    kind = enum_opt!(opts, :kind, :pairwise, [:pairwise, :center])
    metric = enum_opt!(opts, :metric, :manhattan, [:manhattan, :euclidean, :sqeuclidean])

    bin
    |> Native.region_objective(p, relabel(labels), Atom.to_string(kind), Atom.to_string(metric))
    |> unwrap()
  end

  @doc """
  True when every region of `labels` induces a connected subgraph of `graph`.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2], [1, 0, 2, 1])
      iex> {Spatex.Region.contiguous?([0, 0, 1], g), Spatex.Region.contiguous?([0, 1, 0], g)}
      {true, false}
  """
  @spec contiguous?([term()], Graph.t()) :: boolean()
  def contiguous?(labels, %Graph{} = graph) when is_list(labels) do
    if length(labels) != graph.n do
      raise ArgumentError,
            "labels has #{length(labels)} entries but the graph has #{graph.n} nodes"
    end

    graph.ref |> Native.region_non_contiguous(relabel(labels)) |> unwrap() |> Enum.empty?()
  end

  # -- random regions -------------------------------------------------------------------------

  @random_opts [
    :cardinality,
    :compact,
    :max_swaps,
    :maxiter,
    :num_regions,
    :permutations,
    :seed,
    :symmetrize
  ]

  @doc """
  One random partition of `n` areas into regions, after spopt's `RandomRegion`.

  `area` is the number of areas, or a `Spatex.Graph` to require every region to be contiguous
  on it. Without a graph, contiguity is ignored: the areas are shuffled and cut at random
  breaks. With a graph, regions are grown one at a time (largest cardinality first) from
  random areas; when the remaining areas cannot complete a region, an area on the border of an
  existing region is swapped out of it (keeping that region connected) and offered again. The
  construction is retried up to `:maxiter` times; if it never succeeds the result has
  `feasible: false`, as in spopt, instead of raising.

  Without `:cardinality` the sizes are random: `:num_regions` regions (drawn between `2` and
  `n - 1` when omitted) whose sizes come from `num_regions - 1` distinct random cut points
  between `1` and `n - 2`, so the last cut region always holds at least two areas (spopt's
  scheme).

  ## Options

    * `:num_regions` - number of regions. With `:cardinality` it must equal its length. Without
      it, at most `n - 1` (spopt loops forever for more).
    * `:cardinality` - region sizes, positive integers adding up to `n`. The regions are
      built largest first; without a graph they keep the given order.
    * `:maxiter` - attempts at a contiguous solution (default `1000`; only with a graph).
    * `:compact` - grow compact regions by exhausting the current frontier before opening
      a new one (default `false`; only with a graph).
    * `:max_swaps` - cap on swaps per attempt (default `1_000_000`; only with a graph). spopt
      applies the cap between swaps only and can loop forever when no swap is possible; Spatex
      also stops inside a swap search, and then retries.
    * `:seed` - integer seed (default `nil`: a random seed, recorded in `seed`). ChaCha8
      streams: a seed gives the same partition on every run and platform but not numpy's.
    * `:symmetrize` - use `A or A'` for an asymmetric graph.

  Labels are 0-based and areas are positions in the graph (or `0..n-1`); spopt takes arbitrary
  area ids and requires `area_ids` to equal the weights' `id_order`.

  ## Examples

      iex> r = Spatex.Region.random_region(6, cardinality: [3, 2, 1], seed: 1)
      iex> {r.feasible, Enum.map(r.regions, &length/1), r.n_regions}
      {true, [3, 2, 1], 3}
      iex> r.labels |> Enum.sort()
      [0, 0, 0, 1, 1, 2]
  """
  @spec random_region(pos_integer() | Graph.t(), keyword()) :: RandomRegion.t()
  def random_region(area, opts \\ []) do
    allowed!(opts, @random_opts -- [:permutations])
    seed = seed!(opts)
    [{regions, feasible, attempts}] = random_call(area, opts, 1000, seed, 1)
    random_result(regions, feasible, attempts, seed)
  end

  @doc """
  `:permutations` random partitions, after spopt's `RandomRegions`.

  Takes the options of `random_region/2` (with `:maxiter` defaulting to spopt's `100` here)
  and `:permutations` (default `99`). Permutation `i` draws from ChaCha8 stream `i` of
  `seed:`, so the result does not depend on the thread count, and the first permutation equals
  `random_region/2` with the same seed. Returns the solutions and the feasible ones.

  ## Examples

      iex> r = Spatex.Region.random_regions(8, num_regions: 3, permutations: 4, seed: 7)
      iex> {length(r.solutions), length(r.feasible_solutions)}
      {4, 4}
  """
  @spec random_regions(pos_integer() | Graph.t(), keyword()) :: RandomRegions.t()
  def random_regions(area, opts \\ []) do
    allowed!(opts, @random_opts)
    seed = seed!(opts)

    permutations =
      case Keyword.get(opts, :permutations, 99) do
        p when is_integer(p) and p >= 1 ->
          p

        other ->
          raise ArgumentError, "permutations must be a positive integer, got #{inspect(other)}"
      end

    solutions =
      area
      |> random_call(opts, 100, seed, permutations)
      |> Enum.map(fn {regions, feasible, attempts} ->
        random_result(regions, feasible, attempts, seed)
      end)

    %RandomRegions{
      solutions: solutions,
      feasible_solutions: Enum.filter(solutions, & &1.feasible),
      seed: seed
    }
  end

  defp random_call(area, opts, default_maxiter, seed, permutations) do
    {graph_ref, n} = random_area!(area)

    num_regions = random_num_regions!(Keyword.get(opts, :num_regions))
    cardinality = random_cardinality!(Keyword.get(opts, :cardinality))
    maxiter = random_pos!(opts, :maxiter, default_maxiter)
    max_swaps = random_pos!(opts, :max_swaps, 1_000_000)
    compact = random_bool!(opts, :compact)

    graph_ref
    |> Native.region_random(
      n,
      num_regions,
      cardinality,
      {maxiter, max_swaps, compact, Keyword.get(opts, :symmetrize, false)},
      seed,
      permutations
    )
    |> unwrap()
  end

  defp random_num_regions!(nil), do: nil
  defp random_num_regions!(k) when is_integer(k) and k >= 1, do: k

  defp random_num_regions!(other),
    do: raise(ArgumentError, "num_regions must be a positive integer, got #{inspect(other)}")

  defp random_cardinality!(nil), do: []

  defp random_cardinality!([_ | _] = cards) do
    if Enum.all?(cards, &(is_integer(&1) and &1 >= 1)),
      do: cards,
      else: raise(ArgumentError, "cardinality must hold positive integers, got #{inspect(cards)}")
  end

  defp random_cardinality!(other),
    do: raise(ArgumentError, "cardinality must be a non-empty list, got #{inspect(other)}")

  defp random_area!(%Graph{} = graph), do: {graph.ref, graph.n}
  defp random_area!(n) when is_integer(n) and n >= 1, do: {nil, n}

  defp random_area!(other),
    do:
      raise(
        ArgumentError,
        "expected the number of areas (a positive integer) or a Spatex.Graph, got #{inspect(other, limit: 3)}"
      )

  defp random_pos!(opts, key, default) do
    case Keyword.get(opts, key, default) do
      v when is_integer(v) and v >= 1 -> v
      other -> raise ArgumentError, "#{key} must be a positive integer, got #{inspect(other)}"
    end
  end

  defp random_bool!(opts, key) do
    case Keyword.get(opts, key, false) do
      v when is_boolean(v) -> v
      other -> raise ArgumentError, "#{key} must be a boolean, got #{inspect(other)}"
    end
  end

  defp random_result(regions, feasible, attempts, seed) do
    labels =
      if feasible do
        pairs = for {members, r} <- Enum.with_index(regions), a <- members, do: {a, r}
        pairs |> Enum.sort() |> Enum.map(&elem(&1, 1))
      end

    %RandomRegion{
      labels: labels,
      regions: regions,
      n_regions: length(regions),
      feasible: feasible,
      attempts: attempts,
      seed: seed
    }
  end

  @doc """
  Renumbers labels `0..k-1` by order of first appearance, so partitions compare regardless of
  numbering.

  ## Examples

      iex> Spatex.Region.relabel([5, 5, 2, 9, 2])
      [0, 0, 1, 2, 1]
  """
  @spec relabel([term()]) :: labels()
  def relabel(labels) when is_list(labels) do
    {out, _} =
      Enum.map_reduce(labels, %{}, fn label, seen ->
        case seen do
          %{^label => id} -> {id, seen}
          _ -> {map_size(seen), Map.put(seen, label, map_size(seen))}
        end
      end)

    out
  end

  @doc """
  Z-scores every column (`(x - mean) / std`, population standard deviation). Constant columns
  become zeros. Returns a list of columns, ready for the region functions.

  ## Examples

      iex> Spatex.Region.standardize([[1.0, 3.0]])
      [[-1.0, 1.0]]
  """
  @spec standardize(attrs()) :: [[float()]]
  def standardize(attrs) do
    attrs
    |> columns!()
    |> Enum.map(fn {bin, _n} -> bin |> Input.from_f64_binary() |> zscores() end)
  end

  defp zscores(values) do
    n = length(values)
    mean = Enum.sum(values) / n
    std = :math.sqrt(Enum.sum(Enum.map(values, &((&1 - mean) * (&1 - mean)))) / n)
    scale = fn v -> if std == 0.0, do: 0.0, else: (v - mean) / std end
    Enum.map(values, scale)
  end

  # -- input normalization -----------------------------------------------------------------

  # Column-major f64 binary of the attribute table and its number of columns.
  defp attrs!(attrs, n) do
    cols = columns!(attrs)

    Enum.each(cols, fn {_bin, len} ->
      if len != n do
        raise ArgumentError, "attributes have #{len} rows but the graph has #{n} nodes"
      end
    end)

    {cols |> Enum.map(&elem(&1, 0)) |> IO.iodata_to_binary(), length(cols)}
  end

  defp columns!(attrs) do
    cols = split_columns(attrs)
    if cols == [], do: raise(ArgumentError, "attrs has no columns")

    cols
    |> Enum.with_index(1)
    |> Enum.map(fn {col, i} -> Input.to_f64_binary(col, name: "attrs column #{i}") end)
  end

  defp split_columns(%{__struct__: Explorer.DataFrame} = df) do
    names = call(Explorer.DataFrame, :names, [df])
    Enum.map(names, &call(Explorer.DataFrame, :pull, [df, &1]))
  end

  defp split_columns(%{__struct__: Nx.Tensor} = t) do
    case call(Nx, :rank, [t]) do
      1 -> [t]
      2 -> Nx |> call(:transpose, [t]) |> then(&call(Nx, :to_list, [&1]))
      rank -> raise ArgumentError, "attrs tensor must have rank 1 or 2, got rank #{rank}"
    end
  end

  defp split_columns(%{__struct__: Explorer.Series} = s), do: [s]

  defp split_columns(%{} = map) when not is_struct(map),
    do: map |> Enum.sort_by(fn {k, _} -> to_string(k) end) |> Enum.map(&elem(&1, 1))

  defp split_columns([]), do: []

  defp split_columns(list) when is_list(list) do
    if Enum.all?(list, &(is_number(&1) or &1 in [:nan, :infinity, :neg_infinity])),
      do: [list],
      else: list
  end

  defp split_columns(other),
    do: raise(ArgumentError, "unsupported attrs #{inspect(other, limit: 3)}; expected columns")

  # -- option validation -------------------------------------------------------------------

  defp n_clusters!(opts, n) do
    case Keyword.get(opts, :n_clusters) do
      k when is_integer(k) and k >= 1 and k <= n ->
        k

      other ->
        raise ArgumentError,
              "n_clusters must be an integer between 1 and #{n} (number of areas), got #{inspect(other)}"
    end
  end

  # `nil` (the default) draws a seed, recorded in the result's `seed`; an integer is used as given.
  defp seed!(opts) do
    case Keyword.get(opts, :seed) do
      nil -> :rand.uniform(4_294_967_296) - 1
      s when is_integer(s) and s >= 0 -> s
      other -> raise ArgumentError, "seed must be a non-negative integer, got #{inspect(other)}"
    end
  end

  defp required!(opts, key) do
    case Keyword.fetch(opts, key) do
      {:ok, value} when not is_nil(value) -> value
      _ -> raise ArgumentError, "missing required option #{inspect(key)}"
    end
  end

  defp number!(value, _key) when is_number(value), do: value

  defp number!(value, key),
    do: raise(ArgumentError, "#{key} must be a number, got #{inspect(value)}")

  defp number_opt!(opts, key, default) do
    case Keyword.get(opts, key) do
      nil -> default
      value -> number!(value, key)
    end
  end

  defp enum_opt!(opts, key, default, allowed) do
    value = Keyword.get(opts, key, default)

    if value in allowed,
      do: value,
      else:
        raise(ArgumentError, "#{key} must be one of #{inspect(allowed)}, got #{inspect(value)}")
  end

  # Dispatch through a runtime module argument so the optional dependencies are never
  # referenced as compile-time remote calls.
  defp call(mod, fun, args), do: apply(mod, fun, args)

  defp unwrap({:ok, value}), do: value
  defp unwrap({:error, reason}), do: raise(ArgumentError, to_string(reason))
end
