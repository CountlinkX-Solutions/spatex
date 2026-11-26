defmodule Spatex.Graph do
  @moduledoc """
  Spatial weights as an immutable graph, mirroring `libpysal.graph.Graph`.

  The adjacency lives in Rust memory as a compressed sparse row (CSR) structure behind a
  resource reference. The Elixir struct holds only the reference, the node count `n`, the
  node `ids` and the current `transform`:

      %Spatex.Graph{ref: ref, n: 3, ids: [:a, :b, :c], transform: :o}

  ## Nodes and ordering

  Nodes are numbered by their position in `ids`. Rows are ordered by focal position and,
  within a row, by neighbor position (libpysal's canonical order). Every node appears as a
  focal node: a node without outgoing edges is an *island* and is stored, and reported by
  `to_adjacency/1`, as a zero-weight self-loop `{id, id, 0.0}`. A neighbor only counts when
  its weight is non-zero, so `cardinalities/1` and `isolates/1` ignore zero weights.

  ## Weights and transforms

  Weights are numbers (integers are converted to floats); duplicate `{focal, neighbor}`
  pairs are an error and negative weights are kept as given. `transform/2` returns a new
  graph and applies to the current weights: `:r` row-standardizes (rows that sum to zero
  stay zero), `:b` makes weights binary, `:d` scales so all weights sum to one, `:v` is the
  variance-stabilizing transform. `:o` is the identity; original weights cannot be restored
  from a transformed graph.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([:a, :a, :b, :c], [:b, :c, :a, :a], [1, 3, 2, 4])
      iex> Spatex.Graph.ids(g)
      [:a, :b, :c]
      iex> Spatex.Graph.to_adjacency(g)
      [{:a, :b, 1.0}, {:a, :c, 3.0}, {:b, :a, 2.0}, {:c, :a, 4.0}]
      iex> r = Spatex.Graph.transform(g, :r)
      iex> Spatex.Graph.to_adjacency(r)
      [{:a, :b, 0.25}, {:a, :c, 0.75}, {:b, :a, 1.0}, {:c, :a, 1.0}]
      iex> Spatex.Graph.lag(r, [10, 20, 30])
      [27.5, 10.0, 10.0]
      iex> g |> Spatex.Graph.s0()
      10.0
      iex> inspect(r)
      "#Spatex.Graph<n: 3, nnz: 4, transform: :r>"
  """

  alias Spatex.Graph.Builders
  alias Spatex.Graph.Contiguity
  alias Spatex.Graph.IO, as: GraphIO
  alias Spatex.Input
  alias Spatex.Native

  @enforce_keys [:ref, :n, :ids, :transform]
  defstruct [:ref, :n, :ids, :transform]

  @typedoc "A node identifier: any term, unique within the graph."
  @type id :: term()

  @typedoc "Weight transform label."
  @type transform :: :o | :r | :b | :d | :v

  @type t :: %__MODULE__{
          ref: reference(),
          n: non_neg_integer(),
          ids: [id()],
          transform: transform()
        }

  @transforms [:o, :r, :b, :d, :v]
  @default_symmetry_tol 1.0e-12

  @doc """
  Builds a graph from parallel `focal`, `neighbor` and `weight` lists.

  `focal` and `neighbor` hold node ids; `weight` defaults to `1.0` for every edge. A
  keyword list may be given in place of `weight`.

  ## Options

    * `:ids` - the node order. Must be unique and contain every focal and neighbor id.
      Nodes in `:ids` without outgoing edges become islands. Defaults to the ids in order
      of first appearance, scanning `focal` and then `neighbor` (libpysal's `unique_ids`).

  Raises `ArgumentError` on length mismatches, non-numeric weights, duplicate edges, ids
  missing from `:ids`, or a non-unique `:ids`.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([:a, :b], [:b, :c])
      iex> {Spatex.Graph.ids(g), Spatex.Graph.isolates(g)}
      {[:a, :b, :c], [:c]}

      iex> g = Spatex.Graph.from_arrays([1], [2], [0.5], ids: [2, 1, 3])
      iex> Spatex.Graph.to_adjacency(g)
      [{2, 2, 0.0}, {1, 2, 0.5}, {3, 3, 0.0}]
  """
  @spec from_arrays([id()], [id()], [number()] | nil | keyword(), keyword()) :: t()
  def from_arrays(focal, neighbor, weight \\ nil, opts \\ [])

  def from_arrays(focal, neighbor, opts, []) when is_list(opts) and opts != [] do
    if Keyword.keyword?(opts),
      do: build(focal, neighbor, nil, opts),
      else: build(focal, neighbor, opts, [])
  end

  def from_arrays(focal, neighbor, weight, opts), do: build(focal, neighbor, weight, opts)

  @doc """
  Builds a graph from an adjacency list or map.

  Accepts a list of `{focal, neighbor, weight}` triples (as returned by `to_adjacency/1`)
  or a map `%{focal => [{neighbor, weight}]}`. For a map the focal keys are taken in term
  order, followed by neighbors that are not keys, in order of first appearance; a key with
  an empty list is an island. Takes the same options as `from_arrays/4`; an explicit `:ids`
  overrides the derived order.

  ## Examples

      iex> g = Spatex.Graph.from_adjacency(%{b: [{:a, 2}], a: [{:b, 1}, {:c, 3}]})
      iex> Spatex.Graph.ids(g)
      [:a, :b, :c]
  """
  @spec from_adjacency([{id(), id(), number()}] | %{id() => [{id(), number()}]}, keyword()) ::
          t()
  def from_adjacency(adjacency, opts \\ [])

  def from_adjacency(adjacency, opts) when is_list(adjacency) do
    triples = Enum.map(adjacency, &triple!/1)
    {focal, neighbor, weight} = unzip3(triples)
    build(focal, neighbor, weight, opts)
  end

  def from_adjacency(adjacency, opts) when is_map(adjacency) do
    keys = adjacency |> Map.keys() |> Enum.sort()

    triples =
      Enum.flat_map(keys, fn focal ->
        case Map.fetch!(adjacency, focal) do
          list when is_list(list) -> Enum.map(list, &pair!(focal, &1))
          _ -> bad_adjacency!()
        end
      end)

    {focal, neighbor, weight} = unzip3(triples)
    derived = Enum.uniq(keys ++ neighbor)
    build(focal, neighbor, weight, Keyword.put_new(opts, :ids, derived))
  end

  def from_adjacency(_other, _opts), do: bad_adjacency!()

  @doc "Returns the edges as `{focal, neighbor, weight}` triples in canonical order, islands included."
  @spec to_adjacency(t()) :: [{id(), id(), float()}]
  def to_adjacency(%__MODULE__{ref: ref, ids: ids}) do
    {focal, neighbor, weight} = Native.graph_to_arrays(ref)
    lookup = List.to_tuple(ids)

    zip3(u32_list(focal), u32_list(neighbor), f64_list(weight), fn f, t, w ->
      {elem(lookup, f), elem(lookup, t), w}
    end)
  end

  @doc """
  Exports the adjacency as the compressed sparse row (CSR) structure of libpysal's
  `Graph.sparse`, as `Nx` tensors: `%{indptr: _, indices: _, data: _, shape: {n, n}}`.

  Row `i` holds the entries `indices[indptr[i]..indptr[i + 1]]` with weights
  `data[indptr[i]..indptr[i + 1]]`; rows follow the node order of `ids/1` and `indices` are
  sorted within each row. Exactly the stored entries are exported, as in libpysal: islands
  appear as an explicit zero self-loop and explicit zero weights are kept. The data reflect
  the current transform.

  Tensor types: `indptr` is `{:s, 64}`, `indices` is `{:u, 32}` and `data` is `{:f, 64}`
  (scipy uses `int32` indices; these hold every value losslessly).

  Needs the optional `:nx` dependency; raises `ArgumentError` when it is not available.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([:a, :b], [:b, :a], [1.0, 2.0])
      iex> sparse = Spatex.Graph.to_sparse(g)
      iex> {Nx.to_flat_list(sparse.indptr), Nx.to_flat_list(sparse.indices)}
      {[0, 1, 2], [1, 0]}
      iex> {Nx.to_flat_list(sparse.data), sparse.shape}
      {[1.0, 2.0], {2, 2}}
  """
  @spec to_sparse(t()) :: %{
          indptr: Nx.Tensor.t(),
          indices: Nx.Tensor.t(),
          data: Nx.Tensor.t(),
          shape: {non_neg_integer(), non_neg_integer()}
        }
  def to_sparse(%__MODULE__{ref: ref, n: n}) do
    Input.require_nx!("Spatex.Graph.to_sparse/1")
    {indptr, indices, data} = Native.graph_to_csr(ref)

    %{
      indptr: tensor(indptr, {:s, 64}),
      indices: tensor(indices, {:u, 32}),
      data: tensor(data, {:f, 64}),
      shape: {n, n}
    }
  end

  # Nx is an optional dependency: call it dynamically so Spatex compiles without it.
  defp tensor(bin, type) do
    nx = Nx
    nx.from_binary(bin, type)
  end

  @doc "Number of nodes."
  @spec n(t()) :: non_neg_integer()
  def n(%__MODULE__{n: n}), do: n

  @doc "Node ids in graph order."
  @spec ids(t()) :: [id()]
  def ids(%__MODULE__{ids: ids}), do: ids

  @doc "Number of edges with a non-zero weight (island self-loops excluded)."
  @spec nnz(t()) :: non_neg_integer()
  def nnz(%__MODULE__{ref: ref}), do: Native.graph_nnz(ref)

  @doc """
  Number of non-zero-weight neighbors per node, as a list aligned with `ids/1`.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([:a, :a, :b], [:b, :c, :a])
      iex> Spatex.Graph.cardinalities(g)
      [2, 1, 0]
  """
  @spec cardinalities(t()) :: [non_neg_integer()]
  def cardinalities(%__MODULE__{ref: ref}), do: ref |> Native.graph_cardinalities() |> u32_list()

  @doc "Ids of the nodes without any (non-zero-weight) neighbor, in graph order."
  @spec isolates(t()) :: [id()]
  def isolates(%__MODULE__{ref: ref, ids: ids}) do
    lookup = List.to_tuple(ids)
    ref |> Native.graph_isolates() |> u32_list() |> Enum.map(&elem(lookup, &1))
  end

  @doc """
  True when `w_ij == w_ji` for every pair within `tol` (relative to `max(1, |w_ij|, |w_ji|)`,
  default `#{@default_symmetry_tol}`). A missing reverse edge counts as weight zero.
  """
  @spec symmetric?(t(), number()) :: boolean()
  def symmetric?(%__MODULE__{ref: ref}, tol \\ @default_symmetry_tol)
      when is_number(tol) and tol >= 0,
      do: Native.graph_symmetric(ref, tol * 1.0)

  @doc "`S0 = sum_ij w_ij` of the current weights."
  @spec s0(t()) :: float()
  def s0(%__MODULE__{ref: ref}), do: elem(Native.graph_stats(ref), 0)

  @doc "`S1 = 1/2 sum_ij (w_ij + w_ji)^2` of the current weights."
  @spec s1(t()) :: float()
  def s1(%__MODULE__{ref: ref}), do: elem(Native.graph_stats(ref), 1)

  @doc "`S2 = sum_i (sum_j w_ij + sum_j w_ji)^2` of the current weights."
  @spec s2(t()) :: float()
  def s2(%__MODULE__{ref: ref}), do: elem(Native.graph_stats(ref), 2)

  @doc """
  Spatial lag `(Wy)_i = sum_j w_ij * y_j` of `y` under the current weights.

  `y` is a list of numbers aligned with `ids/1`, or a binary of native-endian 64-bit floats.
  Islands get a lag of `0.0`. Returns a list of floats.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 0, 1], [1, 2, 0]) |> Spatex.Graph.transform(:r)
      iex> Spatex.Graph.lag(g, [1.0, 2.0, 3.0])
      [2.5, 1.0, 0.0]
  """
  @spec lag(t(), [number()] | binary()) :: [float()]
  def lag(%__MODULE__{ref: ref, n: n}, y) do
    bin = y_binary!(y, n)
    ref |> Native.graph_lag(bin) |> unwrap() |> f64_list()
  end

  @doc """
  Applies a weight transform and returns a new graph.

  `kind` is `:o`, `:r`, `:b`, `:d` or `:v` (or the strings `"O"`, `"r"`, ... in any case).
  The transform is applied to the graph's current weights, like libpysal; asking for the
  graph's current transform returns it unchanged, and `:o` on a transformed graph raises.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([:a, :a, :b], [:b, :c, :a], [2, 2, 5])
      iex> g |> Spatex.Graph.transform("B") |> Spatex.Graph.to_adjacency()
      [{:a, :b, 1.0}, {:a, :c, 1.0}, {:b, :a, 1.0}, {:c, :c, 0.0}]
  """
  @spec transform(t(), transform() | String.t()) :: t()
  def transform(%__MODULE__{} = graph, kind) do
    case normalize_transform!(kind) do
      same when same == graph.transform ->
        graph

      :o ->
        raise ArgumentError,
              "cannot restore the original weights of a #{inspect(graph.transform)} graph; " <>
                "keep the original graph instead"

      other ->
        label = other |> Atom.to_string() |> String.upcase()
        ref = unwrap(Native.graph_transform(graph.ref, label))
        %{graph | ref: ref, transform: other}
    end
  end

  @doc """
  Builds a contiguity graph from polygon geometries given as WKB binaries.

  Follows libpysal's vertex-set intersection (`Graph.build_contiguity`): geometries sharing an
  identical vertex are *queen* neighbors, geometries sharing an identical edge (two consecutive
  vertices) are *rook* neighbors. Coordinates are compared exactly (`-0.0 == 0.0`), so a
  T-junction is not rook contiguity. Holes count: a polygon filling another's hole is its
  neighbor. Weights are `1.0`; polygons without neighbors become islands. Node order is the
  input order.

  Accepts `Polygon`, `MultiPolygon`, `LineString` and `MultiLineString` WKB (Z/M values are
  ignored; lines only with the vertex method); points, a `nil` entry, malformed WKB or a
  non-finite coordinate raise `ArgumentError` naming the geometry index.

  ## Options

    * `:rook` - `true` (default, like libpysal's `Graph`) for shared edges, `false` for queen
      (shared vertices).
    * `:strict` - `true` uses geometric predicates instead of identical vertices / edges:
      queen neighbors touch (boundaries meet, interiors disjoint) and rook neighbors share a
      boundary piece of positive length (DE-9IM `F***1****`), so T-junctions count. Default
      `false`.
    * `:by_perimeter` - `true` weights every pair by the length of its shared boundary (`0.0`
      for a point contact, which stays stored but does not count as a neighbor); raises when
      two neighbors overlap. Default `false`.
    * `:ids` - unique node ids, one per geometry (default `0..n-1`).
  """
  @spec build_contiguity([binary()], keyword()) :: t()
  def build_contiguity(wkbs, opts \\ []), do: Contiguity.build(wkbs, opts)

  @doc """
  Builds a fuzzy contiguity graph from polygon WKB binaries: libpysal's
  `Graph.build_fuzzy_contiguity`, which relaxes contiguity for geometries that do not share
  exact edges (slivers, gaps, digitizing errors).

  Every pair `{i, j}` for which `predicate(geometry_i, geometry_j)` holds becomes an edge with
  weight `1.0`. By default (`:intersects`) the polygons must touch or overlap. With `:buffer`
  (or `:tolerance`) every polygon is first grown by that distance, so polygons at most twice
  the buffer apart become neighbors. Polygons without a neighbor become islands; node order is
  the input order.

  Spatex does not build buffered geometries (there is no GEOS). It uses the fact that two round
  buffers of radius `b` intersect exactly when the polygons are at most `2b` apart, computed
  with exact polygon-to-polygon distances. GEOS draws arcs with 16 chords per quadrant, which
  makes its buffers up to 0.12 % smaller, so a pair whose distance is within about 0.15 % below
  `2b` can be a neighbor here and not in libpysal (see the parity notes). Everything else
  matched libpysal on its committed fixtures.

  Accepts `Polygon` and `MultiPolygon` WKB; anything else, malformed WKB or a non-finite
  coordinate raises `ArgumentError` naming the geometry index.

  ## Options

    * `:tolerance` - the buffer as a share of half the shorter side of the total bounds of the
      geometries: `buffer = tolerance * 0.5 * min(width, height)`. Exclusive with `:buffer`.
    * `:buffer` - the buffer distance, in the units of the coordinates (not negative; `0`
      means no buffer). Exclusive with `:tolerance`.
    * `:predicate` - `:intersects` (default), or `nil` for intersecting bounding boxes (grown
      by the buffer), or without a buffer also `:touches`, `:overlaps`, `:within`, `:contains`,
      `:covers`, `:covered_by`, `:contains_properly` and `:crosses` (never true for
      polygons). The predicates are directed, `predicate(focal, neighbor)`, so `:contains`
      and `:within` give converse graphs. `:dwithin` is not ported, and a buffer combined with
      any predicate other than `:intersects` or `nil` raises, because the buffered geometries
      are never built.
    * `:ids` - unique node ids, one per geometry (default `0..n-1`).

  ## Examples

      iex> square = fn x ->
      ...>   <<1, 3::little-32, 1::little-32, 5::little-32>> <>
      ...>     for({px, py} <- [{x, 0}, {x + 1, 0}, {x + 1, 1}, {x, 1}, {x, 0}], into: <<>>) do
      ...>       <<px * 1.0::little-float-64, py * 1.0::little-float-64>>
      ...>     end
      ...> end
      iex> wkbs = [square.(0), square.(1.5), square.(5)]
      iex> Spatex.Graph.isolates(Spatex.Graph.build_fuzzy_contiguity(wkbs))
      [0, 1, 2]
      iex> graph = Spatex.Graph.build_fuzzy_contiguity(wkbs, buffer: 0.5)
      iex> Spatex.Graph.to_adjacency(graph)
      [{0, 1, 1.0}, {1, 0, 1.0}, {2, 2, 0.0}]
  """
  @spec build_fuzzy_contiguity([binary()], keyword()) :: t()
  def build_fuzzy_contiguity(wkbs, opts \\ []), do: Contiguity.build_fuzzy(wkbs, opts)

  @doc """
  Builds a block contiguity graph: nodes with the same regime are neighbors of each other.

  `regimes` holds one non-`nil` label per node (compared with `===`); nodes alone in their
  regime are islands. Weights are `1.0`.

  Unlike libpysal, which lists the nodes regime by regime, nodes keep their input order (and
  `:ids`, default `0..n-1`, names them); compare against PySAL by id.

  ## Examples

      iex> g = Spatex.Graph.build_block_contiguity(["a", "b", "a", "c", "b"])
      iex> g |> Spatex.Graph.to_adjacency() |> Enum.reject(fn {_, _, w} -> w == 0.0 end)
      [{0, 2, 1.0}, {1, 4, 1.0}, {2, 0, 1.0}, {4, 1, 1.0}]
  """
  @spec build_block_contiguity([term()], keyword()) :: t()
  def build_block_contiguity(regimes, opts \\ []),
    do: Contiguity.build_block(regimes, opts)

  @doc """
  Builds the k-nearest-neighbor graph of a point set, like `libpysal.graph.Graph.build_knn`.

  `points` is anything `Spatex.Input.to_points_binary/1` accepts (`[{x, y}]`, `[[x, y]]`,
  an `{n, 2}` `Nx.Tensor` or a pair `{xs, ys}`). Distances are Euclidean. Every node gets
  its `k` nearest other nodes with weight `1.0`; the graph is generally asymmetric.
  Equidistant candidates are resolved by lowest point index (libpysal follows scikit-learn
  tree order there, see `docs/parity.md`).

  ## Options

    * `:ids` - node ids, a unique list with one entry per point. Defaults to `0..n-1`.
    * `:coincident` - `:raise` (default) raises `ArgumentError` when two points share a
      location; `:clique` links all points on one site to each other and to the neighbors of
      the site, so a row can hold more than `k` neighbors; `:jitter` moves the points of each
      coincident group by a tiny seeded displacement (radius `max(|x|, |y|, 1) * 2^-30`) until
      all points are distinct. Distinct points are untouched; libpysal's jitter moves every
      point, unseeded (see `docs/parity.md`).
    * `:seed` - seed of `coincident: :jitter` (default `0`).
    * `:metric` - `:euclidean` (default) or `:haversine`: points are `{lon, lat}` in degrees
      (`-180 < lon < 180`, `-90 < lat < 90`) and distances are great-circle kilometres
      (Earth radius 6371 km). libpysal leaves haversine distances in radians; see
      `docs/parity.md`.
    * `:max_edges` - edge budget (default 50_000_000). `n * k` above it raises
      `ArgumentError` before the graph is built.

  Raises `ArgumentError` unless `1 <= k <= n - 1` (distinct locations minus one for
  `coincident: :clique`).

  ## Examples

      iex> g = Spatex.Graph.build_knn([{0, 0}, {1, 0}, {5, 0}], 1)
      iex> Spatex.Graph.to_adjacency(g)
      [{0, 1, 1.0}, {1, 0, 1.0}, {2, 1, 1.0}]
  """
  @spec build_knn(term(), pos_integer(), keyword()) :: t()
  def build_knn(points, k, opts \\ []), do: Builders.Knn.knn(points, k, opts)

  @doc """
  Builds a kernel-weighted graph of a point set, like `libpysal.graph.Graph.build_kernel`.

  `points` is as for `build_knn/3`. The weight of the edge `i -> j` is
  `kernel(d_ij, h)` with the Euclidean distance `d_ij`. Self weights are never present
  (add them with `assign_self_weight` once available). Which pairs are candidates depends on
  the options:

    * without `:k`, all pairs within a fixed `:bandwidth` (or every pair when the bandwidth
      is the default or `taper: false` meets a non-compact kernel, which is O(n^2));
    * with `:k`, the `k` nearest neighbors of each node (the graph is then generally
      asymmetric).

  ## Options

    * `:kernel` - `:gaussian` (default), `:triangular`, `:parabolic` (alias `:epanechnikov`),
      `:bisquare`, `:tricube`, `:cosine`, `:boxcar` (aliases `:uniform`, `:discrete`; weight
      `1.0` for `d < h`, not `1/2`), `:exponential` or `:identity` (the raw distance).
    * `:bandwidth` - a positive number (global), `:adaptive` (each node uses its distance to
      its `k`-th neighbor times `1.0000001`; requires `:k`) or `nil` (default): the 25th
      percentile of all pairwise distances, or the largest `k`-nearest distance when `:k` is
      given. libpysal's `"auto"` is not supported.
    * `:k` - number of nearest neighbors, `1 <= k <= n - 1`.
    * `:taper` - `true` (default) drops weights beyond the bandwidth and stored zeros.
    * `:decay` - `true` divides by the kernel value at distance zero (peak 1). Not defined
      for `:identity`.
    * `:ids` - node ids, a unique list with one entry per point. Defaults to `0..n-1`.
    * `:coincident` - `:raise` (default), `:clique` or `:jitter` (with `:seed`), only relevant
      with `:k` (see `build_knn/3`); jittered points also give the kernel distances. Without `:k`, coincident pairs are kept with distance `0`, whereas
      libpysal drops them on its fixed-bandwidth fast path (see `docs/parity.md`).
    * `:metric` - `:euclidean` (default) or `:haversine`: points are `{lon, lat}` in degrees
      (`-180 < lon < 180`, `-90 < lat < 90`) and distances (and a fixed `:bandwidth`) are great-circle kilometres
      (Earth radius 6371 km). libpysal leaves haversine distances in radians; see
      `docs/parity.md`.
    * `:max_edges` - edge budget (default 50_000_000). Without `:k` or a compact kernel
      with `:bandwidth` every ordered pair is an edge (`n * (n - 1)`); more edges than this
      raise `ArgumentError` before (or while) they are collected.

  Raises `ArgumentError` on invalid options.

  ## Examples

      iex> g = Spatex.Graph.build_kernel([{0, 0}, {1, 0}, {5, 0}], kernel: :triangular, bandwidth: 2.0)
      iex> Spatex.Graph.to_adjacency(g)
      [{0, 1, 0.5}, {1, 0, 0.5}, {2, 2, 0.0}]
  """
  @spec build_kernel(term(), keyword()) :: t()
  def build_kernel(points, opts \\ []), do: Builders.Kernel.kernel(points, opts)

  @doc """
  Builds the distance-band graph of a point set, like `libpysal.graph.Graph.build_distance_band`.

  Two nodes are neighbors when their Euclidean distance is at most `threshold` (inclusive).
  Nodes without a neighbor are islands. `points` is as for `build_knn/3`.

  ## Options

    * `:ids` - node ids, a unique list with one entry per point. Defaults to `0..n-1`.
    * `:binary` - `true` (default) gives weight `1.0`; `false` gives `distance ** alpha`.
    * `:alpha` - the power for `binary: false` (default `-1.0`, inverse distance). Coincident
      points raise for negative `alpha` (libpysal yields infinite weights); a zero weight
      (`alpha > 0` at distance zero) drops the edge.

    * `:kernel` - with `binary: false`, weights come from this kernel (see `build_kernel/2`)
      applied to the band distances instead of `distance ** alpha`. `:alpha` is then unused.
    * `:bandwidth` - the kernel bandwidth, a positive number. Defaults to the 25th percentile
      of the band's distances, as in libpysal.
    * `:taper` / `:decay` - as for `build_kernel/2` (default `true` / `false`).
    * `:metric` - `:euclidean` (default) or `:haversine`: points are `{lon, lat}` in degrees
      (`-180 < lon < 180`, `-90 < lat < 90`) and `threshold` and distances are great-circle kilometres
      (Earth radius 6371 km). libpysal leaves haversine distances in radians; see
      `docs/parity.md`.
    * `:max_edges` - edge budget (default 50_000_000). A band holding more edges raises
      `ArgumentError` as soon as the budget is passed.

  Raises `ArgumentError` unless `threshold` is a positive number.

  ## Examples

      iex> g = Spatex.Graph.build_distance_band([{0, 0}, {1, 0}, {5, 0}], 1.0)
      iex> {Spatex.Graph.to_adjacency(g), Spatex.Graph.isolates(g)}
      {[{0, 1, 1.0}, {1, 0, 1.0}, {2, 2, 0.0}], [2]}
  """
  @spec build_distance_band(term(), number(), keyword()) :: t()
  def build_distance_band(points, threshold, opts \\ []),
    do: Builders.Knn.distance_band(points, threshold, opts)

  @doc """
  Builds a triangulation-based graph of a point set, like
  `libpysal.graph.Graph.build_triangulation`.

  `points` is as for `build_knn/3` (at least 3 distinct, non-collinear points). Edges are
  weighted by `kernel(d_ij, bandwidth)` on the Euclidean edge length; with the defaults
  (`:boxcar`, `:infinity`) every edge has weight `1.0`.

  ## Options

    * `:method` - `:delaunay` (default), `:gabriel` (the Delaunay edges whose diametral circle
      holds no Delaunay neighbor of an endpoint) or `:relative_neighborhood` (the edges whose
      lune holds no point; boundary points prune) or `:voronoi` (rook contiguity of the
      Voronoi cells clipped to `:clip`: the Delaunay edges whose dual Voronoi edge has positive
      length inside the clip region).
    * `:clip` - Voronoi clip region: `:bounding_box` (default, the points' bounding box),
      `:convex_hull` or `nil` (no clip; as with GEOS, the diagram is still bounded by the
      bounding box grown by its largest side on every side). Ignored by the other methods.
    * `:rook` - must be `true` (default) with `:voronoi`; queen cell contiguity is not
      supported. Ignored by the other methods.
    * `:kernel`, `:bandwidth`, `:taper`, `:decay` - as for `build_kernel/2`, except that
      `:bandwidth` is `:infinity` (default, no tapering) or a positive number.
    * `:coincident` - `:raise` (default) or `:clique`: the triangulation is built over distinct
      locations, points on one site take the site's edges and are linked with weight
      `kernel(0, bandwidth)`.
    * `:ids` - node labels, one per point (default `0..n-1`).

  Edges are those of the exact Delaunay triangulation for points in general position, matching
  libpysal (scipy/Qhull). For cocircular points, such as the corners of a square, the diagonal
  is implementation-defined and Gabriel/relative neighborhood results follow it. Unlike
  libpysal, a node whose edges are all tapered away by a finite bandwidth stays in the graph as
  an island instead of disappearing.

  ## Examples

      iex> g = Spatex.Graph.build_triangulation([{0, 0}, {3, 0}, {0, 4}])
      iex> Spatex.Graph.cardinalities(g)
      [2, 2, 2]
      iex> g = Spatex.Graph.build_triangulation([{0, 0}, {3, 0}, {0, 4}], method: :relative_neighborhood)
      iex> Spatex.Graph.to_adjacency(g)
      [{0, 1, 1.0}, {0, 2, 1.0}, {1, 0, 1.0}, {2, 0, 1.0}]
  """
  @spec build_triangulation(term(), keyword()) :: t()
  def build_triangulation(points, opts \\ []),
    do: Builders.Triangulation.triangulation(points, opts)

  @doc """
  Reads a GAL file. Weights are binary. See `Spatex.Graph.IO` for formats and the `:ids`
  option.
  """
  @spec read_gal(Path.t(), keyword()) :: t()
  def read_gal(path, opts \\ []), do: GraphIO.read_gal(path, opts)

  @doc "Reads a GWT file. See `Spatex.Graph.IO`."
  @spec read_gwt(Path.t(), keyword()) :: t()
  def read_gwt(path, opts \\ []), do: GraphIO.read_gwt(path, opts)

  @doc "Writes the graph as a GAL file (weights are not stored). See `Spatex.Graph.IO`."
  @spec write_gal(t(), Path.t()) :: :ok
  def write_gal(graph, path), do: GraphIO.write_gal(graph, path)

  @doc "Writes the graph as a GWT file (weights with seven decimals). See `Spatex.Graph.IO`."
  @spec write_gwt(t(), Path.t()) :: :ok
  def write_gwt(graph, path), do: GraphIO.write_gwt(graph, path)

  @doc "Renders the graph as GAL text (iodata). See `Spatex.Graph.IO`."
  @spec to_gal(t()) :: iodata()
  def to_gal(graph), do: GraphIO.to_gal(graph)

  @doc "Renders the graph as GWT text (iodata). See `Spatex.Graph.IO`."
  @spec to_gwt(t()) :: iodata()
  def to_gwt(graph), do: GraphIO.to_gwt(graph)

  defimpl Inspect do
    def inspect(%Spatex.Graph{n: n, transform: transform} = graph, _opts) do
      "#Spatex.Graph<n: #{n}, nnz: #{Spatex.Graph.nnz(graph)}, transform: #{Kernel.inspect(transform)}>"
    end
  end

  # -- construction ---------------------------------------------------------------------

  defp build(focal, neighbor, weight, opts) do
    unless is_list(focal) and is_list(neighbor) do
      raise ArgumentError, "focal and neighbor must be lists of node ids"
    end

    weight = weight || List.duplicate(1.0, length(focal))

    unless is_list(weight) do
      raise ArgumentError, "weight must be a list of numbers or nil"
    end

    if length(focal) != length(neighbor) or length(focal) != length(weight) do
      raise ArgumentError,
            "focal, neighbor and weight must have the same length, got " <>
              "#{length(focal)}, #{length(neighbor)} and #{length(weight)}"
    end

    ids = resolve_ids!(focal, neighbor, Keyword.get(opts, :ids))
    index = ids |> Enum.with_index() |> Map.new()

    focal_bin = positions_binary!(focal, index)
    neighbor_bin = positions_binary!(neighbor, index)
    weight_bin = weights_binary!(weight)

    n = length(ids)
    ref = unwrap(Native.graph_from_arrays(n, focal_bin, neighbor_bin, weight_bin))
    %__MODULE__{ref: ref, n: n, ids: ids, transform: :o}
  end

  defp resolve_ids!(focal, neighbor, nil), do: Enum.uniq(focal ++ neighbor)

  defp resolve_ids!(focal, neighbor, ids) when is_list(ids) do
    if length(Enum.uniq(ids)) != length(ids) do
      raise ArgumentError, ":ids must be unique"
    end

    known = MapSet.new(ids)

    case Enum.find(focal ++ neighbor, &(not MapSet.member?(known, &1))) do
      nil -> ids
      missing -> raise ArgumentError, "node #{inspect(missing)} is missing from :ids"
    end
  end

  defp resolve_ids!(_focal, _neighbor, _other), do: raise(ArgumentError, ":ids must be a list")

  defp positions_binary!(nodes, index) do
    {bin, _n} =
      nodes
      |> Enum.map(&Map.fetch!(index, &1))
      |> Spatex.Input.to_index_binary(:u32, name: "node position")

    bin
  end

  defp weights_binary!(weights) do
    {bin, _n} = Spatex.Input.to_f64_binary(weights, name: "weight")
    bin
  end

  defp y_binary!(y, n) when is_binary(y), do: y_binary!({:f64, y}, n)

  defp y_binary!(y, n) do
    {bin, _n} = Spatex.Input.to_f64_binary(y, name: "y", length: n)
    bin
  end

  defp triple!({f, t, w}), do: {f, t, w}
  defp triple!(_other), do: bad_adjacency!()

  defp pair!(focal, {t, w}), do: {focal, t, w}
  defp pair!(_focal, _other), do: bad_adjacency!()

  defp bad_adjacency! do
    raise ArgumentError,
          "adjacency must be a list of {focal, neighbor, weight} triples " <>
            "or a map of focal => [{neighbor, weight}]"
  end

  defp unzip3(triples) do
    {f, t, w} =
      Enum.reduce(triples, {[], [], []}, fn {f, t, w}, {fs, ts, ws} ->
        {[f | fs], [t | ts], [w | ws]}
      end)

    {Enum.reverse(f), Enum.reverse(t), Enum.reverse(w)}
  end

  defp zip3(a, b, c, fun), do: do_zip3(a, b, c, fun, [])
  defp do_zip3([], [], [], _fun, acc), do: Enum.reverse(acc)

  defp do_zip3([x | xs], [y | ys], [z | zs], fun, acc),
    do: do_zip3(xs, ys, zs, fun, [fun.(x, y, z) | acc])

  # -- transforms and NIF plumbing ------------------------------------------------------

  defp normalize_transform!(kind) when kind in @transforms, do: kind

  defp normalize_transform!(kind) when is_binary(kind) do
    case String.downcase(kind) do
      "o" -> :o
      "r" -> :r
      "b" -> :b
      "d" -> :d
      "v" -> :v
      _ -> bad_transform!(kind)
    end
  end

  defp normalize_transform!(kind), do: bad_transform!(kind)

  defp bad_transform!(kind) do
    raise ArgumentError,
          "unknown transform #{inspect(kind)}, expected one of :o, :r, :b, :d, :v"
  end

  defp unwrap({:ok, value}), do: value
  defp unwrap({:error, reason}), do: raise(ArgumentError, to_string(reason))

  defp u32_list(bin), do: for(<<v::unsigned-integer-32-native <- bin>>, do: v)
  defp f64_list(bin), do: for(<<v::float-64-native <- bin>>, do: v)
end
