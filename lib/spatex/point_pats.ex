defmodule Spatex.PointPats do
  @moduledoc """
  Point pattern analysis, after `pointpats` 2.6.0: centrography, Ripley's distance functions
  with seeded simulation envelopes, random point processes and quadrat statistics.

  Points are `[{x, y}]`, `[[x, y]]`, an `{n, 2}` `Nx.Tensor` or a pair `{xs, ys}`
  (see `Spatex.Input`). Non-finite coordinates raise `ArgumentError`.

  ## Centrography

  `mean_center/1`, `weighted_mean_center/2`, `manhattan_median/1`, `euclidean_median/1`,
  `std_distance/1`, `ellipse/2`, `dtot/2`, `hull/1`, `mbr/1`, `minimum_rotated_rectangle/2`,
  `minimum_bounding_circle/1` (`skyum/1`), plus the bundle `centrography/1`.

  `alpha_shape/2` and `alpha_shape_auto/2` return the alpha shape of the points as WKB polygons.

  The standard distance uses pointpats' raw-moment formula, which loses digits for large
  coordinate offsets (UTM metres, say); center the coordinates first when that matters. The
  Euclidean median is the exact Weiszfeld optimum, while pointpats stops a BFGS search at a
  gradient tolerance of `1.0e-5`, so the two agree only to about four digits.

  ## Distance statistics

  `g/2`, `f/2`, `j/2`, `k/2` and `l/2` return `{support, statistic}`; `g_test/2`, `f_test/2`,
  `j_test/2`, `k_test/2` and `l_test/2` return a `Spatex.PointPats.DistanceTest` with a seeded
  Monte Carlo p-value and an envelope. The estimators are **uncorrected for edge effects**,
  exactly as in pointpats 2.6.0, which has no edge correction.

  Quirks of pointpats that are replicated on purpose:

    * The default `support` is 20 values from 0 to the largest nearest-neighbour distance of the
      pattern, for K and L as well.
    * `g/2` and `f/2` normalize by the number of distances inside the support and start at 0.
    * `k/2` counts pairs strictly closer than each distance and always uses the bounding box of
      the points for the area, also inside the simulations of `k_test/2`.
    * The empty-space points of `f_test/2` and `j_test/2` are always drawn from the bounding box
      of the observed pattern.
    * The p-value is `min(p, 1 - p)`, so it is at most 0.5 and may be exactly 0.0.

  Random draws use ChaCha8 streams derived from `(seed, simulation)`: results are reproducible
  per `seed:` and independent of the number of threads, but they do not reproduce numpy's
  generators. Compare simulated quantities with pointpats statistically.

  ### The `support` option

  `nil` (20 values), an integer `m >= 2` (`m` values), `{stop}`, `{start, stop}`,
  `{start, stop, count}` (the third element is a count of values) or an ascending list.

  ### The `hull` option

  The simulation window: `:bbox` (default, bounding box of the points), `:convex` (their convex
  hull) or `{xmin, ymin, xmax, ymax}`. pointpats' `"alpha"` hull is not supported.

  ## Random processes and quadrat statistics

  `poisson/2`, `cluster_poisson/2`, `normal/2`, `cluster_normal/2`, `strauss/2` and `quadrat_test/2` (rectangle and hexagon grids).
  """

  alias Spatex.Input
  alias Spatex.Native
  alias Spatex.PointPats.DistanceTest
  alias Spatex.PointPats.Quadrat

  @type point :: {number(), number()}
  @type points :: term()
  @type support :: nil | pos_integer() | tuple() | [number()]
  @type window :: {number(), number(), number(), number()} | [point()] | {:convex, [point()]}
  @type curve :: {[float()], [float() | :nan | :infinity | :neg_infinity]}

  @default_max_cells 1_000_000

  # -- centrography ------------------------------------------------------------------------

  @doc """
  Mean center `{x, y}`.

  ## Examples

      iex> Spatex.PointPats.mean_center([{0, 0}, {2, 4}])
      {1.0, 2.0}
  """
  @spec mean_center(points()) :: {float(), float()}
  def mean_center(points), do: points |> centro("mean_center") |> pair()

  @doc """
  Center weighted by `weights / sum(weights)`.

  Raises when the weights do not match the points or sum to zero.
  """
  @spec weighted_mean_center(points(), [number()]) :: {float(), float()}
  def weighted_mean_center(points, weights) do
    {w, _} = Input.to_f64_binary(weights, name: "weights")
    points |> centro("weighted_mean_center", w) |> pair()
  end

  @doc """
  Coordinate-wise median. For an even number of points it is the mean of the two middle values
  (not unique; pointpats warns about this, Spatex does not).
  """
  @spec manhattan_median(points()) :: {float(), float()}
  def manhattan_median(points), do: points |> centro("manhattan_median") |> pair()

  @doc """
  Euclidean (geometric) median by Weiszfeld iteration from the mean center.

  This converges to the exact minimizer of `dtot/2`; pointpats' BFGS result is only accurate to
  about `1.0e-4` relative (a documented deviation).
  """
  @spec euclidean_median(points()) :: {float(), float()}
  def euclidean_median(points), do: points |> centro("euclidean_median") |> pair()

  @doc "Standard distance (see the moduledoc on its raw-moment formula)."
  @spec std_distance(points()) :: float()
  def std_distance(points), do: points |> centro("std_distance") |> hd()

  @doc """
  Sum of the Euclidean distances from the center `{x, y}` to every point.

  ## Examples

      iex> Spatex.PointPats.dtot({0, 0}, [{3, 4}, {0, 1}])
      6.0
  """
  @spec dtot({number(), number()}, points()) :: float()
  def dtot({cx, cy}, points) when is_number(cx) and is_number(cy) do
    {aux, 2} = Input.to_f64_binary([cx, cy], name: "center")
    points |> centro("dtot", aux) |> hd()
  end

  @doc """
  Standard deviational ellipse `{major, minor, angle}`.

  `angle` is the direction of the major axis in radians, counter-clockwise from the x axis,
  in `(-pi/2, pi/2]`.

  ## Options

    * `:weights` - one weight per point (the small-sample correction still uses the number of
      points, as in pointpats).
    * `:method` - `:crimestat` (default) or `:yuill`.
    * `:crimestat_corr`, `:degfreed_corr` - the two `:yuill` corrections (default `true`).

  Needs at least 3 points.
  """
  @spec ellipse(points(), keyword()) :: {float(), float(), float()}
  def ellipse(points, opts \\ []) do
    opts = validate!(opts, [:weights, :method, :crimestat_corr, :degfreed_corr])

    yuill? =
      case Keyword.get(opts, :method, :crimestat) do
        :crimestat ->
          false

        :yuill ->
          true

        other ->
          raise ArgumentError, "method: expected :crimestat or :yuill, got #{inspect(other)}"
      end

    flags = [
      yuill?,
      Keyword.get(opts, :crimestat_corr, true),
      Keyword.get(opts, :degfreed_corr, true)
    ]

    weights =
      case Keyword.get(opts, :weights) do
        nil -> <<>>
        w -> w |> Input.to_f64_binary(name: "weights") |> elem(0)
      end

    {major, minor, angle} = points |> centro("ellipse", weights, flags) |> List.to_tuple()
    {major, minor, angle}
  end

  @doc "Convex hull vertices, counter-clockwise from the lowest-x vertex; collinear points are dropped."
  @spec hull(points()) :: [{float(), float()}]
  def hull(points), do: points |> centro("hull") |> to_points()

  @doc "Minimum bounding rectangle `{min_x, min_y, max_x, max_y}`."
  @spec mbr(points()) :: {float(), float(), float(), float()}
  def mbr(points), do: points |> centro("mbr") |> List.to_tuple()

  @doc "Alias of `mbr/1`."
  @spec minimum_bounding_rectangle(points()) :: {float(), float(), float(), float()}
  def minimum_bounding_rectangle(points), do: mbr(points)

  @doc """
  Minimum-area rotated rectangle: its 4 corners, counter-clockwise (pointpats' corner order
  and starting corner differ, compare as a set).

  With `return_angle: true` returns `{corners, angle}` where `angle` is the direction of a side
  in degrees, modulo 90. Raises for collinear points.
  """
  @spec minimum_rotated_rectangle(points(), keyword()) ::
          [{float(), float()}] | {[{float(), float()}], float()}
  def minimum_rotated_rectangle(points, opts \\ []) do
    opts = validate!(opts, [:return_angle])
    [angle, _area | corners] = centro(points, "mrr")

    if Keyword.get(opts, :return_angle, false),
      do: {to_points(corners), angle},
      else: to_points(corners)
  end

  @doc "Smallest enclosing circle `{{cx, cy}, radius}`."
  @spec minimum_bounding_circle(points()) :: {{float(), float()}, float()}
  def minimum_bounding_circle(points) do
    [cx, cy, r] = centro(points, "mbc")
    {{cx, cy}, r}
  end

  @doc """
  Alpha shape of the points (`libpysal.cg.alpha_shape`): the polygons of the Delaunay triangles
  whose circumradius is below `1 / alpha`.

  Returns a list of WKB polygons (little-endian, rings closed), `[]` when no triangle qualifies.
  A shape can have several polygons (separate clusters, an island inside a hole) and polygons with
  holes. Each polygon has its shell clockwise and its holes counter-clockwise, as GEOS returns
  them, every ring starting at its smallest vertex; polygons are ordered by their first vertex.
  Larger `alpha` gives tighter shapes: `0` is the convex hull and a very large value is empty.

  Fewer than four points give their convex hull whatever `alpha` is (`[]` when they are
  collinear), as libpysal does. Duplicate points are ignored. Four or more points that are
  collinear, or fewer than three distinct points, raise `ArgumentError`, where libpysal raises
  `QhullError`; so do a negative or non-numeric `alpha` and non-finite coordinates.

  ## Examples

      iex> pts = [{0.0, 0.0}, {4.0, 0.3}, {4.2, 4.0}, {0.1, 3.9}, {2.1, 1.9}]
      iex> [<<1, 3, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, _::binary>>] = Spatex.PointPats.alpha_shape(pts, 0.0)
      iex> Spatex.PointPats.alpha_shape(pts, 1.0e9)
      []
  """
  @spec alpha_shape(points(), number()) :: [binary()]
  def alpha_shape(points, alpha) do
    {bin, _n} = points!(points, "alpha_shape")
    bin |> Native.pp_alpha_shape(non_neg!(alpha, "alpha")) |> unwrap()
  end

  @doc """
  Alpha shape with an automatically chosen alpha (`libpysal.cg.alpha_shape_auto`): the tightest
  shape that is a single polygon containing every point.

  The triangles' circumradii are ranked from the largest down; every `:step`-th radius gives a
  candidate shape (the triangles with a smaller radius), and the search stops at the first
  candidate that is not one hole-free polygon with all the points, returning the previous one.
  Returns the WKB polygon (shell clockwise, starting at its smallest vertex).

  Options:

    * `:step` - how many radii to jump per candidate, a positive integer (default `1`).
    * `:return_radius` - return `{wkb, radius}`, the circumradius whose threshold gave the
      polygon (default `false`).
    * `:return_circles` - return `{wkb, radius, centers}` where `centers` has one `{x, y}` per
      boundary edge of the shell, in ring order: the centre, on the left of the edge, of the circle
      of that radius through its ends (`construct_bounding_circles`). Implies `:return_radius`.

  Needs three distinct points that are not collinear. The search compares
  `radius < 1 / (1 / radius - eps)` exactly as libpysal does, so which triangle is kept at a
  candidate depends on the last bit of its radius; when that makes even the first candidate
  invalid, `ArgumentError` is raised (libpysal returns the first polygon of the convex hull
  minus its largest triangle, or raises `UnboundLocalError` for the radius).

  ## Examples

      iex> pts = [{0.0, 0.0}, {4.0, 0.3}, {4.2, 4.0}, {0.1, 3.9}, {2.1, 1.9}]
      iex> {wkb, radius} = Spatex.PointPats.alpha_shape_auto(pts, return_radius: true)
      iex> is_binary(wkb) and radius > 0
      true
  """
  @spec alpha_shape_auto(points(), keyword()) ::
          binary()
          | {binary(), float()}
          | {binary(), float(), [{float(), float()}]}
  def alpha_shape_auto(points, opts \\ []) do
    opts = validate!(opts, [:step, :return_radius, :return_circles])
    {bin, _n} = points!(points, "alpha_shape_auto")
    step = pos_int!(Keyword.get(opts, :step, 1), "step")
    circles = flag!(opts, :return_circles, false)
    radius? = flag!(opts, :return_radius, false) or circles

    {wkb, radius, centers} = bin |> Native.pp_alpha_shape_auto(step, circles) |> unwrap()

    cond do
      circles -> {wkb, radius, centers |> Input.from_f64_binary() |> to_points()}
      radius? -> {wkb, radius}
      true -> wkb
    end
  end

  @doc "Alias of `minimum_bounding_circle/1` (pointpats implements it with Skyum's algorithm)."
  @spec skyum(points()) :: {{float(), float()}, float()}
  def skyum(points), do: minimum_bounding_circle(points)

  @doc """
  Bundle of the common summaries as a map: `:mean_center`, `:manhattan_median`,
  `:std_distance`, `:ellipse`, `:hull` and `:mbr`.
  """
  @spec centrography(points()) :: map()
  def centrography(points) do
    {bin, _n} = points!(points)
    pts = {:f64, bin}

    %{
      mean_center: mean_center(pts),
      manhattan_median: manhattan_median(pts),
      std_distance: std_distance(pts),
      ellipse: ellipse(pts),
      hull: hull(pts),
      mbr: mbr(pts)
    }
  end

  # -- distance statistics -------------------------------------------------------------------

  @doc """
  Nearest-neighbour distribution `G`: `{support, statistic}`.

  ## Options

    * `:support` - see the moduledoc (default 20 values up to the largest nearest-neighbour
      distance).

  ## Examples

      iex> {support, g} = Spatex.PointPats.g([{0, 0}, {1, 0}, {3, 0}, {5, 0}], support: [0, 1, 2])
      iex> {support, g}
      {[0.0, 1.0, 2.0], [0.0, 0.0, 1.0]}
  """
  @spec g(points(), keyword()) :: curve()
  def g(points, opts \\ []) do
    opts = validate!(opts, [:support])
    distance("g", points, opts, <<>>)
  end

  @doc """
  Empty-space function `F`: `{support, statistic}`.

  The distances come from `:empty_points` (a point set) or from `:n_empty` (default 1000)
  seeded uniform points of the `:hull` window (default `:bbox`).

  ## Options

  `:support`, `:empty_points`, `:n_empty`, `:hull`, `:seed` (default 0).
  """
  @spec f(points(), keyword()) :: curve()
  def f(points, opts \\ []) do
    opts = validate!(opts, [:support, :empty_points, :n_empty, :hull, :seed])
    distance("f", points, opts, empty_points(points, opts))
  end

  @doc """
  `J = (1 - G) / (1 - F)`: `{support, statistic}`.

  Takes the options of `f/2` plus `truncate: true`, which cuts the result at the first infinite
  or NaN value (F reaches 1 before G, so J is normally truncated).
  """
  @spec j(points(), keyword()) :: curve()
  def j(points, opts \\ []) do
    opts = validate!(opts, [:support, :empty_points, :n_empty, :hull, :seed, :truncate])
    distance("j", points, opts, empty_points(points, opts))
  end

  @doc """
  Ripley's K, uncorrected, with the bounding-box area of the points: `{support, statistic}`.
  """
  @spec k(points(), keyword()) :: curve()
  def k(points, opts \\ []) do
    opts = validate!(opts, [:support])
    distance("k", points, opts, <<>>)
  end

  @doc """
  `L = sqrt(K / pi)`: `{support, statistic}`. With `linearized: true` it returns `L - d`.
  """
  @spec l(points(), keyword()) :: curve()
  def l(points, opts \\ []) do
    opts = validate!(opts, [:support, :linearized])
    distance("l", points, opts, <<>>)
  end

  @test_opts [:support, :hull, :permutations, :seed, :keep_simulations, :n_empty]

  @doc """
  Monte Carlo test of `G` against complete spatial randomness. Returns a
  `Spatex.PointPats.DistanceTest`.

  ## Options

    * `:support`, `:hull` - see the moduledoc.
    * `:permutations` - number of CSR simulations, default 9999; `0` disables the test
      (`:nan` p-values).
    * `:seed` - default `nil`: a random seed, recorded in the result's `seed`.
    * `:keep_simulations` - also return every simulated curve (default `false`): `true` as a
      list of `permutations` lists, `:tensor` as a `{permutations, length(support)}`
      `Nx.Tensor` of type `{:f, 64}` (optional `:nx` dependency).
  """
  @spec g_test(points(), keyword()) :: DistanceTest.t()
  def g_test(points, opts \\ []), do: ripley_test(:g, points, opts, @test_opts)

  @doc "Monte Carlo test of `F`; see `g_test/2`. Also takes `:n_empty` (default 1000)."
  @spec f_test(points(), keyword()) :: DistanceTest.t()
  def f_test(points, opts \\ []), do: ripley_test(:f, points, opts, @test_opts)

  @doc """
  Monte Carlo test of `J`; see `g_test/2`. Takes `:truncate` (default `true`).

  DEVIATION: pointpats' `j_test` returns `None` when `truncate: true` and J has no infinite
  value; Spatex returns the result in that case (it cannot occur with a support that ends at
  the maximum nearest-neighbour distance, where J is `0/0` at the last distance).
  """
  @spec j_test(points(), keyword()) :: DistanceTest.t()
  def j_test(points, opts \\ []), do: ripley_test(:j, points, opts, [:truncate | @test_opts])

  @doc "Monte Carlo test of `K`; see `g_test/2`."
  @spec k_test(points(), keyword()) :: DistanceTest.t()
  def k_test(points, opts \\ []), do: ripley_test(:k, points, opts, @test_opts)

  @doc "Monte Carlo test of `L`; see `g_test/2`. Takes `:linearized` (default `false`)."
  @spec l_test(points(), keyword()) :: DistanceTest.t()
  def l_test(points, opts \\ []), do: ripley_test(:l, points, opts, [:linearized | @test_opts])

  defp distance(kind, points, opts, empty) do
    {bin, _n} = points!(points)
    {spec_kind, spec} = support!(Keyword.get(opts, :support))

    {s, v} =
      kind
      |> Native.pp_distance(
        bin,
        spec_kind,
        spec,
        empty,
        Keyword.get(opts, :truncate, true) == true,
        Keyword.get(opts, :linearized, false) == true
      )
      |> unwrap()

    {Input.from_f64_binary(s), Input.from_f64_binary(v)}
  end

  defp empty_points(points, opts) do
    case Keyword.get(opts, :empty_points) do
      nil ->
        {bin, _} = points!(points)
        {kind, data} = window_for(Keyword.get(opts, :hull, :bbox), bin)
        n = pos_int!(Keyword.get(opts, :n_empty, 1000), "n_empty")
        seed = seed!(opts)

        {flat, _, _} =
          kind |> Native.pp_poisson(data, nil, {n, nil}, seed) |> unwrap()

        flat

      given ->
        {bin, n} = points!(given, "empty_points")
        if n == 0, do: raise(ArgumentError, "empty_points: must not be empty")
        bin
    end
  end

  defp ripley_test(function, points, opts, allowed) do
    opts = validate!(opts, allowed)
    {bin, _n} = points!(points)
    {spec_kind, spec} = support!(Keyword.get(opts, :support))
    {wkind, wdata} = window_for(Keyword.get(opts, :hull, :bbox), bin)

    n_sims = Keyword.get(opts, :permutations, 9999)

    unless is_integer(n_sims) and n_sims >= 0,
      do:
        raise(
          ArgumentError,
          "permutations: expected a non-negative integer, got #{inspect(n_sims)}"
        )

    keep = Input.keep_simulations!(Keyword.get(opts, :keep_simulations, false))
    n_empty = pos_int!(Keyword.get(opts, :n_empty, 1000), "n_empty")
    seed = random_seed!(opts)

    {{support, statistic, pvalue}, {sims, low, high}} =
      function
      |> Atom.to_string()
      |> Native.pp_ripley_test(
        bin,
        spec_kind,
        spec,
        wkind,
        wdata,
        {n_sims, seed, Input.keep_flag(keep), flag!(opts, :truncate, true),
         flag!(opts, :linearized, false), n_empty}
      )
      |> unwrap()

    support = Input.from_f64_binary(support)
    m = length(support)

    simulations =
      if keep != false and n_sims > 0,
        do: Input.simulations(sims, keep, {n_sims, m}),
        else: nil

    %DistanceTest{
      function: function,
      support: support,
      statistic: Input.from_f64_binary(statistic),
      pvalue: Input.from_f64_binary(pvalue),
      simulations: simulations,
      permutations: n_sims,
      seed: seed,
      envelope_low: Input.from_f64_binary(low),
      envelope_high: Input.from_f64_binary(high)
    }
  end

  # -- random point processes ----------------------------------------------------------------

  @doc """
  Conditional Poisson (complete spatial randomness): points uniform in `window`.

  `window` is `{xmin, ymin, xmax, ymax}`, a point set (its bounding box) or `{:convex, points}`
  (their convex hull). Returns one pattern (a list of `{x, y}`) for a single simulation and a
  list of patterns otherwise; a one-point pattern is never squeezed to a bare point.

  ## Options

    * `:size` - an integer number of points, or `{n_points, n_simulations}`. With `:intensity`
      an integer `size` is the number of simulations instead. Default: 100 points.
    * `:intensity` - points per unit area; the count is `trunc(area * intensity)`. Not allowed
      with a `{n, m}` size.
    * `:seed` - default 0.

  ## Examples

      iex> pts = Spatex.PointPats.poisson({0, 0, 10, 5}, size: 20, seed: 1)
      iex> length(pts)
      20
  """
  @spec poisson(window(), keyword()) :: [point()] | [[point()]]
  def poisson(window, opts \\ []) do
    opts = validate!(opts, [:size, :intensity, :seed])
    {wkind, wdata} = window!(window)
    {sizes, intensity} = sizes!(opts)

    {flat, n_obs, n_sims} =
      wkind |> Native.pp_poisson(wdata, intensity, sizes, seed!(opts)) |> unwrap()

    patterns(flat, n_obs, n_sims)
  end

  @doc """
  Poisson cluster process: `:n_seeds` parent points, each followed by uniform children within
  `:cluster_radius` of the parent and inside the window; points are ordered cluster by cluster
  (sizes as numpy's `array_split`).

  Takes the window and the `:size`, `:intensity` and `:seed` options of `poisson/2`, plus

    * `:n_seeds` - number of clusters (default 2).
    * `:cluster_radius` - a radius, or one per seed. Default: half the smallest distance
      between the parents (required when `n_seeds` is 1).
  """
  @spec cluster_poisson(window(), keyword()) :: [point()] | [[point()]]
  def cluster_poisson(window, opts \\ []) do
    opts = validate!(opts, [:size, :intensity, :seed, :n_seeds, :cluster_radius])
    {wkind, wdata} = window!(window)
    {sizes, intensity} = sizes!(opts)
    n_seeds = pos_int!(Keyword.get(opts, :n_seeds, 2), "n_seeds")

    radii =
      case Keyword.get(opts, :cluster_radius) do
        nil -> []
        r when is_number(r) -> [r * 1.0]
        list when is_list(list) -> Enum.map(list, &(&1 * 1.0))
        other -> raise ArgumentError, "cluster_radius: unsupported #{inspect(other)}"
      end

    {flat, n_obs, n_sims} =
      wkind
      |> Native.pp_cluster_poisson(wdata, intensity, sizes, {n_seeds, radii}, seed!(opts))
      |> unwrap()

    patterns(flat, n_obs, n_sims)
  end

  @doc """
  Bivariate normal points truncated to the window: candidates are drawn around `:center` and
  redrawn until they fall inside, as pointpats' `random.normal`.

  Takes the window and the `:size` and `:seed` options of `poisson/2` (no `:intensity`), plus

    * `:center` - `{x, y}`. Default: the window centre (the middle of a box, the vertex mean of
      a convex polygon).
    * `:cov` - a number `v` for the covariance `v * I` (pointpats names it a standard deviation
      but uses it as a variance, and so do we), or a symmetric positive-definite 2 x 2 matrix
      `[[xx, xy], [xy, yy]]`. Default: `(max(width, height) / 2)^2`.

  A centre that almost never lands in the window raises `ArgumentError` instead of looping.

  ## Examples

      iex> pts = Spatex.PointPats.normal({0, 0, 10, 5}, size: 20, cov: 2, seed: 1)
      iex> length(pts)
      20
  """
  @spec normal(window(), keyword()) :: [point()] | [[point()]]
  def normal(window, opts \\ []) do
    opts = validate!(opts, [:size, :seed, :center, :cov])
    {wkind, wdata} = window!(window)
    {sizes, _} = sizes!(opts)

    {flat, n_obs, n_sims} =
      wkind
      |> Native.pp_normal(wdata, sizes, center!(opts), cov!(opts), seed!(opts))
      |> unwrap()

    patterns(flat, n_obs, n_sims)
  end

  @doc """
  Normal cluster process: `:n_seeds` cluster centres uniform in the window, each followed by
  normal points around it (truncated to the window); points are ordered cluster by cluster
  (sizes as numpy's `array_split`).

  Takes the window and the `:size` and `:seed` options of `normal/2`, plus

    * `:n_seeds` - number of clusters (default 2).
    * `:cov` - as in `normal/2`, shared by all clusters. Default: the squared mean distance
      between the centres of the replication (pointpats computes it once, from the first
      replication, and reuses it; Spatex keeps replications independent). Required when
      `:n_seeds` is 1.
  """
  @spec cluster_normal(window(), keyword()) :: [point()] | [[point()]]
  def cluster_normal(window, opts \\ []) do
    opts = validate!(opts, [:size, :seed, :n_seeds, :cov])
    {wkind, wdata} = window!(window)
    {sizes, _} = sizes!(opts)
    n_seeds = pos_int!(Keyword.get(opts, :n_seeds, 2), "n_seeds")

    {flat, n_obs, n_sims} =
      wkind
      |> Native.pp_cluster_normal(wdata, sizes, n_seeds, cov!(opts), seed!(opts))
      |> unwrap()

    patterns(flat, n_obs, n_sims)
  end

  @doc """
  Strauss process by pointpats' birth-death Metropolis-Hastings sampler: starts from `1.5 * n`
  uniform points, runs `:n_iter` births and deaths weighted by `gamma^k` (`k` the number of pairs
  closer than `:r`) and keeps the first `n` points of the final pattern.

  Takes the window and the `:size`, `:intensity` and `:seed` options of `poisson/2`, plus

      * `:gamma` - interaction parameter: below 1 inhibits close pairs, 1 is Poisson, above 1
        clusters. Default 0.1.
      * `:r` - interaction radius, in window units (default 0.05, as pointpats; pick it for
        your coordinates). Pairs at distance `<= r` interact.
      * `:n_iter` - steps per replication (default 5000).
      * `:max_iter` - restarts per replication when the sampler cannot reach `n` points
        (default 10).

  Raises `ArgumentError` when the requested size is impossible for `:gamma` and `:r` (pointpats'
  `RuntimeError`).

  ## Examples

      iex> pts = Spatex.PointPats.strauss({0, 0, 1, 1}, size: 10, gamma: 0.2, r: 0.1, n_iter: 200, seed: 1)
      iex> length(pts)
      10
  """
  @spec strauss(window(), keyword()) :: [point()] | [[point()]]
  def strauss(window, opts \\ []) do
    opts = validate!(opts, [:size, :intensity, :seed, :gamma, :r, :n_iter, :max_iter])
    {wkind, wdata} = window!(window)
    {sizes, intensity} = sizes!(opts)
    gamma = non_neg!(Keyword.get(opts, :gamma, 0.1), "gamma")
    r = positive!(Keyword.get(opts, :r, 0.05), "r")
    n_iter = non_neg_int!(Keyword.get(opts, :n_iter, 5000), "n_iter")
    max_iter = pos_int!(Keyword.get(opts, :max_iter, 10), "max_iter")

    {flat, n_obs, n_sims} =
      wkind
      |> Native.pp_strauss(wdata, intensity, sizes, {gamma, r, n_iter, max_iter}, seed!(opts))
      |> unwrap()

    patterns(flat, n_obs, n_sims)
  end

  defp positive!(v, _name) when is_number(v) and v > 0, do: v * 1.0

  defp positive!(v, name),
    do: raise(ArgumentError, "#{name}: expected a positive number, got #{inspect(v)}")

  defp non_neg_int!(v, _name) when is_integer(v) and v >= 0, do: v

  defp non_neg_int!(v, name),
    do: raise(ArgumentError, "#{name}: expected a non-negative integer, got #{inspect(v)}")

  defp center!(opts) do
    case Keyword.get(opts, :center) do
      nil -> []
      {x, y} when is_number(x) and is_number(y) -> [x * 1.0, y * 1.0]
      other -> raise ArgumentError, "center: expected {x, y}, got #{inspect(other)}"
    end
  end

  defp cov!(opts) do
    case Keyword.get(opts, :cov) do
      nil ->
        []

      v when is_number(v) and v > 0 ->
        [v * 1.0, 0.0, 0.0, v * 1.0]

      [[a, b], [c, d]] when is_number(a) and is_number(b) and is_number(c) and is_number(d) ->
        [a * 1.0, b * 1.0, c * 1.0, d * 1.0]

      other ->
        raise ArgumentError,
              "cov: expected a positive number or a 2 x 2 matrix [[xx, xy], [yx, yy]], got #{inspect(other)}"
    end
  end

  defp patterns(flat, n_obs, n_sims) do
    all = flat |> Input.from_f64_binary() |> Enum.chunk_every(2) |> Enum.map(&List.to_tuple/1)

    case n_sims do
      1 -> all
      _ -> Enum.chunk_every(all, n_obs)
    end
  end

  defp sizes!(opts) do
    intensity = intensity!(Keyword.get(opts, :intensity))
    {size!(Keyword.get(opts, :size), intensity), intensity}
  end

  defp intensity!(nil), do: nil
  defp intensity!(i) when is_number(i) and i > 0, do: i * 1.0

  defp intensity!(other),
    do: raise(ArgumentError, "intensity: expected a positive number, got #{inspect(other)}")

  defp size!(nil, _intensity), do: {nil, nil}

  defp size!({n, m}, _intensity) when is_integer(n) and n > 0 and is_integer(m) and m > 0,
    do: {n, m}

  defp size!(n, nil) when is_integer(n) and n > 0, do: {n, nil}
  defp size!(n, _intensity) when is_integer(n) and n > 0, do: {nil, n}

  defp size!(other, _intensity) do
    raise ArgumentError,
          "size: expected a positive integer or {n, simulations}, got #{inspect(other)}"
  end

  # -- quadrat statistics --------------------------------------------------------------------

  @doc """
  Quadrat test of complete spatial randomness (Pearson chi-square on the cell counts).

  ## Options

    * `:shape` - `:rectangle` (default) or `:hexagon`.
    * `:nx`, `:ny` - columns and rows of the rectangle grid (default 3).
    * `:rectangle_width`, `:rectangle_height` - when both are non-zero the cells have this
      size and the grid may extend past the bounding box (default 0).
    * `:lh` - side length of the flat-topped hexagons (default 10).
    * `:permutations` - number of seeded CSR realizations in the bounding box for a Monte
      Carlo p-value (default 0). Each realization is gridded on its own bounding box.
    * `:seed` - default `nil`: a random seed, recorded in the result's `seed`.
    * `:max_cells` - largest grid accepted (default 1_000_000). A larger grid raises
      `ArgumentError` before anything is allocated; a small `:lh` or cell size on projected
      coordinates can otherwise ask for billions of cells.

  Returns a `Spatex.PointPats.Quadrat`.
  """
  @spec quadrat_test(points(), keyword()) :: Quadrat.t()
  def quadrat_test(points, opts \\ []) do
    opts =
      validate!(opts, [
        :shape,
        :nx,
        :ny,
        :rectangle_width,
        :rectangle_height,
        :lh,
        :permutations,
        :seed,
        :max_cells
      ])

    {bin, _} = points!(points)
    {shape, geometry, realizations} = quadrat_params!(opts)
    seed = random_seed!(opts)

    {{ids, counts}, {cols, rows, even, odd, df}, {scalars, contrib, reals}} =
      bin
      |> Native.pp_quadrat(
        Atom.to_string(shape),
        geometry,
        realizations,
        seed,
        pos_int!(Keyword.get(opts, :max_cells, @default_max_cells), "max_cells")
      )
      |> unwrap()

    [chi2, p, w, h, x0, y0, x1, y1, rp] = Input.from_f64_binary(scalars)
    simulated? = realizations > 0
    rect? = shape == :rectangle

    %Quadrat{
      shape: shape,
      cell_ids: ids,
      counts: counts,
      num_cells: length(counts),
      chi2: chi2,
      chi2_pvalue: p,
      df: df,
      chi2_contrib: Input.from_f64_binary(contrib),
      chi2_realizations: if(simulated?, do: Input.from_f64_binary(reals)),
      chi2_r_pvalue: if(simulated?, do: rp),
      count_column: cols,
      count_row: if(rect?, do: rows),
      count_row_even: if(rect?, do: nil, else: even),
      count_row_odd: if(rect?, do: nil, else: odd),
      rectangle_width: if(rect?, do: w),
      rectangle_height: if(rect?, do: h),
      h_length: if(rect?, do: nil, else: elem(geometry, 4)),
      mbb: {x0, y0, x1, y1},
      permutations: realizations,
      seed: seed
    }
  end

  defp quadrat_params!(opts) do
    shape =
      case Keyword.get(opts, :shape, :rectangle) do
        s when s in [:rectangle, :hexagon] ->
          s

        other ->
          raise ArgumentError, "shape: expected :rectangle or :hexagon, got #{inspect(other)}"
      end

    {nx, ny, lh} = grid_dimensions!(shape, opts)
    width = non_neg!(Keyword.get(opts, :rectangle_width, 0), "rectangle_width")
    height = non_neg!(Keyword.get(opts, :rectangle_height, 0), "rectangle_height")
    realizations = Keyword.get(opts, :permutations, 0)

    unless is_integer(realizations) and realizations >= 0,
      do: raise(ArgumentError, "permutations: expected a non-negative integer")

    {shape, {nx, ny, width, height, lh}, realizations}
  end

  defp grid_dimensions!(:rectangle, opts) do
    {pos_int!(Keyword.get(opts, :nx, 3), "nx"), pos_int!(Keyword.get(opts, :ny, 3), "ny"), 10.0}
  end

  defp grid_dimensions!(:hexagon, opts) do
    lh = Keyword.get(opts, :lh, 10)

    unless is_number(lh) and lh > 0,
      do: raise(ArgumentError, "lh: expected a positive number, got #{inspect(lh)}")

    {3, 3, lh * 1.0}
  end

  # -- helpers -------------------------------------------------------------------------------

  defp centro(points, op, aux \\ <<>>, flags \\ []) do
    {bin, _n} = points!(points)
    op |> Native.pp_centrography(bin, aux, flags) |> unwrap() |> Input.from_f64_binary()
  end

  defp points!(points, name \\ "points")

  defp points!({:f64, bin}, _name) when is_binary(bin), do: {bin, div(byte_size(bin), 16)}

  defp points!(points, name) do
    {bin, n} = Input.to_points_binary(points)
    if n == 0, do: raise(ArgumentError, "#{name}: must not be empty")
    {bin, n}
  end

  defp pair([x, y]), do: {x, y}

  defp to_points(flat), do: flat |> Enum.chunk_every(2) |> Enum.map(&List.to_tuple/1)

  defp support!(nil), do: {"int", [20.0]}
  defp support!(m) when is_integer(m) and m >= 2, do: {"int", [m * 1.0]}
  defp support!({stop}) when is_number(stop), do: {"lin", [0.0, stop * 1.0, 20.0]}
  defp support!({a, b}) when is_number(a) and is_number(b), do: {"lin", [a * 1.0, b * 1.0, 20.0]}

  defp support!({a, b, c}) when is_number(a) and is_number(b) and is_integer(c) and c >= 2,
    do: {"lin", [a * 1.0, b * 1.0, c * 1.0]}

  defp support!(list) when is_list(list) and length(list) >= 2 do
    values = Enum.map(list, &(&1 * 1.0))

    if values == Enum.sort(values),
      do: {"list", values},
      else: raise(ArgumentError, "support: values must be ascending")
  rescue
    ArithmeticError -> reraise ArgumentError, "support: values must be numbers", __STACKTRACE__
  end

  defp support!(other),
    do:
      raise(
        ArgumentError,
        "support: expected nil, an integer >= 2, {stop}, {start, stop}, " <>
          "{start, stop, count} or an ascending list, got #{inspect(other)}"
      )

  # The simulation window for `:hull`, from the packed points.
  defp window_for(:bbox, bin) do
    {"bbox", pack(bbox_values(bin))}
  end

  defp window_for(:convex, bin) do
    hull = "hull" |> Native.pp_centrography(bin, <<>>, []) |> unwrap()
    {"convex", hull}
  end

  defp window_for({a, b, c, d}, _bin)
       when is_number(a) and is_number(b) and is_number(c) and is_number(d),
       do: {"bbox", pack([a * 1.0, b * 1.0, c * 1.0, d * 1.0])}

  defp window_for(other, _bin),
    do:
      raise(
        ArgumentError,
        "hull: expected :bbox, :convex or {xmin, ymin, xmax, ymax}, got #{inspect(other)}"
      )

  defp bbox_values(bin) do
    "mbr" |> Native.pp_centrography(bin, <<>>, []) |> unwrap() |> Input.from_f64_binary()
  end

  defp window!({a, b, c, d}) when is_number(a) and is_number(b) and is_number(c) and is_number(d),
    do: {"bbox", pack([a * 1.0, b * 1.0, c * 1.0, d * 1.0])}

  defp window!({:convex, points}) do
    {bin, _} = points!(points, "window")
    window_for(:convex, bin)
  end

  defp window!(points) when is_list(points) do
    {bin, _} = points!(points, "window")
    window_for(:bbox, bin)
  end

  defp window!(other),
    do:
      raise(
        ArgumentError,
        "window: expected {xmin, ymin, xmax, ymax}, a point list or {:convex, points}, got #{inspect(other)}"
      )

  defp pack(values), do: for(v <- values, into: <<>>, do: <<v::float-64-native>>)

  defp pos_int!(v, _name) when is_integer(v) and v > 0, do: v

  defp pos_int!(v, name),
    do: raise(ArgumentError, "#{name}: expected a positive integer, got #{inspect(v)}")

  defp non_neg!(v, _name) when is_number(v) and v >= 0, do: v * 1.0

  defp non_neg!(v, name),
    do: raise(ArgumentError, "#{name}: expected a non-negative number, got #{inspect(v)}")

  # Monte Carlo tests whose result records the seed: `nil` (the default) draws one.
  defp random_seed!(opts) do
    case Keyword.get(opts, :seed) do
      nil -> :rand.uniform(4_294_967_296) - 1
      _ -> seed!(opts)
    end
  end

  defp flag!(opts, key, default) do
    case Keyword.get(opts, key, default) do
      v when is_boolean(v) -> v
      other -> raise ArgumentError, "#{key}: expected a boolean, got #{inspect(other)}"
    end
  end

  # Generators that return bare points or curves (nothing to record a seed in) keep 0.
  defp seed!(opts) do
    case Keyword.get(opts, :seed, 0) do
      s when is_integer(s) and s >= 0 -> s
      other -> raise ArgumentError, "seed: expected a non-negative integer, got #{inspect(other)}"
    end
  end

  defp validate!(opts, allowed) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    case Keyword.keys(opts) -- allowed do
      [] ->
        opts

      [bad | _] ->
        raise ArgumentError, "unknown option #{inspect(bad)}; allowed: #{inspect(allowed)}"
    end
  end

  defp unwrap({:ok, value}), do: value
  defp unwrap({:error, reason}), do: raise(ArgumentError, to_string(reason))
end
