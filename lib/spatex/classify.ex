defmodule Spatex.Classify do
  @moduledoc """
  Map classification, mirroring `mapclassify` 2.11.0.

  Every classifier returns a `%Spatex.Classify{}` with the class upper bounds (`bins`), the
  class of each observation (`yb`, zero based), the observations per class (`counts`) and
  fit summaries `tss`, `adcm` and `gadf`.

  Class `i` holds the values in `(bins[i-1], bins[i]]`; the first class is closed at the
  lower edge (`lowest`, the minimum of `y` unless given). This is mapclassify's right-closed
  `bin1d`. A value above the last bin matches no interval and stays in class 0 (this can only
  happen with `percentiles/2` bins that stop below 100, exactly as in mapclassify).

  ## Classifiers

  | Function | mapclassify |
  |---|---|
  | `equal_interval/2` | `EqualInterval` |
  | `quantiles/2` | `Quantiles` |
  | `box_plot/2` | `BoxPlot` |
  | `std_mean/2` | `StdMean` |
  | `percentiles/2` | `Percentiles` |
  | `maximum_breaks/2` | `MaximumBreaks` |
  | `head_tail_breaks/1` | `HeadTailBreaks` |
  | `user_defined/3` | `UserDefined` |
  | `fisher_jenks/2` | `FisherJenks` |
  | `fisher_jenks_sampled/2` | `FisherJenksSampled` |
  | `jenks_caspall/2` | `JenksCaspall` |
  | `jenks_caspall_sampled/2` | `JenksCaspallSampled` |
  | `jenks_caspall_forced/2` | `JenksCaspallForced` |
  | `natural_breaks/2` | `NaturalBreaks` |
  | `pretty_breaks/2` | `PrettyBreaks` |
  | `max_p/2` | `MaxP` |
  | `gadf/2` | `gadf` |
  | `k_classifiers/2` | `KClassifiers` |

  `classify/3` dispatches on mapclassify's scheme names.

  ## Optimal classifiers

  `fisher_jenks/2` is the exact optimum for within-class variance and matches mapclassify bit
  for bit (it replicates mapclassify's `float32` dynamic program). It costs `O(n^2 k)`; for
  large `n` use `fisher_jenks_sampled/2`. `natural_breaks/2` is one-dimensional k-means with
  seeded k-means++ restarts. mapclassify draws its start points from an unseeded random
  generator, so only statistical agreement is possible: on well separated data the results are
  identical, in general Spatex reaches the same or a lower within-class variance. The sampled
  classifiers draw their subsample from a ChaCha stream seeded by `:seed` (default `nil`: a random
  seed, recorded in `info.seed`), so they
  are reproducible but not equal to mapclassify's `numpy` draws.

  ## The `k` field

  `k` follows mapclassify's `k` attribute: the requested number of classes for
  `equal_interval/2` and `maximum_breaks/2`, and the number of bins for the others. This
  deviates from mapclassify for `quantiles/2` on tied data, where mapclassify keeps the
  requested `k` although fewer bins remain; here `k` is the real number of bins.

  ## Warnings

  Conditions that mapclassify reports with `warnings.warn` are collected in `warnings` and
  logged with `Logger.warning/1`: a `quantiles/2` that collapsed to fewer than `k` unique bins,
  and a `maximum_breaks/2` with fewer unique gaps than `k - 1` (ties then break by position;
  `numpy.argsort`, which mapclassify uses, leaves that order unspecified).

  ## Input

  `y` accepts anything `Spatex.Input.to_f64_binary/2` does and must be finite. Invalid
  input and invalid options raise `ArgumentError`.

  ## Examples

      iex> c = Spatex.Classify.equal_interval([1, 7, 4, 12, 4, 31], k: 5)
      iex> c.bins
      [7.0, 13.0, 19.0, 25.0, 31.0]
      iex> c.yb
      [0, 0, 0, 1, 0, 4]
      iex> c.counts
      [4, 1, 0, 0, 1]
      iex> Spatex.Classify.legend(c)
      ["[ 1.00,  7.00]", "( 7.00, 13.00]", "(13.00, 19.00]", "(19.00, 25.00]", "(25.00, 31.00]"]
      iex> Spatex.Classify.find_bin(c, [0, 8, 100])
      [0, 1, 4]
  """

  require Logger

  alias Spatex.{Input, Native}

  @type method ::
          :equal_interval
          | :quantiles
          | :box_plot
          | :std_mean
          | :percentiles
          | :maximum_breaks
          | :head_tail_breaks
          | :user_defined
          | :fisher_jenks
          | :fisher_jenks_sampled
          | :jenks_caspall
          | :jenks_caspall_sampled
          | :jenks_caspall_forced
          | :natural_breaks
          | :pretty_breaks
          | :max_p

  @type t :: %__MODULE__{
          method: method(),
          k: pos_integer(),
          bins: [float()],
          counts: [non_neg_integer()],
          yb: [non_neg_integer()],
          tss: float(),
          adcm: float(),
          gadf: float(),
          lowest: float(),
          info: map(),
          warnings: [String.t()]
        }

  @enforce_keys [:method, :k, :bins, :counts, :yb, :tss, :adcm, :gadf, :lowest]
  defstruct [
    :method,
    :k,
    :bins,
    :counts,
    :yb,
    :tss,
    :adcm,
    :gadf,
    :lowest,
    info: %{},
    warnings: []
  ]

  @doc """
  Equal-width classes between the minimum and maximum (`mapclassify.EqualInterval`).

  Raises when all values are equal. Options: `:k` (default `5`).
  """
  @spec equal_interval(term(), keyword()) :: t()
  def equal_interval(y, opts \\ []) do
    keys!(opts, [:k])
    k = pos_int!(opts, :k, 5)
    run(:equal_interval, y, k, [], false, nil, %{})
  end

  @doc """
  Classes holding (approximately) the same number of observations (`mapclassify.Quantiles`).

  Percentiles use linear interpolation like `scipy.stats.scoreatpercentile`; repeated cut
  points are dropped, which leaves fewer than `k` bins and logs a warning. The top bin is
  always `max(y)`: mapclassify's last percentile can drift just below 100 (for example
  `k = 23` on `1..46`) and then assigns the maximum to class 0; Spatex does not. Options:
  `:k` (default `5`).
  """
  @spec quantiles(term(), keyword()) :: t()
  def quantiles(y, opts \\ []) do
    keys!(opts, [:k])
    k = pos_int!(opts, :k, 5)
    run(:quantiles, y, k, [], false, nil, %{})
  end

  @doc """
  Box-plot classes (`mapclassify.BoxPlot`): lower fence, Q1, median, Q3, upper fence, maximum.

  With a maximum below the upper fence the fence is dropped from the bins, leaving five
  classes. `info` holds `:hinge`, `:iqr`, `:low_outlier_ids` (class 0, values below the lower
  fence) and `:high_outlier_ids` (class 5). Ids are zero-based positions in `y`. Options:
  `:hinge` (default `1.5`).
  """
  @spec box_plot(term(), keyword()) :: t()
  def box_plot(y, opts \\ []) do
    keys!(opts, [:hinge])
    hinge = number!(opts, :hinge, 1.5)
    c = run(:box_plot, y, hinge, [], false, nil, %{hinge: hinge * 1.0})
    ids = fn cls -> for {b, i} <- Enum.with_index(c.yb), b == cls, do: i end
    %{c | info: Map.merge(c.info, %{low_outlier_ids: ids.(0), high_outlier_ids: ids.(5)})}
  end

  @doc """
  Classes at multiples of the sample standard deviation around the mean
  (`mapclassify.StdMean`, standard deviation with `ddof = 1`).

  The maximum is appended as a last bin when the last cut is below it. Options: `:multiples`
  (default `[-2, -1, 1, 2]`) and `:anchor` (default `false`); with `anchor: true` the multiples
  become every integer from `trunc((min - mean) / std)` to `trunc((max - mean) / std)`
  (the given `:multiples` are then ignored).
  """
  @spec std_mean(term(), keyword()) :: t()
  def std_mean(y, opts \\ []) do
    keys!(opts, [:multiples, :anchor])
    multiples = num_list!(opts, :multiples, [-2, -1, 1, 2])
    anchor = Keyword.get(opts, :anchor, false)

    is_boolean(anchor) ||
      raise ArgumentError, ":anchor must be a boolean, got: #{inspect(anchor)}"

    run(:std_mean, y, 0.0, multiples, anchor, nil, %{multiples: multiples, anchor: anchor})
  end

  @doc """
  Classes cut at percentiles (`mapclassify.Percentiles`). Options: `:pct` (default
  `[1, 10, 50, 90, 99, 100]`, each in `0..100`).
  """
  @spec percentiles(term(), keyword()) :: t()
  def percentiles(y, opts \\ []) do
    keys!(opts, [:pct])
    pct = num_list!(opts, :pct, [1, 10, 50, 90, 99, 100])
    run(:percentiles, y, 0.0, pct, false, nil, %{pct: pct})
  end

  @doc """
  Breaks at the `k - 1` largest gaps between sorted values (`mapclassify.MaximumBreaks`).

  Raises when all values are equal or there are fewer than `k` values. Options: `:k`
  (default `5`) and `:mindiff` (default `0`; stored in `info`, unused, as in mapclassify).
  """
  @spec maximum_breaks(term(), keyword()) :: t()
  def maximum_breaks(y, opts \\ []) do
    keys!(opts, [:k, :mindiff])
    k = pos_int!(opts, :k, 5)
    mindiff = number!(opts, :mindiff, 0)
    run(:maximum_breaks, y, k, [], false, nil, %{mindiff: mindiff})
  end

  @doc """
  Head/tail breaks for heavy-tailed data (`mapclassify.HeadTailBreaks`): the mean, then the
  mean of the values above it, until the head is constant.
  """
  @spec head_tail_breaks(term()) :: t()
  def head_tail_breaks(y), do: run(:head_tail_breaks, y, 0.0, [], false, nil, %{})

  @doc """
  Classes with the given upper bounds (`mapclassify.UserDefined`).

  When the last bin is below `max(y)` the maximum is appended as an extra bin. Option
  `:lowest` only changes the lower edge shown by `legend/2` and `intervals/1`.
  """
  @spec user_defined(term(), [number()], keyword()) :: t()
  def user_defined(y, bins, opts \\ []) do
    keys!(opts, [:lowest])
    bins = num_list!([bins: bins], :bins, nil)
    lowest = Keyword.get(opts, :lowest)

    if not (is_nil(lowest) or is_number(lowest)),
      do: raise(ArgumentError, ":lowest must be a number or nil, got: #{inspect(lowest)}")

    run(:user_defined, y, 0.0, bins, false, lowest && lowest * 1.0, %{})
  end

  @doc """
  Exact optimal classes minimising the within-class sum of squares (`mapclassify.FisherJenks`).

  Raises when `y` has fewer than `k` unique values. Cost is `O(n^2 k)` time and `O(n k)`
  memory. Options: `:k` (default `5`).

      iex> Spatex.Classify.fisher_jenks([1, 2, 3, 10, 11, 12, 30, 31], k: 3).bins
      [3.0, 12.0, 31.0]
  """
  @spec fisher_jenks(term(), keyword()) :: t()
  def fisher_jenks(y, opts \\ []) do
    keys!(opts, [:k])
    k = pos_int!(opts, :k, 5)
    run(:fisher_jenks, y, k, [], false, nil, %{})
  end

  @doc """
  Fisher-Jenks bins fitted on a random subsample, then applied to all of `y`
  (`mapclassify.FisherJenksSampled`).

  The subsample has `trunc(n * pct)` draws with replacement, always including `min(y)` and
  `max(y)`; with `truncate: true` it is capped at about 1000 draws. Options: `:k` (default
  `5`), `:pct` (default `0.1`, in `(0, 1]`), `:truncate` (default `true`) and `:seed`
  (default `nil`: a random seed, recorded in `info.seed`).
  """
  @spec fisher_jenks_sampled(term(), keyword()) :: t()
  def fisher_jenks_sampled(y, opts \\ []) do
    keys!(opts, [:k, :pct, :truncate, :seed])
    k = pos_int!(opts, :k, 5)
    pct = pct!(opts)
    truncate = Keyword.get(opts, :truncate, true)

    is_boolean(truncate) ||
      raise ArgumentError, ":truncate must be a boolean, got: #{inspect(truncate)}"

    seed = seed!(opts)
    run(:fisher_jenks_sampled, y, k, [pct], truncate, nil, %{pct: pct, seed: seed}, seed)
  end

  @doc """
  Jenks-Caspall classes: alternate between assigning every value to the nearest class median and
  recomputing the medians, starting from quantiles (`mapclassify.JenksCaspall`).

  `info.iterations` holds the number of rounds. If a class becomes empty mapclassify produces
  undefined results; here the iteration stops and the non-empty classes are returned. Options:
  `:k` (default `5`).
  """
  @spec jenks_caspall(term(), keyword()) :: t()
  def jenks_caspall(y, opts \\ []) do
    keys!(opts, [:k])
    k = pos_int!(opts, :k, 5)
    run(:jenks_caspall, y, k, [], false, nil, %{})
  end

  @doc """
  Jenks-Caspall bins fitted on a random subsample (`mapclassify.JenksCaspallSampled`); the
  subsample always includes `max(y)`. Options: `:k` (default `5`), `:pct` (default `0.1`) and
  `:seed` (default `nil`: drawn and recorded in `info.seed`). The subsample is capped at about
  1000 draws.
  """
  @spec jenks_caspall_sampled(term(), keyword()) :: t()
  def jenks_caspall_sampled(y, opts \\ []) do
    keys!(opts, [:k, :pct, :seed])
    k = pos_int!(opts, :k, 5)
    pct = pct!(opts)
    seed = seed!(opts)
    run(:jenks_caspall_sampled, y, k, [pct], false, nil, %{pct: pct, seed: seed}, seed)
  end

  @doc """
  Jenks-Caspall with forced single-observation moves between neighbouring quantile classes
  while the within-class sum of squares decreases (`mapclassify.JenksCaspallForced`).

  Raises when all values are equal. Options: `:k` (default `5`).

  Moves are by position in `y`, so classes need not be contiguous in value and `mapclassify`
  can return unordered class maxima. Spatex deviates there and always returns ascending `bins`.
  """
  @spec jenks_caspall_forced(term(), keyword()) :: t()
  def jenks_caspall_forced(y, opts \\ []) do
    keys!(opts, [:k])
    k = pos_int!(opts, :k, 5)
    run(:jenks_caspall_forced, y, k, [], false, nil, %{})
  end

  @doc """
  Natural breaks: one-dimensional k-means (`mapclassify.NaturalBreaks`).

  Runs `:initial` k-means++ restarts from a ChaCha stream seeded by `:seed` and keeps the
  lowest within-class variance; the bins are the class maxima. With fewer than `k` unique values
  every unique value becomes a class (logged as a warning). Options: `:k` (default `5`),
  `:initial` (default `10`) and `:seed` (default `nil`: a random seed, recorded in `info.seed`).
  Unlike mapclassify, whose start points are unseeded, an explicit seed makes the result a pure
  function of the arguments.
  """
  @spec natural_breaks(term(), keyword()) :: t()
  def natural_breaks(y, opts \\ []) do
    keys!(opts, [:k, :initial, :seed])
    k = pos_int!(opts, :k, 5)
    initial = pos_int!(opts, :initial, 10)
    seed = seed!(opts)
    run(:natural_breaks, y, k, [initial], false, nil, %{initial: initial, seed: seed}, seed)
  end

  @doc """
  Pretty breaks: equally spaced round bounds (`mapclassify.PrettyBreaks`).

  The step is 1, 2 or 5 times a power of ten, chosen from the range and `k`; the bounds are
  the multiples of the step from below `min(y)` up to at least `max(y)`, so the number of
  classes can differ from `k`. As in mapclassify, `k` stays the requested value and the fit
  statistics (`tss`, `adcm`, `gadf`) cover classes `0..k-1`. Options: `:k` (default `5`, at
  least `2`).

      iex> Spatex.Classify.pretty_breaks([1, 7, 4, 12, 4, 31], k: 6).bins
      [10.0, 20.0, 30.0, 40.0]
  """
  @spec pretty_breaks(term(), keyword()) :: t()
  def pretty_breaks(y, opts \\ []) do
    keys!(opts, [:k])
    k = pos_int!(opts, :k, 5)
    run(:pretty_breaks, y, k, [], false, nil, %{})
  end

  @doc """
  Max-p classes (`mapclassify.MaxP`): `k` contiguous classes of the sorted values grown from
  quantile seeds, then one round of border swaps that do not raise the sum of squares.

  Options: `:k` (default `5`), `:initial` (default `1000`), `:seed1` (default `0`) and
  `:seed2` (default `1`). The growth and swap orders replay numpy's legacy generator seeded with
  `:seed1` / `:seed2`, as mapclassify does, so results match it. mapclassify mixes sorted
  positions with unsorted values, so it only agrees with this function on sorted input (see
  `docs/parity.md`). Raises when fewer than `k` distinct quantiles exist.
  """
  @spec max_p(term(), keyword()) :: t()
  def max_p(y, opts \\ []) do
    keys!(opts, [:k, :initial, :seed1, :seed2])
    k = pos_int!(opts, :k, 5)
    initial = pos_int!(opts, :initial, 1000)
    seed1 = u32!(opts, :seed1, 0)
    seed2 = u32!(opts, :seed2, 1)

    run(:max_p, y, k, [initial, seed1, seed2], false, nil, %{
      initial: initial,
      seed1: seed1,
      seed2: seed2
    })
  end

  @k_methods [:quantiles, :fisher_jenks, :natural_breaks, :maximum_breaks]

  @doc """
  The smallest `k` whose classification reaches a goodness of absolute deviation fit above
  `:pct` (`mapclassify.gadf`).

  Tries `k = 2..maxk` with `:method` (a method atom or scheme name, default `:quantiles`) and
  stops at the first classification with `gadf > pct`; when none does, the `maxk` one is
  returned. Options: `:method`, `:maxk` (default `15`, at least `2`), `:pct` (default `0.8`,
  in `(0, 1)`) and the method's own options. Returns `{k, classification, gadf}`.

      iex> {k, _c, _gadf} = Spatex.Classify.gadf([1, 2, 3, 10, 11, 12, 30, 31], method: :fisher_jenks)
      iex> k
      3
  """
  @spec gadf(term(), keyword()) :: {pos_integer(), t(), float()}
  def gadf(y, opts \\ []) do
    {method, opts} = Keyword.pop(opts, :method, :quantiles)
    {maxk, opts} = Keyword.pop(opts, :maxk, 15)
    {pct, opts} = Keyword.pop(opts, :pct, 0.8)

    unless is_integer(maxk) and maxk >= 2,
      do: raise(ArgumentError, ":maxk must be an integer of at least 2, got: #{inspect(maxk)}")

    unless is_number(pct) and pct > 0 and pct < 1,
      do: raise(ArgumentError, ":pct must be a number in (0, 1), got: #{inspect(pct)}")

    Enum.reduce_while(2..maxk, nil, fn k, _acc ->
      c = classify(y, method, Keyword.put(opts, :k, k))
      result = {k, c, c.gadf}
      if c.gadf > pct, do: {:halt, result}, else: {:cont, result}
    end)
  end

  @doc """
  Runs `gadf/2` for Quantiles, FisherJenks, NaturalBreaks and MaximumBreaks
  (`mapclassify.KClassifiers`) with `maxk = n - 1`.

  Returns `%{results: %{method => {k, classification, gadf}}, best: {method, k,
  classification, gadf}}`, where the best has the smallest `k`, ties broken by the largest
  `gadf` (FisherJenks first). mapclassify swaps `k` and `gadf` in that comparison, so its `best`
  can differ (see `docs/parity.md`); NaturalBreaks uses Spatex's seeded k-means. Options:
  `:pct` (default `0.8`).
  """
  @spec k_classifiers(term(), keyword()) :: %{
          results: %{method() => {pos_integer(), t(), float()}},
          best: {method(), pos_integer(), t(), float()}
        }
  def k_classifiers(y, opts \\ []) do
    keys!(opts, [:pct])
    pct = Keyword.get(opts, :pct, 0.8)
    {_bin, n} = Input.to_f64_binary(y, name: "y")

    if n < 3,
      do: raise(ArgumentError, "k_classifiers needs at least 3 values, got #{n}")

    results = Map.new(@k_methods, &{&1, gadf(y, method: &1, maxk: n - 1, pct: pct)})

    {method, {k, c, g}} =
      [:fisher_jenks | @k_methods -- [:fisher_jenks]]
      |> Enum.map(&{&1, results[&1]})
      |> Enum.reduce(fn {_m, {k1, _c1, g1}} = cand, {_bm, {k0, _c0, g0}} = best ->
        if k1 < k0 or (k1 == k0 and g1 > g0), do: cand, else: best
      end)

    %{results: results, best: {method, k, c, g}}
  end

  defp u32!(opts, key, default) do
    case Keyword.get(opts, key, default) do
      s when is_integer(s) and s >= 0 and s < 4_294_967_296 ->
        s

      other ->
        raise ArgumentError,
              "#{inspect(key)} must be an integer in 0..4294967295, got: #{inspect(other)}"
    end
  end

  @schemes %{
    "equalinterval" => :equal_interval,
    "quantiles" => :quantiles,
    "boxplot" => :box_plot,
    "stdmean" => :std_mean,
    "percentiles" => :percentiles,
    "maximumbreaks" => :maximum_breaks,
    "headtailbreaks" => :head_tail_breaks,
    "userdefined" => :user_defined,
    "fisherjenks" => :fisher_jenks,
    "fisherjenkssampled" => :fisher_jenks_sampled,
    "jenkscaspall" => :jenks_caspall,
    "jenkscaspallsampled" => :jenks_caspall_sampled,
    "jenkscaspallforced" => :jenks_caspall_forced,
    "naturalbreaks" => :natural_breaks,
    "prettybreaks" => :pretty_breaks,
    "maxp" => :max_p
  }

  @doc """
  Classifies `y` with the named scheme, like `mapclassify.classify`.

  `scheme` is a method atom (`:fisher_jenks`, `:quantiles`, ...) or a mapclassify scheme name
  such as `"FisherJenks"` or `"natural_breaks"` (case and underscores are ignored). `opts` are
  those of the matching function; `:user_defined` takes its bins as `bins:`.

      iex> Spatex.Classify.classify([1, 2, 3, 10, 11, 12, 30, 31], :fisher_jenks, k: 3).bins
      [3.0, 12.0, 31.0]
  """
  @spec classify(term(), method() | String.t(), keyword()) :: t()
  def classify(y, scheme, opts \\ []) do
    case scheme_method(scheme) do
      :head_tail_breaks -> head_tail_breaks(y)
      :user_defined -> user_defined(y, Keyword.get(opts, :bins), Keyword.delete(opts, :bins))
      method -> apply(__MODULE__, method, [y, opts])
    end
  end

  defp scheme_method(scheme) when is_atom(scheme) or is_binary(scheme) do
    key = scheme |> to_string() |> String.downcase() |> String.replace(["_", "-", " "], "")

    Map.get(@schemes, key) ||
      raise ArgumentError,
            "unknown classification scheme #{inspect(scheme)}, expected one of: " <>
              Enum.map_join(Enum.sort(Map.values(@schemes)), ", ", &inspect/1)
  end

  defp scheme_method(scheme),
    do: raise(ArgumentError, "scheme must be an atom or a string, got: #{inspect(scheme)}")

  # Unknown option keys raise; boolean flags must be booleans.
  defp keys!(opts, allowed) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    case Keyword.keys(opts) -- allowed do
      [] ->
        :ok

      bad ->
        raise ArgumentError, "unknown option(s) #{inspect(bad)}; expected #{inspect(allowed)}"
    end

    for key <- [:anchor, :truncate],
        Keyword.has_key?(opts, key),
        not is_boolean(opts[key]),
        do: raise(ArgumentError, "#{inspect(key)} must be a boolean, got #{inspect(opts[key])}")

    :ok
  end

  defp pct!(opts) do
    case Keyword.get(opts, :pct, 0.1) do
      p when is_number(p) and p > 0 and p <= 1 ->
        p * 1.0

      other ->
        raise ArgumentError, ":pct must be a number in (0, 1], got: #{inspect(other)}"
    end
  end

  # `nil` (the default) draws a seed, recorded in `info.seed`; an integer is used as given.
  defp seed!(opts) do
    case Keyword.get(opts, :seed) do
      nil ->
        :rand.uniform(4_294_967_296) - 1

      s when is_integer(s) and s >= 0 ->
        s

      other ->
        raise ArgumentError, ":seed must be a non-negative integer, got: #{inspect(other)}"
    end
  end

  @doc """
  Class index of each value against the classification's bins (`MapClassifier.find_bin`).

  Like `numpy.digitize(x, bins, right: true)`, but values above the last bin fall in the last
  class and values below the first in class 0. A number returns an integer, anything else a list.
  """
  @spec find_bin(t(), number() | term()) :: non_neg_integer() | [non_neg_integer()]
  def find_bin(%__MODULE__{bins: bins}, value) when is_number(value) do
    [bin] = find_bin_list(bins, [value])
    bin
  end

  def find_bin(%__MODULE__{bins: bins}, values), do: find_bin_list(bins, values)

  defp find_bin_list(bins, values) do
    {bin, _n} = Input.to_f64_binary(values, name: "values")
    bins |> Native.classify_find_bin(bin) |> unwrap() |> u32_list()
  end

  @doc """
  The class intervals as `{low, high}` pairs. The first `low` is `lowest`, or `:neg_infinity`
  when `lowest` is above the first bin. All intervals are closed on the right; the others are
  open on the left, and so is the first when its `low` is `:neg_infinity`.
  """
  @spec intervals(t()) :: [{float() | :neg_infinity, float()}]
  def intervals(%__MODULE__{bins: [first | _] = bins, lowest: lowest}) do
    low = if lowest > first, do: :neg_infinity, else: lowest
    Enum.zip([low | bins], bins)
  end

  @doc """
  Legend labels identical to `mapclassify`'s `get_legend_classes/1`, such as `"[ 1.00,  7.00]"`.

  Edges are formatted with `:decimals` digits (default `2`, mapclassify's `"{:.2f}"`) and
  right-aligned to a common width; the first label starts with `[` unless the lower edge is
  open (`-inf`), then with `(`.
  """
  @spec legend(t(), keyword()) :: [String.t()]
  def legend(%__MODULE__{} = c, opts \\ []) do
    keys!(opts, [:decimals])
    decimals = Keyword.get(opts, :decimals, 2)

    (is_integer(decimals) and decimals >= 0) ||
      raise ArgumentError, ":decimals must be a non-negative integer"

    [{low, _} | _] = pairs = intervals(c)

    fmt = fn
      :neg_infinity -> "-inf"
      x -> format_fixed(x, decimals)
    end

    edges = Enum.map([low | Enum.map(pairs, &elem(&1, 1))], fmt)
    width = edges |> Enum.map(&String.length/1) |> Enum.max()
    pad = &String.pad_leading(&1, width)
    open = if low == :neg_infinity, do: "(", else: "["

    edges
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.with_index()
    |> Enum.map(fn {[l, r], i} ->
      if(i == 0, do: open, else: "(") <> pad.(l) <> ", " <> pad.(r) <> "]"
    end)
  end

  # Python's `"{:.Nf}".format(x)`: the exact binary value rounded half to even. Erlang's
  # float_to_binary rounds exact ties away from zero, so format with integer arithmetic.
  defp format_fixed(x, decimals) do
    <<sign::1, exp::11, frac::52>> = <<x::float-64>>
    {mant, e} = if exp == 0, do: {frac, -1074}, else: {frac + 0x10000000000000, exp - 1075}
    scaled = mant * Integer.pow(10, decimals)

    {num, den} =
      if e >= 0, do: {scaled * Integer.pow(2, e), 1}, else: {scaled, Integer.pow(2, -e)}

    q = div(num, den)
    r2 = 2 * rem(num, den)
    q = if r2 > den or (r2 == den and rem(q, 2) == 1), do: q + 1, else: q
    digits = q |> Integer.to_string() |> String.pad_leading(decimals + 1, "0")
    {int, dec} = String.split_at(digits, String.length(digits) - decimals)
    body = if decimals == 0, do: int, else: int <> "." <> dec
    if sign == 1, do: "-" <> body, else: body
  end

  defp run(method, y, scalar, list, flag, lowest, info, seed \\ 0) do
    {bin, _n} = Input.to_f64_binary(y, name: "y")

    {bins, k, yb, counts, {tss, adcm, gadf}, {low, iqr, warnings, iterations}} =
      method
      |> Atom.to_string()
      |> Native.classify_run(bin, scalar * 1.0, Enum.map(list, &(&1 * 1.0)), flag, lowest, seed)
      |> unwrap()

    Enum.each(warnings, &Logger.warning/1)
    info = if iqr, do: Map.put(info, :iqr, iqr), else: info
    info = if iterations, do: Map.put(info, :iterations, iterations), else: info

    %__MODULE__{
      method: method,
      k: k,
      bins: bins,
      counts: counts,
      yb: u32_list(yb),
      tss: tss,
      adcm: adcm,
      gadf: gadf,
      lowest: low,
      info: info,
      warnings: warnings
    }
  end

  defp u32_list(bin), do: for(<<x::unsigned-native-32 <- bin>>, do: x)

  defp unwrap({:ok, value}), do: value
  defp unwrap({:error, reason}), do: raise(ArgumentError, to_string(reason))

  defp pos_int!(opts, key, default) do
    case Keyword.get(opts, key, default) do
      k when is_integer(k) and k > 0 ->
        k

      other ->
        raise ArgumentError, "#{inspect(key)} must be a positive integer, got: #{inspect(other)}"
    end
  end

  defp number!(opts, key, default) do
    case Keyword.get(opts, key, default) do
      x when is_number(x) -> x
      other -> raise ArgumentError, "#{inspect(key)} must be a number, got: #{inspect(other)}"
    end
  end

  defp num_list!(opts, key, default) do
    case Keyword.get(opts, key, default) do
      [_ | _] = list ->
        if Enum.all?(list, &is_number/1),
          do: list,
          else:
            raise(
              ArgumentError,
              "#{inspect(key)} must be a list of numbers, got: #{inspect(list)}"
            )

      other ->
        raise ArgumentError,
              "#{inspect(key)} must be a non-empty list of numbers, got: #{inspect(other)}"
    end
  end
end
