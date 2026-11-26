defmodule Spatex.Graph.Contiguity do
  @moduledoc false
  # Internals of `Spatex.Graph.build_contiguity/2` and `build_block_contiguity/2`: option and
  # input validation in Elixir, the geometry work in the `spatex_core::contiguity` Rust module.

  alias Spatex.Graph
  alias Spatex.Native

  @contiguity_opts [:rook, :strict, :by_perimeter, :ids]
  @block_opts [:ids]
  @fuzzy_opts [:tolerance, :buffer, :predicate, :ids]
  @predicates ~w(intersects touches overlaps within contains covers covered_by contains_properly
                 crosses)a

  @spec build_fuzzy(term(), keyword()) :: Graph.t()
  def build_fuzzy(wkbs, opts) do
    opts = validate_opts!(opts, @fuzzy_opts)
    tolerance = distance_opt!(opts, :tolerance)
    buffer = distance_opt!(opts, :buffer)

    if tolerance && buffer,
      do: raise(ArgumentError, "only one of :tolerance and :buffer can be given, not both")

    predicate = predicate!(Keyword.get(opts, :predicate, :intersects))
    wkbs = check_wkbs!(wkbs)
    n = length(wkbs)
    ids = resolve_ids!(Keyword.get(opts, :ids), n)

    wkbs
    |> Native.fuzzy_contiguity_edges(tolerance, buffer, predicate)
    |> unwrap()
    |> to_graph(n, ids)
  end

  @spec build(term(), keyword()) :: Graph.t()
  def build(wkbs, opts) do
    opts = validate_opts!(opts, @contiguity_opts)
    rook = Keyword.get(opts, :rook, true)
    strict = Keyword.get(opts, :strict, false)
    by_perimeter = Keyword.get(opts, :by_perimeter, false)

    for {name, value} <- [rook: rook, strict: strict, by_perimeter: by_perimeter],
        not is_boolean(value),
        do: raise(ArgumentError, ":#{name} must be a boolean, got #{inspect(value)}")

    wkbs = check_wkbs!(wkbs)
    n = length(wkbs)
    ids = resolve_ids!(Keyword.get(opts, :ids), n)
    wkbs |> Native.contiguity_edges(rook, strict, by_perimeter) |> unwrap() |> to_graph(n, ids)
  end

  @spec build_block(term(), keyword()) :: Graph.t()
  def build_block(regimes, opts) do
    opts = validate_opts!(opts, @block_opts)
    unless is_list(regimes), do: raise(ArgumentError, "regimes must be a list")
    if regimes == [], do: raise(ArgumentError, "regimes must contain at least one value")

    n = length(regimes)
    ids = resolve_ids!(Keyword.get(opts, :ids), n)
    codes = regime_codes!(regimes)
    codes |> Native.block_edges() |> unwrap() |> to_graph(n, ids)
  end

  # -- helpers --------------------------------------------------------------------------

  defp to_graph({focal, neighbor, <<>>}, n, ids), do: to_graph({focal, neighbor}, n, ids)
  defp to_graph({focal, neighbor, weights}, n, ids), do: graph(focal, neighbor, weights, n, ids)

  defp to_graph({focal, neighbor}, n, ids) do
    nnz = div(byte_size(focal), 4)
    graph(focal, neighbor, :binary.copy(<<1.0::float-native-64>>, nnz), n, ids)
  end

  defp graph(focal, neighbor, weights, n, ids) do
    ref = unwrap(Native.graph_from_arrays(n, focal, neighbor, weights))
    %Graph{ref: ref, n: n, ids: ids, transform: :o}
  end

  defp check_wkbs!(wkbs) do
    unless is_list(wkbs), do: raise(ArgumentError, "geometries must be a list of WKB binaries")
    if wkbs == [], do: raise(ArgumentError, "geometries must contain at least one WKB binary")
    wkbs |> Enum.with_index() |> Enum.each(&check_wkb!/1)
    wkbs
  end

  defp distance_opt!(opts, key) do
    case Keyword.get(opts, key) do
      nil ->
        nil

      value when is_number(value) and value >= 0 and value < 1.0e308 ->
        value * 1.0

      value ->
        raise ArgumentError,
              ":#{key} must be a finite number that is not negative, got #{inspect(value)}"
    end
  end

  defp predicate!(nil), do: nil
  defp predicate!(predicate) when predicate in @predicates, do: Atom.to_string(predicate)

  defp predicate!(:dwithin) do
    raise ArgumentError,
          ":dwithin is not ported (it needs a distance); use :intersects with :buffer instead"
  end

  defp predicate!(other) do
    raise ArgumentError,
          ":predicate must be nil (bounding boxes) or one of #{inspect(@predicates)}, " <>
            "got #{inspect(other)}"
  end

  defp check_wkb!({wkb, _index}) when is_binary(wkb), do: :ok

  defp check_wkb!({other, index}) do
    raise ArgumentError, "geometry #{index}: expected a WKB binary, got #{inspect(other)}"
  end

  # First-appearance regime codes as a u32 binary. Regimes are compared with `===`.
  defp regime_codes!(regimes) do
    {codes, _seen} =
      regimes
      |> Enum.with_index()
      |> Enum.map_reduce(%{}, fn
        {nil, index}, _seen ->
          raise ArgumentError, "regime #{index}: nil is not allowed"

        {regime, _index}, seen ->
          code = Map.get_lazy(seen, regime, fn -> map_size(seen) end)
          {code, Map.put(seen, regime, code)}
      end)

    for code <- codes, into: <<>>, do: <<code::unsigned-integer-32-native>>
  end

  defp resolve_ids!(nil, n), do: Enum.to_list(0..(n - 1)//1)

  defp resolve_ids!(ids, n) when is_list(ids) do
    cond do
      length(ids) != n ->
        raise ArgumentError, ":ids must have one id per node (#{n}), got #{length(ids)}"

      length(Enum.uniq(ids)) != n ->
        raise ArgumentError, ":ids must be unique"

      true ->
        ids
    end
  end

  defp resolve_ids!(_other, _n), do: raise(ArgumentError, ":ids must be a list")

  defp validate_opts!(opts, allowed) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    case Keyword.keys(opts) -- allowed do
      [] ->
        opts

      [key | _] ->
        raise ArgumentError, "unknown option #{inspect(key)}, expected one of #{inspect(allowed)}"
    end
  end

  defp unwrap({:ok, value}), do: value
  defp unwrap({:error, reason}), do: raise(ArgumentError, to_string(reason))
end
