defmodule Spatex.Graph.Builders.Knn do
  @moduledoc false
  # Internal builders behind `Spatex.Graph.build_knn/3` and `Spatex.Graph.build_distance_band/3`.
  # Validates options, normalizes points through `Spatex.Input`, and wraps the Rust graph.

  alias Spatex.Graph
  alias Spatex.Graph.Builders.Kernel, as: KernelBuilder
  alias Spatex.Input
  alias Spatex.Native

  @coincident [:raise, :clique, :jitter]
  @default_max_edges 50_000_000

  @spec knn(term(), term(), keyword()) :: Graph.t()
  def knn(points, k, opts) do
    opts =
      validate!(opts,
        ids: nil,
        coincident: :raise,
        seed: 0,
        metric: :euclidean,
        max_edges: @default_max_edges
      )

    unless is_integer(k) and k >= 1 do
      raise ArgumentError, "k must be a positive integer, got #{inspect(k)}"
    end

    coincident = coincident!(opts)

    {bin, n} = Input.to_points_binary(points)
    ids = resolve_ids!(Keyword.fetch!(opts, :ids), n)

    ref =
      bin
      |> Native.weights_knn(k, coincident, metric!(opts), max_edges!(opts))
      |> unwrap()

    graph(ref, n, ids)
  end

  @spec distance_band(term(), term(), keyword()) :: Graph.t()
  def distance_band(points, threshold, opts) do
    opts =
      validate!(opts,
        ids: nil,
        binary: true,
        alpha: -1.0,
        kernel: nil,
        bandwidth: nil,
        taper: true,
        decay: false,
        metric: :euclidean,
        max_edges: @default_max_edges
      )

    unless is_number(threshold) and threshold > 0 do
      raise ArgumentError, "threshold must be a positive number, got #{inspect(threshold)}"
    end

    binary = Keyword.fetch!(opts, :binary)

    unless is_boolean(binary) do
      raise ArgumentError, ":binary must be true or false, got #{inspect(binary)}"
    end

    alpha = Keyword.fetch!(opts, :alpha)

    unless is_number(alpha) do
      raise ArgumentError, ":alpha must be a number, got #{inspect(alpha)}"
    end

    kernel_opts?(opts, binary)

    {bin, n} = Input.to_points_binary(points)
    ids = resolve_ids!(Keyword.fetch!(opts, :ids), n)

    ref =
      if Keyword.fetch!(opts, :kernel) do
        KernelBuilder.band(bin, n, threshold, opts)
      else
        bin
        |> Native.weights_distance_band(
          threshold * 1.0,
          binary,
          alpha * 1.0,
          metric!(opts),
          max_edges!(opts)
        )
        |> unwrap()
      end

    graph(ref, n, ids)
  end

  # `:kernel` selects kernel weights and, like libpysal, only applies to `binary: false`.
  defp kernel_opts?(opts, binary) do
    kernel = Keyword.fetch!(opts, :kernel)
    set = [:bandwidth] |> Enum.reject(&is_nil(Keyword.fetch!(opts, &1)))

    cond do
      kernel && binary ->
        raise ArgumentError, ":kernel needs binary: false (binary bands ignore kernels)"

      kernel == nil and set != [] ->
        raise ArgumentError, ":bandwidth is only used together with a :kernel"

      true ->
        :ok
    end
  end

  @doc false
  @spec validate!(term(), keyword()) :: keyword()
  def validate!(opts, defaults) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    case Keyword.validate(opts, defaults) do
      {:ok, opts} ->
        opts

      {:error, invalid} ->
        raise ArgumentError,
              "unknown option(s) #{inspect(invalid)}; expected #{inspect(Keyword.keys(defaults))}"
    end
  end

  @doc false
  @spec default_max_edges() :: pos_integer()
  def default_max_edges, do: @default_max_edges

  @doc false
  # The validated `:max_edges` option: a positive integer edge budget.
  @spec max_edges!(keyword()) :: pos_integer()
  def max_edges!(opts) do
    case Keyword.fetch!(opts, :max_edges) do
      m when is_integer(m) and m > 0 ->
        m

      other ->
        raise ArgumentError, ":max_edges must be a positive integer, got #{inspect(other)}"
    end
  end

  @doc false
  # The validated `:coincident` and `:seed` options as the NIF's `{mode, seed}`.
  @spec coincident!(keyword()) :: {String.t(), non_neg_integer()}
  def coincident!(opts) do
    coincident = Keyword.fetch!(opts, :coincident)
    seed = Keyword.fetch!(opts, :seed)

    unless coincident in @coincident do
      raise ArgumentError,
            ":coincident must be one of #{inspect(@coincident)}, got #{inspect(coincident)}"
    end

    unless is_integer(seed) and seed >= 0 do
      raise ArgumentError, ":seed must be a non-negative integer, got #{inspect(seed)}"
    end

    {Atom.to_string(coincident), seed}
  end

  @doc false
  # The validated `:metric` option as the NIF's metric name.
  @spec metric!(keyword()) :: String.t()
  def metric!(opts) do
    case Keyword.fetch!(opts, :metric) do
      m when m in [:euclidean, :haversine] ->
        Atom.to_string(m)

      other ->
        raise ArgumentError,
              ":metric must be :euclidean or :haversine, got #{inspect(other)}"
    end
  end

  @doc false
  @spec resolve_ids!(term(), non_neg_integer()) :: [term()]
  def resolve_ids!(nil, n), do: Enum.to_list(0..(n - 1)//1)

  def resolve_ids!(ids, n) when is_list(ids) do
    cond do
      length(ids) != n ->
        raise ArgumentError, ":ids has #{length(ids)} entries but there are #{n} points"

      length(Enum.uniq(ids)) != n ->
        raise ArgumentError, ":ids must be unique"

      true ->
        ids
    end
  end

  def resolve_ids!(_other, _n), do: raise(ArgumentError, ":ids must be a list")

  @doc false
  @spec graph(reference(), non_neg_integer(), [term()]) :: Graph.t()
  def graph(ref, n, ids), do: %Graph{ref: ref, n: n, ids: ids, transform: :o}

  @doc false
  @spec unwrap(term()) :: reference()
  def unwrap({:ok, value}), do: value
  def unwrap({:error, reason}), do: raise(ArgumentError, to_string(reason))
  def unwrap(ref) when is_reference(ref), do: ref
end
