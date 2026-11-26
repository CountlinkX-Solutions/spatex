defmodule Spatex.Graph.Builders.Kernel do
  @moduledoc false
  # Internal builder behind `Spatex.Graph.build_kernel/2` and the `:kernel` option of
  # `Spatex.Graph.build_distance_band/3`.

  alias Spatex.Graph
  alias Spatex.Graph.Builders.Knn
  alias Spatex.Input
  alias Spatex.Native

  @kernels [
    :gaussian,
    :triangular,
    :parabolic,
    :bisquare,
    :tricube,
    :cosine,
    :boxcar,
    :exponential,
    :identity
  ]
  @aliases %{epanechnikov: :parabolic, uniform: :boxcar, discrete: :boxcar}

  @spec kernels() :: [atom()]
  def kernels, do: @kernels

  @spec kernel(term(), keyword()) :: Graph.t()
  def kernel(points, opts) do
    opts =
      Knn.validate!(opts,
        kernel: :gaussian,
        bandwidth: nil,
        k: nil,
        ids: nil,
        taper: true,
        decay: false,
        coincident: :raise,
        seed: 0,
        metric: :euclidean,
        max_edges: Knn.default_max_edges()
      )

    kernel = kernel_name!(Keyword.fetch!(opts, :kernel))
    k = k!(Keyword.fetch!(opts, :k))
    {mode, value} = bandwidth!(Keyword.fetch!(opts, :bandwidth), k)
    taper = boolean!(opts, :taper)
    decay = boolean!(opts, :decay)
    coincident = Knn.coincident!(opts)
    max_edges = Knn.max_edges!(opts)

    {bin, n} = Input.to_points_binary(points)
    ids = Knn.resolve_ids!(Keyword.fetch!(opts, :ids), n)

    ref =
      Knn.unwrap(
        Native.weights_kernel(
          bin,
          Atom.to_string(kernel),
          {mode, value},
          k,
          {taper, decay},
          coincident,
          Knn.metric!(opts),
          max_edges
        )
      )

    Knn.graph(ref, n, ids)
  end

  @doc false
  # Kernel weights over distance-band candidates; `opts` are the validated band options.
  @spec band(binary(), non_neg_integer(), number(), keyword()) :: reference()
  def band(bin, _n, threshold, opts) do
    kernel = kernel_name!(Keyword.fetch!(opts, :kernel))
    bandwidth = band_bandwidth!(Keyword.fetch!(opts, :bandwidth))
    taper = boolean!(opts, :taper)
    decay = boolean!(opts, :decay)

    Knn.unwrap(
      Native.weights_distance_band_kernel(
        bin,
        threshold * 1.0,
        Atom.to_string(kernel),
        bandwidth,
        {taper, decay},
        Knn.metric!(opts),
        Knn.max_edges!(opts)
      )
    )
  end

  @doc false
  @spec kernel_name!(term()) :: atom()
  def kernel_name!(name) when is_atom(name) do
    name = Map.get(@aliases, name, name)

    if name in @kernels do
      name
    else
      raise ArgumentError,
            "unknown kernel #{inspect(name)}; expected one of #{inspect(@kernels)} " <>
              "(aliases :epanechnikov, :uniform)"
    end
  end

  def kernel_name!(name),
    do:
      raise(
        ArgumentError,
        "kernel must be an atom, one of #{inspect(@kernels)}, got #{inspect(name)}"
      )

  defp k!(nil), do: nil
  defp k!(k) when is_integer(k) and k >= 1, do: k
  defp k!(k), do: raise(ArgumentError, "k must be a positive integer, got #{inspect(k)}")

  defp bandwidth!(nil, _k), do: {"default", 0.0}

  defp bandwidth!(:adaptive, nil),
    do: raise(ArgumentError, "bandwidth: :adaptive requires k (the neighbor count)")

  defp bandwidth!(:adaptive, _k), do: {"adaptive", 0.0}

  defp bandwidth!(:auto, _k) do
    raise ArgumentError,
          "bandwidth: :auto (entropy optimisation) is not supported; " <>
            "use a number, :adaptive with k, or nil"
  end

  defp bandwidth!(h, _k) when is_number(h) and h > 0, do: {"fixed", h * 1.0}

  defp bandwidth!(h, _k) do
    raise ArgumentError,
          ":bandwidth must be nil, :adaptive or a positive number, got #{inspect(h)}"
  end

  defp band_bandwidth!(nil), do: nil
  defp band_bandwidth!(h) when is_number(h) and h > 0, do: h * 1.0

  defp band_bandwidth!(h) do
    raise ArgumentError, ":bandwidth must be nil or a positive number, got #{inspect(h)}"
  end

  defp boolean!(opts, key) do
    value = Keyword.fetch!(opts, key)

    unless is_boolean(value) do
      raise ArgumentError, ":#{key} must be true or false, got #{inspect(value)}"
    end

    value
  end
end
