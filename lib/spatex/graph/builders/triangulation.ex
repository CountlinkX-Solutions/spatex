defmodule Spatex.Graph.Builders.Triangulation do
  @moduledoc false
  # Internal builder behind `Spatex.Graph.build_triangulation/2`.

  alias Spatex.Graph
  alias Spatex.Graph.Builders.Kernel, as: KernelBuilder
  alias Spatex.Graph.Builders.Knn
  alias Spatex.Input
  alias Spatex.Native

  @methods [:delaunay, :gabriel, :relative_neighborhood, :voronoi]
  @coincident [:raise, :clique]

  @spec triangulation(term(), keyword()) :: Graph.t()
  def triangulation(points, opts) do
    opts =
      Knn.validate!(opts,
        method: :delaunay,
        bandwidth: :infinity,
        kernel: :boxcar,
        clip: :bounding_box,
        rook: true,
        taper: true,
        decay: false,
        coincident: :raise,
        ids: nil
      )

    method = method!(Keyword.fetch!(opts, :method))
    kernel = KernelBuilder.kernel_name!(Keyword.fetch!(opts, :kernel))
    bandwidth = bandwidth!(Keyword.fetch!(opts, :bandwidth))
    taper = boolean!(opts, :taper)
    decay = boolean!(opts, :decay)
    coincident = Keyword.fetch!(opts, :coincident)

    unless coincident in @coincident do
      raise ArgumentError,
            ":coincident must be one of #{inspect(@coincident)} (:jitter is not supported), " <>
              "got #{inspect(coincident)}"
    end

    {bin, n} = Input.to_points_binary(points)
    ids = Knn.resolve_ids!(Keyword.fetch!(opts, :ids), n)

    ref =
      Knn.unwrap(
        Native.weights_triangulation(
          bin,
          Atom.to_string(method),
          Atom.to_string(kernel),
          bandwidth,
          taper,
          decay,
          Atom.to_string(coincident),
          clip!(method, opts)
        )
      )

    Knn.graph(ref, n, ids)
  end

  defp method!(method) when method in @methods, do: method

  defp method!(other) do
    raise ArgumentError, ":method must be one of #{inspect(@methods)}, got #{inspect(other)}"
  end

  # The Voronoi clip region; the other methods ignore `:clip` and `:rook`, as libpysal does.
  defp clip!(:voronoi, opts) do
    unless Keyword.fetch!(opts, :rook) == true,
      do: raise(ArgumentError, "method: :voronoi supports only rook: true (shared cell edges)")

    case Keyword.fetch!(opts, :clip) do
      :bounding_box ->
        "bounding_box"

      :convex_hull ->
        "convex_hull"

      nil ->
        "none"

      other ->
        raise ArgumentError,
              ":clip must be :bounding_box, :convex_hull or nil, got #{inspect(other)}"
    end
  end

  defp clip!(_method, _opts), do: "bounding_box"

  # `nil` crosses the NIF boundary as an infinite bandwidth.
  defp bandwidth!(:infinity), do: nil
  defp bandwidth!(h) when is_number(h) and h > 0, do: h * 1.0

  defp bandwidth!(h) do
    raise ArgumentError, ":bandwidth must be :infinity or a positive number, got #{inspect(h)}"
  end

  defp boolean!(opts, key) do
    value = Keyword.fetch!(opts, key)

    unless is_boolean(value) do
      raise ArgumentError, ":#{key} must be true or false, got #{inspect(value)}"
    end

    value
  end
end
