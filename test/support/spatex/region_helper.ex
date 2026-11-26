defmodule Spatex.RegionHelper do
  @moduledoc false
  # Test-only helpers shared by the regionalization tests.

  alias Spatex.Graph

  @doc "Builds a graph from a fixture block with `n`, `focal` and `neighbor` keys."
  def graph(%{"n" => n, "focal" => f, "neighbor" => nb}) do
    Graph.from_arrays(f, nb, List.duplicate(1.0, length(f)), ids: Enum.to_list(0..(n - 1)))
  end

  @doc "A rook lattice graph with `rows * cols` nodes in row-major order."
  def lattice(rows, cols) do
    edges =
      for r <- 0..(rows - 1),
          c <- 0..(cols - 1),
          {dr, dc} <- [{0, 1}, {1, 0}, {0, -1}, {-1, 0}],
          r2 = r + dr,
          c2 = c + dc,
          r2 in 0..(rows - 1)//1,
          c2 in 0..(cols - 1)//1,
          do: {r * cols + c, r2 * cols + c2}

    {f, nb} = Enum.unzip(edges)

    Graph.from_arrays(f, nb, List.duplicate(1.0, length(f)),
      ids: Enum.to_list(0..(rows * cols - 1))
    )
  end

  @doc "Regions as a sorted list of sorted member lists."
  def regions(labels) do
    labels
    |> Enum.with_index()
    |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
    |> Map.values()
    |> Enum.map(&Enum.sort/1)
    |> Enum.sort()
  end

  @doc "Sum of pairwise L1 distances within regions for one-dimensional values."
  def pairwise_l1(x, labels) do
    labels
    |> Enum.zip(x)
    |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
    |> Map.values()
    |> Enum.map(fn vals ->
      for {a, i} <- Enum.with_index(vals), {b, j} <- Enum.with_index(vals), i < j, do: abs(a - b)
    end)
    |> List.flatten()
    |> Enum.sum()
  end

  @doc "Adjusted Rand index between two labelings."
  def ari(a, b) do
    n = length(a)
    pairs = fn m -> m * (m - 1) / 2 end
    cont = a |> Enum.zip(b) |> Enum.frequencies() |> Map.values()
    sum_ij = cont |> Enum.map(pairs) |> Enum.sum()
    sum_a = a |> Enum.frequencies() |> Map.values() |> Enum.map(pairs) |> Enum.sum()
    sum_b = b |> Enum.frequencies() |> Map.values() |> Enum.map(pairs) |> Enum.sum()
    expected = sum_a * sum_b / pairs.(n)
    max = (sum_a + sum_b) / 2

    if max == expected, do: 1.0, else: (sum_ij - expected) / (max - expected)
  end
end
