# Shared helpers for the Benchee scripts. Run a script with `mix run bench/<name>.exs`
# (Benchee is a dev-only dependency). Set RAYON_NUM_THREADS=1 to measure one thread; results are
# then recorded under "<name> [1 thread]".
defmodule Bench do
  @moduledoc false

  @results Path.expand("results/spatex.json", __DIR__)

  # --- data -----------------------------------------------------------------------------------

  @doc "Columbus polygons (49) as WKB binaries and the CRIME/INC/HOVAL columns."
  def columbus do
    path =
      Path.expand("../test/reference/fixtures/datasets/columbus.json", __DIR__)

    data = path |> File.read!() |> Jason.decode!()
    wkbs = Enum.map(data["wkb_hex"], &Base.decode16!(&1, case: :mixed))
    {wkbs, data["attributes"]}
  end

  @doc "WKB of an `rows x cols` lattice of unit square polygons (row-major cell order)."
  def lattice_wkb(rows, cols) do
    for r <- 0..(rows - 1), c <- 0..(cols - 1) do
      {x0, y0, x1, y1} = {c * 1.0, r * 1.0, c + 1.0, r + 1.0}
      ring = [{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}, {x0, y0}]
      pts = for {x, y} <- ring, into: <<>>, do: <<x::little-float-64, y::little-float-64>>
      <<1, 3::little-32, 1::little-32, 5::little-32>> <> pts
    end
  end

  @doc "Row-standardized rook lattice graph (n = side^2), built by Spatex."
  def lattice_graph(side) do
    side
    |> lattice_wkb(side)
    |> Spatex.Graph.build_contiguity(rook: true)
    |> Spatex.Graph.transform(:r)
  end

  @doc "`n` seeded uniform points in the unit square."
  def points(n, seed) do
    :rand.seed(:exsss, {seed, seed + 1, seed + 2})
    for _ <- 1..n, do: {:rand.uniform(), :rand.uniform()}
  end

  @doc "`n` seeded standard-normal values."
  def normal(n, seed) do
    :rand.seed(:exsss, {seed, seed + 1, seed + 2})
    for _ <- 1..n, do: :rand.normal()
  end

  # --- running --------------------------------------------------------------------------------

  @doc """
  Runs one Benchee benchmark and records median/mean (milliseconds) and sample count in
  `bench/results/spatex.json` under `key`.
  """
  def run(key, fun, opts \\ []) do
    suite =
      Benchee.run(
        %{key => fun},
        [warmup: 2, time: 8, memory_time: 0, print: [fast_warning: false]] ++ opts
      )

    stats = hd(suite.scenarios).run_time_data.statistics

    record(key, %{
      median_ms: stats.median / 1.0e6,
      mean_ms: stats.average / 1.0e6,
      min_ms: stats.minimum / 1.0e6,
      samples: stats.sample_size
    })
  end

  defp suffix do
    case System.get_env("RAYON_NUM_THREADS") do
      nil -> ""
      n -> " [#{n} thread#{if n == "1", do: "", else: "s"}]"
    end
  end

  defp record(key, entry) do
    key = key <> suffix()

    existing =
      case File.read(@results) do
        {:ok, text} -> Jason.decode!(text)
        _ -> %{}
      end

    File.mkdir_p!(Path.dirname(@results))
    File.write!(@results, Jason.encode!(Map.put(existing, key, entry), pretty: true))
  end
end
