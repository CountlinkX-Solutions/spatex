defmodule Spatex.GWR.Variability do
  @moduledoc false
  # Implementation of `Spatex.GWR.spatial_variability/4`: mgwr's Monte Carlo test. The
  # coordinates are re-ordered among the observations (seeded ChaCha8 permutations from the
  # native `perm` engine), the bandwidth is searched again and the model refitted; the standard
  # deviations of the permuted surfaces are compared with the observed ones.

  alias Spatex.GWR
  alias Spatex.GWR.{Design, SpatialVariability}
  alias Spatex.{Input, Native}

  @own [:permutations, :seed, :keep_simulations]
  @fit_keys [
    :kernel,
    :fixed,
    :constant,
    :spherical,
    :max_n,
    :family,
    :offset,
    :iwls_tol,
    :iwls_max_iter
  ]
  @default_max_n 2_000
  @max_seed 18_446_744_073_709_551_615

  def run(points, y, x, opts) do
    Design.check_known!(opts, @own ++ GWR.select_option_names())
    {own, rest} = Keyword.split(opts, @own)
    perms = permutations!(Keyword.get(own, :permutations, 1000))
    seed = seed!(Keyword.get(own, :seed))
    keep = Input.keep_simulations!(Keyword.get(own, :keep_simulations, true))
    rest = Keyword.put_new(rest, :max_n, @default_max_n)

    selected = GWR.select_bandwidth(points, y, x, rest)
    fit_opts = Keyword.take(rest, @fit_keys)
    observed = surfaces(points, y, x, fit_opts, selected.bw)
    observed_sd = deviations(observed)

    {coords, n} = Input.to_points_binary(points)
    located = List.to_tuple(for <<a::float-64-native, b::float-64-native <- coords>>, do: {a, b})

    sims =
      for r <- 0..(perms - 1) do
        permuted = permute(located, n, seed, r)
        bw = GWR.select_bandwidth(permuted, y, x, rest).bw
        permuted |> surfaces(y, x, fit_opts, bw) |> deviations()
      end

    %SpatialVariability{
      p_values: p_values(sims, observed_sd, perms),
      observed_sd: observed_sd,
      simulated_sd: if(keep, do: simulations(sims, keep, perms)),
      bw: selected.bw,
      permutations: perms,
      seed: seed,
      n: n,
      k: length(observed_sd),
      constant: Keyword.get(rest, :constant, true)
    }
  end

  defp surfaces(points, y, x, fit_opts, bw) do
    GWR.fit(points, y, x, [bw: bw] ++ fit_opts).params
  end

  defp permute(located, n, seed, r) do
    for v <- n |> Native.gwr_permutation(seed, r) |> Input.from_f64_binary(),
        do: elem(located, trunc(v))
  end

  # `np.std(params, axis=0)` (population standard deviation) of each column.
  defp deviations(params) do
    n = length(params)

    params
    |> Enum.zip_with(& &1)
    |> Enum.map(fn column ->
      mean = Enum.sum(column) / n
      :math.sqrt(Enum.sum(for v <- column, do: (v - mean) * (v - mean)) / n)
    end)
  end

  defp p_values(sims, observed_sd, perms) do
    sims
    |> Enum.zip_with(& &1)
    |> Enum.zip(observed_sd)
    |> Enum.map(fn {column, obs} -> Enum.count(column, &(&1 > obs)) / perms end)
  end

  defp simulations(sims, keep, perms) do
    k = length(hd(sims))
    bin = IO.iodata_to_binary(for row <- sims, v <- row, do: <<v::float-64-native>>)
    Input.simulations(bin, keep, {perms, k})
  end

  defp permutations!(p) when is_integer(p) and p >= 1, do: p

  defp permutations!(p),
    do: raise(ArgumentError, ":permutations must be a positive integer, got #{inspect(p)}")

  defp seed!(nil), do: :rand.uniform(4_294_967_296) - 1
  defp seed!(seed) when is_integer(seed) and seed in 0..@max_seed//1, do: seed

  defp seed!(seed) do
    raise ArgumentError, ":seed must be an integer in 0..2^64-1 or nil, got #{inspect(seed)}"
  end
end
