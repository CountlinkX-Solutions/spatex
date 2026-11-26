defmodule Spatex.ESDA.MoranLocal do
  @moduledoc """
  Result of `Spatex.ESDA.moran_local/3` and `Spatex.ESDA.moran_local_bv/4`: local Moran's I
  (LISA, `esda.Moran_Local` / `esda.Moran_Local_BV`).

  Every per-site field is a list aligned with the graph's node order (`ids`); Explorer
  conversion arrives with M5. Field names are PySAL's attributes in snake_case (`Is` is `is`,
  `seI_sim` is `se_i_sim`, `EIc` is `ei_c`).

    * `is` - local statistic, `q` - quadrant (1 High-High, 2 Low-High, 3 Low-Low, 4 High-Low;
      2 and 3 swapped with `geoda_quads: true`), `z` - standardized values (ddof 0; `x` for the
      bivariate statistic, whose standardized `y` is `z_y`), `lag` - spatial lag of `z` (of
      `z_y` for the bivariate statistic).
    * `ei`, `vi`, `se_i` - expectation, variance and standard deviation under total
      randomization; `ei_c`, `vi_c` - conditional randomization moments (Sokal et al. 1998).
      They are `nil` for the bivariate statistic, as esda has no analytic moments for it.
    * `p_sim` - pseudo p-value of the conditional randomization (see `:alternative`), `sim` -
      the simulated statistics, one list of `permutations` values per site (esda's `rlisas`),
      `ei_sim`, `se_i_sim`, `vi_sim`, `z_sim`, `p_z_sim` - their moments and the one-tailed
      normal p-value of `z_sim`. All are `nil` with `permutations: 0`; with
      `keep_simulations: false`, `sim` and the moments are `nil` (as in esda) and only `p_sim`
      remains.

  Islands: the statistic is `0`, and (as in esda) their `p_sim` is the smallest possible
  `1 / (permutations + 1)` and `z_sim` is `:nan`. Non-finite floats are the atoms `:nan`,
  `:infinity` and `:neg_infinity`.

  With `keep_simulations: :tensor`, `sim` is an `Nx.Tensor` of type `{:f, 64}` and shape
  `{n, permutations}` instead of nested lists.
  """

  @enforce_keys [:n, :permutations]
  defstruct [
    :is,
    :q,
    :z,
    :z_y,
    :lag,
    :ei,
    :vi,
    :se_i,
    :ei_c,
    :vi_c,
    :p_sim,
    :sim,
    :ei_sim,
    :se_i_sim,
    :vi_sim,
    :z_sim,
    :p_z_sim,
    :n,
    :permutations,
    :seed,
    :ids,
    geoda_quads: false,
    alternative: :directed
  ]

  @type value :: float() | :nan | :infinity | :neg_infinity

  @type t :: %__MODULE__{
          is: [value()],
          q: [1..4],
          z: [value()],
          z_y: [value()] | nil,
          lag: [value()],
          ei: [value()] | nil,
          vi: [value()] | nil,
          se_i: [value()] | nil,
          ei_c: [value()] | nil,
          vi_c: [value()] | nil,
          p_sim: [float()] | nil,
          sim: [[float()]] | Nx.Tensor.t() | nil,
          ei_sim: [value()] | nil,
          se_i_sim: [value()] | nil,
          vi_sim: [value()] | nil,
          z_sim: [value()] | nil,
          p_z_sim: [value()] | nil,
          n: pos_integer(),
          permutations: non_neg_integer(),
          seed: non_neg_integer() | nil,
          ids: [term()] | nil,
          geoda_quads: boolean(),
          alternative: atom()
        }

  @labels %{1 => "High-High", 2 => "Low-High", 3 => "Low-Low", 4 => "High-Low"}

  @doc """
  LISA cluster labels (`esda`'s `get_cluster_labels`).

  A site is labelled by its quadrant (`"High-High"`, `"Low-High"`, `"Low-Low"`, `"High-Low"`)
  when `p_sim < crit` (strictly), otherwise `"Insignificant"`. The labels follow `q` as
  computed, so with `geoda_quads: true` the names of quadrants 2 and 3 are swapped, exactly as
  in esda. Raises `ArgumentError` for a result computed with `permutations: 0`.

  ## Examples

      iex> m = %Spatex.ESDA.MoranLocal{q: [1, 3, 2], p_sim: [0.01, 0.2, 0.049], n: 3, permutations: 99}
      iex> Spatex.ESDA.MoranLocal.cluster_labels(m)
      ["High-High", "Insignificant", "Low-High"]
      iex> Spatex.ESDA.MoranLocal.cluster_labels(m, 0.5)
      ["High-High", "Low-Low", "Low-High"]
  """
  @spec cluster_labels(t(), number()) :: [String.t()]
  def cluster_labels(result, crit \\ 0.05)

  def cluster_labels(%__MODULE__{p_sim: nil}, _crit) do
    raise ArgumentError, "cluster labels need p_sim: compute the statistic with permutations > 0"
  end

  def cluster_labels(%__MODULE__{q: q, p_sim: p_sim}, crit) when is_number(crit) do
    Enum.zip_with(q, p_sim, fn quadrant, p ->
      if is_number(p) and p < crit, do: Map.fetch!(@labels, quadrant), else: "Insignificant"
    end)
  end

  def cluster_labels(_result, crit) do
    raise ArgumentError, "crit must be a number, got #{inspect(crit)}"
  end
end
