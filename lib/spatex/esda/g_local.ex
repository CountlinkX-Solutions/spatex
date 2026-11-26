defmodule Spatex.ESDA.GLocal do
  @moduledoc """
  Result of `Spatex.ESDA.g_local/3`: the local Getis-Ord statistics `G_i` and `G*_i`
  (`esda.G_Local`).

  Every per-site field is a list aligned with the graph's node order (`ids`). Field names are
  esda's attributes in snake_case (`Gs` is `gs`, `EGs` is `egs`, `seG_sim` is `se_g_sim`);
  `sim` is esda's `rGs`, one list of `permutations` values per site.

  With `keep_simulations: :tensor`, `sim` is an `Nx.Tensor` of type `{:f, 64}` and shape
  `{n, permutations}` instead of nested lists.

    * `gs` - local statistic. `G_i` excludes the focal value (`star: false`), `G*_i` includes it.
    * `egs`, `vgs`, `zs`, `p_norm` - expectation, variance and z-score under the normality
      assumption and its one-tailed p-value `sf(|zs|)`. Islands have `zs` and `p_norm` of
      `:nan`.
    * `p_sim` - pseudo p-value of the conditional randomization (see the `:alternative` option
      of `Spatex.ESDA.g_local/3`), `sim` - the simulated statistics, `eg_sim`, `se_g_sim`,
      `vg_sim`, `z_sim`, `p_z_sim` - their moments and the one-tailed normal p-value of
      `z_sim`. All are `nil` with `permutations: 0`; with `keep_simulations: false`, `sim` and
      the moments are `nil` (as in esda) and only `p_sim` remains.
    * `star` - whether `G*` was computed (the prepared weights have a positive diagonal),
      `star_arg` - the `:star` option as given, `w_transform` - `:r` or `:b`, `y_sum` - `sum(y)`.

  Islands: the statistic is `0`, and (as in esda) their `p_sim` is the smallest possible
  `1 / (permutations + 1)`. esda's `G_Local` has no cluster labels, so there is no
  `hot_cold` helper; combine `p_sim` with `zs` (positive: hot spot, negative: cold spot) and,
  to control for multiple testing, `Spatex.ESDA.fdr/2`.

  Non-finite floats are the atoms `:nan`, `:infinity` and `:neg_infinity`.
  """

  @enforce_keys [:n, :permutations]
  defstruct [
    :gs,
    :egs,
    :vgs,
    :zs,
    :p_norm,
    :p_sim,
    :sim,
    :eg_sim,
    :se_g_sim,
    :vg_sim,
    :z_sim,
    :p_z_sim,
    :y_sum,
    :n,
    :permutations,
    :seed,
    :ids,
    star: false,
    star_arg: false,
    w_transform: :r,
    alternative: :directed,
    island_weight: 0.0
  ]

  @type value :: float() | :nan | :infinity | :neg_infinity

  @type t :: %__MODULE__{
          gs: [value()],
          egs: [value()],
          vgs: [value()],
          zs: [value()],
          p_norm: [value()],
          p_sim: [float()] | nil,
          sim: [[float()]] | Nx.Tensor.t() | nil,
          eg_sim: [value()] | nil,
          se_g_sim: [value()] | nil,
          vg_sim: [value()] | nil,
          z_sim: [value()] | nil,
          p_z_sim: [value()] | nil,
          y_sum: float(),
          n: pos_integer(),
          permutations: non_neg_integer(),
          seed: non_neg_integer() | nil,
          ids: [term()] | nil,
          star: boolean(),
          star_arg: boolean() | nil | number(),
          w_transform: :r | :b,
          alternative: atom(),
          island_weight: number()
        }
end
