defmodule Spatex.ESDA.G do
  @moduledoc """
  Result of `Spatex.ESDA.getis_ord_g/3`: the Getis-Ord general G with analytic and permutation
  inference. Field names are PySAL's `esda.G` attributes in snake_case (`seG_sim` is
  `se_g_sim`); `den_sum` is `(sum y)^2 - sum y^2`.

  With `keep_simulations: :tensor`, `sim` is an `Nx.Tensor` of type `{:f, 64}` and shape
  `{permutations}` instead of a list.

  `z_norm` and `p_norm` are `:nan` when the variance `vg` is not positive (a complete graph).
  Simulation fields (`sim`, `p_sim`, `eg_sim`, `se_g_sim`, `vg_sim`, `z_sim`, `p_z_sim`,
  `seed`) are `nil` when `permutations: 0`; `sim` is also `nil` with `keep_simulations: false`.
  Non-finite values are the atoms `:nan`, `:infinity` and `:neg_infinity`.
  """

  @enforce_keys [:g, :n, :permutations]
  defstruct [
    :g,
    :eg,
    :vg,
    :se_g,
    :z_norm,
    :p_norm,
    :den_sum,
    :sim,
    :eg_sim,
    :vg_sim,
    :se_g_sim,
    :z_sim,
    :p_sim,
    :p_z_sim,
    :n,
    :permutations,
    :seed
  ]

  @type value :: float() | :nan | :infinity | :neg_infinity

  @type t :: %__MODULE__{
          g: value(),
          eg: value(),
          vg: value(),
          se_g: value(),
          z_norm: value(),
          p_norm: value(),
          den_sum: value(),
          sim: [float()] | Nx.Tensor.t() | nil,
          eg_sim: value() | nil,
          vg_sim: value() | nil,
          se_g_sim: value() | nil,
          z_sim: value() | nil,
          p_sim: float() | nil,
          p_z_sim: value() | nil,
          n: pos_integer(),
          permutations: non_neg_integer(),
          seed: non_neg_integer() | nil
        }
end
