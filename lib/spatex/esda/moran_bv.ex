defmodule Spatex.ESDA.MoranBV do
  @moduledoc """
  Result of `Spatex.ESDA.moran_bv/4`: bivariate Moran's I (`esda.Moran_BV`).

  PySAL's bivariate statistic has no analytic moments. Simulation fields are `nil` when
  `permutations: 0`, `sim` also with `keep_simulations: false`.

  With `keep_simulations: :tensor`, `sim` is an `Nx.Tensor` of type `{:f, 64}` and shape
  `{permutations}` instead of a list.
  """

  @enforce_keys [:i, :n, :permutations]
  defstruct [
    :i,
    :sim,
    :ei_sim,
    :vi_sim,
    :se_i_sim,
    :z_sim,
    :p_sim,
    :p_z_sim,
    :n,
    :permutations,
    :seed
  ]

  @type value :: float() | :nan | :infinity | :neg_infinity

  @type t :: %__MODULE__{
          i: value(),
          sim: [float()] | Nx.Tensor.t() | nil,
          ei_sim: value() | nil,
          vi_sim: value() | nil,
          se_i_sim: value() | nil,
          z_sim: value() | nil,
          p_sim: float() | nil,
          p_z_sim: value() | nil,
          n: pos_integer(),
          permutations: non_neg_integer(),
          seed: non_neg_integer() | nil
        }
end
