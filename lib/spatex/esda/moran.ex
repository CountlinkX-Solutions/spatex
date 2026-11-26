defmodule Spatex.ESDA.Moran do
  @moduledoc """
  Result of `Spatex.ESDA.moran/3` and `Spatex.ESDA.moran_rate/4`: global Moran's I with
  analytic and permutation inference. Field names are PySAL's `esda.Moran` attributes in
  snake_case (`seI_norm` is `se_i_norm`).

  Simulation fields (`sim`, `p_sim`, `ei_sim`, `se_i_sim`, `vi_sim`, `z_sim`, `p_z_sim`,
  `seed`) are `nil` when `permutations: 0`; `sim` is also `nil` with `keep_simulations: false`.
  Non-finite values (for example `vi_rand` when `n < 4`) are the atoms `:nan`, `:infinity`
  and `:neg_infinity`.

  With `keep_simulations: :tensor`, `sim` is an `Nx.Tensor` of type `{:f, 64}` and shape
  `{permutations}` instead of a list.
  """

  @enforce_keys [:i, :n, :permutations]
  defstruct [
    :i,
    :ei,
    :vi_norm,
    :se_i_norm,
    :z_norm,
    :p_norm,
    :vi_rand,
    :se_i_rand,
    :z_rand,
    :p_rand,
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
          ei: value(),
          vi_norm: value(),
          se_i_norm: value(),
          z_norm: value(),
          p_norm: value(),
          vi_rand: value(),
          se_i_rand: value(),
          z_rand: value(),
          p_rand: value(),
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
