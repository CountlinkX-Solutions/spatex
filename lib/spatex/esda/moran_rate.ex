defmodule Spatex.ESDA.MoranRate do
  @moduledoc """
  Result of `Spatex.ESDA.moran_rate/4`: Moran's I of event rates (`esda.Moran_Rate`).

  Carries every field of `Spatex.ESDA.Moran` plus the rates it was computed on (`y`, the
  empirical-Bayes standardized rates when `adjusted` is true) and the `adjusted` flag.

  With `keep_simulations: :tensor`, `sim` is an `Nx.Tensor` of type `{:f, 64}` and shape
  `{permutations}` instead of a list.
  """

  @enforce_keys [:i, :n, :permutations, :adjusted, :y]
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
    :seed,
    :adjusted,
    :y
  ]

  @type t :: %__MODULE__{
          i: Spatex.ESDA.Moran.value(),
          ei: Spatex.ESDA.Moran.value(),
          vi_norm: Spatex.ESDA.Moran.value(),
          se_i_norm: Spatex.ESDA.Moran.value(),
          z_norm: Spatex.ESDA.Moran.value(),
          p_norm: Spatex.ESDA.Moran.value(),
          vi_rand: Spatex.ESDA.Moran.value(),
          se_i_rand: Spatex.ESDA.Moran.value(),
          z_rand: Spatex.ESDA.Moran.value(),
          p_rand: Spatex.ESDA.Moran.value(),
          sim: [float()] | Nx.Tensor.t() | nil,
          ei_sim: Spatex.ESDA.Moran.value() | nil,
          vi_sim: Spatex.ESDA.Moran.value() | nil,
          se_i_sim: Spatex.ESDA.Moran.value() | nil,
          z_sim: Spatex.ESDA.Moran.value() | nil,
          p_sim: float() | nil,
          p_z_sim: Spatex.ESDA.Moran.value() | nil,
          n: pos_integer(),
          permutations: non_neg_integer(),
          seed: non_neg_integer() | nil,
          adjusted: boolean(),
          y: [float()]
        }
end
