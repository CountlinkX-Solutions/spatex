defmodule Spatex.ESDA.Geary do
  @moduledoc """
  Result of `Spatex.ESDA.geary/3`: global Geary's C with analytic and permutation inference.
  Field names are PySAL's `esda.Geary` attributes in snake_case (`seC_norm` is `se_c_norm`).

  Inference is one-tailed, as in esda. Simulation fields (`sim`, `p_sim`, `ec_sim`,
  `se_c_sim`, `vc_sim`, `z_sim`, `p_z_sim`, `seed`) are `nil` when `permutations: 0`; `sim` is
  also `nil` with `keep_simulations: false`. Non-finite values (for example `vc_rand` when
  `n < 4`) are the atoms `:nan`, `:infinity` and `:neg_infinity`.

  With `keep_simulations: :tensor`, `sim` is an `Nx.Tensor` of type `{:f, 64}` and shape
  `{permutations}` instead of a list.
  """

  @enforce_keys [:c, :n, :permutations]
  defstruct [
    :c,
    :ec,
    :vc_norm,
    :se_c_norm,
    :z_norm,
    :p_norm,
    :vc_rand,
    :se_c_rand,
    :z_rand,
    :p_rand,
    :sim,
    :ec_sim,
    :vc_sim,
    :se_c_sim,
    :z_sim,
    :p_sim,
    :p_z_sim,
    :n,
    :permutations,
    :seed
  ]

  @type value :: float() | :nan | :infinity | :neg_infinity

  @type t :: %__MODULE__{
          c: value(),
          ec: value(),
          vc_norm: value(),
          se_c_norm: value(),
          z_norm: value(),
          p_norm: value(),
          vc_rand: value(),
          se_c_rand: value(),
          z_rand: value(),
          p_rand: value(),
          sim: [float()] | Nx.Tensor.t() | nil,
          ec_sim: value() | nil,
          vc_sim: value() | nil,
          se_c_sim: value() | nil,
          z_sim: value() | nil,
          p_sim: float() | nil,
          p_z_sim: value() | nil,
          n: pos_integer(),
          permutations: non_neg_integer(),
          seed: non_neg_integer() | nil
        }
end
