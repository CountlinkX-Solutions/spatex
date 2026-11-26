defmodule Spatex.Inequality.SpatialGini do
  @moduledoc """
  Result of `Spatex.Inequality.spatial_gini/3` (`inequality.gini.Gini_Spatial`).

    * `g` - Gini coefficient; `dtotal` - the sum of absolute differences over all ordered pairs
      (`g * den`, `den = 2 n^2 mean`).
    * `wg` - the part of `dtotal` between neighbors, `wcg = dtotal - wg` between non-neighbors,
      `wcg_share = wcg / den`.
    * `polarization` - `(wcg / wg) (neighbor pairs / non-neighbor pairs)`, expected 1 when
      inequality is spatially random.
    * With permutations: `p_sim` (folded pseudo p-value of `wcg`), `e_wcg`, `s_wcg`, `z_wcg`,
      `p_z_sim = 1 - Phi(z_wcg)`, `polarization_p_sim`, and (with `keep_simulations: true`) the
      simulated values `wcgp` and `polarization_sim`. All `nil` otherwise. With
      `keep_simulations: :tensor` the two simulated vectors are `{permutations}` `Nx.Tensor`s of
      type `{:f, 64}` instead of lists.
  """
  defstruct [
    :g,
    :wg,
    :wcg,
    :wcg_share,
    :dtotal,
    :den,
    :polarization,
    :p_sim,
    :e_wcg,
    :s_wcg,
    :z_wcg,
    :p_z_sim,
    :polarization_p_sim,
    :wcgp,
    :polarization_sim,
    :seed,
    permutations: 0
  ]

  @type value :: float() | :nan | :infinity | :neg_infinity

  @type t :: %__MODULE__{
          g: value(),
          wg: value(),
          wcg: value(),
          wcg_share: value(),
          dtotal: value(),
          den: value(),
          polarization: value(),
          p_sim: value() | nil,
          e_wcg: value() | nil,
          s_wcg: value() | nil,
          z_wcg: value() | nil,
          p_z_sim: value() | nil,
          polarization_p_sim: value() | nil,
          wcgp: [float()] | Nx.Tensor.t() | nil,
          polarization_sim: [value()] | Nx.Tensor.t() | nil,
          seed: non_neg_integer() | nil,
          permutations: non_neg_integer()
        }
end
