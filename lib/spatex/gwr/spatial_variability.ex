defmodule Spatex.GWR.SpatialVariability do
  @moduledoc """
  Result of `Spatex.GWR.spatial_variability/4` (`GWRResults.spatial_variability` in mgwr).

    * `p_values` - one pseudo p-value per coefficient surface (intercept first when the model has
      one): the share of permutations whose standard deviation of the surface exceeds the
      observed one, `count(sd_sim > sd_obs) / permutations` (mgwr's rule, no `+ 1` correction).
      A small value means the surface varies in space more than chance would.
    * `observed_sd` - the standard deviation (`ddof = 0`) of each observed coefficient surface.
    * `simulated_sd` - the standard deviations of the permuted fits, `permutations` rows of `k`
      values (a list of lists, an `{permutations, k}` `Nx.Tensor` with `keep_simulations:
      :tensor`, or `nil` with `keep_simulations: false`).
    * `bw` - the bandwidth selected for the observed data (the permuted fits select their own).
    * `permutations`, `seed` - the number of permutations and the seed that reproduces them.
    * `n`, `k`, `constant` - the model that was tested.
  """

  defstruct [
    :p_values,
    :observed_sd,
    :simulated_sd,
    :bw,
    :permutations,
    :seed,
    :n,
    :k,
    :constant
  ]

  @type t :: %__MODULE__{
          p_values: [float()],
          observed_sd: [float()],
          simulated_sd: [[float()]] | Nx.Tensor.t() | nil,
          bw: pos_integer() | float() | :infinity,
          permutations: pos_integer(),
          seed: non_neg_integer(),
          n: pos_integer(),
          k: pos_integer(),
          constant: boolean()
        }
end
