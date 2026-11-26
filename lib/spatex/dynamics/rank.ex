defmodule Spatex.Dynamics.Tau do
  @moduledoc """
  Result of `Spatex.Dynamics.kendall_tau/2` (`giddy.rank.Tau`): Kendall's tau-b with its
  asymptotic two-sided p-value `p` (`tau_p` in giddy) and the pair counts it is built from
  (`extra_x`: pairs tied in `y` only, `extra_y`: pairs tied in `x` only).
  """
  defstruct [:tau, :p, :concordant, :discordant, :extra_x, :extra_y]

  @type t :: %__MODULE__{
          tau: float() | :nan,
          p: float() | :nan,
          concordant: float(),
          discordant: float(),
          extra_x: float(),
          extra_y: float()
        }
end

defmodule Spatex.Dynamics.SpatialTau do
  @moduledoc """
  Result of `Spatex.Dynamics.spatial_tau/4` (`giddy.rank.SpatialTau`).

  `tau`, `tau_p`, `concordant`, `discordant`, `extra_x` and `extra_y` are those of the classic
  statistic (see `Spatex.Dynamics.Tau`); `tau_spatial`, `concordant_spatial`,
  `discordant_spatial` and `pairs_spatial` restrict it to neighbor pairs. With permutations,
  `taus` holds the statistic under random relabelling of locations and `tau_spatial_psim` the
  one-sided folded pseudo p-value; both are `nil` otherwise.
  """
  defstruct [
    :tau,
    :tau_p,
    :concordant,
    :discordant,
    :extra_x,
    :extra_y,
    :tau_spatial,
    :pairs_spatial,
    :concordant_spatial,
    :discordant_spatial,
    :taus,
    :tau_spatial_psim,
    :seed,
    permutations: 0
  ]

  @type t :: %__MODULE__{
          tau: float() | :nan,
          tau_p: float() | :nan,
          concordant: float(),
          discordant: float(),
          extra_x: float(),
          extra_y: float(),
          tau_spatial: float() | :nan,
          pairs_spatial: non_neg_integer(),
          concordant_spatial: float(),
          discordant_spatial: float(),
          taus: [float() | :nan] | nil,
          tau_spatial_psim: float() | nil,
          seed: non_neg_integer() | nil,
          permutations: non_neg_integer()
        }
end

defmodule Spatex.Dynamics.Theta do
  @moduledoc """
  Result of `Spatex.Dynamics.theta/3` (`giddy.rank.Theta`): the share of rank changes that move
  in the same direction within regimes, per interval (`t - 1` values).

  `ranks` are the per-period ranks (average ties), `total` the absolute rank changes per
  interval, `max_total` their theoretical maximum. `pvalue_left`, `pvalue_right` and `z` come from
  permuting the regimes and are `nil` with `permutations: 0`.
  """
  defstruct [
    :theta,
    :total,
    :max_total,
    :ranks,
    :regimes,
    :pvalue_left,
    :pvalue_right,
    :z,
    :seed,
    permutations: 0
  ]

  @type t :: %__MODULE__{
          theta: [float() | :nan],
          total: [float()],
          max_total: float(),
          ranks: [[float()]],
          regimes: [term()],
          pvalue_left: [float()] | nil,
          pvalue_right: [float()] | nil,
          z: [float() | :nan] | nil,
          seed: non_neg_integer() | nil,
          permutations: non_neg_integer()
        }
end

defmodule Spatex.Dynamics.TauLocal do
  @moduledoc """
  Result of `Spatex.Dynamics.tau_local/2` (`giddy.rank.Tau_Local`): `tau` is the share of
  concordant minus discordant ordered pairs, `sum S / (n (n - 1))`, and `tau_local` the row
  means `sum_j S_ij / (n - 1)` of the concordance `S_ij = sign((x_i - x_j)(y_i - y_j))`.
  """
  @enforce_keys [:n]
  defstruct [:tau, :tau_local, :n]
  @type t :: %__MODULE__{tau: float(), tau_local: [float()], n: pos_integer()}
end

defmodule Spatex.Dynamics.TauLocalNeighbor do
  @moduledoc """
  Result of `Spatex.Dynamics.tau_local_neighbor/4` (`giddy.rank.Tau_Local_Neighbor`).

    * `tau_ln` - mean concordance of every site with its neighbors (`:nan` for islands);
      `sign` its sign (`1` or `-1`, `1` for zero and islands, as in giddy).
    * `weights` - `|N(i)| / s0`, the site's share of the (binary) joins.
    * `p_sim` - `(min(#sim >= obs, #sim <= obs) + 1) / (P + 1)` and `sim` the `n x P`
      simulations; `nil` without permutations (`sim` also with `keep_simulations: false`).
      With `keep_simulations: :tensor`, `sim` is an `Nx.Tensor` of type `{:f, 64}` and shape
      `{n, permutations}` instead of nested lists.
  """
  @enforce_keys [:n, :permutations]
  defstruct [:tau_ln, :weights, :sign, :p_sim, :sim, :n, :permutations, :seed, :ids]

  @type t :: %__MODULE__{
          tau_ln: [float() | :nan],
          weights: [float()],
          sign: [integer()],
          p_sim: [float() | :nan] | nil,
          sim: [[float() | :nan]] | Nx.Tensor.t() | nil,
          n: pos_integer(),
          permutations: non_neg_integer(),
          seed: non_neg_integer() | nil,
          ids: [term()]
        }
end

defmodule Spatex.Dynamics.TauLocalNeighborhood do
  @moduledoc """
  Result of `Spatex.Dynamics.tau_local_neighborhood/4` (`giddy.rank.Tau_Local_Neighborhood`).

    * `tau_lnhood` - the tau of every neighborhood `{i} + N(i)` (`:nan` for islands); `sign`
      its sign.
    * `p_sim`, `sim` - as in `Spatex.Dynamics.TauLocalNeighbor`.
  """
  @enforce_keys [:n, :permutations]
  defstruct [:tau_lnhood, :sign, :p_sim, :sim, :n, :permutations, :seed, :ids]

  @type t :: %__MODULE__{
          tau_lnhood: [float() | :nan],
          sign: [integer()],
          p_sim: [float() | :nan] | nil,
          sim: [[float() | :nan]] | Nx.Tensor.t() | nil,
          n: pos_integer(),
          permutations: non_neg_integer(),
          seed: non_neg_integer() | nil,
          ids: [term()]
        }
end

defmodule Spatex.Dynamics.TauRegional do
  @moduledoc """
  Result of `Spatex.Dynamics.tau_regional/4` (`giddy.rank.Tau_Regional`).

    * `regimes` - the distinct regime labels in ascending order; `tau_reg` the `k x k` matrix of
      mean concordance between (row) and within (diagonal) regimes, `:nan` for a regime with
      one member on the diagonal.
    * `p_sim` - `k x k` pseudo p-values (`nil` without permutations); `sim` - the `P` simulated
      `k x k` matrices (with `keep_simulations: true`). With `keep_simulations: :tensor`, `sim`
      is one `Nx.Tensor` of type `{:f, 64}` and shape `{permutations, k, k}`.
  """
  @enforce_keys [:regimes, :permutations]
  defstruct [:regimes, :tau_reg, :p_sim, :sim, :permutations, :seed]
  @type matrix :: [[float() | :nan]]
  @type t :: %__MODULE__{
          regimes: [term()],
          tau_reg: matrix(),
          p_sim: matrix() | nil,
          sim: [matrix()] | Nx.Tensor.t() | nil,
          permutations: non_neg_integer(),
          seed: non_neg_integer() | nil
        }
end
