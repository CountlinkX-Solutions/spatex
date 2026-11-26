defmodule Spatex.ESDA.JoinCounts do
  @moduledoc """
  Result of `Spatex.ESDA.join_counts/3`: binary join counts with a Yates-corrected chi-square
  test and permutation inference. Field names follow PySAL's `esda.Join_Counts`.

    * `j` - number of joins; `bb`, `ww`, `bw` - black-black, white-white and black-white joins.
    * `crosstab` / `expected` - the observed and expected 2x2 tables `[[ww, wb], [bw, bb]]`
      (focal rows, neighbor columns); entries are halves of directed counts.
    * `chi2`, `chi2_p`, `chi2_dof` - `scipy.stats.chi2_contingency` with its default Yates
      correction.
    * `autocorr_pos = bb + ww`, `autocorr_neg = bw`.

  Permutation fields are `nil` when `permutations: 0`, and the `sim_*` vectors are also `nil`
  with `keep_simulations: false`. `sim_autocurr_pos` and `sim_autocurr_neg` keep PySAL's
  spelling. The `p_sim_*` values are upper-tail, not folded: `(count(sim >= observed) + 1) / (P + 1)`.

  With `keep_simulations: :tensor`, each `sim_*` vector is an `Nx.Tensor` of type `{:f, 64}` and
  shape `{permutations}` instead of a list.

  `sim_autocorr_chi2` deviates from PySAL on purpose: esda divides both squared deviations
  by the squared mean simulated positive autocorrelation (a bug); Spatex reports the survival
  function (one degree of freedom) of the Pearson statistic
  `(pos - mean_pos)^2 / mean_pos + (neg - mean_neg)^2 / mean_neg`.
  """

  @enforce_keys [:j, :n, :permutations]
  defstruct [
    :j,
    :bb,
    :ww,
    :bw,
    :chi2,
    :chi2_p,
    :chi2_dof,
    :crosstab,
    :expected,
    :autocorr_pos,
    :autocorr_neg,
    :sim_bb,
    :sim_bw,
    :sim_chi2,
    :sim_autocurr_pos,
    :sim_autocurr_neg,
    :min_bb,
    :mean_bb,
    :max_bb,
    :min_bw,
    :mean_bw,
    :max_bw,
    :p_sim_bb,
    :p_sim_bw,
    :p_sim_chi2,
    :p_sim_autocorr_pos,
    :p_sim_autocorr_neg,
    :sim_autocorr_chi2,
    :n,
    :permutations,
    :seed
  ]

  @type value :: float() | :nan | :infinity | :neg_infinity
  @type table :: [[float()]]

  @type t :: %__MODULE__{
          j: float(),
          bb: float(),
          ww: float(),
          bw: float(),
          chi2: value(),
          chi2_p: value(),
          chi2_dof: float(),
          crosstab: table(),
          expected: table(),
          autocorr_pos: float(),
          autocorr_neg: float(),
          sim_bb: [float()] | Nx.Tensor.t() | nil,
          sim_bw: [float()] | Nx.Tensor.t() | nil,
          sim_chi2: [float()] | Nx.Tensor.t() | nil,
          sim_autocurr_pos: [float()] | Nx.Tensor.t() | nil,
          sim_autocurr_neg: [float()] | Nx.Tensor.t() | nil,
          min_bb: float() | nil,
          mean_bb: float() | nil,
          max_bb: float() | nil,
          min_bw: float() | nil,
          mean_bw: float() | nil,
          max_bw: float() | nil,
          p_sim_bb: float() | nil,
          p_sim_bw: float() | nil,
          p_sim_chi2: float() | nil,
          p_sim_autocorr_pos: float() | nil,
          p_sim_autocorr_neg: float() | nil,
          sim_autocorr_chi2: value() | nil,
          n: pos_integer(),
          permutations: non_neg_integer(),
          seed: non_neg_integer() | nil
        }
end
