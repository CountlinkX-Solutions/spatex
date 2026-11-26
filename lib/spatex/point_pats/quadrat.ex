defmodule Spatex.PointPats.Quadrat do
  @moduledoc """
  Result of `Spatex.PointPats.quadrat_test/2` (`pointpats` `QStatistic`).

    * `shape` - `:rectangle` or `:hexagon`; `cell_ids` and `counts` list the cells in
      pointpats' order (rows from the lower left); `num_cells` is the real number of cells.
    * `chi2`, `chi2_pvalue`, `df`, `chi2_contrib` - Pearson statistic against a uniform
      expectation, its chi-square p-value, `num_cells - 1` and the per-cell contributions.
    * `permutations`, `seed` - the number of CSR realizations (the `:permutations` option)
      and the seed used.
    * `chi2_realizations`, `chi2_r_pvalue` - simulated statistics and
      `(count + 1) / (permutations + 1)` with `count` the simulated statistics at least `chi2`, `nil` when no realizations were requested.
    * Rectangles: `count_column`, `count_row`, `rectangle_width`, `rectangle_height`.
      Hexagons: `count_column`, `count_row_even`, `count_row_odd`, `h_length`.
    * `mbb` - `{xmin, ymin, xmax, ymax}` of the points.

  pointpats' `HexagonM.num` is not exposed: its formula swaps the even and odd row counts and
  can disagree with the real number of cells (`num_cells` is always right).
  """

  defstruct [
    :shape,
    :cell_ids,
    :counts,
    :num_cells,
    :chi2,
    :chi2_pvalue,
    :df,
    :chi2_contrib,
    :chi2_realizations,
    :chi2_r_pvalue,
    :count_column,
    :count_row,
    :count_row_even,
    :count_row_odd,
    :rectangle_width,
    :rectangle_height,
    :h_length,
    :mbb,
    :permutations,
    :seed
  ]

  @type value :: float() | :nan | :infinity | :neg_infinity

  @type t :: %__MODULE__{
          shape: :rectangle | :hexagon,
          cell_ids: [non_neg_integer()],
          counts: [non_neg_integer()],
          num_cells: pos_integer(),
          chi2: value(),
          chi2_pvalue: value(),
          df: non_neg_integer(),
          chi2_contrib: [value()],
          chi2_realizations: [value()] | nil,
          chi2_r_pvalue: float() | nil,
          count_column: pos_integer(),
          count_row: pos_integer() | nil,
          count_row_even: pos_integer() | nil,
          count_row_odd: non_neg_integer() | nil,
          rectangle_width: float() | nil,
          rectangle_height: float() | nil,
          h_length: float() | nil,
          permutations: non_neg_integer(),
          seed: non_neg_integer(),
          mbb: {float(), float(), float(), float()}
        }
end
