defmodule Spatex.Region.AZP do
  @moduledoc """
  Result of `Spatex.Region.azp/3`.

    * `labels` - region of each area, numbered by first appearance.
    * `n_regions` - number of regions.
    * `objective` - final pairwise objective (`metric` of the call).
    * `initial_labels` - the starting solution (given or generated).
    * `iterations` - number of accepted moves.
    * `seed` - the seed used.
    * `method` - the local search (`:basic`, `:tabu` or `:annealing`).
  """

  defstruct labels: [],
            n_regions: 0,
            objective: 0.0,
            initial_labels: [],
            iterations: 0,
            seed: 0,
            method: :basic

  @type t :: %__MODULE__{
          labels: [non_neg_integer()],
          n_regions: pos_integer(),
          objective: float(),
          initial_labels: [non_neg_integer()],
          iterations: non_neg_integer(),
          seed: non_neg_integer(),
          method: :basic | :tabu | :annealing
        }
end
