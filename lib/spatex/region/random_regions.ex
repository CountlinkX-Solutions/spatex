defmodule Spatex.Region.RandomRegions do
  @moduledoc """
  Result of `Spatex.Region.random_regions/2`.

    * `solutions` - one `Spatex.Region.RandomRegion` per permutation (spopt's `solutions`).
    * `feasible_solutions` - the feasible ones (spopt's `solutions_feas`).
    * `seed` - the seed used; permutation `i` draws from ChaCha8 stream `i` of it.
  """

  defstruct solutions: [], feasible_solutions: [], seed: 0

  @type t :: %__MODULE__{
          solutions: [Spatex.Region.RandomRegion.t()],
          feasible_solutions: [Spatex.Region.RandomRegion.t()],
          seed: non_neg_integer()
        }
end
