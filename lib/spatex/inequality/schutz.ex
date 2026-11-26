defmodule Spatex.Inequality.Schutz do
  @moduledoc """
  Result of `Spatex.Inequality.schutz/1` (`inequality.schutz.Schutz`).

    * `distance` - the largest gap between the line of perfect equality and the Lorenz curve (the
      Schutz, or Robin Hood, index as a share).
    * `intersection_point` - the cumulative population share at which that gap is first reached.
    * `coefficient` - the original Schutz coefficient, `sum 10 (share_i / population_share - 1)`
      over the observations above the mean (the factor 10 is the Python package's).
  """
  @enforce_keys [:distance, :intersection_point, :coefficient]
  defstruct [:distance, :intersection_point, :coefficient]

  @type t :: %__MODULE__{
          distance: float(),
          intersection_point: float(),
          coefficient: float()
        }
end
