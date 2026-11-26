defmodule Spatex.Inequality.Atkinson do
  @moduledoc """
  Result of `Spatex.Inequality.atkinson/2` (`inequality.atkinson.Atkinson`).

    * `a` - the Atkinson index, 0 for perfect equality up to 1 for maximum inequality.
    * `ede` - the equally distributed equivalent: the income that, shared equally, gives the same
      social welfare as the actual distribution (`mean * (1 - a)`).
    * `epsilon` - the inequality aversion the index was computed with.
  """
  @enforce_keys [:a, :ede, :epsilon]
  defstruct [:a, :ede, :epsilon]

  @type t :: %__MODULE__{a: float(), ede: float(), epsilon: number()}
end
