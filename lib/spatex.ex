defmodule Spatex do
  @moduledoc """
  PySAL-style spatial analysis for Elixir, backed by a Rust core.

  Spatex brings the PySAL toolbox to the BEAM: spatial weights (graphs), exploratory
  spatial data analysis, map classification, point patterns, spatial regression, spatial
  dynamics, inequality and regionalization. Algorithms follow PySAL and are verified
  against it numerically.

  Heavy computation runs in a Rust workspace (`spatex_core` for the pure algorithms,
  `spatex_native` for the thin Rustler layer). Numeric vectors cross the NIF boundary as
  native-endian `f64` binaries.

  See the README for what is available and the PySAL parity matrix (`docs/parity.md`) for
  what is not.
  """

  @doc """
  Returns the Spatex version string.

  ## Examples

      iex> Spatex.version() =~ ~r/^\\d+\\.\\d+\\.\\d+/
      true

  """
  @spec version() :: String.t()
  def version, do: Application.spec(:spatex, :vsn) |> to_string()

  @doc false
  @spec native_version() :: non_neg_integer()
  def native_version, do: Spatex.Native.nif_version()

  @doc false
  @spec sum_f64([number()]) :: float()
  def sum_f64(values) when is_list(values) do
    bin = for v <- values, into: <<>>, do: <<v * 1.0::float-64-native>>

    case Spatex.Native.sum_f64(bin) do
      {:ok, sum} -> sum
      {:error, msg} -> raise ArgumentError, msg
    end
  end
end
