defmodule Spatex.Explorer do
  @moduledoc """
  Per-site results as `Explorer.DataFrame`s.

  `to_dataframe/2` turns a local ESDA result (`Spatex.ESDA.MoranLocal`,
  `Spatex.ESDA.GLocal`, and any other result struct whose per-site fields are lists with one
  entry per site), a `Spatex.Classify` result or a `Spatex.Region` result into a dataframe
  with one row per site.
  Explorer is an optional dependency: it is only touched when this module is called, through
  runtime dispatch, so Spatex compiles and runs without it.

  ## Columns

  Every field that is a list with one scalar per site becomes a column named after the field
  (`is`, `q`, `z`, `lag`, `p_sim`, `gs`, `zs`, ...). Fields that are `nil` (for example
  `p_sim` with `permutations: 0`), the simulation matrices (`sim`) and non-per-site fields are
  skipped. In addition:

    * `id` - the graph node ids, first column, when the result carries them (ids that are not
      integers or strings are converted with `to_string/1`/`inspect/1`).
    * `cluster` - for `Spatex.ESDA.MoranLocal` with a `p_sim`, the quadrant label from
      `Spatex.ESDA.MoranLocal.cluster_labels/2`.
    * `yb` - for `Spatex.Classify`, the class of each observation.
    * `labels` - for the `Spatex.Region` results (`Skater`, `Ward`, `RegionKMeans`, `AZP`,
      `MaxP`), the region of each area; the only column. `nil` (areas `Spatex.Region.max_p/3`
      dropped) becomes a missing value.

  Non-finite floats stay `:nan`, `:infinity` and `:neg_infinity`, which Explorer reads as
  float special values.

  ## Options

    * `:crit` - significance level for the `cluster` column (default `0.05`).

  ## Examples

      iex> g = Spatex.Graph.from_arrays([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])
      iex> m = Spatex.ESDA.moran_local([1, 5, 2, 8], g, permutations: 0)
      iex> df = Spatex.Explorer.to_dataframe(m)
      iex> Explorer.Series.to_list(df["q"])
      [2, 4, 2, 4]
  """

  alias Spatex.Classify
  alias Spatex.ESDA.MoranLocal
  alias Spatex.Region.{AZP, MaxP, RegionKMeans, Skater, Ward}

  @regions [AZP, MaxP, RegionKMeans, Skater, Ward]

  @doc """
  Converts a per-site result to an `Explorer.DataFrame`. See the module documentation.
  """
  @spec to_dataframe(struct(), keyword()) :: struct()
  def to_dataframe(result, opts \\ []) do
    crit = Keyword.get(opts, :crit, 0.05)

    unless is_number(crit),
      do: raise(ArgumentError, ":crit must be a number, got #{inspect(crit)}")

    ensure_explorer!()
    result |> columns(crit) |> then(&call(Explorer.DataFrame, :new, [&1]))
  end

  @doc false
  # Ordered `[{name, list}]` for a result; the shared core of `to_dataframe/2` and
  # `Spatex.GeoFrame.put_results/3`.
  @spec columns(struct(), number()) :: [{String.t(), list()}]
  def columns(%Classify{yb: yb}, _crit), do: [{"yb", yb}]

  def columns(%module{labels: labels}, _crit) when module in @regions and is_list(labels),
    do: [{"labels", labels}]

  def columns(%{__struct__: module, n: n} = result, crit) when is_integer(n) do
    fields =
      result
      |> Map.from_struct()
      |> Enum.filter(fn {key, value} -> key != :ids and per_site?(value, n) end)
      |> Enum.sort_by(fn {key, _} -> Atom.to_string(key) end)
      |> Enum.map(fn {key, value} -> {Atom.to_string(key), value} end)

    id_column(result) ++ fields ++ cluster_column(module, result, crit)
  end

  def columns(other, _crit) do
    raise ArgumentError,
          "unsupported result #{inspect(other, limit: 3)}; expected a Spatex ESDA result " <>
            "struct, a Spatex.Classify result or a Spatex.Region result"
  end

  defp per_site?([_ | _] = list, n),
    do: length(list) == n and Enum.all?(list, &scalar?/1)

  defp per_site?(_other, _n), do: false

  defp scalar?(value),
    do: is_number(value) or is_binary(value) or value in [:nan, :infinity, :neg_infinity]

  defp id_column(%{ids: ids}) when is_list(ids), do: [{"id", normalize_ids(ids)}]
  defp id_column(_result), do: []

  defp normalize_ids(ids) do
    cond do
      Enum.all?(ids, &is_integer/1) -> ids
      Enum.all?(ids, &is_binary/1) -> ids
      true -> Enum.map(ids, &id_to_string/1)
    end
  end

  defp id_to_string(id) when is_atom(id) or is_binary(id) or is_number(id), do: to_string(id)
  defp id_to_string(id), do: inspect(id)

  defp cluster_column(MoranLocal, %MoranLocal{p_sim: p_sim} = result, crit)
       when is_list(p_sim),
       do: [{"cluster", MoranLocal.cluster_labels(result, crit)}]

  defp cluster_column(_module, _result, _crit), do: []

  defp ensure_explorer! do
    unless Code.ensure_loaded?(Explorer.DataFrame) do
      raise ArgumentError,
            "Explorer is not available: add {:explorer, \"~> 0.13\"} to your dependencies " <>
              "to convert Spatex results to dataframes"
    end
  end

  # Runtime dispatch keeps the optional dependency out of compile-time xref (see Spatex.Input).
  defp call(mod, fun, args), do: apply(mod, fun, args)
end
