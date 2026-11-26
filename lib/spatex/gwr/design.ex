defmodule Spatex.GWR.Design do
  @moduledoc false
  # Input normalization shared by `Spatex.GWR.fit/4`, `select_bandwidth/4` and `mgwr/4`: points,
  # response and regressors become binaries, with the intercept handling of mgwr.

  alias Spatex.Input

  @kernels [:bisquare, :gaussian, :exponential]

  @doc "Raises on unknown options."
  def check_known!(opts, allowed) do
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "options must be a keyword list")

    case Keyword.keys(opts) -- allowed do
      [] ->
        :ok

      bad ->
        raise ArgumentError, "unknown option(s) #{inspect(bad)}; expected #{inspect(allowed)}"
    end
  end

  @doc "Validates the options every GWR entry point shares."
  def check_common!(opts) do
    unless opts[:kernel] in @kernels do
      raise ArgumentError,
            "kernel must be one of #{inspect(@kernels)}, got #{inspect(opts[:kernel])}"
    end

    for key <- [:fixed, :constant, :spherical] do
      unless is_boolean(opts[key]), do: raise(ArgumentError, "#{key} must be a boolean")
    end

    unless is_integer(opts[:max_n]) and opts[:max_n] > 0 do
      raise ArgumentError, "max_n must be a positive integer"
    end

    :ok
  end

  @doc """
  Builds the model binaries. Raises when `n` exceeds `max_n` and `guard_n?` is true. A `nil`
  response gives zeros (the local collinearity diagnostics do not use it).
  """
  def model!(points, y, x, opts, guard_n? \\ true) do
    {coords, n} = Input.to_points_binary(points)

    if guard_n? and n > opts[:max_n] do
      raise ArgumentError,
            "n = #{n} exceeds max_n = #{opts[:max_n]}: a fit costs O(n^2 k^2) time; " <>
              "pass a larger max_n to proceed"
    end

    ybin = response!(y, n)
    columns = columns!(x, n)
    n_loc = length(columns)
    {columns, warnings, kept} = design(columns, opts[:constant])

    %{
      coords: coords,
      y: ybin,
      x: IO.iodata_to_binary(columns),
      n: n,
      k: length(columns),
      n_loc: n_loc,
      kept: kept,
      warnings: warnings
    }
  end

  defp response!(nil, n), do: :binary.copy(<<0.0::float-64-native>>, n)

  defp response!(y, n) do
    {ybin, ^n} = Input.to_f64_binary(y, name: "y", length: n)
    ybin
  end

  @doc """
  The design matrix (`k` columns of `m` values) of `m` new rows: the columns of `x_new`
  that survived in the calibration design (constant columns were dropped there) and the
  intercept first when the model has one.
  """
  def regressors!(x_new, model, constant, m) do
    cols = split_columns(x_new)

    if length(cols) != model.n_loc do
      raise ArgumentError,
            "x_new has #{length(cols)} columns, expected #{model.n_loc} (the columns of x)"
    end

    bins =
      cols
      |> Enum.with_index(1)
      |> Enum.map(fn {col, i} ->
        {bin, _} = Input.to_f64_binary(col, name: "x_new column #{i}", length: m)
        bin
      end)

    kept = Enum.map(model.kept, &Enum.at(bins, &1))
    ones = if constant, do: [:binary.copy(<<1.0::float-64-native>>, m)], else: []
    IO.iodata_to_binary(ones ++ kept)
  end

  @doc """
  Number of columns of mgwr's `X_glob` (`nil` and `[]` count as none), checking that each has
  `n` values. Only the count matters: `Sel_BW` uses it for the default search bounds.
  """
  def glob_columns!(nil, _n), do: 0
  def glob_columns!([], _n), do: 0

  def glob_columns!(x_glob, n) do
    cols = split_columns(x_glob)

    cols
    |> Enum.with_index(1)
    |> Enum.each(fn {col, i} ->
      Input.to_f64_binary(col, name: "x_glob column #{i}", length: n)
    end)

    length(cols)
  end

  defp columns!(x, n) do
    cols = split_columns(x)
    if cols == [], do: raise(ArgumentError, "x has no columns")

    cols
    |> Enum.with_index(1)
    |> Enum.map(fn {col, i} ->
      {bin, _} = Input.to_f64_binary(col, name: "x column #{i}", length: n)
      bin
    end)
  end

  defp split_columns(%{__struct__: Nx.Tensor} = t) do
    case call(Nx, :rank, [t]) do
      1 -> [t]
      2 -> t |> then(&call(Nx, :transpose, [&1])) |> then(&call(Nx, :to_list, [&1]))
      rank -> raise ArgumentError, "x tensor must have rank 1 or 2, got rank #{rank}"
    end
  end

  defp split_columns(%{__struct__: Explorer.Series} = s), do: [s]

  defp split_columns(%{} = map) when not is_struct(map),
    do: map |> Enum.sort_by(fn {k, _} -> to_string(k) end) |> Enum.map(&elem(&1, 1))

  defp split_columns([]), do: []

  defp split_columns(list) when is_list(list) do
    if Enum.all?(list, &(is_number(&1) or &1 in [:nan, :infinity, :neg_infinity])),
      do: [list],
      else: list
  end

  defp split_columns(other),
    do: raise(ArgumentError, "unsupported x #{inspect(other, limit: 3)}; expected columns")

  # mgwr (through spreg's check_constant) drops constant columns and prepends the intercept.
  defp design(bins, false), do: {bins, [], Enum.to_list(0..(length(bins) - 1))}

  defp design(bins, true) do
    {kept_pairs, dropped} =
      bins |> Enum.with_index() |> Enum.split_with(fn {bin, _} -> varies?(bin) end)

    kept = Enum.map(kept_pairs, &elem(&1, 0))

    if kept == [], do: raise(ArgumentError, "every x column is constant; nothing to regress on")

    warnings =
      case dropped do
        [] -> []
        _ -> ["#{length(dropped)} constant x column(s) removed; an intercept is added instead."]
      end

    ones = :binary.copy(<<1.0::float-64-native>>, div(byte_size(hd(bins)), 8))
    {[ones | kept], warnings, Enum.map(kept_pairs, &elem(&1, 1))}
  end

  defp varies?(bin) do
    [first | rest] = for <<v::float-64-native <- bin>>, do: v
    Enum.any?(rest, &(&1 != first))
  end

  # Dispatch through a runtime module argument so the optional dependencies are never
  # referenced as compile-time remote calls.
  defp call(mod, fun, args), do: apply(mod, fun, args)
end
