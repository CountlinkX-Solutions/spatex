defmodule Spatex.Input do
  @moduledoc """
  The input contract shared by every Spatex function.

  User data is normalized in Elixir, before any NIF call, into flat native-endian
  binaries (`f64` for numeric vectors and points, `u32` for index vectors). All
  validation errors raise `ArgumentError` naming the argument and the offending index.

  ## Accepted inputs

  | Input | `to_f64_binary/2` | `to_points_binary/1` |
  |---|---|---|
  | List of numbers (integers are coerced) | yes | `[{x, y}]` or `[[x, y]]` |
  | `Nx.Tensor` (optional dep) | rank 1, any numeric type | `{n, 2}` tensor |
  | `Explorer.Series` (optional dep) | numeric dtypes | `{xs, ys}` pair |
  | Packed `{:f64, binary}` | passed through (validated) | - |

  Labels (regimes, partitions) go through `to_labels/2`: a list of any terms, a rank-1 `Nx`
  tensor or an `Explorer.Series` of any dtype. Panels and other `n x t` matrices go through
  `to_matrix/2`: a list of numeric rows, a rank-2 `Nx` tensor or an `Explorer.DataFrame` (its
  numeric columns in order).

  `Nx` and `Explorer` are optional dependencies; they are only touched when the user
  passes one of their structs, so Spatex compiles and runs without them.

  ## Options (`to_f64_binary/2`)

    * `:name` - argument name used in error messages (default `"values"`).
    * `:allow_nan` - accept NaN and infinities (default `false`).
    * `:allow_nil` - for `Explorer.Series`, map `nil` to NaN instead of raising
      (default `false`). The resulting NaN is still subject to `:allow_nan`.
    * `:length` - expected length; a mismatch raises.

  ## NaN and infinity

  Elixir floats cannot hold NaN or infinity, so those values are written to binaries
  from raw bit patterns and decoded by `from_f64_binary/1` as the atoms `:nan`,
  `:infinity` and `:neg_infinity`. Lists may contain those atoms when `allow_nan: true`.
  """

  import Bitwise

  @nan_bits 0x7FF8000000000000
  @max_u32 4_294_967_295

  @type f64_result :: {binary(), non_neg_integer()}
  @type special :: :nan | :infinity | :neg_infinity

  @doc """
  Normalizes a numeric vector to a native-endian `f64` binary, returning `{binary, n}`.

  ## Examples

      iex> {bin, 2} = Spatex.Input.to_f64_binary([1, 2.5])
      iex> Spatex.Input.from_f64_binary(bin)
      [1.0, 2.5]
  """
  @spec to_f64_binary(term(), keyword()) :: f64_result()
  def to_f64_binary(values, opts \\ []) do
    name = Keyword.get(opts, :name, "values")
    allow_nan = Keyword.get(opts, :allow_nan, false)
    {bin, n} = values |> normalize(name, opts) |> check_length(name, opts[:length])
    check_finite(bin, name, allow_nan)
    {bin, n}
  end

  @doc """
  Normalizes points to an interleaved `x0, y0, x1, y1, ...` `f64` binary, returning `{binary, n}`.

  Accepts `[{x, y}]`, `[[x, y]]`, an `{n, 2}` `Nx.Tensor`, or a pair `{xs, ys}` of
  anything `to_f64_binary/2` accepts. Non-finite coordinates are rejected.

  ## Examples

      iex> {bin, 2} = Spatex.Input.to_points_binary([{0, 1}, {2, 3}])
      iex> Spatex.Input.from_f64_binary(bin)
      [0.0, 1.0, 2.0, 3.0]
  """
  @spec to_points_binary(term()) :: f64_result()
  def to_points_binary(%{__struct__: Nx.Tensor} = tensor) do
    case call(Nx, :shape, [tensor]) do
      {n, 2} ->
        {bin, _} =
          to_f64_binary(call(Nx, :reshape, [tensor, {n * 2}]), name: "points", allow_nan: true)

        check_finite(bin, "points", false, 2)
        {bin, n}

      shape ->
        raise ArgumentError, "points: expected a {n, 2} tensor, got shape #{inspect(shape)}"
    end
  end

  def to_points_binary({xs, ys}) do
    {xb, n} = to_f64_binary(xs, name: "x")
    {yb, ^n} = to_f64_binary(ys, name: "y", length: n)
    {interleave(xb, yb, <<>>), n}
  end

  def to_points_binary(points) when is_list(points) do
    {acc, n} =
      Enum.reduce(points, {<<>>, 0}, fn point, {acc, i} ->
        {x, y} = point_xy(point, i)

        {<<acc::binary, encode_point_coord(x, i)::binary, encode_point_coord(y, i)::binary>>,
         i + 1}
      end)

    {acc, n}
  end

  def to_points_binary(other) do
    raise ArgumentError,
          "points: unsupported input #{inspect(other, limit: 3)}; " <>
            "expected a list of {x, y}, an {n, 2} tensor or a pair {xs, ys}"
  end

  @doc """
  Normalizes a vector of non-negative integer indices to a native-endian `u32` binary.

  ## Examples

      iex> Spatex.Input.to_index_binary([0, 2], :u32) |> elem(1)
      2
  """
  @spec to_index_binary(term(), :u32, keyword()) :: {binary(), non_neg_integer()}
  def to_index_binary(list, type, opts \\ [])

  def to_index_binary(%{__struct__: Nx.Tensor} = tensor, :u32, opts) do
    to_index_binary(call(Nx, :to_flat_list, [tensor]), :u32, opts)
  end

  def to_index_binary(list, :u32, opts) when is_list(list) do
    name = Keyword.get(opts, :name, "indices")

    {bin, n} =
      list
      |> Enum.with_index()
      |> Enum.reduce({<<>>, 0}, fn {v, i}, {acc, _} ->
        {<<acc::binary, encode_index(v, i, name)::binary>>, i + 1}
      end)

    check_length({bin, n}, name, opts[:length])
  end

  def to_index_binary(other, _type, opts) do
    raise ArgumentError,
          "#{Keyword.get(opts, :name, "indices")}: unsupported input #{inspect(other, limit: 3)}"
  end

  @doc """
  Decodes a native-endian `f64` binary to a list.

  NaN and infinity bit patterns decode to `:nan`, `:infinity` and `:neg_infinity`
  because Elixir floats cannot represent them.

  ## Examples

      iex> Spatex.Input.from_f64_binary(<<1.5::float-64-native>>)
      [1.5]
  """
  @spec from_f64_binary(binary()) :: [float() | special()]
  def from_f64_binary(bin) when is_binary(bin) and rem(byte_size(bin), 8) == 0 do
    for <<bits::unsigned-native-64 <- bin>>, do: decode(bits)
  end

  def from_f64_binary(bin) when is_binary(bin) do
    raise ArgumentError, "expected a binary with a multiple of 8 bytes, got #{byte_size(bin)}"
  end

  @doc """
  Validates a `:keep_simulations` option value: `true`, `false` or `:tensor`.

  `:tensor` needs the optional `:nx` dependency; the second argument (default: whether `Nx` is
  loaded) lets that check be exercised without it. Returns the value.
  """
  @spec keep_simulations!(term(), boolean()) :: boolean() | :tensor
  def keep_simulations!(value, nx_loaded? \\ Code.ensure_loaded?(Nx))

  def keep_simulations!(value, _nx_loaded?) when is_boolean(value), do: value

  def keep_simulations!(:tensor, true), do: :tensor

  def keep_simulations!(:tensor, false) do
    raise ArgumentError,
          ":keep_simulations :tensor needs the optional :nx dependency; add {:nx, \"~> 0.10\"} to your deps"
  end

  def keep_simulations!(other, _nx_loaded?) do
    raise ArgumentError,
          ":keep_simulations must be true, false or :tensor, got #{inspect(other)}"
  end

  @doc """
  Raises an `ArgumentError` naming the optional `:nx` dependency unless `Nx` is loaded.

  `what` names the function that needs it; the second argument (default: whether `Nx` is
  loaded) lets the failure be exercised without removing `Nx`.
  """
  @spec require_nx!(String.t(), boolean()) :: :ok
  def require_nx!(what, nx_loaded? \\ Code.ensure_loaded?(Nx))

  def require_nx!(_what, true), do: :ok

  def require_nx!(what, false) do
    raise ArgumentError,
          "#{what} needs the optional :nx dependency; add {:nx, \"~> 0.10\"} to your deps"
  end

  @doc """
  The boolean the native side takes for a validated `:keep_simulations` value: `:tensor` keeps
  the draws like `true`.
  """
  @spec keep_flag(boolean() | :tensor) :: boolean()
  def keep_flag(keep), do: keep != false

  @doc """
  Decodes a simulation binary into the form `keep` asks for: nested lists (`true`) or an
  `Nx.Tensor` of type `{:f, 64}` built straight from the binary (`:tensor`), both of `shape`.

  The list form nests by `shape`, so `{n, p}` is `n` lists of `p` values.
  """
  @spec simulations(binary(), true | :tensor, tuple()) :: list() | Nx.Tensor.t()
  def simulations(bin, :tensor, shape) do
    call(Nx, :reshape, [call(Nx, :from_binary, [bin, {:f, 64}]), shape])
  end

  def simulations(bin, true, shape) do
    bin |> from_f64_binary() |> nest(Tuple.to_list(shape))
  end

  @doc """
  Splits a binary of `count` equal simulation vectors of `len` draws into `count` values, each
  a list (`true`) or a `{len}` tensor (`:tensor`).
  """
  @spec simulation_vectors(binary(), true | :tensor, pos_integer(), non_neg_integer()) :: [
          list() | Nx.Tensor.t()
        ]
  def simulation_vectors(bin, keep, count, len) do
    for i <- 0..(count - 1)//1,
        do: simulations(binary_part(bin, i * len * 8, len * 8), keep, {len})
  end

  @doc """
  Element-wise sum of two simulation vectors of the same form (two lists or two tensors).
  """
  @spec add_simulations(list() | Nx.Tensor.t(), list() | Nx.Tensor.t()) ::
          list() | Nx.Tensor.t()
  def add_simulations(%{__struct__: Nx.Tensor} = a, b), do: call(Nx, :add, [a, b])
  def add_simulations(a, b), do: Enum.zip_with(a, b, &(&1 + &2))

  defp nest(flat, [_last]), do: flat
  defp nest(flat, [_first, last]), do: Enum.chunk_every(flat, last)

  defp nest(flat, [_first | rest]) do
    size = Enum.product(rest)
    flat |> Enum.chunk_every(size) |> Enum.map(&nest(&1, rest))
  end

  @doc """
  Normalizes per-observation labels (regimes, partitions, groups) to a list of terms.

  Accepts a list of any terms, a rank-1 `Nx.Tensor` or an `Explorer.Series` of any dtype
  (`nil` raises). Options: `:name` (default `"labels"`) and `:length`, the expected count.
  """
  @spec to_labels(term(), keyword()) :: [term()]
  def to_labels(values, opts \\ []) do
    name = Keyword.get(opts, :name, "labels")
    labels = labels(values, name)

    case Keyword.fetch(opts, :length) do
      {:ok, n} when length(labels) != n ->
        raise ArgumentError, "#{name} must hold one label per observation (#{n})"

      _ ->
        labels
    end
  end

  defp labels(list, _name) when is_list(list), do: list

  defp labels(%{__struct__: Nx.Tensor} = t, name) do
    case call(Nx, :rank, [t]) do
      1 -> call(Nx, :to_flat_list, [t])
      rank -> raise ArgumentError, "#{name}: expected a rank-1 tensor, got rank #{rank}"
    end
  end

  defp labels(%{__struct__: Explorer.Series} = s, name) do
    list = call(Explorer.Series, :to_list, [s])

    case Enum.find_index(list, &is_nil/1) do
      nil -> list
      i -> raise ArgumentError, "#{name}: missing value (nil) at index #{i}"
    end
  end

  defp labels(other, name) do
    raise ArgumentError,
          "#{name}: unsupported input #{inspect(other, limit: 3)}; expected a list, " <>
            "an Nx.Tensor or an Explorer.Series"
  end

  @doc """
  Normalizes an `n x t` matrix (one row per observation) to a list of float rows.

  Accepts a list of numeric rows, a rank-2 `Nx.Tensor` or an `Explorer.DataFrame` whose numeric
  columns are taken in order (one column per period for panels). Options: `:name` (default
  `"matrix"`) and `:rows`, the expected row count.
  """
  @spec to_matrix(term(), keyword()) :: [[float()]]
  def to_matrix(m, opts \\ []) do
    name = Keyword.get(opts, :name, "matrix")
    rows = rows(m, name)

    case Keyword.fetch(opts, :rows) do
      {:ok, n} when length(rows) != n ->
        raise ArgumentError,
              "#{name}: expected #{n} rows (one per observation), got #{length(rows)}"

      _ ->
        check_rows!(rows, name)
    end
  end

  defp rows(%{__struct__: Nx.Tensor} = t, name) do
    case call(Nx, :rank, [t]) do
      2 -> call(Nx, :to_list, [call(Nx, :as_type, [t, :f64])])
      rank -> raise ArgumentError, "#{name}: expected a rank-2 tensor, got rank #{rank}"
    end
  end

  defp rows(%{__struct__: Explorer.DataFrame} = df, name) do
    cols =
      Enum.map(call(Explorer.DataFrame, :names, [df]), fn col ->
        s = call(Explorer.DataFrame, :pull, [df, col])

        unless numeric_dtype?(call(Explorer.Series, :dtype, [s])),
          do: raise(ArgumentError, "#{name}: column #{inspect(col)} is not numeric")

        call(Explorer.Series, :to_list, [s])
      end)

    cols |> Enum.zip() |> Enum.map(&Tuple.to_list/1)
  end

  defp rows(list, name) when is_list(list) and list != [] do
    if Enum.all?(list, &is_list/1),
      do: list,
      else: raise(ArgumentError, "#{name} must be a list of rows")
  end

  defp rows(other, name) do
    raise ArgumentError,
          "#{name}: unsupported input #{inspect(other, limit: 3)}; expected a non-empty list " <>
            "of rows, a rank-2 Nx.Tensor or an Explorer.DataFrame"
  end

  defp check_rows!(rows, name) do
    t = length(hd(rows))

    unless Enum.all?(rows, &(length(&1) == t)),
      do: raise(ArgumentError, "#{name}: rows must all hold the same number of columns")

    Enum.map(rows, fn row ->
      Enum.map(row, fn
        v when is_number(v) -> v * 1.0
        _ -> raise ArgumentError, "#{name} must contain only numbers"
      end)
    end)
  end

  # -- normalization ---------------------------------------------------------------------

  defp normalize(list, name, _opts) when is_list(list) do
    Enum.reduce(list, {<<>>, 0}, fn v, {acc, i} ->
      {<<acc::binary, encode_value(v, i, name)::binary>>, i + 1}
    end)
  end

  defp normalize({:f64, bin}, name, _opts) when is_binary(bin) do
    if rem(byte_size(bin), 8) != 0 do
      raise ArgumentError,
            "#{name}: packed f64 binary must be a multiple of 8 bytes, got #{byte_size(bin)}"
    end

    {bin, div(byte_size(bin), 8)}
  end

  defp normalize(%{__struct__: Nx.Tensor} = t, name, _opts) do
    case call(Nx, :rank, [t]) do
      1 -> :ok
      rank -> raise ArgumentError, "#{name}: expected a rank-1 tensor, got rank #{rank}"
    end

    case call(Nx, :type, [t]) do
      {:c, _} -> raise ArgumentError, "#{name}: complex tensors are not supported"
      _ -> :ok
    end

    bin = call(Nx, :to_binary, [call(Nx, :as_type, [t, :f64])])
    {bin, div(byte_size(bin), 8)}
  end

  defp normalize(%{__struct__: Explorer.Series} = s, name, opts) do
    dtype = call(Explorer.Series, :dtype, [s])

    unless numeric_dtype?(dtype) do
      raise ArgumentError, "#{name}: expected a numeric series, got dtype #{inspect(dtype)}"
    end

    allow_nil = Keyword.get(opts, :allow_nil, false)

    {bin, n} =
      call(Explorer.Series, :to_list, [s])
      |> Enum.reduce({<<>>, 0}, fn
        nil, {acc, i} ->
          if allow_nil do
            {<<acc::binary, @nan_bits::unsigned-native-64>>, i + 1}
          else
            raise ArgumentError, "#{name}: missing value (nil) at index #{i}"
          end

        v, {acc, i} ->
          {<<acc::binary, encode_value(v, i, name)::binary>>, i + 1}
      end)

    {bin, n}
  end

  defp normalize(other, name, _opts) do
    raise ArgumentError,
          "#{name}: unsupported input #{inspect(other, limit: 3)}; expected a list of numbers, " <>
            "an Nx.Tensor, an Explorer.Series or {:f64, binary}"
  end

  defp numeric_dtype?({kind, _}) when kind in [:s, :u, :f], do: true
  defp numeric_dtype?(dtype), do: dtype in [:integer, :float]

  defp check_length({_bin, _n} = result, _name, nil), do: result
  defp check_length({_bin, n} = result, _name, n), do: result

  defp check_length({_bin, n}, name, expected) do
    raise ArgumentError, "#{name}: expected length #{expected}, got #{n}"
  end

  # -- encoding --------------------------------------------------------------------------

  defp encode_value(v, _i, _name) when is_float(v), do: <<v::float-64-native>>
  defp encode_value(:nan, _i, _name), do: <<@nan_bits::unsigned-native-64>>
  defp encode_value(:infinity, _i, _name), do: <<0x7FF0000000000000::unsigned-native-64>>
  defp encode_value(:neg_infinity, _i, _name), do: <<0xFFF0000000000000::unsigned-native-64>>

  defp encode_value(v, i, name) when is_integer(v) do
    <<v * 1.0::float-64-native>>
  rescue
    ArithmeticError ->
      reraise ArgumentError, "#{name}: integer at index #{i} is too large for f64", __STACKTRACE__
  end

  defp encode_value(v, i, name) do
    raise ArgumentError, "#{name}: expected a number at index #{i}, got #{inspect(v)}"
  end

  defp encode_point_coord(v, i) when is_number(v), do: encode_value(v, i, "points")

  defp encode_point_coord(v, i),
    do: raise(ArgumentError, "points: expected a number at index #{i}, got #{inspect(v)}")

  defp point_xy({x, y}, _i), do: {x, y}
  defp point_xy([x, y], _i), do: {x, y}

  defp point_xy(other, i),
    do: raise(ArgumentError, "points: expected {x, y} at index #{i}, got #{inspect(other)}")

  defp encode_index(v, _i, _name) when is_integer(v) and v >= 0 and v <= @max_u32,
    do: <<v::unsigned-native-32>>

  defp encode_index(v, i, name) when is_integer(v) and v > @max_u32,
    do: raise(ArgumentError, "#{name}: value #{v} at index #{i} exceeds u32")

  defp encode_index(v, i, name),
    do:
      raise(
        ArgumentError,
        "#{name}: expected a non-negative integer at index #{i}, got #{inspect(v)}"
      )

  defp interleave(<<x::binary-8, xr::binary>>, <<y::binary-8, yr::binary>>, acc),
    do: interleave(xr, yr, <<acc::binary, x::binary, y::binary>>)

  defp interleave(<<>>, <<>>, acc), do: acc

  # -- finiteness ------------------------------------------------------------------------

  defp check_finite(bin, name, allow_nan, per \\ 1)
  defp check_finite(_bin, _name, true, _per), do: :ok
  defp check_finite(bin, name, false, per), do: scan(bin, name, 0, per)

  defp scan(<<bits::unsigned-native-64, rest::binary>>, name, i, per) do
    if (bits >>> 52 &&& 0x7FF) == 0x7FF do
      raise ArgumentError, "#{name}: non-finite value (#{describe(bits)}) at index #{div(i, per)}"
    end

    scan(rest, name, i + 1, per)
  end

  defp scan(<<>>, _name, _i, _per), do: :ok

  defp describe(bits) do
    cond do
      (bits &&& 0xFFFFFFFFFFFFF) != 0 -> "NaN"
      bits >>> 63 == 1 -> "-infinity"
      true -> "infinity"
    end
  end

  defp decode(bits) do
    if (bits >>> 52 &&& 0x7FF) == 0x7FF do
      cond do
        (bits &&& 0xFFFFFFFFFFFFF) != 0 -> :nan
        bits >>> 63 == 1 -> :neg_infinity
        true -> :infinity
      end
    else
      <<f::float-64-native>> = <<bits::unsigned-native-64>>
      f
    end
  end

  # Dispatch through a runtime module argument so the optional dependencies are never
  # referenced as compile-time remote calls (no undefined-module warnings when absent).
  defp call(mod, fun, args), do: apply(mod, fun, args)
end
