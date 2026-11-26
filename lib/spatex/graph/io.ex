defmodule Spatex.Graph.IO do
  @moduledoc """
  Reads and writes spatial weights in the GeoDa/PySAL GAL and GWT text formats.

  The writers reproduce `libpysal.graph.Graph.to_gal/1` and `to_gwt/1` byte for byte; the
  readers accept the files written by libpysal, GeoDa and legacy `libpysal.io`.

  ## GAL

  Adjacency only, no weights:

      4
      a 2
      b c
      b 1
      a
      ...

  The first line is the node count `n` (legacy files use `0 n shapefile id_var`; the reader
  takes `n` from the second token whenever the header has more than one). Each node then
  takes two lines: `id cardinality` and its neighbor ids (an empty line for an island).
  The declared cardinality is ignored when reading. A graph read from GAL has binary
  weights (`1.0`); **weights are lost when writing GAL**, and a weighted graph writes the
  same file as its binary version. Ids must not contain whitespace.

  ## GWT

  One edge per line, `focal neighbor weight`, after a one-line header (written as
  `0 n Unknown Unknown`; any first line is skipped when reading). Weights are written with
  seven decimals (`%.7f`, as libpysal does), so a roundtrip preserves them to within
  `5.0e-8`. libpysal writes integer-typed weights (for example binary contiguity graphs) as
  `1` rather than `1.0000000`; Spatex stores floats only and always writes seven decimals (both
  forms read back identically). Whitespace in ids is replaced by `_` when writing, and islands are written as
  a zero-weight self row `id id 0.0000000`.

  ## Ids

  Both writers accept integer, binary and atom ids and write them with `to_string/1`.
  The readers take an `:ids` option:

    * `:auto` (default) - integers when every id parses as one, strings otherwise, matching
      libpysal's lossless integer roundtrip;
    * `:integer` - integers; raises when an id is not an integer;
    * `:string` - always strings.

  Node order is the file order: GAL row order, or first appearance in a GWT file.

  Readers take a path (`read_gal/2`, `read_gwt/2`) or already-loaded text (`parse_gal/2`,
  `parse_gwt/2`). All functions raise `ArgumentError` on malformed input.

  ## Examples

      iex> g = Spatex.Graph.from_arrays([1, 2], [2, 1], [0.5, 2.0])
      iex> g |> Spatex.Graph.IO.to_gal() |> IO.iodata_to_binary()
      "2\\n1 1\\n2\\n2 1\\n1\\n"
      iex> g |> Spatex.Graph.IO.to_gwt() |> IO.iodata_to_binary()
      "0 2 Unknown Unknown\\n1 2 0.5000000\\n2 1 2.0000000\\n"
      iex> g |> Spatex.Graph.IO.to_gwt() |> Spatex.Graph.IO.parse_gwt() |> Spatex.Graph.to_adjacency()
      [{1, 2, 0.5}, {2, 1, 2.0}]
  """

  alias Spatex.Graph

  @type ids_opt :: :auto | :integer | :string

  # -- Writing --------------------------------------------------------------------------

  @doc """
  Renders `graph` as GAL text (weights are dropped, see the module documentation).
  """
  @spec to_gal(Graph.t()) :: iodata()
  def to_gal(%Graph{ids: ids} = graph) do
    isolates = MapSet.new(Graph.isolates(graph))

    rows =
      graph
      |> Graph.to_adjacency()
      |> Enum.chunk_by(&elem(&1, 0))
      |> Map.new(fn [{focal, _, _} | _] = chunk -> {focal, Enum.map(chunk, &elem(&1, 1))} end)

    body =
      Enum.map(ids, fn id ->
        neighbors = if MapSet.member?(isolates, id), do: [], else: Map.get(rows, id, [])
        names = Enum.map(neighbors, &gal_id!/1)

        [
          gal_id!(id),
          " ",
          Integer.to_string(length(names)),
          "\n",
          Enum.intersperse(names, " "),
          "\n"
        ]
      end)

    [Integer.to_string(graph.n), "\n" | body]
  end

  @doc """
  Renders `graph` as GWT text: header, then `focal neighbor weight` per edge with seven
  decimals.
  """
  @spec to_gwt(Graph.t()) :: iodata()
  def to_gwt(%Graph{} = graph) do
    rows =
      for {focal, neighbor, weight} <- Graph.to_adjacency(graph) do
        [gwt_id!(focal), " ", gwt_id!(neighbor), " ", float(weight), "\n"]
      end

    ["0 ", Integer.to_string(graph.n), " Unknown Unknown\n" | rows]
  end

  @doc "Writes `graph` to `path` in GAL format. Returns `:ok`."
  @spec write_gal(Graph.t(), Path.t()) :: :ok
  def write_gal(graph, path), do: write(path, to_gal(graph))

  @doc "Writes `graph` to `path` in GWT format. Returns `:ok`."
  @spec write_gwt(Graph.t(), Path.t()) :: :ok
  def write_gwt(graph, path), do: write(path, to_gwt(graph))

  # -- Reading --------------------------------------------------------------------------

  @doc "Reads a GAL file. See the module documentation for the `:ids` option."
  @spec read_gal(Path.t(), keyword()) :: Graph.t()
  def read_gal(path, opts \\ []), do: path |> read!() |> parse_gal(opts)

  @doc "Reads a GWT file. See the module documentation for the `:ids` option."
  @spec read_gwt(Path.t(), keyword()) :: Graph.t()
  def read_gwt(path, opts \\ []), do: path |> read!() |> parse_gwt(opts)

  @doc "Parses GAL text (a binary or iodata)."
  @spec parse_gal(iodata(), keyword()) :: Graph.t()
  def parse_gal(text, opts \\ []) do
    mode = ids_mode!(opts)

    case lines(text) do
      [header | rest] ->
        n = gal_count!(header)
        {ids, neighbors} = gal_rows!(rest, n, [], [])
        build_gal(ids, neighbors, mode)

      [] ->
        raise ArgumentError, "invalid GAL: empty input"
    end
  end

  @doc "Parses GWT text (a binary or iodata); the first line is a header and is skipped."
  @spec parse_gwt(iodata(), keyword()) :: Graph.t()
  def parse_gwt(text, opts \\ []) do
    mode = ids_mode!(opts)

    rows =
      case lines(text) do
        [] -> raise ArgumentError, "invalid GWT: empty input"
        [_header | rest] -> rest |> Enum.reject(&(&1 == "")) |> Enum.map(&gwt_row!/1)
      end

    {focal, neighbor, weight} = unzip3(rows)
    convert = converter(focal ++ neighbor, mode)
    Graph.from_arrays(Enum.map(focal, convert), Enum.map(neighbor, convert), weight)
  end

  # -- GAL internals --------------------------------------------------------------------

  defp gal_count!(header) do
    tokens = String.split(header)

    token =
      case tokens do
        [n] -> n
        [_, n | _] -> n
        [] -> raise ArgumentError, "invalid GAL: missing header"
      end

    case Integer.parse(token) do
      {n, ""} when n >= 0 -> n
      _ -> raise ArgumentError, "invalid GAL header: #{inspect(header)}"
    end
  end

  defp gal_rows!(_lines, 0, ids, neighbors), do: {Enum.reverse(ids), Enum.reverse(neighbors)}

  defp gal_rows!([id_line, neighbor_line | rest], n, ids, neighbors) do
    case String.split(id_line) do
      [id, _cardinality] ->
        gal_rows!(rest, n - 1, [id | ids], [String.split(neighbor_line) | neighbors])

      _ ->
        raise ArgumentError, "invalid GAL row (expected `id cardinality`): #{inspect(id_line)}"
    end
  end

  defp gal_rows!(_lines, _n, _ids, _neighbors),
    do: raise(ArgumentError, "invalid GAL: file ends before all declared rows were read")

  defp build_gal(ids, neighbors, mode) do
    convert = converter(ids ++ List.flatten(neighbors), mode)
    ids = Enum.map(ids, convert)

    {focal, neighbor} =
      ids
      |> Enum.zip(neighbors)
      |> Enum.flat_map(fn {id, ns} -> Enum.map(ns, &{id, convert.(&1)}) end)
      |> Enum.unzip()

    Graph.from_arrays(focal, neighbor, nil, ids: ids)
  end

  # -- GWT internals --------------------------------------------------------------------

  defp gwt_row!(line) do
    with [focal, neighbor, weight] <- String.split(line),
         {w, ""} <- Float.parse(weight) do
      {focal, neighbor, w}
    else
      _ ->
        raise ArgumentError,
              "invalid GWT row (expected `focal neighbor weight`): #{inspect(line)}"
    end
  end

  defp unzip3(rows) do
    {f, n, w} =
      Enum.reduce(rows, {[], [], []}, fn {a, b, c}, {fa, na, wa} ->
        {[a | fa], [b | na], [c | wa]}
      end)

    {Enum.reverse(f), Enum.reverse(n), Enum.reverse(w)}
  end

  # -- Shared ---------------------------------------------------------------------------

  defp ids_mode!(opts) do
    case Keyword.get(opts, :ids, :auto) do
      mode when mode in [:auto, :integer, :string] ->
        mode

      other ->
        raise ArgumentError, ":ids must be :auto, :integer or :string, got: #{inspect(other)}"
    end
  end

  # Returns a function converting id tokens according to `mode`, deciding `:auto` over `tokens`.
  defp converter(_tokens, :string), do: & &1

  defp converter(tokens, :integer) do
    if Enum.all?(tokens, &integer?/1),
      do: &String.to_integer/1,
      else: raise(ArgumentError, "ids: :integer, but the file has ids that are not integers")
  end

  defp converter(tokens, :auto) do
    if Enum.all?(tokens, &integer?/1), do: &String.to_integer/1, else: & &1
  end

  defp integer?(token), do: match?({_, ""}, Integer.parse(token))

  defp lines(text) do
    text
    |> IO.iodata_to_binary()
    |> String.split("\n")
    |> Enum.map(&String.trim_trailing(&1, "\r"))
    |> drop_final_empty()
  end

  # A trailing newline yields one empty final element; island rows need the others kept.
  defp drop_final_empty(lines) do
    case List.last(lines) do
      "" -> Enum.drop(lines, -1)
      _ -> lines
    end
  end

  defp read!(path) do
    case File.read(path) do
      {:ok, data} ->
        data

      {:error, reason} ->
        raise ArgumentError, "cannot read #{path}: #{:file.format_error(reason)}"
    end
  end

  defp write(path, iodata) do
    case File.write(path, iodata) do
      :ok ->
        :ok

      {:error, reason} ->
        raise ArgumentError, "cannot write #{path}: #{:file.format_error(reason)}"
    end
  end

  defp float(w), do: :erlang.float_to_binary(w, decimals: 7)

  defp gal_id!(id) do
    name = id_string!(id)

    if String.match?(name, ~r/\s/),
      do: raise(ArgumentError, "GAL ids must not contain whitespace, got: #{inspect(id)}"),
      else: name
  end

  defp gwt_id!(id), do: id |> id_string!() |> String.replace(" ", "_")

  defp id_string!(id) when is_integer(id) or is_binary(id) or is_atom(id), do: to_string(id)

  defp id_string!(id),
    do:
      raise(
        ArgumentError,
        "cannot write id #{inspect(id)}: ids must be integers, strings or atoms"
      )
end
