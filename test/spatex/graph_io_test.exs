defmodule Spatex.GraphIOTest do
  use ExUnit.Case, async: true

  alias Spatex.Graph

  doctest Spatex.Graph.IO

  @moduletag :tmp_dir

  defp sample do
    Graph.from_adjacency(
      [{:a, :b, 1.0}, {:a, :c, 0.5}, {:b, :a, 0.25}, {:c, :a, 2.0}, {:d, :d, 0.0}],
      ids: [:a, :b, :c, :d]
    )
  end

  describe "to_gal/1" do
    test "writes the simple header, cardinalities and empty island rows" do
      assert IO.iodata_to_binary(Graph.IO.to_gal(sample())) ==
               "4\na 2\nb c\nb 1\na\nc 1\na\nd 0\n\n"
    end

    test "weights are dropped: weighted and binary graphs write identical GAL" do
      assert Graph.IO.to_gal(sample()) |> IO.iodata_to_binary() ==
               sample() |> Graph.transform(:b) |> Graph.IO.to_gal() |> IO.iodata_to_binary()
    end

    test "ids containing whitespace raise" do
      g = Graph.from_arrays(["a b"], ["c"])
      assert_raise ArgumentError, ~r/whitespace/, fn -> Graph.IO.to_gal(g) end
    end

    test "unsupported id terms raise" do
      g = Graph.from_arrays([{1, 2}], [{3, 4}])
      assert_raise ArgumentError, ~r/id/, fn -> Graph.IO.to_gal(g) end
    end
  end

  describe "to_gwt/1" do
    test "writes the libpysal header and %.7f weights, islands as zero self rows" do
      assert IO.iodata_to_binary(Graph.IO.to_gwt(sample())) ==
               "0 4 Unknown Unknown\n" <>
                 "a b 1.0000000\na c 0.5000000\nb a 0.2500000\nc a 2.0000000\nd d 0.0000000\n"
    end

    test "spaces in ids become underscores" do
      g = Graph.from_arrays(["a b"], ["c d"], [1.5])

      assert IO.iodata_to_binary(Graph.IO.to_gwt(g)) ==
               "0 2 Unknown Unknown\na_b c_d 1.5000000\nc_d c_d 0.0000000\n"
    end
  end

  describe "parse_gal/2" do
    test "simple header, integer ids detected" do
      g = Graph.IO.parse_gal("3\n0 1\n1\n1 2\n0 2\n2 1\n1\n")
      assert Graph.ids(g) == [0, 1, 2]
      assert Graph.to_adjacency(g) == [{0, 1, 1.0}, {1, 0, 1.0}, {1, 2, 1.0}, {2, 1, 1.0}]
    end

    test "legacy header takes n from the second token" do
      g = Graph.IO.parse_gal("0 2 shapefile id\n1 1\n2\n2 1\n1\n")
      assert Graph.ids(g) == [1, 2]
      assert Graph.n(g) == 2
    end

    test "islands are rows with no neighbors" do
      g = Graph.IO.parse_gal("2\na 1\nb\nb 0\n\n")
      assert Graph.isolates(g) == ["b"]
      assert Graph.ids(g) == ["a", "b"]
    end

    test "mixed ids stay strings; ids: :string keeps numeric strings" do
      assert Graph.ids(Graph.IO.parse_gal("2\n1 1\nx\nx 1\n1\n")) == ["1", "x"]
      assert Graph.ids(Graph.IO.parse_gal("2\n1 1\n2\n2 1\n1\n", ids: :string)) == ["1", "2"]
    end

    test "ids: :integer raises on non-integer ids" do
      assert_raise ArgumentError, ~r/integer/, fn ->
        Graph.IO.parse_gal("2\na 1\nb\nb 1\na\n", ids: :integer)
      end
    end

    test "tolerates CRLF line endings and ignores the declared cardinality" do
      g = Graph.IO.parse_gal("2\r\n0 5\r\n1\r\n1 0\r\n0\r\n")
      assert Graph.to_adjacency(g) == [{0, 1, 1.0}, {1, 0, 1.0}]
    end

    test "truncated or malformed files raise" do
      assert_raise ArgumentError, ~r/GAL/, fn -> Graph.IO.parse_gal("3\n0 1\n1\n") end
      assert_raise ArgumentError, ~r/GAL/, fn -> Graph.IO.parse_gal("") end
      assert_raise ArgumentError, ~r/GAL/, fn -> Graph.IO.parse_gal("x\n") end
      assert_raise ArgumentError, ~r/GAL/, fn -> Graph.IO.parse_gal("1\n0 1 9\n1\n") end
    end

    test "invalid :ids option raises" do
      assert_raise ArgumentError, ~r/:ids/, fn -> Graph.IO.parse_gal("0\n", ids: :float) end
    end
  end

  describe "parse_gwt/2" do
    test "skips the header and reads weighted triples" do
      g = Graph.IO.parse_gwt("0 3 shp id\n1 2 0.5\n2 1 0.5\n2 3 1\n3 3 0.0000000\n")
      assert Graph.ids(g) == [1, 2, 3]
      assert Graph.to_adjacency(g) == [{1, 2, 0.5}, {2, 1, 0.5}, {2, 3, 1.0}, {3, 3, 0.0}]
      assert Graph.isolates(g) == [3]
    end

    test "string ids and ids: :string" do
      assert Graph.ids(Graph.IO.parse_gwt("h\na b 1\nb a 1\n")) == ["a", "b"]
      assert Graph.ids(Graph.IO.parse_gwt("h\n1 2 1\n", ids: :string)) == ["1", "2"]
    end

    test "malformed rows raise" do
      assert_raise ArgumentError, ~r/GWT/, fn -> Graph.IO.parse_gwt("h\n1 2\n") end
      assert_raise ArgumentError, ~r/GWT/, fn -> Graph.IO.parse_gwt("h\n1 2 x\n") end
    end
  end

  describe "files" do
    test "write then read roundtrips through disk", %{tmp_dir: dir} do
      gal = Path.join(dir, "g.gal")
      gwt = Path.join(dir, "g.gwt")
      g = Graph.from_arrays([1, 1, 2], [2, 3, 1], [0.5, 0.25, 1.0], ids: [1, 2, 3, 4])

      assert :ok = Graph.write_gal(g, gal)
      assert :ok = Graph.write_gwt(g, gwt)

      assert Graph.to_adjacency(Graph.read_gal(gal)) ==
               [{1, 2, 1.0}, {1, 3, 1.0}, {2, 1, 1.0}, {3, 3, 0.0}, {4, 4, 0.0}]

      assert Graph.to_adjacency(Graph.read_gwt(gwt)) == Graph.to_adjacency(g)
    end

    test "missing file raises ArgumentError" do
      assert_raise ArgumentError, ~r/cannot read/, fn -> Graph.read_gal("/nonexistent/x.gal") end
      assert_raise ArgumentError, ~r/cannot read/, fn -> Graph.read_gwt("/nonexistent/x.gwt") end
    end

    test "Graph delegates match Graph.IO" do
      assert Graph.to_gal(sample()) == Graph.IO.to_gal(sample())
      assert Graph.to_gwt(sample()) == Graph.IO.to_gwt(sample())
    end
  end
end
