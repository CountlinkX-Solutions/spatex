defmodule Spatex.InputTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Spatex.Input

  doctest Spatex.Input

  describe "to_f64_binary/2 with lists" do
    test "packs floats and coerces integers" do
      assert {bin, 3} = Input.to_f64_binary([1, 2.5, -3])
      assert bin == <<1.0::float-64-native, 2.5::float-64-native, -3.0::float-64-native>>
    end

    test "empty list" do
      assert {<<>>, 0} = Input.to_f64_binary([])
    end

    test "rejects non-numbers naming argument and index" do
      assert_raise ArgumentError, ~r/y: expected a number at index 1, got "a"/, fn ->
        Input.to_f64_binary([1, "a"], name: "y")
      end
    end

    test "rejects non-list, non-supported input" do
      assert_raise ArgumentError, ~r/values: unsupported input/, fn ->
        Input.to_f64_binary(:nope)
      end
    end

    test "length check" do
      assert {_, 2} = Input.to_f64_binary([1, 2], length: 2)

      assert_raise ArgumentError, ~r/y: expected length 3, got 2/, fn ->
        Input.to_f64_binary([1, 2], name: "y", length: 3)
      end
    end

    test "allow_nan does not change plain lists" do
      assert {_, 1} = Input.to_f64_binary([1.0], allow_nan: true)
    end
  end

  describe "to_f64_binary/2 with packed binaries" do
    test "passes through {:f64, binary}" do
      bin = <<1.0::float-64-native, 2.0::float-64-native>>
      assert {^bin, 2} = Input.to_f64_binary({:f64, bin})
    end

    test "rejects misaligned binaries" do
      assert_raise ArgumentError, ~r/multiple of 8 bytes/, fn ->
        Input.to_f64_binary({:f64, <<1, 2, 3>>})
      end
    end

    test "rejects NaN and infinity with index unless allowed" do
      nan = <<0x7FF8000000000000::unsigned-native-64>>
      bin = <<1.0::float-64-native>> <> nan

      assert_raise ArgumentError, ~r/x: non-finite value \(NaN\) at index 1/, fn ->
        Input.to_f64_binary({:f64, bin}, name: "x")
      end

      assert {^bin, 2} = Input.to_f64_binary({:f64, bin}, allow_nan: true)

      inf = <<0x7FF0000000000000::unsigned-native-64>>

      assert_raise ArgumentError, ~r/non-finite value \(infinity\) at index 0/, fn ->
        Input.to_f64_binary({:f64, inf})
      end
    end

    test "length check applies" do
      assert_raise ArgumentError, ~r/expected length 2, got 1/, fn ->
        Input.to_f64_binary({:f64, <<1.0::float-64-native>>}, length: 2)
      end
    end
  end

  describe "to_f64_binary/2 with Nx" do
    test "casts numeric types to f64" do
      assert {bin, 3} = Input.to_f64_binary(Nx.tensor([1, 2, 3], type: :s32))
      assert Input.from_f64_binary(bin) == [1.0, 2.0, 3.0]

      assert {bin, 2} = Input.to_f64_binary(Nx.tensor([0.5, 1.5], type: :f32))
      assert Input.from_f64_binary(bin) == [0.5, 1.5]

      assert {bin, 2} = Input.to_f64_binary(Nx.tensor([1, 2], type: :u8))
      assert Input.from_f64_binary(bin) == [1.0, 2.0]
    end

    test "rejects rank != 1" do
      assert_raise ArgumentError, ~r/x: expected a rank-1 tensor, got rank 2/, fn ->
        Input.to_f64_binary(Nx.tensor([[1, 2], [3, 4]]), name: "x")
      end
    end

    test "rejects NaN unless allowed" do
      t = Nx.tensor([1.0, :nan, :infinity], type: :f64)

      assert_raise ArgumentError, ~r/non-finite value \(NaN\) at index 1/, fn ->
        Input.to_f64_binary(t)
      end

      assert {bin, 3} = Input.to_f64_binary(t, allow_nan: true)
      assert Input.from_f64_binary(bin) == [1.0, :nan, :infinity]
    end

    test "length check applies" do
      assert_raise ArgumentError, ~r/expected length 3, got 2/, fn ->
        Input.to_f64_binary(Nx.tensor([1, 2]), length: 3)
      end
    end
  end

  describe "to_f64_binary/2 with Explorer" do
    test "numeric series" do
      assert {bin, 3} = Input.to_f64_binary(Explorer.Series.from_list([1, 2, 3]))
      assert Input.from_f64_binary(bin) == [1.0, 2.0, 3.0]

      assert {bin, 2} = Input.to_f64_binary(Explorer.Series.from_list([1.5, 2.5]))
      assert Input.from_f64_binary(bin) == [1.5, 2.5]
    end

    test "nil is an error unless allow_nil: true" do
      s = Explorer.Series.from_list([1.0, nil, 3.0])

      assert_raise ArgumentError, ~r/x: missing value \(nil\) at index 1/, fn ->
        Input.to_f64_binary(s, name: "x")
      end

      assert {bin, 3} = Input.to_f64_binary(s, allow_nil: true, allow_nan: true)
      assert Input.from_f64_binary(bin) == [1.0, :nan, 3.0]
    end

    test "allow_nil without allow_nan still rejects the NaN it creates" do
      s = Explorer.Series.from_list([1.0, nil])

      assert_raise ArgumentError, ~r/non-finite value \(NaN\) at index 1/, fn ->
        Input.to_f64_binary(s, allow_nil: true)
      end
    end

    test "NaN values follow allow_nan" do
      s = Explorer.Series.from_list([1.0, :nan])
      assert_raise ArgumentError, ~r/NaN\) at index 1/, fn -> Input.to_f64_binary(s) end
      assert {bin, 2} = Input.to_f64_binary(s, allow_nan: true)
      assert Input.from_f64_binary(bin) == [1.0, :nan]
    end

    test "non-numeric dtype" do
      assert_raise ArgumentError, ~r/x: expected a numeric series, got dtype :string/, fn ->
        Input.to_f64_binary(Explorer.Series.from_list(["a"]), name: "x")
      end
    end
  end

  describe "to_points_binary/1" do
    test "tuples" do
      assert {bin, 2} = Input.to_points_binary([{1, 2}, {3.5, 4}])
      assert Input.from_f64_binary(bin) == [1.0, 2.0, 3.5, 4.0]
    end

    test "two-element lists" do
      assert {bin, 2} = Input.to_points_binary([[1, 2], [3, 4]])
      assert Input.from_f64_binary(bin) == [1.0, 2.0, 3.0, 4.0]
    end

    test "empty" do
      assert {<<>>, 0} = Input.to_points_binary([])
    end

    test "bad point names index" do
      assert_raise ArgumentError, ~r/points: expected \{x, y\} at index 1, got \{1, 2, 3\}/, fn ->
        Input.to_points_binary([{1, 2}, {1, 2, 3}])
      end

      assert_raise ArgumentError, ~r/points: expected a number at index 0, got "a"/, fn ->
        Input.to_points_binary([{"a", 2}])
      end
    end

    test "{n, 2} tensor" do
      assert {bin, 2} = Input.to_points_binary(Nx.tensor([[1, 2], [3, 4]]))
      assert Input.from_f64_binary(bin) == [1.0, 2.0, 3.0, 4.0]
    end

    test "tensor with wrong shape" do
      assert_raise ArgumentError, ~r/expected a \{n, 2\} tensor/, fn ->
        Input.to_points_binary(Nx.tensor([[1, 2, 3]]))
      end
    end

    test "tensor with NaN rejected with point index" do
      assert_raise ArgumentError, ~r/non-finite value \(NaN\) at index 1/, fn ->
        Input.to_points_binary(Nx.tensor([[1.0, 2.0], [:nan, 1.0]], type: :f64))
      end
    end

    test "pair of lists and of series" do
      assert {bin, 2} = Input.to_points_binary({[1, 3], [2, 4]})
      assert Input.from_f64_binary(bin) == [1.0, 2.0, 3.0, 4.0]

      xs = Explorer.Series.from_list([1.0, 3.0])
      ys = Explorer.Series.from_list([2, 4])
      assert {bin, 2} = Input.to_points_binary({xs, ys})
      assert Input.from_f64_binary(bin) == [1.0, 2.0, 3.0, 4.0]
    end

    test "pair length mismatch" do
      assert_raise ArgumentError, ~r/y: expected length 2, got 3/, fn ->
        Input.to_points_binary({[1, 2], [1, 2, 3]})
      end
    end
  end

  describe "to_index_binary/2" do
    test "packs u32" do
      assert {bin, 3} = Input.to_index_binary([0, 5, 4_294_967_295], :u32)

      assert bin ==
               <<0::unsigned-native-32, 5::unsigned-native-32, 4_294_967_295::unsigned-native-32>>
    end

    test "negative, non-integer and too large raise with index" do
      assert_raise ArgumentError,
                   ~r/indices: expected a non-negative integer at index 1, got -1/,
                   fn ->
                     Input.to_index_binary([0, -1], :u32)
                   end

      assert_raise ArgumentError, ~r/at index 0, got 1.5/, fn ->
        Input.to_index_binary([1.5], :u32)
      end

      assert_raise ArgumentError, ~r/exceeds u32/, fn ->
        Input.to_index_binary([4_294_967_296], :u32)
      end
    end

    test "accepts integer tensors" do
      assert {bin, 2} = Input.to_index_binary(Nx.tensor([1, 2], type: :s64), :u32)
      assert bin == <<1::unsigned-native-32, 2::unsigned-native-32>>
    end
  end

  describe "from_f64_binary/1" do
    test "decodes special patterns as atoms" do
      bin =
        <<0x7FF8000000000000::unsigned-native-64, 0x7FF0000000000000::unsigned-native-64,
          0xFFF0000000000000::unsigned-native-64, 2.0::float-64-native>>

      assert Input.from_f64_binary(bin) == [:nan, :infinity, :neg_infinity, 2.0]
    end

    test "rejects misaligned binaries" do
      assert_raise ArgumentError, fn -> Input.from_f64_binary(<<1, 2, 3>>) end
    end
  end

  property "from_f64_binary(to_f64_binary(list)) == floats(list)" do
    check all(list <- list_of(one_of([float(), integer(-1_000_000..1_000_000)]), max_length: 50)) do
      {bin, n} = Input.to_f64_binary(list)
      assert n == length(list)
      assert Input.from_f64_binary(bin) == Enum.map(list, &(&1 * 1.0))
    end
  end

  property "points roundtrip interleaved" do
    check all(pts <- list_of({float(), float()}, max_length: 30)) do
      {bin, n} = Input.to_points_binary(pts)
      assert n == length(pts)
      assert Input.from_f64_binary(bin) == Enum.flat_map(pts, fn {x, y} -> [x, y] end)
    end
  end

  describe "to_labels/2" do
    test "lists, Nx vectors and Explorer series of any dtype become term lists" do
      assert Input.to_labels([:a, "b", 1]) == [:a, "b", 1]
      assert Input.to_labels(Nx.tensor([2, 0, 2])) == [2, 0, 2]
      assert Input.to_labels(Explorer.Series.from_list(["n", "s", "n"])) == ["n", "s", "n"]
      assert Input.to_labels(Explorer.Series.from_list([1, 2]), length: 2) == [1, 2]
    end

    test "rejects wrong lengths, missing values and other inputs, naming the argument" do
      assert_raise ArgumentError, ~r/regimes.*one label per observation \(3\)/, fn ->
        Input.to_labels([1, 2], name: "regimes", length: 3)
      end

      assert_raise ArgumentError, ~r/regimes: missing value \(nil\) at index 1/, fn ->
        Input.to_labels(Explorer.Series.from_list(["a", nil]), name: "regimes")
      end

      assert_raise ArgumentError, ~r/rank-1/, fn -> Input.to_labels(Nx.tensor([[1, 2]])) end
      assert_raise ArgumentError, ~r/labels: unsupported/, fn -> Input.to_labels(%{a: 1}) end
    end
  end

  describe "to_matrix/2" do
    test "rows of numbers, {n, t} tensors and data frames give float rows" do
      assert Input.to_matrix([[1, 2], [3, 4.5]]) == [[1.0, 2.0], [3.0, 4.5]]
      assert Input.to_matrix(Nx.tensor([[1, 2], [3, 4]])) == [[1.0, 2.0], [3.0, 4.0]]

      df = Explorer.DataFrame.new(t1: [1, 3], t2: [2.0, 4.0])
      assert Input.to_matrix(df, rows: 2) == [[1.0, 2.0], [3.0, 4.0]]
    end

    test "rejects ragged rows, wrong row counts and non-numbers" do
      assert_raise ArgumentError, ~r/y: rows must all hold the same number/, fn ->
        Input.to_matrix([[1, 2], [3]], name: "y")
      end

      assert_raise ArgumentError, ~r/y: expected 3 rows/, fn ->
        Input.to_matrix([[1, 2], [3, 4]], name: "y", rows: 3)
      end

      assert_raise ArgumentError, ~r/only numbers/, fn -> Input.to_matrix([[1, :a]]) end

      assert_raise ArgumentError, ~r/numeric/, fn ->
        Input.to_matrix(Explorer.DataFrame.new(a: ["x"], b: ["y"]))
      end

      assert_raise ArgumentError, ~r/rank-2/, fn -> Input.to_matrix(Nx.tensor([1, 2])) end
    end
  end
end
