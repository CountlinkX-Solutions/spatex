defmodule Spatex.ReferenceHelperTest do
  use ExUnit.Case, async: true

  alias Spatex.Reference

  describe "assert_close/3" do
    test "accepts equal numbers, mixed ints and floats, within tolerance" do
      assert Reference.assert_close(1.0, 1.0, 1.0e-12) == :ok
      assert Reference.assert_close(1, 1.0, 1.0e-12) == :ok
      assert Reference.assert_close(1.0 + 1.0e-13, 1.0, 1.0e-12) == :ok
    end

    test "applies the tolerance relative to large expected values" do
      assert Reference.assert_close(1.0e9 + 1.0e-4, 1.0e9, 1.0e-12) == :ok
    end

    test "walks lists and nested maps and treats markers as exact" do
      expected = %{"a" => [1.0, 2.0, %{"b" => :nan}], "c" => :inf, "d" => "text"}
      actual = %{"a" => [1.0, 2.0 + 1.0e-14, %{"b" => :nan}], "c" => :inf, "d" => "text"}
      assert Reference.assert_close(actual, expected, 1.0e-12) == :ok
    end

    test "reports the path of the mismatching element" do
      expected = %{"transforms" => %{"R" => %{"lag" => [1.0, 2.0, 3.0]}}}
      actual = %{"transforms" => %{"R" => %{"lag" => [1.0, 2.5, 3.0]}}}

      error =
        assert_raise ExUnit.AssertionError, fn ->
          Reference.assert_close(actual, expected, 1.0e-12)
        end

      assert error.message =~ ~s(transforms.R.lag[1])
      assert error.message =~ "2.5"
      assert error.message =~ "2.0"
    end

    test "reports length and key mismatches with their path" do
      error =
        assert_raise ExUnit.AssertionError, fn ->
          Reference.assert_close(%{"x" => [1.0]}, %{"x" => [1.0, 2.0]}, 1.0e-12)
        end

      assert error.message =~ "x"
      assert error.message =~ "length"

      error =
        assert_raise ExUnit.AssertionError, fn ->
          Reference.assert_close(%{"x" => 1.0}, %{"y" => 1.0}, 1.0e-12)
        end

      assert error.message =~ "keys"
    end

    test "nan only matches nan" do
      assert_raise ExUnit.AssertionError, fn -> Reference.assert_close(1.0, :nan, 1.0e-12) end

      assert_raise ExUnit.AssertionError, fn ->
        Reference.assert_close(:inf, :neg_inf, 1.0e-12)
      end
    end
  end

  describe "load!/1 and fixture_path/1" do
    test "fixture_path resolves under test/reference/fixtures" do
      assert Reference.fixture_path("graph/x.json") =~ "test/reference/fixtures/graph/x.json"
    end

    test "load! raises on a missing fixture" do
      assert_raise File.Error, fn -> Reference.load!("graph/does_not_exist.json") end
    end
  end
end
