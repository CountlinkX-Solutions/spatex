defmodule SpatexTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  doctest Spatex

  describe "native smoke" do
    test "the NIF library loads and reports its ABI version" do
      assert Spatex.native_version() == 1
    end

    test "sum_f64 adds a native-endian f64 binary" do
      assert Spatex.sum_f64([1.0, 2.5, -0.5]) == 3.0
      assert Spatex.sum_f64([]) == 0.0
    end

    test "sum_f64 rejects binaries whose length is not a multiple of 8" do
      assert {:error, msg} = Spatex.Native.sum_f64(<<1, 2, 3>>)
      assert msg =~ "multiple of 8"
    end

    property "sum_f64 matches Enum.sum on finite floats" do
      check all(values <- list_of(float(min: -1.0e6, max: 1.0e6), max_length: 50)) do
        assert_in_delta Spatex.sum_f64(values), Enum.sum(values), 1.0e-6
      end
    end
  end
end
