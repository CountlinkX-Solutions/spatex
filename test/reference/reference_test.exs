defmodule Spatex.ReferenceTest do
  use ExUnit.Case, async: true

  @moduletag :oracle

  alias Spatex.Reference

  @graphs ~w(columbus_queen lattice4x4_rook asymmetric_island)
  @transforms ~w(O R B D V)

  for name <- @graphs do
    describe "graph fixture #{name}" do
      setup do
        {:ok, fx: Reference.load!("graph/#{unquote(name)}.json")}
      end

      test "records library versions", %{fx: fx} do
        for key <- ~w(libpysal esda numpy), do: assert(is_binary(fx["meta"][key]))
        assert fx["meta"]["libpysal"] == "4.15.1"
      end

      test "array lengths agree with n", %{fx: fx} do
        n = fx["n"]
        assert length(fx["ids"]) == n
        assert length(fx["cardinalities"]) == n
        assert length(fx["variable"]["values"]) == n
        assert is_boolean(fx["symmetric"])
        assert Enum.all?(fx["isolates"], &(&1 in 0..(n - 1)))
      end

      test "every transform is internally consistent", %{fx: fx} do
        n = fx["n"]
        assert Map.keys(fx["transforms"]) |> Enum.sort() == Enum.sort(@transforms)

        for {t, tr} <- fx["transforms"] do
          %{"focal" => f, "neighbor" => nb, "weight" => w} = tr["adjacency"]
          assert length(f) == length(nb) and length(nb) == length(w), "#{t} adjacency lengths"
          assert Enum.all?(f ++ nb, &(&1 in 0..(n - 1))), "#{t} positions in range"
          assert length(tr["lag"]) == n
          # s0 is the sum of all weights.
          Reference.assert_close(tr["s0"], Enum.sum(w), 1.0e-12)
        end
      end

      test "row-standardised rows sum to one except isolates", %{fx: fx} do
        %{"focal" => f, "weight" => w} = fx["transforms"]["R"]["adjacency"]
        sums = f |> Enum.zip(w) |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))

        for {i, ws} <- sums do
          expected = if i in fx["isolates"], do: 0.0, else: 1.0
          Reference.assert_close(Enum.sum(ws), expected, 1.0e-12)
        end
      end
    end
  end

  describe "graph fixture contents" do
    test "columbus queen contiguity" do
      fx = Reference.load!("graph/columbus_queen.json")
      assert fx["n"] == 49
      assert length(fx["transforms"]["O"]["adjacency"]["focal"]) == 236
      assert fx["symmetric"] == true
      assert fx["isolates"] == []
      Reference.assert_close(fx["transforms"]["O"]["s0"], 236.0, 1.0e-12)
    end

    test "4x4 rook lattice has 48 directed edges" do
      fx = Reference.load!("graph/lattice4x4_rook.json")
      assert fx["n"] == 16
      assert length(fx["transforms"]["O"]["adjacency"]["focal"]) == 48
    end

    test "asymmetric graph keeps its island as a zero-weight self-loop" do
      fx = Reference.load!("graph/asymmetric_island.json")
      assert fx["symmetric"] == false
      assert [island] = fx["isolates"]
      %{"focal" => f, "neighbor" => nb, "weight" => w} = fx["transforms"]["O"]["adjacency"]
      assert {island, island, 0.0} in Enum.zip([f, nb, w])
    end
  end

  describe "columbus dataset fixture" do
    test "has 49 polygons as WKB hex and the attribute columns" do
      ds = Reference.load!("datasets/columbus.json")
      assert ds["n"] == 49
      assert length(ds["wkb_hex"]) == 49
      assert Enum.all?(ds["wkb_hex"], &(&1 =~ ~r/^[0-9a-f]+$/))
      for col <- ~w(CRIME INC HOVAL), do: assert(length(ds["attributes"][col]) == 49)
    end

    test "the graph fixture lag input is the dataset CRIME column" do
      ds = Reference.load!("datasets/columbus.json")
      fx = Reference.load!("graph/columbus_queen.json")
      assert fx["variable"]["name"] == "CRIME"
      Reference.assert_close(fx["variable"]["values"], ds["attributes"]["CRIME"], 0.0)
    end
  end
end
