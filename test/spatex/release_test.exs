defmodule Spatex.ReleaseTest do
  use ExUnit.Case, async: true

  @moduledoc false

  @root Path.expand("../..", __DIR__)

  describe "NIF loading" do
    test "the NIF library loads on the build-from-source path" do
      assert Spatex.Native.nif_version() == 1
    end

    test "the compiled library is named after the single cdylib, spatex_native" do
      libs = Path.wildcard(Path.join(@root, "priv/native/*spatex_native*"))
      assert libs != []
      assert Enum.all?(libs, &(Path.basename(&1) =~ ~r/^(lib)?spatex_native\.(so|dll|dylib)$/))
    end

    test "rustler_precompiled is a runtime dependency" do
      assert {:ok, _} = Application.ensure_all_started(:rustler_precompiled)
      assert Code.ensure_loaded?(RustlerPrecompiled)
    end
  end

  describe "Hex package" do
    setup do
      {:ok, package: Mix.Project.config()[:package]}
    end

    test "carries the metadata Hex requires", %{package: package} do
      assert package[:licenses] == ["Apache-2.0"]
      assert Map.has_key?(package[:links], "GitHub")
      assert Map.has_key?(package[:links], "Changelog")
    end

    test "ships the native sources, lockfile, changelog and legal files", %{
      package: package
    } do
      files = package[:files]

      for entry <-
            ~w(lib native/spatex_core/src native/spatex_native/src Cargo.toml Cargo.lock
               .cargo/config.toml CHANGELOG.md LICENSE LICENSES NOTICE README.md mix.exs) do
        assert entry in files, "#{entry} missing from package files"
      end

      refute Enum.any?(files, &String.contains?(&1, "priv"))
    end

    test "every package entry exists, so mix hex.build cannot report missing files", %{
      package: package
    } do
      for entry <- package[:files] do
        assert File.exists?(Path.join(@root, entry)), "#{entry} does not exist"
      end
    end
  end

  describe "checksum file" do
    test "is packaged exactly when it exists" do
      packaged = Enum.filter(Mix.Project.config()[:package][:files], &(&1 =~ ~r/^checksum-/))

      assert packaged ==
               Path.wildcard(Path.join(@root, "checksum-*.exs")) |> Enum.map(&Path.basename/1)
    end

    test "is git-ignored, so a checkout always builds the NIF from source" do
      assert File.read!(Path.join(@root, ".gitignore")) =~ ~r/^checksum-\*\.exs$/m
    end
  end

  describe "dependencies" do
    setup do
      {:ok, deps: Map.new(Mix.Project.config()[:deps], &{elem(&1, 0), &1})}
    end

    test "rustler is optional, so Hex users of the precompiled NIF do not fetch it", %{
      deps: deps
    } do
      assert {:rustler, _requirement, opts} = deps[:rustler]
      assert opts[:optional] == true
      assert opts[:runtime] == false
    end

    test "rustler_precompiled is a required dependency", %{deps: deps} do
      assert {:rustler_precompiled, _requirement} = deps[:rustler_precompiled]
    end

    # Rustler needs Jason to build the NIF, so a checkout has it in every environment and an
    # `:only` restriction would diverge from Rustler's requirement; optional keeps it out of the
    # dependency tree of Hex users.
    test "jason is optional, so Hex users do not fetch it", %{deps: deps} do
      assert {:jason, _requirement, opts} = deps[:jason]
      assert opts[:optional] == true
      refute Keyword.has_key?(opts, :only)
    end

    test "the library itself does not use Jason" do
      for file <- Path.wildcard(Path.join(@root, "lib/**/*.ex")) do
        refute File.read!(file) =~ "Jason.", "#{file} uses Jason"
      end
    end
  end

  describe "docs" do
    test "link the sources of the release tag" do
      docs = Mix.Project.config()[:docs]
      assert docs[:source_ref] == "v" <> Mix.Project.config()[:version]
    end
  end

  describe "Cargo workspace" do
    test "declares a rust-version that every member inherits" do
      assert File.read!(Path.join(@root, "Cargo.toml")) =~ ~r/^rust-version = "1\.\d+"/m

      for crate <- ~w(spatex_core spatex_native) do
        manifest = File.read!(Path.join(@root, "native/#{crate}/Cargo.toml"))
        assert manifest =~ "rust-version.workspace = true"
      end
    end
  end

  describe "changelog" do
    setup do
      {:ok, changelog: File.read!(Path.join(@root, "CHANGELOG.md"))}
    end

    test "has an Unreleased section above the 0.1.0 release section", %{changelog: changelog} do
      {unreleased, _} = :binary.match(changelog, "## [Unreleased]")
      {released, _} = :binary.match(changelog, "## [0.1.0] - ")
      assert unreleased < released
    end

    test "the mix.exs version is a non-pre-release one with a matching changelog section", %{
      changelog: changelog
    } do
      version = Mix.Project.config()[:version]
      assert version =~ ~r/^\d+\.\d+\.\d+$/
      assert changelog =~ "## [#{version}] - "
    end
  end

  describe "release workflow and NIF features" do
    setup do
      native = File.read!(Path.join(@root, "lib/spatex/native.ex"))
      workflow = File.read!(Path.join(@root, ".github/workflows/release.yml"))
      cargo = File.read!(Path.join(@root, "native/spatex_native/Cargo.toml"))

      [declared_nifs] = Regex.run(~r/nif_versions: \[([^\]]*)\]/, native, capture: :all_but_first)
      [declared_targets] = Regex.run(~r/targets: ~w\(([^)]*)\)/s, native, capture: :all_but_first)

      {:ok,
       workflow: workflow,
       cargo: cargo,
       nifs: Regex.scan(~r/\d+\.\d+/, declared_nifs) |> List.flatten() |> Enum.sort(),
       targets: declared_targets |> String.split() |> Enum.sort()}
    end

    test "Cargo.toml declares a rustler nif_version feature for every precompiled NIF version", %{
      cargo: cargo,
      nifs: nifs
    } do
      assert nifs != []

      for nif <- nifs do
        feature = "nif_version_" <> String.replace(nif, ".", "_")

        assert cargo =~ ~s(#{feature} = ["rustler/#{feature}"]),
               "#{feature} feature missing from native/spatex_native/Cargo.toml"
      end
    end

    test "the release matrix builds exactly the NIF versions and targets of Spatex.Native", %{
      workflow: workflow,
      nifs: nifs,
      targets: targets
    } do
      [matrix_nifs] = Regex.run(~r/^\s+nif: \[([^\]]*)\]/m, workflow, capture: :all_but_first)
      matrix_targets = Regex.scan(~r/target: ([\w-]+),/, workflow, capture: :all_but_first)

      assert matrix_nifs |> then(&Regex.scan(~r/\d+\.\d+/, &1)) |> List.flatten() |> Enum.sort() ==
               nifs

      assert matrix_targets |> List.flatten() |> Enum.sort() == targets
    end

    test "the workflow does not coerce use-cross with || false", %{workflow: workflow} do
      assert workflow =~ "use-cross: ${{ matrix.job.use-cross }}"
      refute workflow =~ ~r/use-cross:[^\n]*\|\|/
    end
  end
end
