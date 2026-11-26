defmodule Spatex.MixProject do
  use Mix.Project

  @version "0.2.1"
  @source_url "https://github.com/CountlinkX-Solutions/spatex"

  def project do
    [
      app: :spatex,
      version: @version,
      elixir: "~> 1.17",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: "PySAL-style spatial analysis for Elixir, backed by a Rust core.",
      name: "Spatex",
      source_url: @source_url,
      package: package(),
      docs: docs(),
      # known_deviations.exs is data read by the oracle tests, not a test file.
      test_ignore_filters: [&String.ends_with?(&1, "known_deviations.exs")]
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp docs do
    [
      main: "readme",
      source_ref: "v#{@version}",
      extras: [
        "README.md",
        "CHANGELOG.md": [title: "Changelog"],
        "docs/benchmarks.md": [title: "Benchmarks"],
        "docs/guide_esda.md": [title: "ESDA guide"],
        "notebooks/tour.livemd": [title: "Tour"],
        "notebooks/geoframe.livemd": [title: "GeoFrame"],
        "notebooks/nc_sids_case.livemd": [title: "Use case: NC SIDS"],
        "notebooks/us_income_case.livemd": [title: "Use case: US income"],
        "notebooks/baltimore_case.livemd": [title: "Use case: Baltimore housing"],
        "notebooks/georgia_case.livemd": [title: "Use case: Georgia GWR"],
        "notebooks/snow_case.livemd": [title: "Use case: John Snow's cholera map"],
        "notebooks/mexico_case.livemd": [title: "Use case: Mexico maps and regions"],
        "notebooks/es/caso_sids_carolina_del_norte.livemd": [title: "Caso de uso: SIDS (es)"],
        "notebooks/es/caso_ingresos_eeuu.livemd": [title: "Caso de uso: renta EE. UU. (es)"],
        "notebooks/es/caso_vivienda_baltimore.livemd": [
          title: "Caso de uso: vivienda Baltimore (es)"
        ],
        "notebooks/es/caso_educacion_georgia.livemd": [
          title: "Caso de uso: educación Georgia (es)"
        ],
        "notebooks/es/caso_colera_snow.livemd": [title: "Caso de uso: cólera de Snow (es)"],
        "notebooks/es/caso_regiones_mexico.livemd": [title: "Caso de uso: regiones México (es)"],
        "notebooks/pysal_comparison.livemd": [title: "PySAL comparison"],
        "docs/architecture.md": [title: "Architecture"],
        "docs/parity.md": [title: "PySAL parity"],
        "docs/research.md": [title: "Research"],
        "docs/roadmap.md": [title: "Roadmap"],
        LICENSE: [title: "License"],
        NOTICE: [title: "Notice"]
      ],
      # Spatex.Native is internal (@moduledoc false) but the architecture doc names it.
      skip_undefined_reference_warnings_on: ["docs/architecture.md"],
      groups_for_extras: [
        Guides: ~r{docs/},
        Notebooks: ~r{notebooks/},
        Legal: ["LICENSE", "NOTICE"]
      ]
    ]
  end

  # Ship the NIF sources, not the build output in `priv/native`. `.cargo/config.toml` holds the
  # target-specific link settings a source build needs on macOS and musl.
  defp package do
    [
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => @source_url,
        "Changelog" => @source_url <> "/blob/main/CHANGELOG.md",
        "Docs" => "https://hexdocs.pm/spatex"
      },
      files:
        ~w(lib native/spatex_core/src native/spatex_core/Cargo.toml
           native/spatex_native/src native/spatex_native/Cargo.toml
           Cargo.toml Cargo.lock .cargo/config.toml mix.exs docs notebooks
           README.md CHANGELOG.md LICENSE LICENSES NOTICE) ++
          checksum_files()
    ]
  end

  # The checksum file is generated from the published release artifacts right before
  # `mix hex.build` (README, "Releasing") and is git-ignored, so it only ever ships inside the Hex
  # package. Hex rejects a glob that matches nothing, so it is listed only once it exists.
  defp checksum_files, do: Path.wildcard("checksum-*.exs")

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      # Only needed to build the NIF from source (a checkout, or SPATEX_BUILD=1), so it is optional,
      # as rustler_precompiled recommends: Hex users of the precompiled NIF do not fetch it.
      # The crate is rustler 0.39 (native/spatex_native/Cargo.toml; 0.38's OwnedBinary API does not
      # compile with our NIFs). Hex only has the Mix package 0.38.0, which builds the 0.39 crate
      # fine (CI proves it); raise this to "~> 0.39" once that is published.
      {:rustler, "~> 0.38", optional: true, runtime: false},
      # Downloads the precompiled NIF of a release, verified against the packaged checksum file.
      {:rustler_precompiled, "~> 0.8"},
      {:nx, "~> 1.0", optional: true},
      {:explorer, "~> 0.13", optional: true},
      {:stream_data, "~> 1.4", only: [:dev, :test]},
      # The library does not use Jason; the oracle-test helper and the benchmarks decode JSON with
      # it. Rustler needs it to build the NIF, so it cannot be limited to :dev/:test; optional
      # keeps it out of the dependency tree of Hex users.
      {:jason, "~> 1.4", optional: true},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      # Also in :test so CI can run `mix docs` in its single MIX_ENV (and reuse the test build).
      {:ex_doc, "~> 0.40", only: [:dev, :test], runtime: false},
      {:benchee, "~> 1.3", only: :dev, runtime: false}
    ] ++ geoframe_dep()
  end

  # GeoFrame is not on Hex yet, so the lock never lists it and CI does not fetch it: the dependency
  # exists only when SPATEX_GEOFRAME_PATH points at a local checkout. Spatex.GeoFrame calls it at
  # runtime, so nothing here is needed to compile Spatex.
  defp geoframe_dep do
    case System.get_env("SPATEX_GEOFRAME_PATH") do
      path when path in [nil, ""] ->
        []

      path ->
        [{:geoframe, path: path, only: :test, optional: true}]
    end
  end
end
