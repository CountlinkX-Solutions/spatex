# Contributing to Spatex

Thank you for helping. Bug reports, PySAL functions to port, fixes and documentation are all
welcome. Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md); report
vulnerabilities privately as described in [SECURITY.md](SECURITY.md), not in an issue.

## Development setup

- Erlang/OTP 28.5 and Elixir 1.19.5 (`.tool-versions`, usable with asdf or mise). CI also tests
  Elixir 1.17 with OTP 27, the oldest supported pair.
- Rust stable, 1.91 or newer (`rust-version` in `Cargo.toml`), with `rustfmt` and `clippy`. A
  checkout always compiles the NIF from source: the precompiled NIF is only used by the Hex
  package.

```
mix deps.get
mix test
```

The first build compiles the Rust crates in release mode and takes a while. On a machine with
little memory, limit Cargo's parallelism with `CARGO_BUILD_JOBS=1`.

## Checks

CI runs these with `MIX_ENV=test`; run them before opening a pull request:

```
mix format --check-formatted
mix compile --warnings-as-errors
mix credo --strict
mix test
mix docs --warnings-as-errors
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

CI also runs the code cells of the offline notebooks:

```
MIX_ENV=dev elixir scripts/run_livemd.exs notebooks/tour.livemd notebooks/pysal_comparison.livemd
```

The GeoFrame integration tests and notebooks need a local GeoFrame checkout (GeoFrame is not on
Hex yet) and are skipped otherwise:

```
SPATEX_GEOFRAME_PATH=/path/to/geoframe mix test --only geoframe
```

## Tests and oracle fixtures

Spatex is developed test first: a change in behavior starts with a failing test. Results are
checked against PySAL through JSON fixtures in `test/reference/fixtures`, produced by the Python
generators in `test/reference/*.py` and committed, so `mix test` and CI need no Python. Oracle
tests carry `@moduletag :oracle` (`mix test --only oracle`).

- To add or change a PySAL comparison, extend the matching generator and regenerate its fixture
  with the pinned package versions listed in [test/reference/README.md](test/reference/README.md).
  Commit the regenerated JSON together with the reason.
- Record every accepted difference from PySAL in `test/reference/known_deviations.exs` and in
  [docs/parity.md](docs/parity.md).
- New PySAL functionality also gets a row in `docs/parity.md`.

## Commits and pull requests

- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org) with a scope,
  for example `feat(esda): add local join counts` or `fix(regression): correct the HAC weights`.
  Scopes: `graph`, `esda`, `classify`, `pointpats`, `regression`, `gwr`, `dynamics`, `inequality`,
  `region`, `geoframe`, `native`, `ci`, `deps`, `docs`.
- Each commit carries its tests and documentation.
- Open pull requests against `main`. CI must pass, and user-visible changes need an entry under
  `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md).
