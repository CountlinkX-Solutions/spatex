# GeoFrame is not on Hex yet and is an optional test-only dependency (see mix.exs); its
# integration tests run only when SPATEX_GEOFRAME_PATH points at a local checkout.
unless Code.ensure_loaded?(GeoFrame.GeoDataFrame), do: ExUnit.configure(exclude: [:geoframe])

ExUnit.start()
