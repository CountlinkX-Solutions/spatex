# Runs the code cells of Livebook notebooks, in order, in one shared binding and environment.
# Livebook has no headless runner, so this verifies that a notebook's code works.
#
#     elixir scripts/run_livemd.exs notebooks/tour.livemd [notebooks/pysal_comparison.livemd ...]
#
# Each notebook runs in its own fresh binding. Each cell's value is also passed through
# `Kino.Render.to_livebook/1` when Kino is loaded, which exercises the rendering a notebook
# would show. The first cell is expected to be `Mix.install/2`; a notebook can only call it once
# per VM, so with several notebooks each one is run in a separate `elixir` process.
# Adapted from GeoFrame's runner (Elixir cells only; Spatex notebooks need no Python).

defmodule RunLivemd do
  def cells(markdown) do
    ~r/^```elixir[^\n]*\n(.*?)^```\s*$/ms
    |> Regex.scan(markdown, capture: :all_but_first)
    |> Enum.map(fn [code] -> code end)
  end

  def run(path) do
    path = Path.expand(path)
    cells = path |> File.read!() |> cells()
    IO.puts("#{length(cells)} code cells in #{path}")

    state = %{binding: [], env: Code.env_for_eval(file: path)}

    cells
    |> Enum.with_index(1)
    |> Enum.reduce(state, fn {code, index}, state ->
      IO.puts("--- cell #{index}")
      {value, state} = eval(code, state, path)
      render(value)
      state
    end)

    IO.puts("OK: all cells evaluated")
  end

  defp eval(code, state, path) do
    quoted = Code.string_to_quoted!(code, file: path)
    {value, binding, env} = Code.eval_quoted_with_env(quoted, state.binding, state.env)
    {value, %{state | binding: binding, env: env}}
  end

  # Mimics the implicit output of a Livebook cell: renders the value when Kino is loaded.
  defp render(value) do
    if Code.ensure_loaded?(Kino.Render) do
      output = Kino.Render.to_livebook(value)
      IO.puts("  rendered: " <> preview(output))
    else
      IO.puts("  value: " <> preview(value))
    end
  end

  defp preview(term), do: term |> inspect(limit: 4, printable_limit: 60) |> String.slice(0, 110)
end

case System.argv() do
  [path] ->
    RunLivemd.run(path)

  [_ | _] = paths ->
    script = __ENV__.file

    for path <- paths do
      {_, status} = System.cmd("elixir", [script, path], into: IO.stream(), stderr_to_stdout: true)
      if status != 0, do: System.halt(status)
    end

  [] ->
    IO.puts(:stderr, "usage: elixir scripts/run_livemd.exs NOTEBOOK.livemd [...]")
    System.halt(1)
end
