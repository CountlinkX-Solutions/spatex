defmodule Spatex.GWR.Intervals do
  @moduledoc false
  # Implementation of `Spatex.GWR.bandwidth_intervals/2`: mgwr's `get_bws_intervals`, a
  # confidence interval for the bandwidth from the Akaike weights of the search history
  # (Li et al., 2020).

  alias Spatex.GWR.{Bandwidth, Design, MGWRResult}

  def run(source, opts) do
    Design.check_known!(opts, [:level])
    level = Keyword.get(opts, :level, 0.95)

    unless is_number(level) and level > 0 and level <= 1 do
      raise ArgumentError, "level must be a number in (0, 1], got #{inspect(level)}"
    end

    case source do
      %Bandwidth{history: history} ->
        interval(history, level)

      %MGWRResult{k: k, sel_hist: sel_hist} when is_list(sel_hist) ->
        # mgwr reads `sel_hist[-k + j]`: the searches of the last backfitting iteration that
        # searched at all (the last `k` of them).
        if length(sel_hist) < k do
          raise ArgumentError,
                "the MGWR result has #{length(sel_hist)} search histories, fewer than its #{k} covariates"
        end

        sel_hist |> Enum.take(-k) |> Enum.map(&interval(&1, level))

      other ->
        raise ArgumentError,
              "expected a Spatex.GWR.Bandwidth or Spatex.GWR.MGWRResult, got #{inspect(other)}"
    end
  end

  defp interval([], _level) do
    raise ArgumentError,
          "the search history is empty (search: :scipy keeps none, as mgwr's sel_hist); " <>
            "use a golden-section or interval search"
  end

  defp interval(history, level) do
    # pandas `sort_values` by AICc; ties keep their search order.
    sorted = Enum.sort_by(history, fn {_bw, score} -> score end)
    best = sorted |> hd() |> elem(1)
    raw = for {_bw, score} <- sorted, do: :math.exp(-0.5 * (score - best))
    total = Enum.sum(raw)
    weights = Enum.map(raw, &(&1 / total))
    scale = Enum.sum(weights)

    cumulative =
      weights |> Enum.map(&(&1 / scale)) |> Enum.scan(&(&1 + &2))

    # `len(cum_w < level) + 1` entries (never more than there are) bound the interval.
    keep = min(Enum.count(cumulative, &(&1 < level)) + 1, length(sorted))
    bws = sorted |> Enum.take(keep) |> Enum.map(&elem(&1, 0))
    {Enum.min(bws), Enum.max(bws)}
  end
end
