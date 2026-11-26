defmodule Spatex.GWR.MGWR do
  @moduledoc false
  # Implementation of `Spatex.GWR.mgwr/4`: option handling, the n guard, standardization and
  # the conversion of the NIF output into a `Spatex.GWR.MGWRResult`.

  alias Spatex.GWR.{Design, MGWRResult}
  alias Spatex.{Input, Native}

  @criteria [:aicc, :aic, :bic, :cv]
  @default_max_n 5_000
  @default_chunk 256

  @allowed [
    :kernel,
    :fixed,
    :constant,
    :spherical,
    :max_n,
    :criterion,
    :search,
    :interval,
    :tol,
    :max_iter,
    :multi_bw_min,
    :multi_bw_max,
    :init_bw,
    :tol_multi,
    :max_iter_multi,
    :rss_score,
    :bws_same_times,
    :standardize,
    :inference,
    :chunk_size
  ]

  def run(points, y, x, opts) do
    opts = validate!(opts)
    model = Design.model!(points, y, x, opts, false)
    guard!(model, opts)
    {ybin, xbin} = if opts[:standardize], do: standardize(model, opts), else: {model.y, model.x}
    k = model.k
    {step, init_bw} = search!(opts)

    entries =
      model.coords
      |> Native.mgwr_fit(
        ybin,
        xbin,
        {Atom.to_string(opts[:kernel]), opts[:fixed], opts[:spherical], k},
        {Atom.to_string(opts[:criterion]), step, opts[:tol] * 1.0, opts[:max_iter]},
        {per_column!(opts, :multi_bw_min, k), per_column!(opts, :multi_bw_max, k), init_bw},
        {opts[:tol_multi] * 1.0, opts[:max_iter_multi], opts[:rss_score], opts[:bws_same_times]},
        {opts[:inference], opts[:chunk_size]}
      )
      |> unwrap()
      |> Map.new(fn {key, bin} -> {String.to_existing_atom(key), Input.from_f64_binary(bin)} end)

    build(entries, model, xbin, opts)
  end

  # -- options -------------------------------------------------------------------------

  defp validate!(opts) do
    Design.check_known!(opts, @allowed)

    opts =
      Keyword.merge(
        [
          kernel: :bisquare,
          fixed: false,
          constant: true,
          spherical: false,
          max_n: @default_max_n,
          criterion: :aicc,
          search: :golden_section,
          interval: nil,
          tol: 1.0e-6,
          max_iter: 200,
          multi_bw_min: nil,
          multi_bw_max: nil,
          init_bw: nil,
          tol_multi: 1.0e-5,
          max_iter_multi: 200,
          rss_score: false,
          bws_same_times: 5,
          standardize: false,
          inference: true,
          chunk_size: @default_chunk
        ],
        opts
      )

    Design.check_common!(opts)
    check_numbers!(opts)
    opts
  end

  defp check_numbers!(opts) do
    unless opts[:criterion] in @criteria do
      raise ArgumentError,
            "criterion must be one of #{inspect(@criteria)}, got #{inspect(opts[:criterion])}"
    end

    unless opts[:search] in [:golden_section, :interval] do
      raise ArgumentError, "search must be :golden_section or :interval"
    end

    for key <- [:rss_score, :standardize, :inference] do
      unless is_boolean(opts[key]), do: raise(ArgumentError, "#{key} must be a boolean")
    end

    check_tolerances!(opts)
    check_counts!(opts)
    check_optional_positive!(opts, [:init_bw, :interval])
  end

  defp check_tolerances!(opts) do
    unless is_number(opts[:tol]) and opts[:tol] >= 0, do: raise(ArgumentError, "tol must be >= 0")

    unless is_number(opts[:tol_multi]) and opts[:tol_multi] > 0,
      do: raise(ArgumentError, "tol_multi must be positive")
  end

  defp check_counts!(opts) do
    for key <- [:max_iter, :max_iter_multi, :chunk_size] do
      unless is_integer(opts[key]) and opts[key] >= 1,
        do: raise(ArgumentError, "#{key} must be a positive integer")
    end

    unless is_integer(opts[:bws_same_times]) and opts[:bws_same_times] >= 0,
      do: raise(ArgumentError, "bws_same_times must be a non-negative integer")
  end

  defp check_optional_positive!(opts, keys) do
    for key <- keys do
      v = opts[key]

      unless is_nil(v) or (is_number(v) and v > 0),
        do: raise(ArgumentError, "#{key} must be a positive number")
    end
  end

  defp guard!(model, opts) do
    if opts[:inference] and model.n > opts[:max_n] do
      raise ArgumentError,
            "n = #{model.n} exceeds max_n = #{opts[:max_n]}: MGWR inference builds one n x n hat " <>
              "matrix per covariate (O(iterations * k * n^3) time); pass a larger max_n " <>
              "deliberately or inference: false to skip standard errors and criteria"
    end
  end

  defp search!(opts) do
    case opts[:search] do
      :golden_section ->
        {0.0, opts[:init_bw] && opts[:init_bw] * 1.0}

      :interval ->
        unless opts[:interval], do: raise(ArgumentError, "the interval search needs :interval")

        unless opts[:init_bw] && opts[:multi_bw_min] && opts[:multi_bw_max] do
          raise ArgumentError,
                "the interval search needs :init_bw, :multi_bw_min and :multi_bw_max"
        end

        {opts[:interval] * 1.0, opts[:init_bw] * 1.0}
    end
  end

  # A number applies to every column; a list gives one value (or nil) per column.
  defp per_column!(opts, key, k) do
    case opts[key] do
      nil ->
        List.duplicate(nil, k)

      v when is_number(v) and v > 0 ->
        List.duplicate(v * 1.0, k)

      list when is_list(list) and length(list) == k ->
        Enum.map(list, fn
          nil ->
            nil

          v when is_number(v) and v > 0 ->
            v * 1.0

          other ->
            raise ArgumentError, "#{key} entries must be positive numbers, got #{inspect(other)}"
        end)

      other ->
        raise ArgumentError,
              "#{key} must be a positive number or a list of #{k} numbers or nils (one per " <>
                "column, intercept first), got #{inspect(other, limit: 4)}"
    end
  end

  # -- standardization -----------------------------------------------------------------

  defp standardize(model, opts) do
    ybin = standardize_column(model.y, "y")
    n = model.n
    columns = for <<col::binary-size(n * 8) <- model.x>>, do: col
    skip = if opts[:constant], do: 1, else: 0
    {head, tail} = Enum.split(columns, skip)

    tail =
      tail
      |> Enum.with_index(skip + 1)
      |> Enum.map(fn {col, i} -> standardize_column(col, "x column #{i}") end)

    {ybin, IO.iodata_to_binary(head ++ tail)}
  end

  defp standardize_column(bin, name) do
    values = for <<v::float-64-native <- bin>>, do: v
    n = length(values)
    mean = Enum.sum(values) / n
    sd = :math.sqrt(Enum.sum(Enum.map(values, &((&1 - mean) * (&1 - mean)))) / n)

    if sd == 0.0, do: raise(ArgumentError, "cannot standardize #{name}: it is constant")

    IO.iodata_to_binary(for v <- values, do: <<(v - mean) / sd::float-64-native>>)
  end

  # -- output --------------------------------------------------------------------------

  defp unwrap({:ok, value}), do: value
  defp unwrap({:error, reason}), do: raise(ArgumentError, to_string(reason))

  defp build(e, model, xbin, opts) do
    k = model.k
    history = e.bws_history |> Enum.chunk_every(k) |> Enum.map(&bandwidths(&1, opts))
    converged = hd(e.converged) == 1.0
    inference? = Map.has_key?(e, :enp_j)

    %MGWRResult{
      n: model.n,
      k: k,
      kernel: opts[:kernel],
      fixed: opts[:fixed],
      spherical: opts[:spherical],
      constant: opts[:constant],
      bws: bandwidths(e.bws, opts),
      bw_init: hd(bandwidths(e.bw_init, opts)),
      bws_history: history,
      sel_hist: sel_hist(e.sel_hist, opts),
      scores: e.scores,
      iterations: length(history),
      converged: converged,
      params: rows(e.params, k),
      predy: e.predy,
      resid: e.resid,
      resid_ss: hd(e.resid_ss),
      r2: if(inference?, do: hd(e.r2), else: r2(model, e)),
      warnings: model.warnings ++ warnings(converged, history, xbin, model, opts)
    }
    |> put_inference(e, inference?, k)
  end

  defp put_inference(result, _e, false, _k), do: result

  defp put_inference(result, e, true, k) do
    %{
      result
      | bse: rows(e.bse, k),
        tvalues: rows(e.tvalues, k),
        cct: rows(e.cct, k),
        enp_j: e.enp_j,
        tr_s: hd(e.tr_s),
        sigma2: hd(e.sigma2),
        llf: hd(e.llf),
        aic: hd(e.aic),
        aicc: hd(e.aicc),
        bic: hd(e.bic),
        adj_r2: hd(e.adj_r2),
        adj_alpha_j: rows(e.adj_alpha_j, 3),
        critical_tval: e.critical_tval
    }
  end

  # R-squared without inference: 1 - RSS / TSS.
  defp r2(model, e) do
    ys = for <<v::float-64-native <- model.y>>, do: v
    mean = Enum.sum(ys) / model.n
    tss = ys |> Enum.map(&((&1 - mean) * (&1 - mean))) |> Enum.sum()
    1.0 - hd(e.resid_ss) / tss
  end

  # `[count, bw0, score0, ..., count, ...]` back into one `{bw, score}` list per search.
  defp sel_hist([], _opts), do: []

  defp sel_hist([count | rest], opts) do
    {flat, rest} = Enum.split(rest, 2 * trunc(count))

    pairs =
      flat
      |> Enum.chunk_every(2)
      |> Enum.map(fn [bw, score] -> {hd(bandwidths(bw, opts)), score} end)

    [pairs | sel_hist(rest, opts)]
  end

  defp bandwidths(values, opts) do
    if opts[:fixed], do: List.wrap(values), else: adaptive(values)
  end

  defp adaptive(values),
    do: values |> List.wrap() |> Enum.map(fn v -> if is_float(v), do: trunc(v), else: v end)

  defp warnings(converged, history, xbin, model, opts) do
    not_converged =
      if converged,
        do: [],
        else: [
          "backfitting did not converge within #{length(history)} iteration(s) " <>
            "(max_iter_multi = #{opts[:max_iter_multi]}); the bandwidths may not be final"
        ]

    not_converged ++ if(opts[:inference], do: zero_warning(xbin, model.n), else: [])
  end

  defp zero_warning(xbin, n) do
    columns = for <<col::binary-size(n * 8) <- xbin>>, do: col

    zero_columns =
      for {col, j} <- Enum.with_index(columns), has_zero?(col), do: j

    case zero_columns do
      [] ->
        []

      cols ->
        [
          "column(s) #{inspect(cols)} contain exact zeros; mgwr divides by the regressor " <>
            "to get standard errors, so bse and tvalues are non-finite where it is zero"
        ]
    end
  end

  defp has_zero?(bin) do
    Enum.any?(for(<<v::float-64-native <- bin>>, do: v), &(&1 == 0.0))
  end

  defp rows(values, k), do: Enum.chunk_every(values, k)
end
