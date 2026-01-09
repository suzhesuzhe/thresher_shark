

'''

  
bokeh serve --show klines_tail_app.py --args \
    --minibar DATA/organized/BTCUSDT/klines_1m_with_events.csv \
    --agg-trades DATA/organized/BTCUSDT/aggTrades.parquet \
    --value-column first_bool_colname,second_bool_colname \
    --indicator indicator_column_name

e.g
bokeh serve --show klines_tail_app.py --args \
    --minibar DATA/klines_1m_with_events.csv \
    --agg-trades DATA/organized/BTCUSDT/aggTrades.parquet \
    --value-column polynomial2_coeff_bool,polynomial2_coeff_bool2 \
    --indicator ema20,ema50

    
'''

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from bokeh.events import Tap
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    BoxAnnotation,
    Button,
    CheckboxGroup,
    ColumnDataSource,
    DataRange1d,
    Div,
    HoverTool,
    Range1d,
    TextInput,
)
from bokeh.plotting import figure

MINUTE_DELTA = pd.Timedelta(minutes=1)
MINUTE_HALF = MINUTE_DELTA / 2
BAR_WIDTH = MINUTE_DELTA * 0.8
BAR_WIDTH_MS = int(BAR_WIDTH.total_seconds() * 1000)
SELECTION_MARKERS = [
    "circle",
    "triangle",
    "square",
    "diamond",
    "inverted_triangle",
    "cross",
    "x",
    "asterisk",
]
SELECTION_COLORS = [
    "#FFD700",
    "#26A69A",
    "#EF5350",
    "#5C6BC0",
    "#FF7043",
    "#8D6E63",
    "#26C6DA",
    "#7CB342",
]
INDICATOR_COLORS = ["#2196F3", "#8E24AA", "#00897B", "#F9A825"]


def _marker_svg(marker: str, color: str) -> str:
    if marker == "circle":
        shape = f"<circle cx='6' cy='6' r='4.5' fill='{color}' />"
    elif marker == "triangle":
        shape = f"<polygon points='6,1.5 10.5,10.5 1.5,10.5' fill='{color}' />"
    else:
        shape = f"<rect x='2' y='2' width='8' height='8' fill='{color}' />"
    return f"<svg width='12' height='12' style='vertical-align:middle'>{shape}</svg>"


def prepare_minibar_dataset(
    minibar: pd.DataFrame,
    *,
    value_columns: list[str],
    indicator_columns: list[str],
    require_volume: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]], list[int]]:
    required_cols = {"open", "high", "low", "close"}
    volume_cols = {"buy_volume", "sell_volume"}
    if require_volume:
        required_cols |= volume_cols
    if not required_cols.issubset(minibar.columns):
        missing = required_cols - set(minibar.columns)
        raise ValueError(f"minibar is missing columns: {missing}")
    for col in value_columns:
        if col not in minibar.columns:
            raise ValueError(f"{col} column not found.")
    for col in indicator_columns:
        if col not in minibar.columns:
            raise ValueError(f"{col} column not found.")

    drop_columns = required_cols if require_volume else {"open", "high", "low", "close"}
    base = minibar.sort_index().dropna(subset=list(drop_columns)).copy()
    if not require_volume:
        for col in volume_cols:
            if col not in base.columns:
                base[col] = 0.0
    base = base.rename_axis("ts").reset_index()
    if base.empty:
        raise ValueError("Not enough data to plot.")

    base["ts_right"] = base["ts"] + MINUTE_DELTA
    base["ts_center"] = base["ts"] + MINUTE_HALF
    base["mid_price"] = (base["open"] + base["close"]) / 2

    selection_meta: list[dict[str, object]] = []
    combined_mask = pd.Series(False, index=base.index)
    total_selections = max(1, len(value_columns))
    for idx, col in enumerate(value_columns):
        mask = base[col].fillna(False).astype(bool)
        size_field = f"selection_{idx}_size"
        alpha_field = f"selection_{idx}_alpha"
        base[size_field] = np.where(mask, 9, 0)
        base[alpha_field] = 0.8 * mask.to_numpy()
        combined_mask = combined_mask | mask
        selection_meta.append(
            {
                "name": col,
                "size_field": size_field,
                "alpha_field": alpha_field,
                "count": int(mask.sum()),
                "tail_marker_field": f"tail_marker_{idx}",
                "close_marker_field": f"close_marker_{idx}",
            }
        )

    indicator_meta: list[dict[str, object]] = []
    for idx, col in enumerate(indicator_columns):
        indicator_series = base[col]
        if not np.issubdtype(indicator_series.dtype, np.number):
            indicator_series = indicator_series.astype(str).str.replace(",", "", regex=False)
        indicator_series = pd.to_numeric(indicator_series, errors="coerce")
        if indicator_series.isna().all():
            raise ValueError(f"Indicator column '{col}' has no numeric values to plot.")
        field = f"indicator_{idx}"
        base[field] = indicator_series
        indicator_meta.append({"name": col, "field": field})
    if not indicator_meta:
        base["indicator_0"] = np.nan
    base["bar_top"] = base[["open", "close"]].max(axis=1)
    base["bar_bottom"] = base[["open", "close"]].min(axis=1)
    base["bar_color"] = np.where(base["close"] >= base["open"], "#4CAF50", "#F44336")
    base["is_tail"] = combined_mask.to_numpy()
    base["tail_marker"] = base["high"] + (base["high"] - base["low"]) * 0.05
    bar_range = (base["high"] - base["low"]).replace(0, np.nan)
    range_fallback = float(bar_range.median()) if bar_range.notna().any() else 0.0
    bar_range = bar_range.fillna(range_fallback)
    for idx in range(len(value_columns)):
        offset = (idx - (total_selections - 1) / 2) * bar_range * 0.1
        base[f"tail_marker_{idx}"] = base["tail_marker"] + offset
        base[f"close_marker_{idx}"] = base["close"] + offset
    sample_positions = np.flatnonzero(base["is_tail"]).tolist()
    return base, selection_meta, indicator_meta, sample_positions


@dataclass
class TailAppConfig:
    minibar_path: Path
    agg_trades_path: Optional[Path]
    value_columns: tuple[str, ...]
    indicator_columns: tuple[str, ...]
    window_bars: int
    step_bars: int
    context_bars: int
    day_bars: int


def _load_minibar_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Minibar file not found: {path}")
    if path.suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "ts" not in df.columns:
        raise ValueError("Minibar file must include a 'ts' column")
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.set_index("ts").sort_index()
    return df


class AggTradeLoader:
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"AggTrades file not found: {path}")
        self.path = path

    def load_window(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        filters = [[("ts", ">=", start), ("ts", "<=", end)]]
        try:
            trades = pd.read_parquet(self.path, filters=filters)
        except Exception:
            trades = pd.read_parquet(self.path)
        trades = trades.rename(columns={"qty": "size"}).copy()
        if "ts" not in trades.columns:
            raise ValueError("AggTrades parquet must have a 'ts' column")
        trades["ts"] = pd.to_datetime(trades["ts"], errors="coerce")
        trades = trades.dropna(subset=["ts", "price", "size", "isBuyerMaker"])
        trades = trades[(trades["ts"] >= start) & (trades["ts"] <= end)]
        return trades.sort_values("ts").reset_index(drop=True)

    def minute_volume(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        if start > end:
            start, end = end, start
        padded_start = start - pd.Timedelta(minutes=1)
        padded_end = end + pd.Timedelta(minutes=1)
        trades = self.load_window(padded_start, padded_end)
        minute_index = pd.date_range(start.floor("min"), end.floor("min"), freq="1min")
        if trades.empty:
            return pd.DataFrame(
                {
                    "ts": minute_index,
                    "sell_volume": np.zeros(len(minute_index)),
                    "buy_volume": np.zeros(len(minute_index)),
                }
            )
        trades = trades.copy()
        trades.set_index("ts", inplace=True)
        trades["sell_size"] = trades["size"] * trades["isBuyerMaker"]
        trades["buy_size"] = trades["size"] * (1 - trades["isBuyerMaker"])
        minute_volume = trades[["sell_size", "buy_size"]].resample("1min").sum()
        minute_volume = minute_volume.rename(columns={"sell_size": "sell_volume", "buy_size": "buy_volume"})
        minute_volume.index = minute_volume.index.tz_localize(None)
        minute_volume = minute_volume.reindex(minute_index, fill_value=0.0)
        minute_volume = minute_volume.reset_index().rename(columns={"index": "ts"})
        return minute_volume


def _cds_payload(df: pd.DataFrame, columns: Iterable[str]) -> Dict[str, List]:
    data: Dict[str, List] = {}
    for col in columns:
        if col in df.columns:
            data[col] = df[col].tolist()
        else:
            data[col] = []
    return data


def _format_trade_source(trades: pd.DataFrame) -> Dict[str, List]:
    if trades.empty:
        empty_cols = ["ts", "price", "size", "marker_size", "marker_color", "side_label"]
        return {col: [] for col in empty_cols}
    trades = trades.copy()
    magnitude = trades["size"].astype(float).abs()
    scale_ref = np.nanpercentile(magnitude, 95)
    if not np.isfinite(scale_ref) or scale_ref <= 0:
        scale_ref = magnitude.max() if magnitude.max() > 0 else 1.0
    trades["marker_size"] = 4 + 6 * np.sqrt(np.clip(magnitude / scale_ref, 0, 1))
    trades["marker_color"] = np.where(trades["isBuyerMaker"], "#EF5350", "#42A5F5")
    trades["side_label"] = np.where(trades["isBuyerMaker"], "Sell aggression", "Buy aggression")
    cols = ["ts", "price", "size", "marker_size", "marker_color", "side_label"]
    return _cds_payload(trades, cols)


def _event_ts(value: float) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(value, unit="ms")
    except Exception:
        return None


def build_tail_layout(config: TailAppConfig):
    minibar = _load_minibar_frame(config.minibar_path)
    loader = AggTradeLoader(config.agg_trades_path) if config.agg_trades_path else None
    base, selection_meta, indicator_meta, sample_positions = prepare_minibar_dataset(
        minibar,
        value_columns=list(config.value_columns),
        indicator_columns=list(config.indicator_columns),
        require_volume=False,
    )
    ts_numeric = base["ts"].astype("int64").to_numpy()
    ts_center_numeric = base["ts_center"].astype("int64").to_numpy()

    total_len = len(base)
    if total_len == 0:
        raise ValueError("Minibar dataframe has no rows")
    window_size = min(config.window_bars, total_len)
    context_window = min(config.context_bars, total_len)
    start_idx = max(0, total_len - window_size)
    trade_window = pd.Timedelta(minutes=5)
    trade_half = trade_window / 2

    detail_source = ColumnDataSource(base.iloc[start_idx : start_idx + window_size])
    volume_source = ColumnDataSource(dict(ts=[], ts_center=[], sell_volume=[], buy_volume=[], total_volume=[]))
    context_start = max(0, min(start_idx - (context_window - window_size) // 2, total_len - context_window))
    context_df = base.iloc[context_start : context_start + context_window]
    context_source = ColumnDataSource(context_df)
    tail_cols = ["ts_center", "low", "high"]
    context_selection_sources: list[ColumnDataSource] = []
    selection_renderers: list[list] = [[] for _ in selection_meta]
    indicator_renderers: list[list] = [[] for _ in indicator_meta]
    for idx, meta in enumerate(selection_meta):
        color = SELECTION_COLORS[idx % len(SELECTION_COLORS)]
        marker = SELECTION_MARKERS[idx % len(SELECTION_MARKERS)]
        meta["color"] = color
        meta["marker"] = marker
        mask = context_df[meta["name"]].fillna(False).astype(bool)
        marker_field = meta["tail_marker_field"]
        cols = tail_cols + [marker_field]
        context_selection_sources.append(
            ColumnDataSource(_cds_payload(context_df.loc[mask, cols], cols))
        )
    trade_source = ColumnDataSource(dict(ts=[], price=[], size=[], marker_size=[], marker_color=[], side_label=[]))

    highlight_box = BoxAnnotation(fill_alpha=0.15, fill_color="#FFEB3B", line_color=None)
    trade_box = None
    volume_trade_box = None
    trade_range = None
    has_trades = loader is not None
    if not has_trades and "volume" not in base.columns:
        raise ValueError("Minibar data must include a 'volume' column when --agg-trades is omitted.")
    if has_trades:
        trade_box = BoxAnnotation(fill_color="#64B5F6", fill_alpha=0.15, line_color=None, level="underlay")
        volume_trade_box = BoxAnnotation(fill_color="#64B5F6", fill_alpha=0.15, line_color=None, level="underlay")
        trade_range = Range1d()

    tail_enabled = len(selection_meta) > 0
    columns_label = ", ".join(config.value_columns)
    context_title = (
        f"36h overview ({columns_label} selections highlighted in 2h view)"
        if tail_enabled
        else "36h overview (highlighting off)"
    )
    detail_range = Range1d()

    context_plot = figure(
        x_axis_type="datetime",
        width=1400,
        height=400,
        tools="reset,save,tap",
        active_drag=None,
        y_range=DataRange1d(),
    )
    context_plot.segment("ts_center", "high", "ts_center", "low", color="gray", source=context_source)
    context_plot.vbar(
        x="ts_center",
        width=60 * 1000 * 0.6,
        top="bar_top",
        bottom="bar_bottom",
        fill_color="bar_color",
        line_color="bar_color",
        source=context_source,
    )
    for idx, meta in enumerate(indicator_meta):
        color = INDICATOR_COLORS[idx % len(INDICATOR_COLORS)]
        renderer = context_plot.line(
            x="ts",
            y=meta["field"],
            source=context_source,
            color=color,
            line_width=2,
        )
        indicator_renderers[idx].append(renderer)
    context_plot.add_layout(highlight_box)
    for idx, (meta, source) in enumerate(zip(selection_meta, context_selection_sources)):
        segment_renderer = context_plot.segment(
            x0="ts_center",
            y0="low",
            x1="ts_center",
            y1="high",
            source=source,
            line_color=meta["color"],
            line_dash="dashed",
            line_width=2,
        )
        scatter_renderer = context_plot.scatter(
            x="ts_center",
            y=meta["tail_marker_field"],
            size=10,
            marker=meta["marker"],
            color=meta["color"],
            alpha=0.9,
            source=source,
        )
        selection_renderers[idx].extend([segment_renderer, scatter_renderer])

    detail_title = f"Minibar Candles with {columns_label} selections highlighted" if tail_enabled else "Minibar Candles (highlighting off)"
    detail_plot = figure(
        x_axis_type="datetime",
        width=1400,
        height=400,
        tools="reset,save,tap",
        active_drag=None,
        x_range=detail_range,
        y_range=DataRange1d(range_padding=0.05),
    )
    if trade_box is not None:
        detail_plot.add_layout(trade_box)
    width_ms = BAR_WIDTH_MS
    detail_plot.segment("ts_center", "high", "ts_center", "low", color="gray", source=detail_source)
    detail_plot.vbar(
        x="ts_center",
        width=width_ms,
        top="bar_top",
        bottom="bar_bottom",
        fill_color="bar_color",
        line_color="bar_color",
        source=detail_source,
    )
    for idx, meta in enumerate(indicator_meta):
        color = INDICATOR_COLORS[idx % len(INDICATOR_COLORS)]
        renderer = detail_plot.line(
            x="ts",
            y=meta["field"],
            source=detail_source,
            color=color,
            line_width=2,
        )
        indicator_renderers[idx].append(renderer)
    if tail_enabled:
        sampler_bits = []
        for idx, meta in enumerate(selection_meta):
            color = meta["color"]
            marker = meta["marker"]
            detail_renderer = detail_plot.scatter(
                x="ts_center",
                y=meta["close_marker_field"],
                size=meta["size_field"],
                marker=marker,
                color=color,
                fill_alpha=meta["alpha_field"],
                line_alpha=meta["alpha_field"],
                line_color="black",
                source=detail_source,
            )
            selection_renderers[idx].append(detail_renderer)
            sampler_bits.append(
                f"{meta['name']} "
                f"<span style='display:inline-block;width:10px;height:10px;"
                f"background:{color};border:1px solid #222;margin-right:6px;'></span>"
                f"({marker})"
            )
    hover_anchor = detail_plot.circle(
        x="ts_center",
        y="mid_price",
        size=8,
        alpha=0,
        hover_alpha=0.4,
        color="#000000",
        source=detail_source,
    )
    detail_tooltips = [
        ("Time", "@ts{%F %T}"),
        ("Open", "@open{0.2f}"),
        ("High", "@high{0.2f}"),
        ("Low", "@low{0.2f}"),
        ("Close", "@close{0.2f}"),
    ]
    if tail_enabled:
        for column_name in config.value_columns:
            detail_tooltips.append((column_name, f"@{column_name}"))
    detail_plot.add_tools(
        HoverTool(
            tooltips=detail_tooltips,
            formatters={"@ts": "datetime"},
            mode="mouse",
            renderers=[hover_anchor],
        )
    )

    volume_plot = figure(
        x_axis_type="datetime",
        x_range=detail_range,
        width=1400,
        height=220,
        tools="reset,save",
        active_drag=None,
        title="Stacked volume",
        y_range=DataRange1d(),
    )
    if has_trades:
        volume_plot.vbar_stack(
            ["sell_volume", "buy_volume"],
            x="ts_center",
            width=width_ms,
            color=["#D32F2F", "#2E7D32"],
            source=volume_source,
            legend_label=["Sell volume", "Buy volume"],
        )
        if volume_trade_box is not None:
            volume_plot.add_layout(volume_trade_box)
        volume_plot.legend.location = "top_left"
        volume_plot.legend.click_policy = "hide"
        volume_plot.add_tools(
            HoverTool(
                tooltips=[
                    ("Time", "@ts{%F %T}"),
                    ("Sell volume", "@sell_volume{0.00 a}"),
                    ("Buy volume", "@buy_volume{0.00 a}"),
                ],
                formatters={"@ts": "datetime"},
                mode="vline",
            )
        )
    else:
        volume_plot.vbar(
            x="ts_center",
            width=width_ms,
            top="total_volume",
            color="#90A4AE",
            source=volume_source,
        )
        volume_plot.add_tools(
            HoverTool(
                tooltips=[
                    ("Time", "@ts{%F %T}"),
                    ("Volume", "@total_volume{0.00 a}"),
                ],
                formatters={"@ts": "datetime"},
                mode="vline",
            )
        )

    trade_plot = None
    if has_trades:
        trade_plot = figure(
            x_axis_type="datetime",
            x_range=trade_range,
            width=1400,
            height=260,
            tools="reset,save",
            active_drag=None,
            title="AggTrades (±5 minutes)",
            y_range=DataRange1d(),
        )
        trade_plot.scatter(
            x="ts",
            y="price",
            size="marker_size",
            fill_color="marker_color",
            line_color=None,
            fill_alpha=0.25,
            source=trade_source,
        )
        trade_plot.add_tools(
            HoverTool(
                tooltips=[
                    ("Time", "@ts{%F %T.%3N}"),
                    ("Price", "@price{0.2f}"),
                    ("Size", "@size{0.4f}"),
                    ("Aggressor", "@side_label"),
                ],
                formatters={"@ts": "datetime"},
                mode="mouse",
            )
        )

    indicator_suffix = "" if indicator_meta else " Indicators disabled (no --indicator)."
    if tail_enabled:
        counts = " | ".join(f"{meta['name']}: {meta['count']}" for meta in selection_meta)
        info_text = f"<b>Selections highlighted</b> — {counts}.{indicator_suffix}"
        sampler_text = " | ".join(sampler_bits) if sampler_bits else "No sampler markers."
    else:
        info_text = f"Highlighting disabled (no value columns provided).{indicator_suffix}"
        sampler_text = "Sampler markers disabled (no value columns provided)."
    info_div = Div(text=info_text, width=950)

    if tail_enabled:
        sampler_checkboxes = CheckboxGroup(
            labels=[meta["name"] for meta in selection_meta],
            active=list(range(len(selection_meta))),
        )
        sampler_checkboxes.styles = {"line-height": "24px"}
        sampler_icons = "<div style='display:grid;grid-auto-rows:24px;align-items:center;'>"
        sampler_icons += "".join(
            f"<div>{_marker_svg(meta['marker'], meta['color'])}</div>" for meta in selection_meta
        )
        sampler_icons += "</div>"
        sampler_icon_div = Div(text=sampler_icons, width=30)

        def _toggle_samplers(attr: str, old: List[int], new: List[int]) -> None:
            active = set(sampler_checkboxes.active)
            for idx, renderers in enumerate(selection_renderers):
                visible = idx in active
                for renderer in renderers:
                    renderer.visible = visible
            if active:
                active_cols = [selection_meta[idx]["name"] for idx in sorted(active)]
                active_mask = base[active_cols].fillna(False).astype(bool).any(axis=1)
                state["sample_positions"] = np.flatnonzero(active_mask.to_numpy()).tolist()
            else:
                state["sample_positions"] = []

        sampler_checkboxes.on_change("active", _toggle_samplers)
        sampler_tool_column = column(
            Div(text="<b>Samplers</b>"),
            row(sampler_checkboxes, sampler_icon_div),
            width=260,
        )
    else:
        sampler_tool_column = column(Div(text="Samplers disabled."), width=220)

    if indicator_meta:
        indicator_checkboxes = CheckboxGroup(
            labels=[meta["name"] for meta in indicator_meta],
            active=list(range(len(indicator_meta))),
        )
        indicator_checkboxes.styles = {"line-height": "24px"}
        indicator_icons = "<div style='display:grid;grid-auto-rows:24px;align-items:center;'>"
        indicator_icons += "".join(
            (
                "<div>"
                "<svg width='22' height='12' style='vertical-align:middle'>"
                f"<line x1='1' y1='6' x2='21' y2='6' "
                f"stroke='{INDICATOR_COLORS[idx % len(INDICATOR_COLORS)]}' stroke-width='2' />"
                "</svg>"
                "</div>"
            )
            for idx, _ in enumerate(indicator_meta)
        )
        indicator_icons += "</div>"
        indicator_icon_div = Div(text=indicator_icons, width=30)

        def _toggle_indicators(attr: str, old: List[int], new: List[int]) -> None:
            active = set(indicator_checkboxes.active)
            for idx, renderers in enumerate(indicator_renderers):
                visible = idx in active
                for renderer in renderers:
                    renderer.visible = visible

        indicator_checkboxes.on_change("active", _toggle_indicators)
        indicator_tool_column = column(
            Div(text="<b>Indicators</b>"),
            row(indicator_checkboxes, indicator_icon_div),
            width=260,
        )
    else:
        indicator_tool_column = column(Div(text="Indicators disabled."), width=220)

    state = {
        "start_idx": start_idx,
        "window_size": window_size,
        "context_window": context_window,
        "sample_positions": sample_positions,
        "sample_ptr": -1,
        "trade_idx": min(total_len - 1, start_idx + window_size // 2),
    }

    def _update_detail(new_start: int, *, recenter_trade: bool = True) -> None:
        total = len(base)
        win = state["window_size"]
        start = max(0, min(new_start, total - win))
        end = min(total, start + win)
        window = base.iloc[start:end]
        detail_source.data = window.to_dict("list")
        if loader is not None:
            volume_frame = loader.minute_volume(window["ts"].iloc[0], window["ts"].iloc[-1])
            aligned_volume = volume_frame.set_index("ts")
            ts_index = window["ts"].dt.floor("min")
            aligned_volume = aligned_volume.reindex(ts_index, fill_value=0.0)
            aligned_volume = aligned_volume.reset_index(drop=True)
            aligned_volume["ts"] = window["ts"].values
            aligned_volume["ts_center"] = window["ts_center"].values
            aligned_volume["total_volume"] = aligned_volume["sell_volume"] + aligned_volume["buy_volume"]
            volume_source.data = _cds_payload(
                aligned_volume,
                ["ts", "ts_center", "sell_volume", "buy_volume", "total_volume"],
            )
        else:
            volume_frame = window[["ts", "ts_center", "volume"]].copy()
            volume_frame["total_volume"] = pd.to_numeric(volume_frame["volume"], errors="coerce").fillna(0.0)
            volume_source.data = _cds_payload(volume_frame, ["ts", "ts_center", "total_volume"])
        highlight_box.left = window["ts"].iloc[0]
        highlight_box.right = window["ts_right"].iloc[-1]
        detail_range.start = window["ts"].iloc[0]
        detail_range.end = window["ts_right"].iloc[-1]
        ctx = state["context_window"]
        ctx_start = max(0, min(start - (ctx - win) // 2, total - ctx))
        ctx_end = min(total, ctx_start + ctx)
        context_slice = base.iloc[ctx_start:ctx_end]
        context_source.data = context_slice.to_dict("list")
        for meta, source in zip(selection_meta, context_selection_sources):
            mask = context_slice[meta["name"]].fillna(False).astype(bool)
            cols = tail_cols + [meta["tail_marker_field"]]
            selection_slice = context_slice.loc[mask, cols]
            source.data = _cds_payload(selection_slice, cols)
        state["start_idx"] = start
        center_idx = min(total - 1, start + win // 2)
        if recenter_trade and loader is not None:
            _update_trade(center_idx)

    def _update_trade(target_idx: int) -> None:
        if loader is None or trade_range is None or trade_box is None or volume_trade_box is None:
            return
        total = len(base)
        idx = max(0, min(target_idx, total - 1))
        center_ts = base["ts_center"].iloc[idx]
        start = center_ts - trade_half
        end = center_ts + trade_half
        trades = loader.load_window(start, end)
        trade_source.data = _format_trade_source(trades)
        trade_range.start = start
        trade_range.end = end
        trade_box.left = start
        trade_box.right = end
        volume_trade_box.left = start
        volume_trade_box.right = end
        state["trade_idx"] = idx

    def _ensure_visible(target_idx: int) -> None:
        start = state["start_idx"]
        win = state["window_size"]
        if target_idx < start or target_idx >= start + win:
            _update_detail(target_idx - win // 2, recenter_trade=False)

    def _shift_detail(delta: int) -> None:
        _update_detail(state["start_idx"] + delta)

    def _shift_day(delta: int) -> None:
        _update_detail(state["start_idx"] + delta)

    def _jump_to_sample(direction: int) -> None:
        samples: List[int] = state["sample_positions"]
        if not samples:
            return
        center = state["start_idx"] + state["window_size"] // 2
        target_idx = -1
        if direction < 0:
            for pos in samples:
                if pos < center:
                    target_idx = pos
                else:
                    break
            if target_idx == -1:
                target_idx = samples[-1]
        else:
            for pos in samples:
                if pos > center:
                    target_idx = pos
                    break
            if target_idx == -1:
                target_idx = samples[0]
        _update_detail(target_idx - state["window_size"] // 2, recenter_trade=False)
        _update_trade(target_idx)

    def _jump_to_timestamp() -> None:
        raw = jump_input.value.strip()
        if not raw:
            return
        try:
            target = pd.to_datetime(raw)
        except Exception:
            return
        ts_array = base["ts"].to_numpy()
        idx = int(np.searchsorted(ts_numeric, int(target.value)))
        idx = max(0, min(idx, len(base) - 1))
        _update_detail(idx - state["window_size"] // 2)

    def _shift_trade(delta: int) -> None:
        if loader is None:
            return
        new_idx = state["trade_idx"] + delta
        _ensure_visible(new_idx)
        _update_trade(new_idx)

    def _handle_selection(attr: str, old: List[int], new: List[int]) -> None:
        if loader is None:
            return
        if not new:
            return
        absolute_idx = state["start_idx"] + new[-1]
        _update_trade(absolute_idx)

    def _handle_context_tap(event) -> None:
        ts = _event_ts(event.x)
        if ts is None:
            return
        idx = int(np.searchsorted(ts_numeric, int(ts.value)))
        idx = max(0, min(idx, len(base) - 1))
        _update_detail(idx - state["window_size"] // 2)

    def _handle_detail_tap(event) -> None:
        if loader is None:
            return
        ts = _event_ts(event.x)
        if ts is None:
            return
        click_ns = int(ts.value)
        idx = int(np.searchsorted(ts_center_numeric, click_ns))
        if idx >= len(ts_center_numeric):
            idx = len(ts_center_numeric) - 1
        elif idx > 0:
            prev_diff = abs(click_ns - ts_center_numeric[idx - 1])
            curr_diff = abs(ts_center_numeric[idx] - click_ns)
            if prev_diff <= curr_diff:
                idx -= 1
        _update_trade(idx)

    prev_hour_button = Button(label="Prev 1h", width=90)
    next_hour_button = Button(label="Next 1h", width=90)
    prev_day_button = Button(label="Prev day", width=90)
    next_day_button = Button(label="Next day", width=90)
    prev_sample_button = Button(label="Prev sample", width=110)
    next_sample_button = Button(label="Next sample", width=110)
    prev_trade_button = Button(label="Prev minute", width=120)
    next_trade_button = Button(label="Next minute", width=120)
    jump_input = TextInput(
        title="Jump to timestamp (YYYY-MM-DD HH:MM)",
        placeholder="2024-05-01 12:30",
        width=260,
    )
    jump_button = Button(label="Jump", width=80)

    prev_hour_button.on_click(lambda: _shift_detail(-config.step_bars))
    next_hour_button.on_click(lambda: _shift_detail(config.step_bars))
    prev_day_button.on_click(lambda: _shift_day(-config.day_bars))
    next_day_button.on_click(lambda: _shift_day(config.day_bars))
    prev_sample_button.on_click(lambda: _jump_to_sample(-1))
    next_sample_button.on_click(lambda: _jump_to_sample(1))
    prev_trade_button.on_click(lambda: _shift_trade(-1))
    next_trade_button.on_click(lambda: _shift_trade(1))
    jump_button.on_click(_jump_to_timestamp)
    detail_source.selected.on_change("indices", _handle_selection)
    context_plot.on_event(Tap, _handle_context_tap)
    detail_plot.on_event(Tap, _handle_detail_tap)

    _update_detail(start_idx, recenter_trade=True)

    day_controls = row(prev_day_button, next_day_button, sizing_mode="scale_width")
    jump_controls = row(jump_input, jump_button, sizing_mode="scale_width")
    hour_controls = row(prev_hour_button, next_hour_button, prev_sample_button, next_sample_button, sizing_mode="scale_width")
    trade_controls = row(prev_trade_button, next_trade_button, sizing_mode="scale_width") if has_trades else None

    layout_children = [
        info_div,
        day_controls,
        jump_controls,
        context_plot,
        hour_controls,
        detail_plot,
        volume_plot,
    ]
    if trade_controls is not None and trade_plot is not None:
        layout_children.extend([trade_controls, trade_plot])
    main_column = column(*layout_children)
    tool_column = column(indicator_tool_column, sampler_tool_column, width=220)
    layout = row(tool_column, main_column)
    return layout


def _parse_args() -> TailAppConfig:
    parser = argparse.ArgumentParser(description="Interactive minibar tail viewer")
    parser.add_argument("--minibar", required=True, help="Path to parquet/csv with minibar data")
    parser.add_argument("--agg-trades", dest="agg_trades", help="Path to aggTrades parquet")
    parser.add_argument(
        "--value-column",
        action="append",
        dest="value_columns",
        default=[],
        help="Boolean column used to highlight selections (repeatable)",
    )
    parser.add_argument("--value-column-2", help="Optional secondary boolean column to highlight (deprecated)")
    parser.add_argument(
        "--indicator",
        action="append",
        dest="indicator_columns",
        default=[],
        help="Indicator column(s) to plot (repeatable, comma-separated).",
    )
    parser.add_argument(
        "--anchor-column",
        dest="anchor_column",
        default=None,
        help="Deprecated. Use --indicator instead.",
    )
    parser.add_argument("--window-bars", type=int, default=120)
    parser.add_argument("--step-bars", type=int, default=60)
    parser.add_argument("--context-bars", type=int, default=36 * 60)
    parser.add_argument("--day-bars", type=int, default=24 * 60)
    args, _ = parser.parse_known_args()
    raw_values = list(args.value_columns or [])
    if args.value_column_2:
        raw_values.append(args.value_column_2)
    value_columns: list[str] = []
    for entry in raw_values:
        value_columns.extend(part.strip() for part in entry.split(",") if part.strip())
    value_columns = list(dict.fromkeys(value_columns))

    indicator_columns: list[str] = []
    for entry in args.indicator_columns or []:
        indicator_columns.extend(part.strip() for part in entry.split(",") if part.strip())
    if args.anchor_column:
        indicator_columns.extend(part.strip() for part in args.anchor_column.split(",") if part.strip())
    indicator_columns = list(dict.fromkeys(indicator_columns))
    return TailAppConfig(
        minibar_path=Path(args.minibar).expanduser(),
        agg_trades_path=Path(args.agg_trades).expanduser() if args.agg_trades else None,
        value_columns=tuple(value_columns),
        indicator_columns=tuple(indicator_columns),
        window_bars=args.window_bars,
        step_bars=args.step_bars,
        context_bars=args.context_bars,
        day_bars=args.day_bars,
    )


def _build_document() -> None:
    config = _parse_args()
    layout = build_tail_layout(config)
    curdoc().add_root(layout)
    curdoc().title = "Minibar Tail Viewer"


_build_document()
