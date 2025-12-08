
#bokeh serve --show klines_tail_app.py --args \
#    --minibar DATA/organized/BTCUSDT/klines_1m.csv \
#    --agg-trades DATA/organized/BTCUSDT/aggTrades.parquet \
#    --ema-window 20 \
#    --left-tail 0.0 \
#    --right-tail 0.025


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


def prepare_minibar_dataset(
    minibar: pd.DataFrame,
    *,
    value_column: str | None,
    left_tail_p: float,
    right_tail_p: float,
    ema_window: int | None,
    require_volume: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[int]]:
    required_cols = {"open", "high", "low", "close"}
    volume_cols = {"buy_volume", "sell_volume"}
    if require_volume:
        required_cols |= volume_cols
    if not required_cols.issubset(minibar.columns):
        missing = required_cols - set(minibar.columns)
        raise ValueError(f"minibar is missing columns: {missing}")
    if value_column is not None and value_column not in minibar.columns:
        raise ValueError(f"{value_column} column not found.")
    if not 0 <= left_tail_p < 0.5:
        raise ValueError("left_tail_p must be between 0 and 0.5.")
    if not 0 <= right_tail_p < 0.5:
        raise ValueError("right_tail_p must be between 0 and 0.5.")
    if ema_window is not None and ema_window <= 0:
        raise ValueError("ema_window must be positive when provided.")

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

    if value_column is not None:
        lower = base[value_column].quantile(left_tail_p) if left_tail_p > 0 else None
        upper = base[value_column].quantile(1 - right_tail_p) if right_tail_p > 0 else None
        lower_mask = base[value_column] <= lower if lower is not None else pd.Series(False, index=base.index)
        upper_mask = base[value_column] >= upper if upper is not None else pd.Series(False, index=base.index)
    else:
        lower = upper = None
        lower_mask = pd.Series(False, index=base.index)
        upper_mask = pd.Series(False, index=base.index)

    if ema_window is not None:
        base["ema"] = base["close"].ewm(span=ema_window, adjust=False).mean()
    else:
        base["ema"] = np.nan
    base["bar_top"] = base[["open", "close"]].max(axis=1)
    base["bar_bottom"] = base[["open", "close"]].min(axis=1)
    base["bar_color"] = np.where(base["close"] >= base["open"], "#4CAF50", "#F44336")
    base["tail_alpha"] = 0.8 * (lower_mask | upper_mask).to_numpy()
    base["tail_size"] = np.where(lower_mask | upper_mask, 9, 0)
    base["is_tail"] = (lower_mask | upper_mask).to_numpy()
    base["tail_marker"] = base["high"] + (base["high"] - base["low"]) * 0.05
    sample_positions = np.flatnonzero(base["is_tail"]).tolist()
    return base, lower_mask, upper_mask, sample_positions


@dataclass
class TailAppConfig:
    minibar_path: Path
    agg_trades_path: Path
    value_column: Optional[str]
    ema_window: Optional[int]
    left_tail_p: float
    right_tail_p: float
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
    trades["ts_ms"] = trades["ts"].astype("int64") // 10**6
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
    loader = AggTradeLoader(config.agg_trades_path)
    base, lower_mask, upper_mask, sample_positions = prepare_minibar_dataset(
        minibar,
        value_column=config.value_column,
        left_tail_p=config.left_tail_p,
        right_tail_p=config.right_tail_p,
        ema_window=config.ema_window,
        require_volume=False,
    )

    total_len = len(base)
    if total_len == 0:
        raise ValueError("Minibar dataframe has no rows")
    window_size = min(config.window_bars, total_len)
    context_window = min(config.context_bars, total_len)
    start_idx = max(0, total_len - window_size)
    trade_window = pd.Timedelta(minutes=5)
    trade_half = trade_window / 2

    detail_source = ColumnDataSource(base.iloc[start_idx : start_idx + window_size])
    volume_source = ColumnDataSource(dict(ts=[], ts_center=[], sell_volume=[], buy_volume=[]))
    context_start = max(0, min(start_idx - (context_window - window_size) // 2, total_len - context_window))
    context_df = base.iloc[context_start : context_start + context_window]
    context_source = ColumnDataSource(context_df)
    tail_cols = ["ts_center", "low", "high", "tail_marker"]
    context_tail_source = ColumnDataSource(
        _cds_payload(context_df.loc[context_df["is_tail"], tail_cols], tail_cols)
    )
    trade_source = ColumnDataSource(dict(ts=[], price=[], size=[], marker_size=[], marker_color=[], side_label=[]))

    highlight_box = BoxAnnotation(fill_alpha=0.15, fill_color="#FFEB3B", line_color=None)
    trade_box = BoxAnnotation(fill_color="#64B5F6", fill_alpha=0.15, line_color=None, level="underlay")
    volume_trade_box = BoxAnnotation(fill_color="#64B5F6", fill_alpha=0.15, line_color=None, level="underlay")
    trade_range = Range1d()

    tail_enabled = config.value_column is not None
    context_title = (
        f"36h overview ({config.value_column} tails highlighted in 2h view)"
        if tail_enabled
        else "36h overview (tail highlighting off)"
    )
    detail_range = Range1d()

    context_plot = figure(
        x_axis_type="datetime",
        width=1400,
        height=200,
        tools="reset,save,tap",
        active_drag=None,
        title=context_title,
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
    context_plot.add_layout(highlight_box)
    context_plot.segment(
        x0="ts_center",
        y0="low",
        x1="ts_center",
        y1="high",
        source=context_tail_source,
        line_color="#E91E63",
        line_dash="dashed",
        line_width=2,
    )
    context_plot.scatter(
        x="ts_center",
        y="tail_marker",
        size=10,
        color="#E91E63",
        alpha=0.9,
        source=context_tail_source,
    )

    detail_title = (
        f"Minibar Candles with {config.value_column} tails "
        f"(left {config.left_tail_p:.0%}, right {config.right_tail_p:.0%})"
        if tail_enabled
        else "Minibar Candles (tail highlighting off)"
    )
    detail_plot = figure(
        x_axis_type="datetime",
        width=1400,
        height=400,
        tools="reset,save,tap",
        active_drag=None,
        title=detail_title,
        x_range=detail_range,
        y_range=DataRange1d(range_padding=0.05),
    )
    detail_plot.add_layout(trade_box)
    width_ms = 60 * 1000 * 0.8
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
    if config.ema_window is not None:
        detail_plot.line(
            x="ts",
            y="ema",
            source=detail_source,
            color="#2196F3",
            line_width=2,
            legend_label=f"EMA({config.ema_window})",
        )
    detail_plot.scatter(
        x="ts_center",
        y="close",
        size="tail_size",
        marker="circle",
        color="#FFD700",
        fill_alpha="tail_alpha",
        line_alpha="tail_alpha",
        line_color="black",
        source=detail_source,
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
        detail_tooltips.append((config.value_column, f"@{config.value_column}{{0.8f}}"))
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
    volume_plot.vbar_stack(
        ["sell_volume", "buy_volume"],
        x="ts_center",
        width=width_ms,
        color=["#D32F2F", "#2E7D32"],
        source=volume_source,
        legend_label=["Sell volume", "Buy volume"],
    )
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

    if tail_enabled:
        info_text = (
            f"<b>{config.value_column}</b> | left tail p = {config.left_tail_p:.2%}"
            f" (samples: {int(lower_mask.sum())}) | right tail p = {config.right_tail_p:.2%}"
            f" (samples: {int(upper_mask.sum())})"
        )
    else:
        info_text = "Tail highlighting disabled (no value column provided)."
    info_div = Div(text=info_text, width=950)

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
        volume_frame = loader.minute_volume(window["ts"].iloc[0], window["ts"].iloc[-1])
        aligned_volume = volume_frame.set_index("ts")
        ts_index = window["ts"].dt.floor("min")
        aligned_volume = aligned_volume.reindex(ts_index, fill_value=0.0)
        aligned_volume = aligned_volume.reset_index(drop=True)
        aligned_volume["ts"] = window["ts"].values
        aligned_volume["ts_center"] = window["ts_center"].values
        volume_source.data = _cds_payload(aligned_volume, ["ts", "ts_center", "sell_volume", "buy_volume"])
        highlight_box.left = window["ts"].iloc[0]
        highlight_box.right = window["ts_right"].iloc[-1]
        detail_range.start = window["ts"].iloc[0]
        detail_range.end = window["ts_right"].iloc[-1]
        ctx = state["context_window"]
        ctx_start = max(0, min(start - (ctx - win) // 2, total - ctx))
        ctx_end = min(total, ctx_start + ctx)
        context_slice = base.iloc[ctx_start:ctx_end]
        context_source.data = context_slice.to_dict("list")
        context_tail = context_slice.loc[context_slice["is_tail"], tail_cols]
        context_tail_source.data = _cds_payload(context_tail, tail_cols)
        state["start_idx"] = start
        center_idx = min(total - 1, start + win // 2)
        if recenter_trade:
            _update_trade(center_idx)

    def _update_trade(target_idx: int) -> None:
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
        idx = int(np.searchsorted(ts_array, target))
        idx = max(0, min(idx, len(base) - 1))
        _update_detail(idx - state["window_size"] // 2)

    def _shift_trade(delta: int) -> None:
        new_idx = state["trade_idx"] + delta
        _ensure_visible(new_idx)
        _update_trade(new_idx)

    def _handle_selection(attr: str, old: List[int], new: List[int]) -> None:
        if not new:
            return
        absolute_idx = state["start_idx"] + new[-1]
        _update_trade(absolute_idx)

    def _handle_context_tap(event) -> None:
        ts = _event_ts(event.x)
        if ts is None:
            return
        ts_array = base["ts"].to_numpy()
        idx = int(np.searchsorted(ts_array, ts))
        idx = max(0, min(idx, len(base) - 1))
        _update_detail(idx - state["window_size"] // 2)

    def _handle_detail_tap(event) -> None:
        ts = _event_ts(event.x)
        if ts is None:
            return
        ts_array = base["ts"].to_numpy()
        idx = int(np.searchsorted(ts_array, ts))
        idx = max(0, min(idx, len(base) - 1))
        _ensure_visible(idx)
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
    trade_controls = row(prev_trade_button, next_trade_button, sizing_mode="scale_width")

    layout = column(
        info_div,
        day_controls,
        jump_controls,
        context_plot,
        hour_controls,
        detail_plot,
        volume_plot,
        trade_controls,
        trade_plot,
    )
    return layout


def _parse_args() -> TailAppConfig:
    parser = argparse.ArgumentParser(description="Interactive minibar tail viewer")
    parser.add_argument("--minibar", required=True, help="Path to parquet/csv with minibar data")
    parser.add_argument("--agg-trades", required=True, dest="agg_trades", help="Path to aggTrades parquet")
    parser.add_argument("--value-column", help="Column to evaluate for tail events")
    parser.add_argument("--ema-window", type=int, default=None)
    parser.add_argument("--left-tail", type=float, default=0.05)
    parser.add_argument("--right-tail", type=float, default=0.05)
    parser.add_argument("--window-bars", type=int, default=120)
    parser.add_argument("--step-bars", type=int, default=60)
    parser.add_argument("--context-bars", type=int, default=36 * 60)
    parser.add_argument("--day-bars", type=int, default=24 * 60)
    args, _ = parser.parse_known_args()
    return TailAppConfig(
        minibar_path=Path(args.minibar).expanduser(),
        agg_trades_path=Path(args.agg_trades).expanduser(),
        value_column=args.value_column,
        ema_window=args.ema_window,
        left_tail_p=args.left_tail,
        right_tail_p=args.right_tail,
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
