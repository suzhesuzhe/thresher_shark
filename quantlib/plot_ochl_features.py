#%%
import numpy as np
import pandas as pd
from pathlib import Path
import re
from bokeh.plotting import figure, show
from bokeh.io import save
from bokeh.resources import INLINE
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    Button,
    CustomJS,
    DatetimeTickFormatter,
    Div,
    TextInput,
)
from bokeh.palettes import Category10, Category20

#%%
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    # Try common name candidates
    candidates = [df.columns[0], 'datetime', 'Datetime', 'date', 'Date', 'timestamp', 'Timestamp']
    for cand in candidates:
        if cand in df.columns:
            try:
                idx = pd.to_datetime(df[cand], errors='raise')
                out = df.copy()
                out.index = idx
                return out.sort_index()
            except Exception:
                continue
    # Fallback: try to parse current index
    try:
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        return out.sort_index()
    except Exception:
        raise ValueError("Could not determine a DatetimeIndex for OHLC data")



def plot_ohlc_basic(
    plot_df: pd.DataFrame,
    *,
    indicators: dict[str, str] | None = None,  # {column: 'overlay'|'separate'}
    rules_dict: dict[str, str] | None = None,  # {rule_column: legend_name}
    window_bars: int = 200,
    step_bars: int = 50,
    title: str = "OHLC",
    height: int = 620,
    output_dir: str | Path = ".",
    output_filename: str | None = None,
    open_in_browser: bool = False,
) -> None:
    """
    Plot OHLC candlesticks with left/right scroll (fixed-size window) and auto y-range.

    - window_bars: number of bars visible at once
    - step_bars: bars to move per left/right click
    - Saves a standalone HTML to `output_dir/output_filename` (defaults to
      current folder with a name derived from title). Set `open_in_browser=True`
      to also open after saving.
    """
    df_full = _ensure_datetime_index(plot_df)
    req = ["Open", "High", "Low", "Close"]
    missing = [c for c in req if c not in df_full.columns]
    if missing:
        raise ValueError(f"OHLC missing required columns: {missing}")
    if 'Volume' not in df_full.columns:
        raise ValueError("OHLC missing required column 'Volume' for volume pane")
    df = df_full[req].astype(float)
    vol_full = df_full['Volume'].astype(float)
    if window_bars and len(df) > window_bars:
        start_idx = len(df) - window_bars
    else:
        start_idx = 0
        window_bars = len(df)

    ts = df.index
    # Median bar width in ms
    if len(ts) >= 2:
        td = (ts[1:] - ts[:-1]).asi8.astype('float64') / 1_000_000.0
        bar_ms = float(np.nanmedian(td)) if len(td) else 60_000.0
        if not np.isfinite(bar_ms) or bar_ms <= 0:
            bar_ms = 60_000.0
    else:
        bar_ms = 60_000.0

    # Full data source (for JS slicing)
    full = ColumnDataSource(dict(
        time=ts.astype('datetime64[ms]'),
        Open=df['Open'].values,
        High=df['High'].values,
        Low=df['Low'].values,
        Close=df['Close'].values,
    ))

    # Visible sources (wick, inc body, dec body)
    src_w = ColumnDataSource(dict(time=[], low=[], high=[]))
    src_i = ColumnDataSource(dict(time=[], top=[], bottom=[]))
    src_d = ColumnDataSource(dict(time=[], top=[], bottom=[]))

    p = figure(
        x_axis_type="datetime",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_drag="pan",
        active_scroll="wheel_zoom",
        height=height,
        title=title,
        sizing_mode="stretch_width",
    )
    p.xaxis.formatter = DatetimeTickFormatter(
        milliseconds="%Y-%m-%d %H:%M:%S.%3N",
        seconds="%Y-%m-%d %H:%M:%S",
        minutes="%Y-%m-%d %H:%M",
        hours="%Y-%m-%d %H:%M",
        days="%Y-%m-%d",
        months="%Y-%m",
        years="%Y",
    )

    # Candles: wick + bodies
    p.segment(x0="time", y0="low", x1="time", y1="high", color="#555", source=src_w)
    p.vbar(x="time", top="top", bottom="bottom", width=bar_ms * 0.7,
           fill_color="#2E7D32", line_color="#2E7D32", source=src_i)
    p.vbar(x="time", top="top", bottom="bottom", width=bar_ms * 0.7,
           fill_color="#C62828", line_color="#C62828", source=src_d)

    # Volume pane (top)
    pv = figure(
        x_axis_type="datetime",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_drag="pan",
        active_scroll="wheel_zoom",
        height=180,
        title="Volume",
        sizing_mode="stretch_width",
        x_range=p.x_range,
    )
    vol_src = ColumnDataSource(dict(time=[], vol=[]))
    pv.vbar(x='time', top='vol', width=bar_ms * 0.7, source=vol_src,
            fill_color='lightsteelblue', line_color='lightsteelblue')

    # Indicators prep (column names come from plot_df directly)
    indicators = indicators or {}
    overlay_cols = [k for k, v in indicators.items() if str(v).lower() == 'overlay']
    separate_cols = [k for k, v in indicators.items() if str(v).lower() == 'separate']
    present_cols = set(df_full.columns)
    missing = [c for c in overlay_cols + separate_cols if c not in present_cols]
    if missing:
        sample = sorted(list(present_cols))[:20]
        raise ValueError(
            "Requested indicator columns not found in plot_df: "
            f"{missing}. Available sample: {sample} (total {len(present_cols)} columns)."
        )
    # Use direct alignment (same index)
    aligned_overlay_full = df_full[overlay_cols] if overlay_cols else pd.DataFrame(index=ts)
    aligned_separate_full = df_full[separate_cols] if separate_cols else pd.DataFrame(index=ts)

    # Create sources for overlays and separate indicators (these will be sliced in JS)
    overlay_names = list(overlay_cols)
    overlay_sources = [ColumnDataSource(dict(time=[], y=[])) for _ in overlay_names]
    separate_names = list(separate_cols)
    separate_sources = [ColumnDataSource(dict(time=[], y=[])) for _ in separate_names]

    # State source for JS (start, window, step)
    state = ColumnDataSource(dict(start=[start_idx], window=[int(window_bars)], step=[int(step_bars)]))

    # JS callback to update visible window and y-range
    # Rules prep: rules_dict maps column_name -> legend_name
    rules_dict = rules_dict or {}
    rule_cols = list(rules_dict.keys())
    rule_names = [rules_dict[c] for c in rule_cols]
    # Validate rule columns
    missing_rules = [c for c in rule_cols if c not in present_cols]
    if missing_rules:
        raise ValueError(f"Requested rule columns not found in plot_df: {missing_rules}")
    # Build boolean flags aligned to ts
    rule_flags = {rules_dict[c]: df_full[c].astype(bool).reindex(ts).fillna(False).tolist() for c in rule_cols}
    rule_sources = {name: ColumnDataSource(dict(x0=[], y0=[], x1=[], y1=[])) for name in rule_names}

    # Build JSON-safe lists (replace NaN/inf with null) for CustomJS args
    def _to_js_list(series: pd.Series) -> list:
        s = pd.to_numeric(series, errors='coerce')
        arr = s.to_numpy(dtype='float64', copy=False)
        out = []
        for v in arr:
            if v is None or (isinstance(v, float) and (not np.isfinite(v))):
                out.append(None)
            else:
                out.append(float(v))
        return out

    vol_js = _to_js_list(vol_full)
    ol_js_data = {name: _to_js_list(aligned_overlay_full[name]) if (len(aligned_overlay_full.columns) > 0 and name in aligned_overlay_full.columns) else [] for name in overlay_names}
    sep_js_data = {name: _to_js_list(aligned_separate_full[name]) if (len(aligned_separate_full.columns) > 0 and name in aligned_separate_full.columns) else [] for name in separate_names}

    js_update = CustomJS(args=dict(
        full=full,
        wsrc=src_w,
        isrc=src_i,
        dsrc=src_d,
        fig=p,
        state=state,
        vol_src=vol_src,
        vol_data=vol_js,
        # overlays
        ol_names=overlay_names,
        ol_sources=overlay_sources,
        ol_data=ol_js_data,
        # separate
        sep_names=separate_names,
        sep_sources=separate_sources,
        sep_data=sep_js_data,
        # rules
        rule_names=rule_names,
        rule_sources=rule_sources,
        rule_data=rule_flags,
        fig_sep=None,
    ), code="""
        const n = full.data['time'].length;
        let start = state.data['start'][0];
        const window = state.data['window'][0];
        const end = Math.min(n - 1, start + window - 1);

        const t = full.data['time'];
        const o = full.data['Open'];
        const h = full.data['High'];
        const l = full.data['Low'];
        const c = full.data['Close'];

        // Slice wick
        const tw = t.slice(start, end + 1);
        const lw = l.slice(start, end + 1);
        const hw = h.slice(start, end + 1);
        wsrc.data['time'] = tw;
        wsrc.data['low'] = lw;
        wsrc.data['high'] = hw;

        // Bodies: compute inc/dec for the window
        const ti = [], topi = [], boti = [];
        const td = [], topd = [], botd = [];
        for (let i = start; i <= end; i++) {
            const top = Math.max(o[i], c[i]);
            const bot = Math.min(o[i], c[i]);
            if (c[i] >= o[i]) { // inc
                ti.push(t[i]); topi.push(top); boti.push(bot);
            } else {
                td.push(t[i]); topd.push(top); botd.push(bot);
            }
        }
        isrc.data['time'] = ti; isrc.data['top'] = topi; isrc.data['bottom'] = boti;
        dsrc.data['time'] = td; dsrc.data['top'] = topd; dsrc.data['bottom'] = botd;

        // Update x-range
        fig.x_range.start = tw[0];
        fig.x_range.end = tw[tw.length - 1];

        // Auto y-range using window lows/highs (and overlays)
        let ymin = Infinity, ymax = -Infinity;
        for (let i = 0; i < lw.length; i++) {
            if (lw[i] < ymin) ymin = lw[i];
            if (hw[i] > ymax) ymax = hw[i];
        }
        if (ol_names && ol_sources) {
            for (let k = 0; k < ol_names.length; k++) {
                const name = ol_names[k];
                const arr = ol_data[name] || [];
                for (let i = start; i <= end && i < arr.length; i++) {
                    const v = arr[i];
                    if (v != null && isFinite(v)) {
                        if (v < ymin) ymin = v;
                        if (v > ymax) ymax = v;
                    }
                }
            }
        }
        if (!isFinite(ymin) || !isFinite(ymax) || ymin === Infinity || ymax === -Infinity) {
            ymin = 0; ymax = 1;
        }
        const pad = (ymax - ymin) * 0.20;
        fig.y_range.start = ymin - (isFinite(pad) && pad > 0 ? pad : 1e-6);
        fig.y_range.end = ymax + (isFinite(pad) && pad > 0 ? pad : 1e-6);

        // Update overlay sources (slice to tw)
        if (ol_names && ol_sources) {
            for (let k = 0; k < ol_names.length; k++) {
                const name = ol_names[k];
                const src = ol_sources[k];
                const arr = ol_data[name] || [];
                const ys = arr.slice(start, end + 1);
                src.data['time'] = tw;
                src.data['y'] = ys;
                src.change.emit();
            }
        }

        // Volume slice
        if (vol_src && vol_data) {
            const vv = vol_data.slice(start, end + 1);
            vol_src.data['time'] = tw;
            vol_src.data['vol'] = vv;
            vol_src.change.emit();
        }

        // Update separate pane sources and y-range
        if (sep_names && sep_sources && fig_sep) {
            let vmin = Infinity, vmax = -Infinity;
            for (let k = 0; k < sep_names.length; k++) {
                const name = sep_names[k];
                const src = sep_sources[k];
                const arr = sep_data[name] || [];
                const ys = arr.slice(start, end + 1);
                src.data['time'] = tw;
                src.data['y'] = ys;
                src.change.emit();
                for (let i = 0; i < ys.length; i++) {
                    const v = ys[i];
                    if (v != null && isFinite(v)) {
                        if (v < vmin) vmin = v;
                        if (v > vmax) vmax = v;
                    }
                }
            }
            if (isFinite(vmin) && isFinite(vmax) && vmin !== Infinity && vmax !== -Infinity) {
                const pad2 = (vmax - vmin) * 0.20;
                fig_sep.y_range.start = vmin - (isFinite(pad2) && pad2 > 0 ? pad2 : 1e-6);
                fig_sep.y_range.end = vmax + (isFinite(pad2) && pad2 > 0 ? pad2 : 1e-6);
            }
        }

        // Update vertical rule lines spanning current y-range
        if (rule_names && rule_sources) {
            for (const name of rule_names) {
                const flags = rule_data[name] || [];
                const xs0 = [], ys0 = [], xs1 = [], ys1 = [];
                for (let i = start, k = 0; i <= end; i++, k++) {
                    if (i < flags.length && flags[i]) {
                        const x = tw[k];
                        xs0.push(x); ys0.push(fig.y_range.start);
                        xs1.push(x); ys1.push(fig.y_range.end);
                    }
                }
                const src = rule_sources[name];
                src.data['x0'] = xs0;
                src.data['y0'] = ys0;
                src.data['x1'] = xs1;
                src.data['y1'] = ys1;
                src.change.emit();
            }
        }

        wsrc.change.emit();
        isrc.change.emit();
        dsrc.change.emit();
    """)

    # Prepare initial visible window from Python (no 'ready' event)
    end_idx = min(len(df) - 1, start_idx + window_bars - 1)
    tw = ts[start_idx:end_idx + 1]
    lw = df['Low'].iloc[start_idx:end_idx + 1].values
    hw = df['High'].iloc[start_idx:end_idx + 1].values

    # Assign wick
    src_w.data = dict(time=tw.astype('datetime64[ms]'), low=lw, high=hw)

    # Assign bodies
    inc_mask_full = (df['Close'] >= df['Open']).values
    win_open = df['Open'].values[start_idx:end_idx + 1]
    win_close = df['Close'].values[start_idx:end_idx + 1]
    win_time = ts[start_idx:end_idx + 1].astype('datetime64[ms]')
    inc_mask = inc_mask_full[start_idx:end_idx + 1]
    const_top = np.maximum(win_open, win_close)
    const_bottom = np.minimum(win_open, win_close)
    src_i.data = dict(time=win_time[inc_mask], top=const_top[inc_mask], bottom=const_bottom[inc_mask])
    dec_mask = ~inc_mask
    src_d.data = dict(time=win_time[dec_mask], top=const_top[dec_mask], bottom=const_bottom[dec_mask])

    # Set initial x/y ranges
    if len(tw) > 0:
        p.x_range.start = pd.to_datetime(tw[0]).to_pydatetime()
        p.x_range.end = pd.to_datetime(tw[-1]).to_pydatetime()
        ymin = float(np.nanmin(lw)) if len(lw) else 0.0
        ymax = float(np.nanmax(hw)) if len(hw) else 1.0
        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
            ymin, ymax = 0.0, 1.0
        pad = (ymax - ymin) * 0.20 if ymax > ymin else 1e-6
        p.y_range.start = ymin - pad
        p.y_range.end = ymax + pad
        # Initial volume (JSON-safe)
        vwin = pd.to_numeric(vol_full.iloc[start_idx:end_idx + 1], errors='coerce')
        vwin_list = [None if (v is None or (isinstance(v, float) and (not np.isfinite(v)))) else float(v) for v in vwin.to_numpy(dtype='float64', copy=False)]
        vol_src.data = dict(time=tw.astype('datetime64[ms]'), vol=vwin_list)

    js_update.args['noop'] = None  # keep reference stable

    # Navigation buttons
    btn_first = Button(label="⏮ First")
    btn_left = Button(label="◀ Left")
    btn_right = Button(label="Right ▶")
    btn_last = Button(label="Last ⏭")
    goto_input = TextInput(title="Go to (YYYY-MM-DD)", placeholder="2023-11-01", value="")
    btn_goto = Button(label="Go")

    cb_left = CustomJS(args=dict(state=state, updater=js_update), code="""
        let s = state.data['start'][0];
        const step = state.data['step'][0];
        const win = state.data['window'][0];
        s = Math.max(0, s - step);
        state.data['start'][0] = s;
        state.change.emit();
        updater.execute();
    """)

    cb_right = CustomJS(args=dict(state=state, full=full, updater=js_update), code="""
        let s = state.data['start'][0];
        const step = state.data['step'][0];
        const win = state.data['window'][0];
        const n = full.data['time'].length;
        s = Math.min(Math.max(0, n - win), s + step);
        state.data['start'][0] = s;
        state.change.emit();
        updater.execute();
    """)

    cb_first = CustomJS(args=dict(state=state, updater=js_update), code="""
        state.data['start'][0] = 0;
        state.change.emit();
        updater.execute();
    """)

    cb_last = CustomJS(args=dict(state=state, full=full, updater=js_update), code="""
        const n = full.data['time'].length;
        const win = state.data['window'][0];
        state.data['start'][0] = Math.max(0, n - win);
        state.change.emit();
        updater.execute();
    """)

    btn_first.js_on_click(cb_first)
    btn_left.js_on_click(cb_left)
    btn_right.js_on_click(cb_right)
    btn_last.js_on_click(cb_last)

    # Go to date callback
    cb_goto = CustomJS(args=dict(state=state, full=full, updater=js_update, inp=goto_input), code="""
        const val = (inp.value || '').trim();
        if (!val) { return; }
        const t = full.data['time'];
        const n = t.length;
        if (n === 0) { return; }
        // Parse target date (assume local time)
        let target = Date.parse(val);
        if (!isFinite(target)) {
            // Try append time
            target = Date.parse(val + 'T00:00:00');
        }
        if (!isFinite(target)) { return; }
        // Binary search for first index >= target
        let lo = 0, hi = n - 1, pos = n;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            const tm = t[mid];
            if (tm >= target) { pos = mid; hi = mid - 1; }
            else { lo = mid + 1; }
        }
        const win = state.data['window'][0];
        let s = pos;
        if (s > n - win) s = Math.max(0, n - win);
        if (s < 0) s = 0;
        state.data['start'][0] = s;
        state.change.emit();
        updater.execute();
    """)
    btn_goto.js_on_click(cb_goto)

    # Initial render already set from Python; JS runs on button clicks

    # Draw overlay lines using sources; fill initial window
    overlay_palette = (Category20[20] if len(overlay_names) > 10 else Category10[10]) if overlay_names else []
    dash_patterns = ['solid', 'dashed', 'dotted', 'dotdash', 'dashdot']
    for idx, name in enumerate(overlay_names):
        src = overlay_sources[idx]
        if name in aligned_overlay_full.columns:
            ys = pd.to_numeric(aligned_overlay_full[name].iloc[start_idx:end_idx + 1], errors='coerce')
            src.data = dict(time=tw.astype('datetime64[ms]'), y=ys.values)
        p.line(
            x='time', y='y', source=src,
            line_width=1.8,
            line_color=(overlay_palette[idx % len(overlay_palette)] if overlay_palette else '#1f77b4'),
            line_dash=dash_patterns[idx % len(dash_patterns)],
            legend_label=name,
        )
    # Only set legend behavior if a legend exists
    if getattr(p, 'legend', None):
        try:
            if isinstance(p.legend, list):
                for lg in p.legend:
                    lg.click_policy = 'hide'
            else:
                p.legend.click_policy = 'hide'
        except Exception:
            pass

    layout_children = [Div(text=f"<b>Window:</b> {window_bars} bars &nbsp; <b>Step:</b> {step_bars} bars"), pv, p]

    # Separate-pane indicators
    if separate_names:
        p2 = figure(
            x_axis_type="datetime",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_drag="pan",
            active_scroll="wheel_zoom",
            height=260,
            title="Indicators",
            sizing_mode="stretch_width",
            x_range=p.x_range,
        )
        sep_palette = (Category20[20] if len(separate_names) > 10 else Category10[10]) if separate_names else []
        for idx, name in enumerate(separate_names):
            src = separate_sources[idx]
            if name in aligned_separate_full.columns:
                ys = pd.to_numeric(aligned_separate_full[name].iloc[start_idx:end_idx + 1], errors='coerce')
                src.data = dict(time=tw.astype('datetime64[ms]'), y=ys.values)
            p2.line(
                x='time', y='y', source=src,
                line_width=1.8,
                line_color=(sep_palette[idx % len(sep_palette)] if sep_palette else '#2ca02c'),
                line_dash=dash_patterns[idx % len(dash_patterns)],
                legend_label=name,
            )
        # Only set legend behavior if a legend exists
        if getattr(p2, 'legend', None):
            try:
                if isinstance(p2.legend, list):
                    for lg in p2.legend:
                        lg.click_policy = 'hide'
                else:
                    p2.legend.click_policy = 'hide'
            except Exception:
                pass
        # autoscale
        if not aligned_separate_full.empty:
            vals2 = pd.to_numeric(aligned_separate_full.iloc[start_idx:end_idx + 1].stack(), errors='coerce')
            if not vals2.empty and np.isfinite(vals2).any():
                vmin = float(vals2.min()); vmax = float(vals2.max())
                pad2 = (vmax - vmin) * 0.20 if vmax > vmin else 1e-6
                p2.y_range.start = vmin - pad2
                p2.y_range.end = vmax + pad2
        # pass fig_sep to JS so y auto-fit updates while stepping
        js_update.args['fig_sep'] = p2
        layout_children.append(p2)

    # Draw rule segments (legend per rule)
    rule_palette = (Category20[20] if len(rule_names) > 10 else Category10[10]) if rule_names else []
    for idx, name in enumerate(rule_names):
        color = rule_palette[idx % len(rule_palette)] if rule_palette else 'lightgray'
        p.segment(x0='x0', y0='y0', x1='x1', y1='y1', source=rule_sources[name],
                  line_dash='dashed', line_color=color, line_alpha=0.8,
                  legend_label=name)

    controls = row(btn_first, btn_left, btn_right, btn_last, goto_input, btn_goto)
    layout_children.append(controls)
    layout = column(*layout_children, sizing_mode="stretch_width")

    # Save to HTML (standalone)
    outdir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If directory cannot be created, fallback to current working directory
        outdir = Path('.')
    if not output_filename:
        # Derive a safe filename from title
        base = re.sub(r"[^A-Za-z0-9_-]+", "_", title.strip()) or "ohlc_plot"
        output_filename = f"{base}.html"
    outfile = outdir / output_filename
    save(layout, filename=str(outfile), resources=INLINE, title=title)

    if open_in_browser:
        show(layout)



#%%
if __name__ == "__main__":
    ochl = pd.read_csv('../projs/crypto_proj/doge_intraday_processed_fill.csv', index_col=0, parse_dates=True)
    features = pd.read_csv('../projs/crypto_proj/doge_features.csv', index_col=0, parse_dates=True)
    indicators = {
        'rsi_14': 'separate',
        'ema_20': 'overlay',
        'k_14_3_3': 'separate',
    }



    def make_rsi_signal_frame(
        ohlc: pd.DataFrame,
        features: pd.DataFrame,
        *,
        rsi_col: str = "rsi_14",
        low_thresh: float = 20.0,
        high_thresh: float = 80.0,
    ) -> pd.DataFrame:
        """
        Return a boolean DataFrame aligned to `ohlc.index` with two columns:
        - rsi_le_{low_thresh}: True where features[rsi_col] <= low_thresh
        - rsi_ge_{high_thresh}: True where features[rsi_col] >= high_thresh

        Notes
        - Aligns features to OHLC via exact index reindexing (no asof); rows not present
        in features become False in both columns.
        - Raises if the RSI column is missing in features.
        """
        if rsi_col not in features.columns:
            sample = sorted(list(features.columns))[:20]
            raise ValueError(
                f"Column '{rsi_col}' not found in features. Sample available: {sample}"
            )

        o = _ensure_datetime_index(ohlc)
        f = _ensure_datetime_index(features)
        rsi = f[rsi_col].reindex(o.index)

        le_low = (rsi <= low_thresh).fillna(False)
        ge_high = (rsi >= high_thresh).fillna(False)
        print(np.sum(le_low))
        out = pd.DataFrame(
            {
                f"rsi_le_{int(low_thresh)}": le_low.astype(bool),
                f"rsi_ge_{int(high_thresh)}": ge_high.astype(bool),
            },
            index=o.index,
        )
        print(np.sum(out.rsi_le_20))
        return out
    rules = make_rsi_signal_frame(ochl, features)
    rules_dict = {
        'rsi_le_20': 'rsi_le_20',
        'rsi_ge_80': 'rsi_ge_80',
    }
    plot_df = pd.concat([ochl, features, rules], axis=1)

#%%
    plot_ohlc_basic(plot_df, 
                    window_bars=300,
                    step_bars=120,
                    indicators=indicators,
                    rules_dict=rules_dict)

# %%
