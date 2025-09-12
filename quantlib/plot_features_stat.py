import io
import base64
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None


@dataclass
class ReportConfig:
    window: int = 240
    step: int = 10
    bins: int = 50
    top_k_corr: int = 3
    title: str = "Feature Report"
    cmap: str = "RdBu_r"  # -1 red, +1 blue, 0 white
    figsize_heatmap: tuple[float, float] = (12.0, 10.0)
    figsize_panel: tuple[float, float] = (12.0, 8.0)
    max_cols_heatmap: int = 200


def _numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()


def _corr_matrix(df_num: pd.DataFrame) -> pd.DataFrame:
    # Pearson correlation; pairwise complete by default
    return df_num.corr(method="pearson")


def _top_k_for(col: str, corr: pd.DataFrame, k: int) -> list[tuple[str, float]]:
    s = corr[col].drop(labels=[col]).dropna()
    if s.empty:
        return []
    top = s.abs().nlargest(min(k, len(s))).index
    return [(name, float(corr.loc[col, name])) for name in top]


def _rolling_quantiles(s: pd.Series, *, window: int, step: int,
                       qs: Iterable[float]) -> pd.DataFrame:
    # Compute rolling quantiles, then sample every `step` and forward-fill
    out = {}
    for q in qs:
        rq = s.rolling(window=window, min_periods=window).quantile(q)
        if step > 1:
            rq = rq.iloc[::step].reindex(s.index).ffill()
        out[q] = rq
    return pd.DataFrame(out, index=s.index)


def _format_stats(s: pd.Series) -> str:
    qs = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0]
    idx = ["min", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "max"]
    qv = s.quantile(qs).to_numpy()
    lines = [f"{name:>4}: {val: .6g}" for name, val in zip(idx, qv)]
    return "\n".join(lines)


def _draw_corr_heatmap(df_num: pd.DataFrame, cfg: ReportConfig) -> plt.Figure:
    corr = _corr_matrix(df_num)
    cols = list(corr.columns)
    if len(cols) > cfg.max_cols_heatmap:
        cols = cols[: cfg.max_cols_heatmap]
        corr = corr.loc[cols, cols]

    fig, ax = plt.subplots(figsize=cfg.figsize_heatmap)
    if sns is not None:
        sns.heatmap(
            corr,
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            cmap=cfg.cmap,
            square=False,
            cbar_kws={"shrink": 0.75},
            ax=ax,
        )
    else:  # Fallback to imshow
        im = ax.imshow(corr.values, vmin=-1.0, vmax=1.0, cmap=cfg.cmap, aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.75)
        ax.set_xticks(np.arange(corr.shape[1]))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(np.arange(corr.shape[0]))
        ax.set_yticklabels(corr.index)
    ax.set_title(f"Correlation Heatmap ({len(corr.columns)} features)")
    fig.tight_layout()
    return fig


def _draw_feature_panel(
    name: str,
    s: pd.Series,
    corr: Optional[pd.DataFrame],
    cfg: ReportConfig,
) -> plt.Figure:
    # Figure and layout: 60% top row, 40% bottom row
    fig = plt.figure(figsize=cfg.figsize_panel)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[60, 40])

    # Top row split into histogram (left) and text (right)
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[2, 1])
    ax_hist = fig.add_subplot(gs_top[0, 0])
    ax_text = fig.add_subplot(gs_top[0, 1])

    # Bottom row: rolling quantiles over time
    ax_quant = fig.add_subplot(gs[1])

    # Histogram
    clean = s.dropna()
    if len(clean) > 0:
        ax_hist.hist(clean.values, bins=cfg.bins, color="#4c72b0", alpha=0.85)
    ax_hist.set_title(f"{name} — distribution")
    ax_hist.set_ylabel("freq")

    # Text block: quantiles and top-k correlations
    ax_text.axis("off")
    stats_txt = _format_stats(clean)
    lines = ["Stats:", stats_txt]
    if corr is not None and name in corr.columns:
        top = _top_k_for(name, corr, cfg.top_k_corr)
        if top:
            lines.append("\nTop-|corr|:")
            for k, v in top:
                lines.append(f"  {k}: {v:+.3f}")
    ax_text.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace")

    # Rolling quantiles
    qs = [0.05, 0.25, 0.50, 0.75, 0.95]
    rq = _rolling_quantiles(s.astype(float), window=cfg.window, step=cfg.step, qs=qs)
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
    for (q, col), color in zip(rq.items(), colors):
        ax_quant.plot(rq.index, col, label=f"q{int(q*100)}", color=color, linewidth=1.2)
    ax_quant.set_title(f"{name} — rolling quantiles (w={cfg.window}, step={cfg.step})")
    # Use a fixed legend location to avoid the expensive 'best' placement
    ax_quant.legend(ncol=5, fontsize=9, frameon=False, loc="upper right")
    ax_quant.grid(True, alpha=0.25)

    fig.suptitle(name, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def visualize_features_report(
    df: pd.DataFrame,
    out_path: str,
    *,
    window: int = 240,
    step: int = 10,
    bins: int = 50,
    title: str = "Feature Report",
    output: str | None = None,
    top_k_corr: int = 3,
    cmap: str = "RdBu_r",
    figsize_heatmap: tuple[float, float] = (12.0, 10.0),
    figsize_panel: tuple[float, float] = (12.0, 8.0),
    max_cols_heatmap: int = 200,
) -> str:
    """
    Generate a correlation heatmap and per-feature panels, exported as PDF or HTML.

    Layout for each feature page:
    - Upper 60%: left histogram, right text (stats + top-K correlations)
    - Lower 40%: rolling quantiles (5/25/50/75/95) over time
    """
    cfg = ReportConfig(
        window=window,
        step=step,
        bins=bins,
        top_k_corr=top_k_corr,
        title=title,
        cmap=cmap,
        figsize_heatmap=figsize_heatmap,
        figsize_panel=figsize_panel,
        max_cols_heatmap=max_cols_heatmap,
    )

    df_num = _numeric_df(df)
    corr_full = _corr_matrix(df_num) if df_num.shape[1] > 1 else None

    # Decide output type
    fmt = output or (out_path.split(".")[-1].lower() if "." in out_path else "pdf")

    if fmt == "pdf":
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(out_path) as pdf:
            # Heatmap page
            if df_num.shape[1] >= 2:
                fig = _draw_corr_heatmap(df_num, cfg)
                fig.suptitle(cfg.title)
                pdf.savefig(fig)
                plt.close(fig)

            # Feature pages
            for name in df_num.columns:
                fig = _draw_feature_panel(name, df_num[name], corr_full, cfg)
                pdf.savefig(fig)
                plt.close(fig)
        return out_path

    # Fallback: HTML with embedded PNGs
    images: list[tuple[str, str]] = []  # (caption, base64_png)

    def fig_to_b64(fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    if df_num.shape[1] >= 2:
        fig = _draw_corr_heatmap(df_num, cfg)
        images.append(("Correlation Heatmap", fig_to_b64(fig)))

    for name in df_num.columns:
        fig = _draw_feature_panel(name, df_num[name], corr_full, cfg)
        images.append((name, fig_to_b64(fig)))

    html_parts = [
        "<html><head><meta charset='utf-8'><title>{}</title>".format(title),
        "<style>body{font-family:Arial,Helvetica,sans-serif} img{max-width:100%} .block{margin:16px 0}</style>",
        "</head><body>",
        f"<h1>{title}</h1>",
    ]
    for caption, b64 in images:
        html_parts.append(f"<div class='block'><h2>{caption}</h2><img src='data:image/png;base64,{b64}'/></div>")
    html_parts.append("</body></html>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    return out_path
