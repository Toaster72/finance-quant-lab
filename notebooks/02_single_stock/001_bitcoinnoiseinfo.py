from __future__ import annotations

"""
Bitcoin Noise vs Information — FULL WORKING SCRIPT (Mac / Python 3.9+ safe)

What this does:
- Downloads BTC-USD daily prices (yfinance)
- Computes daily returns, rolling realized volatility
- Defines surprise = |ret| / sigma
- Buckets days by surprise quantiles (noise / mid / large / extreme)
- Computes forward returns + sign persistence
- Computes full retracement probability within 5/10/20 days
- Saves summary + detail CSVs and charts into outputs/

Run:
  /usr/local/bin/python3 bitcoinnoiseinfo.py
or:
  python3 bitcoinnoiseinfo.py

Dependencies:
  pip3 install yfinance pandas numpy matplotlib
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    ticker: str = "BTC-USD"
    start: str = "2015-01-01"
    end: Optional[str] = None
    vol_window: int = 20
    fwd_horizons: Tuple[int, ...] = (1, 5, 10, 20)
    retrace_horizons: Tuple[int, ...] = (5, 10, 20)
    bucket_quantiles: Tuple[float, ...] = (0.0, 0.50, 0.90, 0.99, 1.0)
    bucket_labels: Tuple[str, ...] = ("noise_0_50", "mid_50_90", "large_90_99", "extreme_99_100")
    outdir: str = "outputs"


# -----------------------------
# IO helpers
# -----------------------------
def ensure_dirs(outdir: str) -> Dict[str, Path]:
    base = Path(outdir)
    charts = base / "charts"
    tables = base / "tables"
    charts.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return {"base": base, "charts": charts, "tables": tables}


# -----------------------------
# Data + features
# -----------------------------
def download_prices(ticker: str, start: str, end: Optional[str]) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df.empty:
        raise RuntimeError("No data returned. Check ticker/start/end or your internet connection.")

    # Robustly extract 1D close series (handles MultiIndex / edge cases)
    if isinstance(df.columns, pd.MultiIndex):
        # Expected format like ('Close', 'BTC-USD')
        if ("Close", ticker) in df.columns:
            px = df[("Close", ticker)]
        else:
            # Fallback: take first column under "Close"
            px = df["Close"].iloc[:, 0]
    else:
        px = df["Close"]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]

    px = px.dropna().sort_index()
    px.name = "px"
    return px


def compute_surprise(px: pd.Series, vol_window: int) -> pd.DataFrame:
    ret = px.pct_change()
    sigma = ret.rolling(vol_window).std()

    surprise = (ret.abs() / sigma)
    surprise.name = "surprise"

    out = pd.DataFrame({"px": px, "ret": ret, "sigma": sigma, "surprise": surprise})
    out = out.dropna().copy()
    out["sign"] = np.sign(out["ret"]).astype(int)
    return out


def bucket_by_quantiles(surprise: pd.Series, qs: Tuple[float, ...], labels: Tuple[str, ...]) -> pd.Series:
    edges = surprise.quantile(list(qs)).values.astype(float)

    # Ensure strictly increasing edges (avoid duplicates due to quantiles)
    eps = 1e-12
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps

    b = pd.cut(
        surprise,
        bins=edges,
        labels=list(labels),
        include_lowest=True,
        right=True,
    )
    return b.astype("category")


def add_forward_returns(df: pd.DataFrame, horizons: Tuple[int, ...]) -> pd.DataFrame:
    px = df["px"]
    ret = df["ret"]
    out = df.copy()

    for h in horizons:
        fwd = px.shift(-h) / px - 1.0
        out[f"fwd_{h}d"] = fwd
        out[f"same_sign_{h}d"] = (np.sign(fwd) == np.sign(ret)).astype(float)

    return out


def retracement_flags(px: pd.Series, ret: pd.Series, horizons: Tuple[int, ...]) -> pd.DataFrame:
    """
    Full retracement back to pre-move price within H days.

    For each day t:
      pre = px[t-1]
      if ret[t] > 0: retrace_H = 1 if min(px[t+1 : t+H]) <= pre else 0
      if ret[t] < 0: retrace_H = 1 if max(px[t+1 : t+H]) >= pre else 0
      if ret[t] == 0: retrace_H = NaN
    """
    idx = px.index
    pxv = px.values
    retv = ret.values
    pre = px.shift(1).values

    out = pd.DataFrame(index=idx)
    n = len(pxv)

    for H in horizons:
        flags = np.full(n, np.nan, dtype=float)

        for i in range(n):
            if i == 0:
                continue
            if np.isnan(retv[i]) or np.isnan(pre[i]):
                continue

            j0 = i + 1
            j1 = min(i + H, n - 1)
            if j0 > j1:
                continue

            window = pxv[j0 : j1 + 1]

            if retv[i] > 0:
                flags[i] = 1.0 if np.nanmin(window) <= pre[i] else 0.0
            elif retv[i] < 0:
                flags[i] = 1.0 if np.nanmax(window) >= pre[i] else 0.0
            else:
                flags[i] = np.nan

        out[f"retrace_{H}d"] = flags

    return out


# -----------------------------
# Summary
# -----------------------------
def summarize_by_bucket(df: pd.DataFrame, fwd_horizons: Tuple[int, ...], retrace_horizons: Tuple[int, ...]) -> pd.DataFrame:
    g = df.groupby("bucket", observed=True)
    rows = []

    for name, sub in g:
        row = {
            "bucket": name,
            "n": int(sub.shape[0]),
            "avg_surprise": float(sub["surprise"].mean()),
            "median_surprise": float(sub["surprise"].median()),
            "avg_abs_ret": float(sub["ret"].abs().mean()),
        }

        for h in fwd_horizons:
            row[f"mean_fwd_{h}d"] = float(sub[f"fwd_{h}d"].mean())
            row[f"median_fwd_{h}d"] = float(sub[f"fwd_{h}d"].median())
            row[f"same_sign_{h}d"] = float(sub[f"same_sign_{h}d"].mean())

        for H in retrace_horizons:
            row[f"retrace_prob_{H}d"] = float(sub[f"retrace_{H}d"].mean())

        # Simple info score example
        if 5 in fwd_horizons and 10 in retrace_horizons:
            row["info_score_5d_minus_retrace10"] = float(sub["same_sign_5d"].mean() - sub["retrace_10d"].mean())

        rows.append(row)

    summary = pd.DataFrame(rows).set_index("bucket")

    # Preserve bucket ordering if categorical
    if isinstance(df["bucket"].dtype, pd.CategoricalDtype):
        summary = summary.reindex(list(df["bucket"].dtype.categories))

    return summary


# -----------------------------
# Plots
# -----------------------------
def plot_surprise_hist(df: pd.DataFrame, outpath: Path) -> None:
    plt.figure()
    plt.hist(df["surprise"].values, bins=60)
    plt.title("BTC Surprise Distribution (|ret| / rolling vol)")
    plt.xlabel("Surprise")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_bucket_bar(summary: pd.DataFrame, col: str, title: str, outpath: Path) -> None:
    plt.figure()
    x = np.arange(len(summary.index))
    y = summary[col].values
    plt.bar(x, y)
    plt.xticks(x, summary.index, rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_forward_medians(summary: pd.DataFrame, horizons: Tuple[int, ...], outpath: Path) -> None:
    plt.figure()
    x = np.arange(len(summary.index))
    for h in horizons:
        col = f"median_fwd_{h}d"
        if col in summary.columns:
            plt.plot(x, summary[col].values, marker="o", label=f"{h}D")
    plt.xticks(x, summary.index, rotation=30, ha="right")
    plt.title("Median Forward Returns by Surprise Bucket")
    plt.xlabel("Bucket")
    plt.ylabel("Median forward return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main(cfg: Config) -> None:
    paths = ensure_dirs(cfg.outdir)

    px = download_prices(cfg.ticker, cfg.start, cfg.end)

    # Safety sanity check
    if not isinstance(px, pd.Series):
        raise TypeError(f"Expected px to be pd.Series, got {type(px)}")

    df = compute_surprise(px, cfg.vol_window)
    df["bucket"] = bucket_by_quantiles(df["surprise"], cfg.bucket_quantiles, cfg.bucket_labels)

    df = add_forward_returns(df, cfg.fwd_horizons)

    retr = retracement_flags(df["px"], df["ret"], cfg.retrace_horizons)
    df = df.join(retr, how="left")

    # Drop last rows where max forward return is unavailable
    max_fwd = max(cfg.fwd_horizons)
    df = df.dropna(subset=[f"fwd_{max_fwd}d"]).copy()

    summary = summarize_by_bucket(df, cfg.fwd_horizons, cfg.retrace_horizons)

    # Save tables
    summary_path = paths["tables"] / "btc_noise_info_summary.csv"
    detail_path = paths["tables"] / "btc_noise_info_detail.csv"
    summary.to_csv(summary_path)
    df.to_csv(detail_path)

    # Print summary
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 60)
    print("\n=== SUMMARY (by surprise bucket) ===")
    print(summary.round(4))

    # Save charts
    plot_surprise_hist(df, paths["charts"] / "surprise_hist.png")

    if "same_sign_5d" in summary.columns:
        plot_bucket_bar(
            summary,
            "same_sign_5d",
            "Directional Persistence (Same Sign) — 5D",
            paths["charts"] / "persistence_5d.png",
        )

    if "retrace_prob_10d" in summary.columns:
        plot_bucket_bar(
            summary,
            "retrace_prob_10d",
            "Full Retracement Probability — within 10D",
            paths["charts"] / "retrace_10d.png",
        )

    plot_forward_medians(summary, cfg.fwd_horizons, paths["charts"] / "median_fwd_returns.png")

    print("\nSaved outputs:")
    print(f" - {summary_path}")
    print(f" - {detail_path}")
    print(f" - {paths['charts'] / 'surprise_hist.png'}")
    if (paths["charts"] / "persistence_5d.png").exists():
        print(f" - {paths['charts'] / 'persistence_5d.png'}")
    if (paths["charts"] / "retrace_10d.png").exists():
        print(f" - {paths['charts'] / 'retrace_10d.png'}")
    print(f" - {paths['charts'] / 'median_fwd_returns.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, default="BTC-USD")
    p.add_argument("--start", type=str, default="2015-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--vol_window", type=int, default=20)
    p.add_argument("--outdir", type=str, default="outputs")
    args = p.parse_args()

    cfg = Config(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        vol_window=args.vol_window,
        outdir=args.outdir,
    )

    main(cfg)
