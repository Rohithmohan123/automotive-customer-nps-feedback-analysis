"""Static charts (PNG) for the report. Colours follow a fixed, colour-blind-checked palette:
brands and countries keep the same colour in every chart, NPS segments use a diverging
red / grey / blue scheme, and heatmaps use a single-hue or diverging scale."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from . import config as C  # noqa: E402
from .metrics import nps, nps_table  # noqa: E402

# Palette (light mode)
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
BRAND_COLOR = dict(zip(C.BRANDS, SERIES))  # TVS blue, Bajaj orange, Haojue aqua
COUNTRY_COLOR = dict(zip(C.COUNTRIES, SERIES))
SEGMENT_COLOR = {"Detractor": "#e34948", "Passive": "#c3c2b7", "Promoter": "#2a78d6"}
DIVERGING = LinearSegmentedColormap.from_list("nps_div", ["#c23b3a", "#e88d8c", "#f0efec", "#86b6ef", "#1c5cab"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#f0f5fc", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 24, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK2,
})

FOOTNOTE = "Synthetic data for demonstration. Not real customer or brand performance."


def _finish(fig, path, note: str = FOOTNOTE) -> None:
    # Place the footnote just below the lowest element (axis labels, legends) so it never overlaps
    renderer = fig.canvas.get_renderer()
    bottom = fig.get_tightbbox(renderer).y0 * fig.dpi / fig.dpi  # inches
    y = (bottom - 0.12) / fig.get_figheight()
    fig.text(0.01, y, note, fontsize=7.5, color=MUTED, ha="left", va="top", transform=fig.transFigure)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def score_distribution(df: pd.DataFrame, path) -> None:
    counts = df["nps_score"].value_counts().reindex(range(11), fill_value=0)
    colors = [SEGMENT_COLOR["Detractor"]] * 7 + [SEGMENT_COLOR["Passive"]] * 2 + [SEGMENT_COLOR["Promoter"]] * 2
    r = nps(df["nps_score"])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(counts.index, counts.values, color=colors, width=0.86, edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(range(11))
    ax.set_xlabel("Likelihood to recommend (0-10)")
    ax.set_ylabel("Respondents")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    for label, x, share in [("Detractors", 3, r.detractors), ("Passives", 7.5, r.passives), ("Promoters", 9.5, r.promoters)]:
        ax.text(x, counts.max() * 1.07, f"{label} {share:.0%}", ha="center", color=INK, fontsize=9.5)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title(f"Score distribution: overall NPS {r.nps:+.0f} (95% CI {r.ci_low:+.0f} to {r.ci_high:+.0f})")
    _subtitle(ax, f"n = {r.n:,} valid responses across four markets")
    _finish(fig, path)


def nps_by_group(tbl: pd.DataFrame, col: str, path, title: str, color_map: dict | None = None) -> None:
    t = tbl.sort_values("nps")
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(t) + 1.6))
    y = np.arange(len(t))
    colors = [color_map.get(v, SERIES[0]) if color_map else SERIES[0] for v in t[col]]
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.hlines(y, t["ci_low"], t["ci_high"], color=colors, linewidth=2.2, alpha=0.55)
    ax.scatter(t["nps"], y, s=70, color=colors, zorder=3, edgecolor=SURFACE, linewidth=2)
    for yi, (_, row) in zip(y, t.iterrows()):
        ax.text(row["ci_high"] + 1.2, yi, f"{row['nps']:+.0f}  (n={row['n']:,})", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, t[col])
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("NPS (dot) with 95% confidence interval (line)")
    lo, hi = t["ci_low"].min(), t["ci_high"].max()
    ax.set_xlim(min(lo - 5, -5), max(hi + 18, 10))
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title(title)
    _finish(fig, path)


def segment_mix(df: pd.DataFrame, col: str, order: list, path, title: str) -> None:
    tbl = nps_table(df, col, order)
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(tbl) + 1.8))
    y = np.arange(len(tbl))[::-1]
    left = np.zeros(len(tbl))
    for seg, key in [("Detractor", "detractors_pct"), ("Passive", "passives_pct"), ("Promoter", "promoters_pct")]:
        vals = tbl[key].to_numpy()
        ax.barh(y, vals, left=left, color=SEGMENT_COLOR[seg], height=0.62, edgecolor=SURFACE, linewidth=2, label=seg)
        for yi, l, v in zip(y, left, vals):
            if v >= 7:
                ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=8.5, color="#ffffff" if seg != "Passive" else INK)
        left += vals
    for yi, (_, row) in zip(y, tbl.iterrows()):
        ax.text(101.5, yi, f"NPS {row['nps']:+.0f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, tbl[col])
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.10))
    ax.set_title(title)
    _finish(fig, path)


def brand_country_heatmap(df: pd.DataFrame, path) -> None:
    tbl = nps_table(df, ["country", "vehicle_brand"])
    mat = tbl.pivot(index="country", columns="vehicle_brand", values="nps").reindex(index=C.COUNTRIES, columns=C.BRANDS)
    nmat = tbl.pivot(index="country", columns="vehicle_brand", values="n").reindex(index=C.COUNTRIES, columns=C.BRANDS)
    lim = max(abs(mat.min().min()), abs(mat.max().max()), 10)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    im = ax.imshow(mat.values, cmap=DIVERGING, norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            dark = abs(v) > lim * 0.55
            ax.text(j, i - 0.08, f"{v:+.0f}", ha="center", va="center", fontsize=12, fontweight="bold", color="#ffffff" if dark else INK)
            ax.text(j, i + 0.22, f"n={int(nmat.iat[i, j])}", ha="center", va="center", fontsize=8, color="#ffffff" if dark else INK2)
    ax.set_xticks(range(len(C.BRANDS)), C.BRANDS)
    ax.set_yticks(range(len(C.COUNTRIES)), C.COUNTRIES)
    ax.tick_params(length=0, colors=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(C.BRANDS)), minor=True)
    ax.set_yticks(np.arange(-.5, len(C.COUNTRIES)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, length=0)
    cb.set_label("NPS", color=INK2)
    ax.set_title("NPS by brand and market")
    _finish(fig, path)


def competitor_gap_chart(gaps: pd.DataFrame, path) -> None:
    rows = ["All markets", *C.COUNTRIES]
    comps = [b for b in C.BRANDS if b != C.FOCUS_BRAND]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    offsets = dict(zip(comps, [-0.14, 0.14]))
    ax.axvline(0, color=AXIS, linewidth=1)
    for comp in comps:
        g = gaps[gaps["competitor"] == comp].set_index("country").reindex(rows)
        y = np.arange(len(rows))[::-1] + offsets[comp]
        ax.hlines(y, g["ci_low"], g["ci_high"], color=BRAND_COLOR[comp], linewidth=2.2, alpha=0.55)
        filled = g["significant_5pct"].to_numpy()
        ax.scatter(g["difference"][filled], y[filled], s=64, color=BRAND_COLOR[comp], zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.scatter(g["difference"][~filled], y[~filled], s=64, facecolor=SURFACE, edgecolor=BRAND_COLOR[comp], linewidth=2, zorder=3,
                   label=f"{C.FOCUS_BRAND} vs {comp}")
    ax.set_yticks(np.arange(len(rows))[::-1], rows)
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"NPS difference ({C.FOCUS_BRAND} minus competitor), 95% CI")
    ax.text(0.99, 0.02, f"{C.FOCUS_BRAND} ahead →", transform=ax.transAxes, ha="right", color=MUTED, fontsize=8.5)
    ax.text(0.01, 0.02, f"← {C.FOCUS_BRAND} behind", transform=ax.transAxes, ha="left", color=MUTED, fontsize=8.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2)
    ax.set_title(f"Competitor gaps: {C.FOCUS_BRAND} vs Bajaj and Haojue")
    _subtitle(ax, "Hollow marker = difference not statistically significant at the 5% level")
    _finish(fig, path)


def ownership_trend(df: pd.DataFrame, path) -> None:
    known = df[df["ownership_band"] != "Unknown"]
    overall = nps_table(known, "ownership_band", C.OWNERSHIP_ORDER)
    x = np.arange(len(C.OWNERSHIP_ORDER))
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.fill_between(x, overall["ci_low"], overall["ci_high"], color="#52514e", alpha=0.10, linewidth=0, label="All brands, 95% CI")
    ax.plot(x, overall["nps"], color=INK, linewidth=2.6, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label="All brands", zorder=4)
    for brand in C.BRANDS:
        t = nps_table(known[known["vehicle_brand"] == brand], "ownership_band", C.OWNERSHIP_ORDER)
        ax.plot(x, t["nps"], color=BRAND_COLOR[brand], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=brand)
    ax.set_xticks(x, C.OWNERSHIP_ORDER)
    ax.set_xlim(-0.3, len(x) - 0.7)
    ax.set_ylabel("NPS")
    ax.set_xlabel("Vehicle ownership duration")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=5)
    ax.set_title("NPS by ownership duration")
    _finish(fig, path)


def theme_heatmap(mentions: pd.DataFrame, path, title: str, subtitle: str) -> None:
    mat = mentions.pivot(index="themes", columns="country", values="pct_of_commented").reindex(columns=C.COUNTRIES).fillna(0)
    mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat.values, cmap=SEQUENTIAL, vmin=0, vmax=max(40, mat.values.max()), aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9.5, color="#ffffff" if v > 24 else INK)
    ax.set_xticks(range(mat.shape[1]), mat.columns)
    ax.set_yticks(range(mat.shape[0]), mat.index)
    ax.tick_params(length=0, colors=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, mat.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, mat.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, length=0)
    ax.set_title(title)
    _subtitle(ax, subtitle)
    _finish(fig, path)


def csat_vs_nps(df: pd.DataFrame, path) -> None:
    sub = df.dropna(subset=["satisfaction_score"]).copy()
    sub["satisfaction_score"] = sub["satisfaction_score"].astype(int)
    segment_mix(sub.assign(csat=sub["satisfaction_score"].map(lambda v: f"CSAT {v}")), "csat",
                [f"CSAT {v}" for v in range(1, 6)], path, "NPS segment mix by satisfaction score (1-5)")


def monthly_trend(df: pd.DataFrame, path) -> None:
    t = nps_table(df.dropna(subset=["survey_month"]), "survey_month").sort_values("survey_month")
    x = np.arange(len(t))
    overall = nps(df["nps_score"]).nps
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.fill_between(x, t["ci_low"], t["ci_high"], color=SERIES[0], alpha=0.12, linewidth=0)
    ax.plot(x, t["nps"], color=SERIES[0], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.axhline(overall, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(len(x) - 0.6, overall, f"Year {overall:+.0f}", color=INK2, fontsize=8.5, va="bottom", ha="right")
    ax.set_xticks(x, [pd.Period(m).strftime("%b") for m in t["survey_month"]])
    ax.set_ylabel("NPS")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title("Monthly NPS, 2024 (shaded band = 95% CI)")
    k = int(((t["ci_low"] > overall) | (t["ci_high"] < overall)).sum())
    msg = (f"{k} of {len(t)} monthly intervals exclude the annual NPS"
           + ("; one or two are expected by chance alone, so there is no clear trend" if k <= 2 else ""))
    _subtitle(ax, msg)
    _finish(fig, path)
