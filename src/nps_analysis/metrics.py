"""NPS calculations with uncertainty, segment tables and group comparisons."""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np
import pandas as pd

from . import config as C


@dataclass(frozen=True)
class NPSResult:
    n: int
    promoters: float  # shares, 0-1
    passives: float
    detractors: float
    nps: float  # -100 to +100
    se: float  # standard error, NPS points
    ci_low: float
    ci_high: float


def nps(scores: pd.Series | np.ndarray, z: float = 1.96) -> NPSResult:
    """Net Promoter Score = %Promoters (9-10) - %Detractors (0-6).

    The standard error treats each response as +1 / 0 / -1, so
    Var(NPS) = [p_P + p_D - (p_P - p_D)^2] / n  (on the -1..1 scale).
    """
    s = pd.Series(scores).dropna().astype(int)
    if ((s < 0) | (s > 10)).any():
        raise ValueError("NPS scores must be between 0 and 10")
    n = len(s)
    if n == 0:
        nan = float("nan")
        return NPSResult(0, nan, nan, nan, nan, nan, nan, nan)
    p_p = float((s >= 9).mean())
    p_d = float((s <= 6).mean())
    p_pass = 1 - p_p - p_d
    value = p_p - p_d
    se = sqrt(max(p_p + p_d - value**2, 0) / n)
    return NPSResult(n, p_p, p_pass, p_d, 100 * value, 100 * se, 100 * (value - z * se), 100 * (value + z * se))


def nps_table(df: pd.DataFrame, by: str | list[str], order: list | None = None) -> pd.DataFrame:
    """One row per group: n, segment shares, NPS with 95% CI, mean CSAT and a small-sample flag."""
    rows = []
    for key, g in df.groupby(by, observed=True, sort=False):
        r = nps(g["nps_score"])
        keys = key if isinstance(key, tuple) else (key,)
        row = dict(zip([by] if isinstance(by, str) else by, keys))
        row.update({
            "n": r.n,
            "promoters_pct": round(100 * r.promoters, 1),
            "passives_pct": round(100 * r.passives, 1),
            "detractors_pct": round(100 * r.detractors, 1),
            "nps": round(r.nps, 1),
            "ci_low": round(r.ci_low, 1),
            "ci_high": round(r.ci_high, 1),
            "mean_csat": round(float(g["satisfaction_score"].astype(float).mean()), 2),
            "low_sample": r.n < C.MIN_SEGMENT_N,
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if order is not None and isinstance(by, str):
        out[by] = pd.Categorical(out[by], categories=order, ordered=True)
        out = out.sort_values(by)
        out[by] = out[by].astype(str)
    return out.reset_index(drop=True)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def compare_nps(a: pd.Series, b: pd.Series) -> dict:
    """Two-sided z-test for the difference between two independent NPS values (A - B)."""
    ra, rb = nps(a), nps(b)
    diff = ra.nps - rb.nps
    se = sqrt(ra.se**2 + rb.se**2)
    z = diff / se if se > 0 else float("nan")
    p = 2 * (1 - _norm_cdf(abs(z))) if se > 0 else float("nan")
    return {"nps_a": round(ra.nps, 1), "nps_b": round(rb.nps, 1), "n_a": ra.n, "n_b": rb.n,
            "difference": round(diff, 1), "ci_low": round(diff - 1.96 * se, 1), "ci_high": round(diff + 1.96 * se, 1),
            "z": round(z, 2), "p_value": round(p, 4), "significant_5pct": bool(p < 0.05)}


def competitor_gaps(df: pd.DataFrame, focus: str = C.FOCUS_BRAND) -> pd.DataFrame:
    """Focus brand minus each competitor, per country and overall, with significance tests."""
    rows = []
    for country in ["All markets", *C.COUNTRIES]:
        sub = df if country == "All markets" else df[df["country"] == country]
        for comp in [b for b in C.BRANDS if b != focus]:
            res = compare_nps(sub.loc[sub["vehicle_brand"] == focus, "nps_score"], sub.loc[sub["vehicle_brand"] == comp, "nps_score"])
            rows.append({"country": country, "focus_brand": focus, "competitor": comp, **res})
    return pd.DataFrame(rows)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """False-discovery-rate adjusted p-values, for when many segment comparisons are tested at once."""
    p = p_values.to_numpy(dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adjusted, 1)
    return pd.Series(out, index=p_values.index)
