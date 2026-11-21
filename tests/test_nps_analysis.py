"""Unit tests for the NPS maths, cleaning rules and theme tagging."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nps_analysis import clean as CL  # noqa: E402
from nps_analysis import generate_data as G  # noqa: E402
from nps_analysis import metrics as M  # noqa: E402
from nps_analysis import themes as T  # noqa: E402


# ---------------------------------------------------------------- NPS maths
def test_nps_known_value():
    # 3 promoters, 2 passives, 5 detractors -> 30% - 50% = -20
    r = M.nps([10, 9, 9, 8, 7, 6, 5, 3, 0, 1])
    assert r.n == 10
    assert r.nps == pytest.approx(-20)
    assert (r.promoters, r.passives, r.detractors) == pytest.approx((0.3, 0.2, 0.5))


def test_nps_boundaries_are_standard():
    assert M.nps([9]).nps == 100 and M.nps([8]).nps == 0 and M.nps([7]).nps == 0 and M.nps([6]).nps == -100


def test_nps_standard_error_formula():
    scores = [10] * 30 + [8] * 20 + [3] * 50
    r = M.nps(scores)
    expected = 100 * np.sqrt((0.3 + 0.5 - (-0.2) ** 2) / 100)
    assert r.se == pytest.approx(expected)
    assert r.ci_low < r.nps < r.ci_high


def test_nps_rejects_out_of_range():
    with pytest.raises(ValueError):
        M.nps([5, 11])


def test_nps_empty_returns_nan():
    assert np.isnan(M.nps([]).nps)


def test_compare_nps_detects_large_difference_and_not_noise():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.choice([10, 9, 8], size=400))
    b = pd.Series(rng.choice([2, 5, 8], size=400))
    assert M.compare_nps(a, b)["significant_5pct"]
    same = M.compare_nps(a.iloc[:200], a.iloc[200:])
    assert abs(same["difference"]) < 15


def test_benjamini_hochberg_monotone_and_bounded():
    p = pd.Series([0.01, 0.04, 0.03, 0.5])
    adj = M.benjamini_hochberg(p)
    assert (adj >= p).all() and (adj <= 1).all()
    assert adj[0] == pytest.approx(0.04)


def test_nps_table_flags_low_sample():
    df = pd.DataFrame({"g": ["a"] * 60 + ["b"] * 10, "nps_score": [9] * 70, "satisfaction_score": [5] * 70})
    t = M.nps_table(df, "g").set_index("g")
    assert not t.loc["a", "low_sample"] and t.loc["b", "low_sample"]


# ---------------------------------------------------------------- Cleaning
@pytest.fixture(scope="module")
def cleaned():
    raw = G.inject_errors(G.generate_clean(n=800, seed=7), seed=7).astype(str).replace({"nan": np.nan, "": np.nan})
    return CL.clean(raw)


def test_clean_output_passes_validation(cleaned):
    df, _ = cleaned
    assert CL.validate(df) == []


def test_clean_standardises_names(cleaned):
    df, _ = cleaned
    assert set(df["country"]) <= {"Nigeria", "Egypt", "Philippines", "DR Congo"}
    assert set(df["vehicle_brand"]) <= {"TVS", "Bajaj", "Haojue"}


def test_cleaning_log_reconciles(cleaned):
    df, log = cleaned
    l = log.set_index("rule")["rows_affected"]
    dropped = l[[r for r in l.index if log.set_index("rule").loc[r, "action"].startswith("dropped")]].sum()
    assert l["Rows in raw export"] - dropped == l["Rows in clean dataset"] == len(df)


def test_segments_and_bands(cleaned):
    df, _ = cleaned
    assert (df.loc[df["nps_score"] >= 9, "nps_segment"] == "Promoter").all()
    assert (df.loc[df["nps_score"] <= 6, "nps_segment"] == "Detractor").all()
    assert CL.ownership_band(0) == "0-6 months" and CL.ownership_band(12) == "12-24 months" and CL.ownership_band(pd.NA) == "Unknown"


def test_generator_is_reproducible():
    a = G.generate_clean(n=50, seed=1)
    b = G.generate_clean(n=50, seed=1)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------- Themes
@pytest.mark.parametrize("text,theme", [
    ("Hard to find genuine spare parts, waited weeks.", "Spare parts availability"),
    ("Very good mileage, I spend less on fuel every week.", "Fuel efficiency"),
    ("Dealer was slow with registration papers.", "Dealer experience"),
    ("Warranty claim was rejected without explanation.", "After-sales service"),
])
def test_tag_themes(text, theme):
    assert theme in T.tag_themes(text)


def test_sentiment():
    assert T.sentiment("Excellent fuel economy compared to my old bike.") == "Positive"
    assert T.sentiment("Seat is uncomfortable on long trips.") == "Negative"
    assert T.sentiment("Expected more.") == "Neutral"
