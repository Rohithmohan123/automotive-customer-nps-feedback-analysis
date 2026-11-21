"""Clean and validate the raw survey export.

Every rule is logged with the number of rows it affected, so the cleaning is auditable
(data/processed/cleaning_log.csv). Rows are dropped only when the primary metric (NPS) or
the respondent identity cannot be trusted; other problems are set to missing and kept.

Run:  python -m nps_analysis.clean
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

BLANK_COMMENTS = {"", "n/a", "na", "-", "none", "nil"}


def nps_segment(score: float) -> str:
    """Standard NPS grouping: 9-10 Promoter, 7-8 Passive, 0-6 Detractor."""
    if score >= 9:
        return "Promoter"
    if score >= 7:
        return "Passive"
    return "Detractor"


def ownership_band(months: float) -> str:
    if pd.isna(months):
        return "Unknown"
    for lo, hi, label in C.OWNERSHIP_BANDS:
        if lo <= months < hi:
            return label
    return C.OWNERSHIP_BANDS[-1][2]


def _parse_dates(s: pd.Series) -> pd.Series:
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    return iso.fillna(dmy)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean dataframe, cleaning log)."""
    log: list[dict] = []
    df = raw.copy()

    def record(rule: str, n: int, action: str) -> None:
        log.append({"rule": rule, "rows_affected": int(n), "action": action})

    record("Rows in raw export", len(df), "-")

    # 1. Trim whitespace in text columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    # 2. Exact duplicate submissions
    dup = df.duplicated()
    record("Exact duplicate rows", dup.sum(), "dropped")
    df = df[~dup]

    # 3. Missing respondent ID
    no_id = df["customer_id"].isna() | (df["customer_id"] == "")
    record("Missing customer_id", no_id.sum(), "dropped (cannot de-duplicate or trace)")
    df = df[~no_id]

    # 4. Same customer ID submitted more than once (keep the first response)
    dup_id = df.duplicated(subset="customer_id", keep="first")
    record("Repeated customer_id", dup_id.sum(), "dropped, first response kept")
    df = df[~dup_id]

    # 5. Standardise country names
    country_key = df["country"].str.lower()
    mapped = country_key.map(C.COUNTRY_ALIASES)
    changed = mapped.notna() & (mapped != df["country"])
    record("Country spelling variants standardised", changed.sum(), "recoded")
    unknown_country = mapped.isna()
    record("Unrecognised country", unknown_country.sum(), "dropped")
    df = df.assign(country=mapped)[~unknown_country]

    # 6. Standardise brand names
    brand_mapped = df["vehicle_brand"].str.lower().map(C.BRAND_ALIASES)
    changed = brand_mapped.notna() & (brand_mapped != df["vehicle_brand"])
    record("Brand spelling variants standardised", changed.sum(), "recoded")
    unknown_brand = brand_mapped.isna()
    record("Unrecognised brand", unknown_brand.sum(), "dropped")
    df = df.assign(vehicle_brand=brand_mapped)[~unknown_brand]

    # 7. NPS score: numeric and within 0-10, otherwise unusable
    nps = pd.to_numeric(df["nps_score"], errors="coerce")
    non_numeric = nps.isna() & df["nps_score"].notna()
    out_of_range = nps.notna() & ~nps.between(*C.NPS_RANGE)
    missing = df["nps_score"].isna()
    record("NPS score missing", missing.sum(), "dropped (primary metric)")
    record("NPS score not a number (e.g. 'ten', 'n/a')", non_numeric.sum(), "dropped (primary metric)")
    record("NPS score outside 0-10", out_of_range.sum(), "dropped (primary metric)")
    bad_nps = missing | non_numeric | out_of_range
    df = df.assign(nps_score=nps)[~bad_nps]
    df["nps_score"] = df["nps_score"].astype(int)

    # 8. Satisfaction (CSAT 1-5): invalid values set to missing, row kept
    csat = pd.to_numeric(df["satisfaction_score"], errors="coerce")
    bad = csat.notna() & ~csat.between(*C.CSAT_RANGE)
    record("Satisfaction score outside 1-5", bad.sum(), "set to missing, row kept")
    record("Satisfaction score missing", csat.isna().sum(), "left missing, row kept")
    df["satisfaction_score"] = csat.mask(bad).astype("Int64")

    # 9. Ownership duration: numeric, 0-120 months
    own = pd.to_numeric(df["ownership_months"], errors="coerce")
    unparseable = own.isna()
    implausible = own.notna() & ~own.between(0, C.OWNERSHIP_MAX_MONTHS)
    record("Ownership duration unparseable or blank", unparseable.sum(), "set to missing, band 'Unknown'")
    record(f"Ownership duration outside 0-{C.OWNERSHIP_MAX_MONTHS} months", implausible.sum(), "set to missing, band 'Unknown'")
    df["ownership_months"] = own.mask(implausible).astype("Int64")

    # 10. Survey date: accept ISO or dd/mm/yyyy, must fall inside the survey period
    dates = _parse_dates(df["survey_date"])
    reformatted = dates.notna() & ~df["survey_date"].str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    outside = dates.notna() & ~dates.between(*C.SURVEY_PERIOD)
    record("Survey date in dd/mm/yyyy format", reformatted.sum(), "converted to ISO date")
    record("Survey date outside survey period", outside.sum(), "set to missing, row kept")
    df["survey_date"] = dates.mask(outside)

    # 11. Blank / placeholder comments
    blank = df["comment"].isna() | df["comment"].str.lower().isin(BLANK_COMMENTS)
    record("Blank or placeholder comment", blank.sum(), "set to empty, row kept")
    df["comment"] = df["comment"].mask(blank, "")

    # Derived fields
    df["nps_segment"] = df["nps_score"].map(nps_segment)
    df["ownership_band"] = df["ownership_months"].map(ownership_band)
    df["survey_month"] = df["survey_date"].dt.to_period("M").astype("string")
    df["has_comment"] = df["comment"] != ""

    record("Rows in clean dataset", len(df), "-")
    df = df.sort_values("customer_id").reset_index(drop=True)
    return df, pd.DataFrame(log)


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of rule violations (empty list = dataset passes)."""
    problems = []
    if df["customer_id"].duplicated().any():
        problems.append("duplicate customer_id")
    if not df["nps_score"].between(0, 10).all():
        problems.append("nps_score outside 0-10")
    if not df["satisfaction_score"].dropna().between(1, 5).all():
        problems.append("satisfaction_score outside 1-5")
    if not set(df["country"]).issubset(C.COUNTRIES):
        problems.append("unexpected country value")
    if not set(df["vehicle_brand"]).issubset(C.BRANDS):
        problems.append("unexpected brand value")
    if not set(df["ownership_band"]).issubset(set(C.OWNERSHIP_ORDER) | {"Unknown"}):
        problems.append("unexpected ownership band")
    own = df["ownership_months"].dropna()
    if not own.between(0, C.OWNERSHIP_MAX_MONTHS).all():
        problems.append("ownership_months out of range")
    return problems


def main() -> None:
    raw = pd.read_csv(C.DATA_RAW, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8-sig")
    df, log = clean(raw)
    problems = validate(df)
    if problems:
        raise SystemExit(f"Validation failed: {problems}")
    C.DATA_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.DATA_CLEAN, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    log.to_csv(C.CLEANING_LOG, index=False)
    print(log.to_string(index=False))
    print(f"\nClean dataset: {len(df):,} rows -> {C.DATA_CLEAN.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
