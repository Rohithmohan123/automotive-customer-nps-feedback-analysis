"""Run every analysis on the clean dataset: tables (CSV + Excel), figures and a findings dict
that the report is generated from. Nothing in the findings is typed in by hand."""
from __future__ import annotations

import pandas as pd

from . import config as C
from . import generate_data as G
from . import themes as T
from . import visualize as V
from .metrics import benjamini_hochberg, competitor_gaps, compare_nps, nps, nps_table

THEME_ACTIONS = {
    "Spare parts availability": "Improve genuine spare-parts stocking outside major cities (regional depots, minimum stock lists for high-wear parts) and track fill rates.",
    "After-sales service": "Shorten service turnaround: appointment slots, a target waiting time per visit and a follow-up call when a fault is not fixed first time.",
    "Price & value for money": "Strengthen value messaging and financing: transparent pricing, longer instalment options and total-cost-of-ownership comparisons.",
    "Dealer experience": "Set dealer service standards for paperwork and delivery times, and score dealers on post-sale follow-up.",
    "Durability & build quality": "Analyse warranty claims by component (chain, clutch, plastics) to find early-wear issues on rough-road use.",
    "Ride comfort": "Test seat and suspension options for long-distance and two-up riding in the affected markets.",
    "Resale value": "Consider certified pre-owned or buy-back programmes to support resale prices.",
    "Fuel efficiency": "Investigate the reported mileage drop and offer a free tune-up check to affected owners.",
}


def run() -> dict:
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    C.TABLES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.DATA_CLEAN, encoding="utf-8-sig", parse_dates=["survey_date"])
    log = pd.read_csv(C.CLEANING_LOG)
    tagged = T.tag_dataframe(df)

    overall = nps(df["nps_score"])
    by_country = nps_table(df, "country", C.COUNTRIES)
    by_brand = nps_table(df, "vehicle_brand", C.BRANDS)
    by_ownership = nps_table(df, "ownership_band", [*C.OWNERSHIP_ORDER, "Unknown"])
    by_usage = nps_table(df, "usage_type", C.USAGE_TYPES)
    by_brand_country = nps_table(df, ["country", "vehicle_brand"]).sort_values(["country", "vehicle_brand"])
    by_brand_ownership = nps_table(df[df["ownership_band"] != "Unknown"], ["vehicle_brand", "ownership_band"])
    by_category = nps_table(df, "feedback_category").sort_values("nps")
    by_channel = nps_table(df, "survey_channel")

    gaps = competitor_gaps(df)
    gaps["p_adjusted_bh"] = benjamini_hochberg(gaps["p_value"]).round(4)

    usage_test = compare_nps(df.loc[df["usage_type"] == C.USAGE_TYPES[0], "nps_score"], df.loc[df["usage_type"] == C.USAGE_TYPES[1], "nps_score"])
    known = df[df["ownership_band"] != "Unknown"]
    worst_band = by_ownership[by_ownership["ownership_band"] != "Unknown"].sort_values("nps").iloc[0]["ownership_band"]
    tenure_test = compare_nps(known.loc[known["ownership_band"] == "0-6 months", "nps_score"], known.loc[known["ownership_band"] == worst_band, "nps_score"])

    csat_corr = float(df[["satisfaction_score", "nps_score"]].dropna().corr(method="spearman").iloc[0, 1])

    det_mentions = T.theme_mentions(tagged[tagged["nps_segment"] == "Detractor"], by="country")
    promo_mentions = T.theme_mentions(tagged[tagged["nps_segment"] == "Promoter"], by="country")
    seg_mentions = T.theme_mentions(tagged, by="nps_segment")
    agreement = T.theme_agreement(tagged)
    det_terms = T.top_terms(tagged.loc[tagged["nps_segment"] == "Detractor", "comment"], n=12)
    promo_terms = T.top_terms(tagged.loc[tagged["nps_segment"] == "Promoter", "comment"], n=12)

    top_det_theme = det_mentions.sort_values("pct_of_commented", ascending=False).groupby("country").head(2)
    top_promo_theme = seg_mentions[seg_mentions["nps_segment"] == "Promoter"].head(3)

    # ---- Method check: does the analysis recover what the generator built in? ----
    planted_rank = sorted(G.COUNTRY_EFFECT, key=G.COUNTRY_EFFECT.get, reverse=True)
    observed_rank = by_country.sort_values("nps", ascending=False)["country"].tolist()
    planted_worst_band = min(G.TENURE_EFFECT, key=G.TENURE_EFFECT.get)
    brand_spread_planted = max(G.BRAND_EFFECT.values()) - min(G.BRAND_EFFECT.values())
    method_check = pd.DataFrame([
        {"assumption built into the generator": "Market ranking (best to worst): " + " > ".join(planted_rank),
         "what the analysis found": " > ".join(observed_rank), "recovered": planted_rank == observed_rank},
        {"assumption built into the generator": f"Lowest advocacy in the {planted_worst_band} ownership band",
         "what the analysis found": f"Lowest NPS in the {worst_band} band", "recovered": planted_worst_band == worst_band},
        {"assumption built into the generator": "Commercial riders less likely to recommend than personal users",
         "what the analysis found": f"Gap {usage_test['difference']:+.1f} NPS points (" + ("p < 0.001" if usage_test["p_value"] < 0.001 else f"p = {usage_test['p_value']:.3f}") + ")",
         "recovered": usage_test["difference"] > 0 and usage_test["significant_5pct"]},
        {"assumption built into the generator": f"Small brand effects (max spread {brand_spread_planted:.2f} points on the 0-10 scale)",
         "what the analysis found": f"{int((gaps['p_adjusted_bh'] < 0.05).sum())} of {len(gaps)} brand comparisons significant after FDR correction",
         "recovered": "n/a (expected to be hard to detect)"},
        {"assumption built into the generator": f"About {int(G.P_MISCATEGORISED * 100)}% of responses filed under the wrong category",
         "what the analysis found": f"Recorded category matches comment text for {agreement['category_matches_text']}% of comments",
         "recovered": agreement["category_matches_text"] < 95},
    ])

    tables = {
        "01_overall": pd.DataFrame([{"n": overall.n, "promoters_pct": round(100 * overall.promoters, 1), "passives_pct": round(100 * overall.passives, 1),
                                      "detractors_pct": round(100 * overall.detractors, 1), "nps": round(overall.nps, 1),
                                      "ci_low": round(overall.ci_low, 1), "ci_high": round(overall.ci_high, 1),
                                      "mean_csat": round(float(df["satisfaction_score"].mean()), 2), "csat_nps_spearman": round(csat_corr, 3)}]),
        "02_by_country": by_country, "03_by_brand": by_brand, "04_by_brand_country": by_brand_country,
        "05_competitor_gaps": gaps, "06_by_ownership": by_ownership, "07_by_brand_ownership": by_brand_ownership,
        "08_by_usage": by_usage, "09_by_feedback_category": by_category, "10_by_survey_channel": by_channel,
        "11_theme_mentions_by_segment": seg_mentions, "12_detractor_themes_by_country": det_mentions,
        "13_top_detractor_terms": det_terms, "14_top_promoter_terms": promo_terms,
        "15_method_check": method_check, "16_cleaning_log": log,
    }
    for name, t in tables.items():
        t.to_csv(C.TABLES / f"{name}.csv", index=False)
    with pd.ExcelWriter(C.TABLES / "nps_analysis_tables.xlsx", engine="openpyxl") as xw:
        pd.DataFrame({"note": ["SYNTHETIC DATA - generated for demonstration only. Not real customers, brands' real performance or client findings.",
                               "Each sheet matches a CSV file in outputs/tables."]}).to_excel(xw, sheet_name="README", index=False)
        for name, t in tables.items():
            t.to_excel(xw, sheet_name=name[:31], index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, *(len(str(c.value or "")) for c in col)) + 2)
            ws.freeze_panes = "A2"

    figs = {
        "score_distribution": C.FIGURES / "01_score_distribution.png",
        "nps_by_country": C.FIGURES / "02_nps_by_country.png",
        "segment_mix_country": C.FIGURES / "03_segment_mix_by_country.png",
        "brand_country": C.FIGURES / "04_nps_brand_by_market.png",
        "competitor_gaps": C.FIGURES / "05_competitor_gaps.png",
        "ownership": C.FIGURES / "06_nps_by_ownership_duration.png",
        "detractor_themes": C.FIGURES / "07_detractor_themes_by_country.png",
        "promoter_themes": C.FIGURES / "08_promoter_themes_by_country.png",
        "csat_vs_nps": C.FIGURES / "09_csat_vs_nps.png",
        "category_nps": C.FIGURES / "10_nps_by_feedback_category.png",
        "monthly": C.FIGURES / "11_monthly_trend.png",
    }
    V.score_distribution(df, figs["score_distribution"])
    V.nps_by_group(by_country, "country", figs["nps_by_country"], "NPS by market", V.COUNTRY_COLOR)
    V.segment_mix(df, "country", C.COUNTRIES, figs["segment_mix_country"], "Promoter / passive / detractor mix by market")
    V.brand_country_heatmap(df, figs["brand_country"])
    V.competitor_gap_chart(gaps, figs["competitor_gaps"])
    V.ownership_trend(df, figs["ownership"])
    V.theme_heatmap(det_mentions, figs["detractor_themes"], "What detractors talk about, by market",
                    "% of detractor comments mentioning each theme (a comment can mention several)")
    V.theme_heatmap(promo_mentions, figs["promoter_themes"], "What promoters talk about, by market",
                    "% of promoter comments mentioning each theme (a comment can mention several)")
    V.csat_vs_nps(df, figs["csat_vs_nps"])
    V.nps_by_group(by_category, "feedback_category", figs["category_nps"], "NPS by recorded feedback category")
    V.monthly_trend(df, figs["monthly"])

    actions = []
    for theme in det_mentions.groupby("themes")["mentions"].sum().sort_values(ascending=False).index[:4]:
        countries = top_det_theme.loc[top_det_theme["themes"] == theme, "country"].tolist()
        actions.append({"theme": theme, "lead_markets": countries, "action": THEME_ACTIONS[theme]})

    return {
        "df": df, "log": log, "overall": overall, "by_country": by_country, "by_brand": by_brand,
        "by_brand_country": by_brand_country, "gaps": gaps, "by_ownership": by_ownership, "by_usage": by_usage,
        "usage_test": usage_test, "tenure_test": tenure_test, "worst_band": worst_band, "csat_corr": csat_corr,
        "by_category": by_category, "by_channel": by_channel, "det_mentions": det_mentions, "seg_mentions": seg_mentions,
        "top_det_theme": top_det_theme, "top_promo_theme": top_promo_theme, "agreement": agreement,
        "det_terms": det_terms, "promo_terms": promo_terms, "method_check": method_check, "figs": figs, "actions": actions,
    }
