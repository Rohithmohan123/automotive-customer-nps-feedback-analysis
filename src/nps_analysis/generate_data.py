"""Generate a SYNTHETIC two-wheeler customer feedback survey dataset.

Nothing here is real customer data. Every effect size below is an assumption chosen for
demonstration and documented in docs/synthetic_data_design.md. The goal is a dataset that
behaves like a real multi-country NPS survey (skewed scores, segment differences, messy
free text and data-entry errors) so the cleaning and analysis code has something
realistic to work on.

Run:  python -m nps_analysis.generate_data
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# ---------------------------------------------------------------------------
# Model assumptions (all synthetic). The latent "advocacy" score is on the 0-10 NPS
# scale before rounding and clipping.
# ---------------------------------------------------------------------------
BASE_ADVOCACY = 7.2

COUNTRY_EFFECT = {"Nigeria": 0.15, "Egypt": -0.10, "Philippines": 0.35, "DR Congo": -0.55}
COUNTRY_WEIGHT = {"Nigeria": 0.30, "Egypt": 0.24, "Philippines": 0.26, "DR Congo": 0.20}

# Brand market mix per country (shares sum to 1) and small brand effects.
BRAND_MIX = {
    "Nigeria": {"TVS": 0.36, "Bajaj": 0.40, "Haojue": 0.24},
    "Egypt": {"TVS": 0.34, "Bajaj": 0.38, "Haojue": 0.28},
    "Philippines": {"TVS": 0.30, "Bajaj": 0.28, "Haojue": 0.42},
    "DR Congo": {"TVS": 0.33, "Bajaj": 0.30, "Haojue": 0.37},
}
BRAND_EFFECT = {"TVS": 0.10, "Bajaj": 0.05, "Haojue": -0.25}
# One interaction so the competitor comparison is not uniform across markets.
BRAND_COUNTRY_INTERACTION = {("Haojue", "Philippines"): 0.45, ("TVS", "DR Congo"): -0.30}

# Satisfaction dips around 12-24 months (first major service, warranty expiry),
# then partially recovers among long-term owners who chose to keep the vehicle.
TENURE_EFFECT = {"0-6 months": 0.45, "6-12 months": 0.15, "12-24 months": -0.45, "24-36 months": -0.20, "36+ months": 0.05}
USAGE_EFFECT = {"Personal commuting": 0.10, "Commercial (taxi / delivery)": -0.25}
USAGE_SHARE = {"Nigeria": 0.55, "Egypt": 0.30, "Philippines": 0.45, "DR Congo": 0.60}  # share commercial

NOISE_SD = 1.9

CITIES = {
    "Nigeria": ["Lagos", "Kano", "Abuja", "Ibadan", "Port Harcourt"],
    "Egypt": ["Cairo", "Alexandria", "Giza", "Mansoura"],
    "Philippines": ["Manila", "Cebu City", "Davao City", "Iloilo City"],
    "DR Congo": ["Kinshasa", "Lubumbashi", "Goma", "Kisangani"],
}

# Which topic a customer talks about depends on whether they are unhappy or happy.
PAIN_TOPICS = {
    "Nigeria": {"Spare parts availability": 0.22, "After-sales service": 0.24, "Price & value for money": 0.18, "Durability & build quality": 0.14, "Dealer experience": 0.10, "Ride comfort": 0.06, "Resale value": 0.04, "Fuel efficiency": 0.02},
    "Egypt": {"Price & value for money": 0.30, "After-sales service": 0.20, "Spare parts availability": 0.14, "Dealer experience": 0.14, "Durability & build quality": 0.10, "Ride comfort": 0.06, "Resale value": 0.04, "Fuel efficiency": 0.02},
    "Philippines": {"Dealer experience": 0.24, "After-sales service": 0.26, "Spare parts availability": 0.12, "Price & value for money": 0.12, "Ride comfort": 0.12, "Durability & build quality": 0.08, "Resale value": 0.04, "Fuel efficiency": 0.02},
    "DR Congo": {"Spare parts availability": 0.38, "After-sales service": 0.22, "Durability & build quality": 0.14, "Price & value for money": 0.12, "Dealer experience": 0.08, "Ride comfort": 0.03, "Resale value": 0.02, "Fuel efficiency": 0.01},
}
PRAISE_TOPICS = {"Fuel efficiency": 0.30, "Durability & build quality": 0.24, "Price & value for money": 0.16, "Ride comfort": 0.12, "Resale value": 0.08, "After-sales service": 0.05, "Dealer experience": 0.04, "Spare parts availability": 0.01}

COMMENTS = {
    ("Fuel efficiency", "pos"): ["Very good mileage, I spend less on fuel every week.", "Fuel consumption is low, good for my daily trips.", "Excellent fuel economy compared to my old bike.", "Mileage is the best thing about this motorcycle."],
    ("Fuel efficiency", "neg"): ["Mileage dropped a lot after the first year.", "Fuel consumption is higher than the dealer promised.", "Not economical on petrol anymore."],
    ("Durability & build quality", "pos"): ["Strong and reliable even on bad roads.", "Very durable, no breakdowns so far.", "Solid build, handles heavy loads well.", "Reliable engine, starts every morning."],
    ("Durability & build quality", "neg"): ["Frame and parts wear out too quickly on rough roads.", "Too many breakdowns for a new motorcycle.", "Chain and clutch failed early.", "Build quality feels weak, plastic parts cracked."],
    ("Spare parts availability", "pos"): ["Spare parts are easy to find in my area.", "Genuine parts always available at the shop."],
    ("Spare parts availability", "neg"): ["Hard to find genuine spare parts, waited weeks.", "Spare parts are not available outside the big city.", "Mechanics use fake parts because original ones are scarce.", "No spare parts in stock, bike was off the road for a long time."],
    ("After-sales service", "pos"): ["Service centre staff are helpful and quick.", "Good after-sales support and fair service charges."],
    ("After-sales service", "neg"): ["Service centre is far and the waiting time is very long.", "Poor after-sales support, the problem was not fixed.", "Warranty claim was rejected without explanation.", "Service quality is inconsistent, depends on the mechanic."],
    ("Price & value for money", "pos"): ["Good value for the price.", "Affordable and worth the money.", "Fair price with easy instalment options."],
    ("Price & value for money", "neg"): ["Price went up too much, not affordable now.", "Too expensive for what you get.", "Instalment payments are hard with rising prices."],
    ("Ride comfort", "pos"): ["Comfortable seat for long rides.", "Smooth ride and good suspension.", "Comfortable even with a passenger."],
    ("Ride comfort", "neg"): ["Seat is uncomfortable on long trips.", "Suspension is too hard on bumpy roads.", "Vibration at high speed is tiring."],
    ("Dealer experience", "pos"): ["Dealer explained everything clearly.", "Smooth buying process at the showroom."],
    ("Dealer experience", "neg"): ["Dealer was slow with registration papers.", "Showroom staff were not helpful after the sale.", "Delivery took longer than the dealer promised."],
    ("Resale value", "pos"): ["Holds its resale value well.", "Easy to sell second hand at a good price."],
    ("Resale value", "neg"): ["Resale value is low compared to other brands.", "Hard to resell, buyers offer very little."],
}
GENERIC_COMMENTS = {"pos": ["Good bike.", "Happy with it.", "No complaints so far.", "Very satisfied."],
                    "neg": ["Not happy.", "Expected more.", "Many problems.", "Disappointed overall."]}
# Share of comments that are vague, mention a second topic, or sit under a mis-chosen category
P_GENERIC, P_SECOND_TOPIC, P_MISCATEGORISED = 0.08, 0.15, 0.08
NEUTRAL_SUFFIX = ["", "", "", " Overall it is okay.", " Overall okay.", " Could be better."]


def _band(months: int) -> str:
    for lo, hi, label in C.OWNERSHIP_BANDS:
        if lo <= months < hi:
            return label
    return C.OWNERSHIP_BANDS[-1][2]


def _pick(rng: np.random.Generator, dist: dict[str, float]) -> str:
    keys = list(dist)
    p = np.array(list(dist.values()), dtype=float)
    return str(rng.choice(keys, p=p / p.sum()))


def generate_clean(n: int = C.N_RESPONDENTS, seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    """Return a clean synthetic dataset before any errors are injected."""
    rng = np.random.default_rng(seed)
    rows = []
    countries = rng.choice(list(COUNTRY_WEIGHT), size=n, p=list(COUNTRY_WEIGHT.values()))
    start = pd.Timestamp(C.SURVEY_PERIOD[0])
    for i, country in enumerate(countries):
        brand = _pick(rng, BRAND_MIX[country])
        usage = C.USAGE_TYPES[1] if rng.random() < USAGE_SHARE[country] else C.USAGE_TYPES[0]
        # Ownership: right-skewed, most respondents own the vehicle < 3 years
        months = int(min(rng.gamma(shape=1.8, scale=10.5), 96))
        band = _band(months)

        latent = (BASE_ADVOCACY + COUNTRY_EFFECT[country] + BRAND_EFFECT[brand]
                  + BRAND_COUNTRY_INTERACTION.get((brand, country), 0.0)
                  + TENURE_EFFECT[band] + USAGE_EFFECT[usage] + rng.normal(0, NOISE_SD))
        nps = int(np.clip(np.round(latent), 0, 10))

        # CSAT (1-5) is correlated with, but not identical to, likelihood to recommend
        csat_latent = 1 + (latent / 10) * 4 + rng.normal(0, 0.55)
        csat = int(np.clip(np.round(csat_latent), 1, 5))

        polarity = "neg" if nps <= 6 else ("pos" if nps >= 9 else rng.choice(["pos", "neg"], p=[0.55, 0.45]))
        topic = _pick(rng, PAIN_TOPICS[country] if polarity == "neg" else PRAISE_TOPICS)
        comment = str(rng.choice(COMMENTS[(topic, polarity)]))
        r = rng.random()
        if r < P_GENERIC:
            comment = str(rng.choice(GENERIC_COMMENTS[polarity]))
        elif r < P_GENERIC + P_SECOND_TOPIC:
            other = _pick(rng, PAIN_TOPICS[country] if polarity == "neg" else PRAISE_TOPICS)
            if other != topic:
                comment += " " + str(rng.choice(COMMENTS[(other, polarity)]))
        recorded_category = topic
        if rng.random() < P_MISCATEGORISED:
            # Respondent (or interviewer) ticked a category that does not match the comment
            recorded_category = str(rng.choice([c for c in C.FEEDBACK_CATEGORIES if c != topic]))
        if 7 <= nps <= 8:
            comment += str(rng.choice(NEUTRAL_SUFFIX))

        rows.append({
            "customer_id": f"CUST-{i + 1:05d}",
            "survey_date": (start + pd.Timedelta(days=int(rng.integers(0, 366)))).strftime("%Y-%m-%d"),
            "country": country,
            "city": str(rng.choice(CITIES[country])),
            "vehicle_brand": brand,
            "usage_type": usage,
            "ownership_months": months,
            "satisfaction_score": csat,
            "nps_score": nps,
            "feedback_category": recorded_category,
            "comment": comment,
            "survey_channel": str(rng.choice(C.SURVEY_CHANNELS, p=[0.35, 0.25, 0.2, 0.2])),
        })
    return pd.DataFrame(rows)


def inject_errors(df: pd.DataFrame, seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    """Add the kinds of problems seen in real survey exports, so cleaning has work to do."""
    rng = np.random.default_rng(seed + 1)
    out = df.copy()
    out["ownership_months"] = out["ownership_months"].astype(object)
    out["nps_score"] = out["nps_score"].astype(object)
    out["satisfaction_score"] = out["satisfaction_score"].astype(object)
    n = len(out)

    def idx(frac: float) -> np.ndarray:
        return rng.choice(n, size=max(1, int(n * frac)), replace=False)

    # Inconsistent country and brand spellings, stray whitespace
    variants = {"Nigeria": ["nigeria", "NG", " Nigeria"], "Egypt": ["egypt", "EG", "Egypt "],
                "Philippines": ["The Philippines", "PH", "philippines"], "DR Congo": ["DRC", "Congo (DRC)", "Democratic Republic of the Congo"]}
    for i in idx(0.06):
        out.at[i, "country"] = str(rng.choice(variants[out.at[i, "country"]]))
    for i in idx(0.05):
        b = out.at[i, "vehicle_brand"]
        out.at[i, "vehicle_brand"] = str(rng.choice([b.upper(), b.lower(), f" {b}", {"TVS": "TVS Motor", "Bajaj": "Bajaj Auto", "Haojue": "HAOJUE"}[b]]))

    # Out-of-range and missing scores
    for i in idx(0.008):
        out.at[i, "nps_score"] = int(rng.choice([11, 12, -1, 99]))
    for i in idx(0.012):
        out.at[i, "nps_score"] = np.nan
    for i in idx(0.006):
        out.at[i, "nps_score"] = str(rng.choice(["ten", "9/10", "n/a"]))
    for i in idx(0.01):
        out.at[i, "satisfaction_score"] = int(rng.choice([0, 6, 7]))
    for i in idx(0.015):
        out.at[i, "satisfaction_score"] = np.nan

    # Implausible or unknown ownership durations
    for i in idx(0.006):
        out.at[i, "ownership_months"] = int(rng.choice([-3, -12, 400, 999]))
    for i in idx(0.008):
        out.at[i, "ownership_months"] = str(rng.choice(["unknown", "", "2 yrs"]))

    # Dates outside the survey window and in a different format
    for i in idx(0.004):
        out.at[i, "survey_date"] = "2031-05-01"
    for i in idx(0.02):
        out.at[i, "survey_date"] = pd.Timestamp(out.at[i, "survey_date"]).strftime("%d/%m/%Y")

    # Blank comments
    for i in idx(0.03):
        out.at[i, "comment"] = str(rng.choice(["", " ", "N/A", "-"]))

    # Missing customer ID on a few rows
    for i in idx(0.002):
        out.at[i, "customer_id"] = ""

    # Exact duplicate submissions (e.g. form resubmitted)
    dupes = out.iloc[idx(0.012)]
    out = pd.concat([out, dupes], ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    C.DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
    raw = inject_errors(generate_clean())
    raw.to_csv(C.DATA_RAW, index=False, encoding="utf-8-sig")  # BOM so Excel opens UTF-8 correctly
    print(f"Wrote {len(raw):,} synthetic rows to {C.DATA_RAW.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
