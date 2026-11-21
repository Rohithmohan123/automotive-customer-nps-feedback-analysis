"""Shared configuration: paths, category lists and validation rules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "customer_feedback_synthetic_raw.csv"
DATA_CLEAN = ROOT / "data" / "processed" / "customer_feedback_clean.csv"
CLEANING_LOG = ROOT / "data" / "processed" / "cleaning_log.csv"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
REPORTS = ROOT / "reports"

RANDOM_SEED = 2024
N_RESPONDENTS = 3200

COUNTRIES = ["Nigeria", "Egypt", "Philippines", "DR Congo"]
BRANDS = ["TVS", "Bajaj", "Haojue"]
FOCUS_BRAND = "TVS"

# Canonical spellings for messy inputs found in raw survey exports.
COUNTRY_ALIASES = {
    "nigeria": "Nigeria", "ng": "Nigeria", "federal republic of nigeria": "Nigeria",
    "egypt": "Egypt", "eg": "Egypt", "arab republic of egypt": "Egypt",
    "philippines": "Philippines", "ph": "Philippines", "the philippines": "Philippines",
    "dr congo": "DR Congo", "drc": "DR Congo", "congo (drc)": "DR Congo",
    "democratic republic of the congo": "DR Congo", "congo-kinshasa": "DR Congo",
}
BRAND_ALIASES = {"tvs": "TVS", "tvs motor": "TVS", "bajaj": "Bajaj", "bajaj auto": "Bajaj", "haojue": "Haojue"}

FEEDBACK_CATEGORIES = [
    "Fuel efficiency",
    "Durability & build quality",
    "Spare parts availability",
    "After-sales service",
    "Price & value for money",
    "Ride comfort",
    "Dealer experience",
    "Resale value",
]

# Ownership duration bands (months, inclusive lower bound, exclusive upper bound)
OWNERSHIP_BANDS = [
    (0, 6, "0-6 months"),
    (6, 12, "6-12 months"),
    (12, 24, "12-24 months"),
    (24, 36, "24-36 months"),
    (36, 1000, "36+ months"),
]
OWNERSHIP_ORDER = [b[2] for b in OWNERSHIP_BANDS]

USAGE_TYPES = ["Personal commuting", "Commercial (taxi / delivery)"]
SURVEY_CHANNELS = ["Phone interview", "SMS link", "Dealer tablet", "WhatsApp link"]

# Validation rules applied by the cleaning step
NPS_RANGE = (0, 10)
CSAT_RANGE = (1, 5)
OWNERSHIP_MAX_MONTHS = 120
SURVEY_PERIOD = ("2024-01-01", "2024-12-31")

# Minimum group size before a segment's NPS is reported without a warning flag
MIN_SEGMENT_N = 50
