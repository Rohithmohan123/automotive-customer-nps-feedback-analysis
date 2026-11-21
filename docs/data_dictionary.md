# Data dictionary

Files: `data/raw/customer_feedback_synthetic_raw.csv` (as exported, with errors) and
`data/processed/customer_feedback_clean.csv` (after cleaning). Both are UTF-8 CSV with a byte-order mark so Excel opens them
correctly. **All values are synthetic.**

| Column | Type | Description | Valid values (clean file) |
|---|---|---|---|
| `customer_id` | text | Respondent identifier | `CUST-00001` ... unique |
| `survey_date` | date | Date the survey was completed | 2024-01-01 to 2024-12-31, or blank if invalid |
| `country` | text | Market | Nigeria, Egypt, Philippines, DR Congo |
| `city` | text | City of the respondent | e.g. Lagos, Cairo, Cebu City, Kinshasa |
| `vehicle_brand` | text | Brand of the two-wheeler owned | TVS, Bajaj, Haojue |
| `usage_type` | text | Main use of the vehicle | Personal commuting; Commercial (taxi / delivery) |
| `ownership_months` | integer | Months since purchase | 0-120, blank if invalid |
| `satisfaction_score` | integer | Overall satisfaction (CSAT) | 1 (very dissatisfied) to 5 (very satisfied), blank if invalid |
| `nps_score` | integer | "How likely are you to recommend [brand] to a friend?" | 0-10 |
| `feedback_category` | text | Main topic the respondent selected | 8 categories, see `config.py` |
| `comment` | text | Open-ended feedback | free text, empty if none |
| `survey_channel` | text | How the survey was completed | Phone interview, SMS link, Dealer tablet, WhatsApp link |
| `nps_segment` | text | *Derived*: Promoter (9-10), Passive (7-8), Detractor (0-6) | |
| `ownership_band` | text | *Derived*: 0-6, 6-12, 12-24, 24-36, 36+ months, or Unknown | |
| `survey_month` | text | *Derived*: `YYYY-MM` | |
| `has_comment` | boolean | *Derived*: comment is not empty | |
