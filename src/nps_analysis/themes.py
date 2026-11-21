"""Rule-based theme tagging for free-text comments.

Each comment is tagged independently of the recorded `feedback_category`, so the two can be
compared: agreement shows whether the survey's category field matches what customers
actually wrote. A comment can match several themes; sentiment comes from cue words.
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

THEME_KEYWORDS: dict[str, list[str]] = {
    "Fuel efficiency": ["mileage", "fuel", "petrol", "economical", "economy"],
    "Durability & build quality": ["durable", "reliable", "breakdown", "build", "solid", "strong", "wear out", "failed", "cracked", "frame", "chain", "clutch", "engine"],
    "Spare parts availability": ["spare part", "spare parts", "genuine part", "parts", "fake part"],
    "After-sales service": ["service", "mechanic", "warranty", "after-sales", "repair"],
    "Price & value for money": ["price", "expensive", "affordable", "value for", "worth the money", "instalment", "cost"],
    "Ride comfort": ["comfortable", "uncomfortable", "seat", "suspension", "vibration", "smooth ride"],
    "Dealer experience": ["dealer", "showroom", "registration", "delivery", "buying process"],
    "Resale value": ["resale", "resell", "second hand"],
}

NEGATIVE_CUES = ["not ", "no ", "too ", "poor", "hard", "weak", "fake", "dropped", "failed", "long", "slow", "low ", "rejected",
                 "uncomfortable", "breakdowns", "wear out", "scarce", "inconsistent", "tiring", "higher than", "cracked", "very little", "far "]
POSITIVE_CUES = ["good", "very good", "excellent", "best", "easy", "helpful", "reliable", "durable", "affordable", "smooth",
                 "comfortable", "fair", "strong", "solid", "worth", "well", "clearly", "quick", "always available"]

_STOPWORDS = set("""a an the and or but is are was were be been it its this that to of in on for with my me i we our
very so too not no at by from as than after before every so far even all only more less just any about over can
could would there their they them you your have has had do does did get got bike motorcycle overall okay better""".split())


def tag_themes(comment: str) -> list[str]:
    text = f" {comment.lower()} "
    return [t for t, kws in THEME_KEYWORDS.items() if any(k in text for k in kws)]


def sentiment(comment: str) -> str:
    text = f" {comment.lower()} "
    neg = sum(c in text for c in NEGATIVE_CUES)
    pos = sum(c in text for c in POSITIVE_CUES)
    # "uncomfortable" contains "comfortable": do not count that as positive
    if "uncomfortable" in text:
        pos -= 1
    if neg == pos == 0:
        return "Neutral"
    return "Negative" if neg >= pos else "Positive"


def tag_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["themes"] = out["comment"].fillna("").map(tag_themes)
    out["primary_theme"] = out["themes"].map(lambda t: t[0] if t else "Untagged")
    out["comment_sentiment"] = out["comment"].fillna("").map(lambda c: sentiment(c) if c else "No comment")
    return out


def theme_agreement(tagged: pd.DataFrame) -> dict:
    """Share of commented responses whose recorded category appears among the text-derived themes."""
    sub = tagged[tagged["has_comment"]]
    hit = sub.apply(lambda r: r["feedback_category"] in r["themes"], axis=1)
    return {"responses_with_comment": int(len(sub)), "category_matches_text": round(100 * float(hit.mean()), 1),
            "untagged_comments": int((sub["primary_theme"] == "Untagged").sum())}


def theme_mentions(tagged: pd.DataFrame, by: str = "nps_segment") -> pd.DataFrame:
    """Count theme mentions (multi-label) per group, as % of commented responses in that group."""
    sub = tagged[tagged["has_comment"]].explode("themes").dropna(subset=["themes"])
    counts = sub.groupby([by, "themes"]).size().rename("mentions").reset_index()
    base = tagged[tagged["has_comment"]].groupby(by).size().rename("commented")
    counts = counts.join(base, on=by)
    counts["pct_of_commented"] = (100 * counts["mentions"] / counts["commented"]).round(1)
    return counts.sort_values([by, "mentions"], ascending=[True, False]).reset_index(drop=True)


def top_terms(comments: pd.Series, n: int = 15, ngram: int = 2) -> pd.DataFrame:
    """Most frequent word n-grams (stopwords removed) — a quick check on what customers repeat."""
    counter: Counter = Counter()
    for c in comments.dropna():
        words = [w for w in re.findall(r"[a-z\-]+", c.lower()) if w not in _STOPWORDS and len(w) > 2]
        counter.update(" ".join(words[i:i + ngram]) for i in range(len(words) - ngram + 1))
    return pd.DataFrame(counter.most_common(n), columns=["term", "count"])
