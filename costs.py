"""
Cost model shared by train.py (to choose the operating point) and app.py (to
show it live), so both use exactly the same arithmetic.

A "case" is one camera photo. It is positive when at least one person in it is
not wearing a hard hat. The model's score for a photo is the highest
confidence of the `head` class (a head without a hard hat) in it.

All costs are per 1,000 photos, after reweighting the test results to the
production base rate (the test split has ~20% positives; production does not).
The figures and their justification live in REPORT.md.
"""

import numpy as np
import pandas as pd

# Cost assumptions (euros). Justified in REPORT.md, section "Cost matrix".
ASSUMPTIONS = {
    "C_FN": 17.0,                  # missed photo with someone bare-headed (accident + fine risk)
    "C_FP": 1.1,                   # false alarm: 3.5 min of the site manager at 19.5 EUR/h
    "C_TP": 1.5,                   # true alarm: 4.5 min (go, check, correct the worker)
    "C_TN": 0.0,                   # nothing happens
    "BASE_RATE": 0.07,             # share of production photos with someone bare-headed
    "CASES_PER_DAY": 720,          # 15 cameras x 6 photos/h x 8 h
    "REVIEW_SECONDS": 10,          # seconds for the guard to check one photo on screen
    "REVIEWER_EUR_PER_HOUR": 10.0, # cost to the company of one hour of the guard
    "HOURS_PER_SHIFT": 8,
    "MAX_REVIEWERS": 1.0,          # most full-time reviewers the company would pay for
    "REVIEWER_MISS_RATE": 0.02,    # share of positives the reviewer still misses
}

THRESHOLDS = np.round(np.arange(0, 1.0001, 0.005), 3)   # single-threshold sweep
GRID = np.round(np.arange(0, 1.0001, 0.01), 2)          # two-threshold search


def review_cost(a: dict) -> float:
    """Euros to have a person look at one photo."""
    return a["REVIEW_SECONDS"] / 3600 * a["REVIEWER_EUR_PER_HOUR"]


def people_needed(reviews_per_day: float, a: dict) -> float:
    """Full-time reviewers needed for a given number of reviews a day."""
    return reviews_per_day * a["REVIEW_SECONDS"] / 3600 / a["HOURS_PER_SHIFT"]


def split(scores: pd.DataFrame, column: str = "score_model"):
    """Scores of the positive and negative test photos."""
    y = scores["y_true"].to_numpy()
    s = scores[column].to_numpy()
    return s[y == 1], s[y == 0]


def cost_do_nothing(a: dict) -> float:
    p = 1000 * a["BASE_RATE"]
    return p * a["C_FN"] + (1000 - p) * a["C_TN"]


def cost_human_checks_all(a: dict) -> float:
    p = 1000 * a["BASE_RATE"]
    f = a["REVIEWER_MISS_RATE"]
    return (1000 * review_cost(a) + p * ((1 - f) * a["C_TP"] + f * a["C_FN"])
            + (1000 - p) * a["C_TN"])


def single_threshold(pos, neg, t: float, a: dict, c_fn=None, base_rate=None) -> dict:
    """Model decides alone: alert if score >= t, otherwise pass."""
    c_fn = a["C_FN"] if c_fn is None else c_fn
    br = a["BASE_RATE"] if base_rate is None else base_rate
    p = 1000 * br
    n = 1000 - p
    tpr = float((pos >= t).mean())
    fpr = float((neg >= t).mean())
    tp, fn, fp, tn = p * tpr, p * (1 - tpr), n * fpr, n * (1 - fpr)
    return {
        "threshold": float(t), "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "recall": tpr, "precision": tp / (tp + fp) if tp + fp else np.nan,
        "cost_per_1000": tp * a["C_TP"] + fn * c_fn + fp * a["C_FP"] + tn * a["C_TN"],
    }


def sweep_single(pos, neg, a: dict, thresholds=THRESHOLDS) -> pd.DataFrame:
    return pd.DataFrame([single_threshold(pos, neg, t, a) for t in thresholds])


def two_thresholds(pos, neg, low: float, high: float, a: dict) -> dict:
    """score < low -> pass; score >= high -> alert; in between -> a person checks."""
    low, high = min(low, high), max(low, high)
    p = 1000 * a["BASE_RATE"]
    n = 1000 - p
    f = a["REVIEWER_MISS_RATE"]
    c_rev = review_cost(a)

    p_pass, p_alert = float((pos < low).mean()), float((pos >= high).mean())
    p_rev = 1 - p_pass - p_alert
    n_pass, n_alert = float((neg < low).mean()), float((neg >= high).mean())
    n_rev = 1 - n_pass - n_alert

    cost = (p * (p_pass * a["C_FN"] + p_alert * a["C_TP"]
                 + p_rev * (c_rev + (1 - f) * a["C_TP"] + f * a["C_FN"]))
            + n * (n_pass * a["C_TN"] + n_alert * a["C_FP"] + n_rev * (c_rev + a["C_TN"])))
    share_review = a["BASE_RATE"] * p_rev + (1 - a["BASE_RATE"]) * n_rev
    reviews_day = share_review * a["CASES_PER_DAY"]
    return {
        "low": float(low), "high": float(high),
        "cost_per_1000": cost,
        "share_review": share_review,
        "reviews_per_day": reviews_day,
        "people": people_needed(reviews_day, a),
        "FN_1000": p * (p_pass + p_rev * f),   # escaped: passed, or missed by the reviewer
        "FP_1000": n * n_alert,                # false alarms that reach the site manager
        # expected photos per 1,000 by real class and outcome
        "matrix_per_1000": {
            "pos": {"pass": p * p_pass, "review": p * p_rev, "alert": p * p_alert},
            "neg": {"pass": n * n_pass, "review": n * n_rev, "alert": n * n_alert},
        },
    }


def search_two(pos, neg, a: dict, grid=GRID) -> pd.DataFrame:
    rows = []
    for i, low in enumerate(grid):
        for high in grid[i:]:
            r = two_thresholds(pos, neg, low, high, a)
            r.pop("matrix_per_1000")
            rows.append(r)
    return pd.DataFrame(rows)


def best_two(grid_df: pd.DataFrame, a: dict) -> pd.Series:
    """Cheapest pair whose review load fits in MAX_REVIEWERS."""
    feasible = grid_df[grid_df.people <= a["MAX_REVIEWERS"]]
    return feasible.loc[feasible.cost_per_1000.idxmin()]


def scenarios(pos, neg, a: dict, t_single: float, low: float, high: float) -> pd.DataFrame:
    k = a["CASES_PER_DAY"] / 1000
    p = 1000 * a["BASE_RATE"]
    one = single_threshold(pos, neg, t_single, a)
    two = two_thresholds(pos, neg, low, high, a)
    df = pd.DataFrame([
        {"scenario": "Do nothing", "cost_per_1000": cost_do_nothing(a),
         "FN_per_day": p * k, "FP_per_day": 0.0, "reviews_per_day": 0.0, "people": 0.0},
        {"scenario": "Person checks everything", "cost_per_1000": cost_human_checks_all(a),
         "FN_per_day": p * a["REVIEWER_MISS_RATE"] * k, "FP_per_day": 0.0,
         "reviews_per_day": float(a["CASES_PER_DAY"]), "people": people_needed(a["CASES_PER_DAY"], a)},
        {"scenario": f"Model alone (threshold {t_single:.3f})", "cost_per_1000": one["cost_per_1000"],
         "FN_per_day": one["FN"] * k, "FP_per_day": one["FP"] * k, "reviews_per_day": 0.0, "people": 0.0},
        {"scenario": f"Model + human review ({low:.2f} / {high:.2f})", "cost_per_1000": two["cost_per_1000"],
         "FN_per_day": two["FN_1000"] * k, "FP_per_day": two["FP_1000"] * k,
         "reviews_per_day": two["reviews_per_day"], "people": two["people"]},
    ])
    df["cost_per_day"] = df.cost_per_1000 * k
    df["cost_per_year_250d"] = df.cost_per_day * 250
    return df
