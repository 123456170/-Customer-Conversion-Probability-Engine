import json
import math
import tempfile
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except Exception:
    HistGradientBoostingClassifier = None

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Customer Conversion Probability Engine",
    page_icon="📈",
    layout="wide",
)

APP_TITLE = "Customer Conversion Probability Engine"
APP_SUBTITLE = "AI Advertising Intelligence and Revenue Optimization"
MODEL_VERSION = "cce-v0.1.4-synthetic"
RANDOM_SEED = 20261108

CHANNELS = ["search", "social", "display", "email", "affiliate"]
DEVICES = ["desktop", "mobile", "tablet"]
REGIONS = ["NA", "EMEA", "APAC", "LATAM"]
PRODUCTS = ["starter", "growth", "enterprise", "addon"]

CAMPAIGN_BY_CHANNEL = {
    "search": ["brand_core", "generic_high_intent", "competitor_compare"],
    "social": ["retargeting", "lookalike_broad", "video_awareness"],
    "display": ["programmatic_contextual", "banner_retarget", "native_content"],
    "email": ["lifecycle_nurture", "promo_flash", "winback"],
    "affiliate": ["partner_review", "coupon_network", "influencer"],
}

BASE_RATE = {
    "search": 0.085,
    "social": 0.032,
    "display": 0.014,
    "email": 0.115,
    "affiliate": 0.052,
}

DEVICE_MULT = {"desktop": 1.18, "mobile": 0.86, "tablet": 1.02}
REGION_MULT = {"NA": 1.08, "EMEA": 1.00, "APAC": 0.92, "LATAM": 0.88}
PRODUCT_MULT = {"starter": 0.90, "growth": 1.05, "enterprise": 1.25, "addon": 0.80}
AOV_BASE = {"starter": 95.0, "growth": 260.0, "enterprise": 1200.0, "addon": 45.0}

TARGET = "converted"
VALUE_TARGET = "revenue_if_converted"

CAT_FEATURES = ["channel", "device", "region", "campaign", "product_interest"]
NUMERIC_FEATURES = [
    "ad_cost_log",
    "landing_seconds",
    "form_completion",
    "prior_sessions",
    "email_opens",
    "page_views",
    "promo_code_flag",
    "hour",
    "dayofweek",
]
FEATURE_COLUMNS = CAT_FEATURES + NUMERIC_FEATURES

REQUIRED_COLUMNS = [
    "lead_id",
    "timestamp",
    "channel",
    "campaign",
    "device",
    "region",
    "product_interest",
    "ad_cost",
    "landing_seconds",
    "form_completion",
    "prior_sessions",
    "email_opens",
    "page_views",
    "promo_code",
    "revenue_if_converted",
    "converted",
]

BASELINE_KEY_SETS = [
    ("channel", "device", "region"),
    ("channel", "device"),
    ("channel",),
    (),
]

SCORED_OUTPUT_COLUMNS = [
    "conversion_probability",
    "probability_uncertainty",
    "score_source",
    "expected_order_value",
    "expected_revenue",
    "attributed_cost",
    "expected_roi",
    "recommendation",
    "input_quality",
]

DATA_DICTIONARY = [
    {"field": "lead_id", "type": "string", "unit": "identifier", "description": "Synthetic lead identifier.", "example": "H-000123"},
    {"field": "timestamp", "type": "datetime UTC", "unit": "ISO-8601", "description": "Lead arrival time.", "example": "2026-10-14T15:22:09Z"},
    {"field": "channel", "type": "category", "unit": "enum", "description": "Acquisition channel.", "example": "search"},
    {"field": "campaign", "type": "category", "unit": "enum", "description": "Campaign name.", "example": "brand_core"},
    {"field": "device", "type": "category", "unit": "enum", "description": "Device type.", "example": "mobile"},
    {"field": "region", "type": "category", "unit": "enum", "description": "Geo region bucket.", "example": "EMEA"},
    {"field": "product_interest", "type": "category", "unit": "enum", "description": "Product tier viewed.", "example": "growth"},
    {"field": "ad_cost", "type": "float", "unit": "USD", "description": "Attributed media cost for the lead.", "example": 2.41},
    {"field": "landing_seconds", "type": "float", "unit": "seconds", "description": "Time spent on landing page.", "example": 28.5},
    {"field": "form_completion", "type": "float", "unit": "0-1 ratio", "description": "Percentage of form fields completed.", "example": 0.72},
    {"field": "prior_sessions", "type": "integer", "unit": "sessions", "description": "Prior anonymous sessions.", "example": 3},
    {"field": "email_opens", "type": "integer", "unit": "events", "description": "Email opens in nurture window.", "example": 2},
    {"field": "page_views", "type": "integer", "unit": "events", "description": "Page views during session.", "example": 5},
    {"field": "promo_code", "type": "boolean", "unit": "flag", "description": "Promo code present.", "example": True},
    {"field": "revenue_if_converted", "type": "float", "unit": "USD", "description": "Estimated order value if conversion occurs.", "example": 260.0},
    {"field": "converted", "type": "integer", "unit": "0/1 label", "description": "Simulated ground-truth conversion label.", "example": 1},
    {"field": "conversion_probability", "type": "float", "unit": "0-1", "description": "Predicted conversion probability.", "example": 0.184},
    {"field": "probability_uncertainty", "type": "float", "unit": "0-0.49", "description": "Approximate uncertainty band for the score.", "example": 0.042},
    {"field": "score_source", "type": "string", "unit": "audit label", "description": "Model or deterministic fallback used.", "example": "model"},
    {"field": "expected_revenue", "type": "float", "unit": "USD", "description": "Probability-weighted expected revenue.", "example": 47.84},
    {"field": "expected_roi", "type": "float", "unit": "ratio", "description": "Expected revenue minus cost divided by cost.", "example": 18.85},
    {"field": "recommendation", "type": "string", "unit": "decision", "description": "Sandbox budget/bid recommendation.", "example": "increase_bid_or_budget"},
    {"field": "quality_flags", "type": "string", "unit": "audit label", "description": "Data-quality flags from validation.", "example": "missing_landing_seconds;"},
]

FAILURE_CASES = [
    {
        "failure_case": "Duplicate lead_id",
        "example_field": "lead_id repeated",
        "expected_behavior": "Quarantine duplicate for training; live duplicate receives fallback and audit flag.",
        "acceptance_check": "Quarantine table contains duplicate_lead_id flag.",
    },
    {
        "failure_case": "Missing channel",
        "example_field": "channel = NaN",
        "expected_behavior": "Do not trust model; use global or segment deterministic fallback.",
        "acceptance_check": "score_source contains baseline or fallback.",
    },
    {
        "failure_case": "Negative ad cost",
        "example_field": "ad_cost = -3.20",
        "expected_behavior": "Replace with segment/global median cost and mark low quality.",
        "acceptance_check": "input_quality != ok and attributed_cost > 0.",
    },
    {
        "failure_case": "Form completion > 1",
        "example_field": "form_completion = 1.35",
        "expected_behavior": "Range violation; fallback scoring and uncertainty penalty.",
        "acceptance_check": "probability_uncertainty increased versus clean row.",
    },
    {
        "failure_case": "Future timestamp",
        "example_field": "timestamp = now + 2 hours",
        "expected_behavior": "Quarantine in batch ingestion; live scorer marks invalid.",
        "acceptance_check": "future_timestamp flag appears in quarantine or input_quality invalid.",
    },
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def finite_number(x):
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def as_int_array(x):
    if x is None:
        return np.array([], dtype=int)

    arr = np.asarray(x)
    if arr.size == 0:
        return np.array([], dtype=int)

    s = pd.Series(arr)
    s = pd.to_numeric(s, errors="coerce")
    s = s.fillna(0).astype(int)
    return s.to_numpy()


def as_float_array(x):
    if x is None:
        return np.array([], dtype=float)

    arr = np.asarray(x)
    if arr.size == 0:
        return np.array([], dtype=float)

    s = pd.Series(arr)
    s = pd.to_numeric(s, errors="coerce")
    out = s.to_numpy(dtype=float)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


def col_series(df, name, default=0):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def safe_tuple(row, keys):
    if not keys:
        return ()
    vals = []
    for k in keys:
        v = row.get(k, "unknown")
        vals.append("unknown" if pd.isna(v) else str(v))
    return tuple(vals)


def calibration_curve_manual(y_true, y_prob, n_bins=10):
    y_true = as_int_array(y_true)
    y_prob = np.clip(as_float_array(y_prob), 0.0, 1.0)

    if len(y_true) == 0 or len(y_prob) == 0:
        return np.array([]), np.array([])

    min_len = min(len(y_true), len(y_prob))
    y_true = y_true[:min_len]
    y_prob = y_prob[:min_len]

    if min_len == 0:
        return np.array([]), np.array([])

    try:
        bins = pd.qcut(y_prob, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        bins = np.zeros(len(y_prob), dtype=int)

    tmp = pd.DataFrame({"y": y_true, "p": y_prob, "bin": bins})
    grouped = (
        tmp.groupby("bin")
        .agg(observed=("y", "mean"), mean_predicted=("p", "mean"), count=("y", "size"))
        .reset_index()
        .sort_values("bin")
    )

    return grouped["observed"].to_numpy(), grouped["mean_predicted"].to_numpy()


def generate_historical(n=4200, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    now = pd.Timestamp.now(tz="UTC").normalize()
    start = now - pd.Timedelta(days=60)
    seconds = np.sort(rng.uniform(0, 60 * 86400, n))
    ts = start + pd.to_timedelta(seconds, unit="s")

    channel_raw = rng.choice(CHANNELS, size=n, p=[0.31, 0.24, 0.19, 0.16, 0.10]).astype(object)
    campaign = np.array([rng.choice(CAMPAIGN_BY_CHANNEL[c]) for c in channel_raw], dtype=object)
    device_raw = rng.choice(DEVICES, size=n, p=[0.42, 0.50, 0.08]).astype(object)
    region_raw = rng.choice(REGIONS, size=n, p=[0.38, 0.29, 0.22, 0.11]).astype(object)
    product_raw = rng.choice(PRODUCTS, size=n, p=[0.38, 0.34, 0.18, 0.10]).astype(object)

    landing = np.clip(
        rng.gamma(shape=2.0, scale=18.0, size=n) + np.where(channel_raw == "email", 12.0, 0.0),
        0.0,
        None,
    )
    form = np.clip(
        rng.beta(2.2, 3.0, size=n) + np.where(channel_raw == "search", 0.08, 0.0),
        0.0,
        1.0,
    )
    prior = rng.poisson(1.2 + np.where(channel_raw == "email", 0.8, 0.0))
    email_opens = rng.poisson(0.7 + np.where(channel_raw == "email", 1.3, 0.0))
    page_views = rng.poisson(2.1 + np.where(channel_raw == "search", 1.1, 0.0))
    promo = rng.random(n) < np.where(channel_raw == "affiliate", 0.25, 0.08)

    cost_mult = np.where(
        channel_raw == "search",
        1.45,
        np.where(
            channel_raw == "social",
            1.05,
            np.where(
                channel_raw == "display",
                0.75,
                np.where(channel_raw == "email", 0.35, 0.90),
            ),
        ),
    )
    ad_cost = np.exp(rng.normal(loc=math.log(1.8), scale=0.75, size=n)) * cost_mult

    landing[rng.random(n) < 0.025] = np.nan
    form[rng.random(n) < 0.015] = np.nan
    ad_cost[rng.random(n) < 0.010] = np.nan

    base = np.array([BASE_RATE.get(c, 0.05) for c in channel_raw])
    dev_mult = np.array([DEVICE_MULT.get(d, 1.0) for d in device_raw])
    reg_mult = np.array([REGION_MULT.get(r, 1.0) for r in region_raw])
    prod_mult = np.array([PRODUCT_MULT.get(p, 1.0) for p in product_raw])

    landing_median = float(np.nanmedian(landing)) if np.isfinite(np.nanmedian(landing)) else 20.0
    form_median = float(np.nanmedian(form)) if np.isfinite(np.nanmedian(form)) else 0.45

    z_landing = (np.nan_to_num(landing, nan=landing_median) - landing_median) / 15.0
    z_form = (np.nan_to_num(form, nan=form_median) - form_median) / 0.20
    z_prior = (prior - 1.5) / 1.2
    z_email = (email_opens - 1.0) / 1.3
    z_page = (page_views - 2.5) / 1.6

    base_logit = np.log(base / (1.0 - base))
    logit = (
        base_logit
        + np.log(np.clip(dev_mult * reg_mult * prod_mult, 0.20, 5.0))
        + 0.55 * z_landing
        + 0.95 * z_form
        + 0.35 * z_prior
        + 0.25 * z_email
        + 0.30 * z_page
        + 0.18 * promo.astype(int)
        + rng.normal(0.0, 0.35, n)
    )
    logit = np.clip(logit, -8.0, 8.0)
    p_true = sigmoid(logit)
    converted = (rng.random(n) < p_true).astype(int)

    aov_scale = np.array([AOV_BASE.get(p, 100.0) for p in product_raw], dtype=float) / 2.0
    revenue = rng.gamma(shape=2.0, scale=aov_scale)
    revenue = np.clip(revenue, 10.0, 5000.0)

    df = pd.DataFrame(
        {
            "lead_id": [f"H-{i:06d}" for i in range(n)],
            "timestamp": ts,
            "channel": channel_raw,
            "campaign": campaign,
            "device": device_raw,
            "region": region_raw,
            "product_interest": product_raw,
            "ad_cost": ad_cost,
            "landing_seconds": landing,
            "form_completion": form,
            "prior_sessions": prior,
            "email_opens": email_opens,
            "page_views": page_views,
            "promo_code": promo,
            "revenue_if_converted": revenue,
            "converted": converted,
        }
    )

    dup_count = max(5, int(n * 0.01))
    dup_idx = rng.choice(n, size=dup_count, replace=False)
    dup_rows = df.iloc[dup_idx].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    return df


def validate_dataframe(df, name="dataset"):
    clean = df.copy()

    for col in REQUIRED_COLUMNS:
        if col not in clean.columns:
            raise ValueError(f"Missing required column: {col}")

    for col in [
        "landing_seconds",
        "form_completion",
        "prior_sessions",
        "email_opens",
        "page_views",
        "ad_cost",
        "revenue_if_converted",
    ]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean["converted"] = pd.to_numeric(clean["converted"], errors="coerce")

    if not pd.api.types.is_datetime64_any_dtype(clean["timestamp"]):
        clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce", utc=True)
    else:
        if clean["timestamp"].dt.tz is None:
            clean["timestamp"] = clean["timestamp"].dt.tz_localize("UTC")
        else:
            clean["timestamp"] = clean["timestamp"].dt.tz_convert("UTC")

    flags = []

    def add(mask, label):
        mask = pd.Series(mask, index=clean.index).fillna(False).astype(bool)
        flags.append((label, mask))
        return mask

    dup = add(clean.duplicated(subset=["lead_id"], keep="first"), "duplicate_lead_id")
    crit = add(clean[["lead_id", "timestamp", "channel", "converted"]].isna().any(axis=1), "missing_critical")
    invalid = add(
        (~clean["channel"].isin(CHANNELS))
        | (~clean["device"].isin(DEVICES))
        | (~clean["region"].isin(REGIONS))
        | (~clean["product_interest"].isin(PRODUCTS)),
        "invalid_category",
    )
    bad_conv = add(~clean["converted"].isin([0, 1]), "invalid_target")
    range_bad = add(
        (
            (clean["form_completion"].notna() & ((clean["form_completion"] < 0) | (clean["form_completion"] > 1)))
            | (clean["landing_seconds"].notna() & (clean["landing_seconds"] < 0))
            | (clean["ad_cost"].notna() & (clean["ad_cost"] < 0))
            | (clean["revenue_if_converted"].notna() & (clean["revenue_if_converted"] < 0))
        ),
        "range_violation",
    )

    now = pd.Timestamp.now(tz="UTC")
    future = add(clean["timestamp"] > now + pd.Timedelta(hours=1), "future_timestamp")
    old = add(clean["timestamp"] < now - pd.Timedelta(days=365), "late_or_old")

    q = clean["ad_cost"].quantile(0.999)
    q = float(q) if np.isfinite(q) else 100.0
    outlier = add(clean["ad_cost"] > max(1000.0, q * 8.0), "outlier_cost")
    missing_cost = add(clean["ad_cost"].isna(), "missing_ad_cost")
    missing_landing = add(clean["landing_seconds"].isna(), "missing_landing_seconds")

    quarantine_mask = dup | crit | invalid | bad_conv | range_bad | future

    flag_series = pd.Series("", index=clean.index, dtype=object)
    for label, mask in flags:
        if mask.any():
            flag_series.loc[mask] = flag_series.loc[mask] + label + ";"

    clean["quality_flags"] = flag_series

    quarantine = clean[quarantine_mask].copy()
    usable = clean[~quarantine_mask].copy()

    summary = pd.DataFrame(
        [
            {"check": "dataset", "value": name},
            {"check": "raw_rows", "value": int(len(df))},
            {"check": "usable_rows", "value": int(len(usable))},
            {"check": "quarantined_rows", "value": int(len(quarantine))},
            {"check": "duplicate_lead_id", "value": int(dup.sum())},
            {"check": "missing_critical", "value": int(crit.sum())},
            {"check": "invalid_category", "value": int(invalid.sum())},
            {"check": "range_violation", "value": int(range_bad.sum())},
            {"check": "future_timestamp", "value": int(future.sum())},
            {"check": "late_or_old", "value": int(old.sum())},
            {"check": "outlier_cost", "value": int(outlier.sum())},
            {"check": "missing_ad_cost", "value": int(missing_cost.sum())},
            {"check": "missing_landing_seconds", "value": int(missing_landing.sum())},
        ]
    )

    return usable, quarantine, summary


def add_derived_features(df):
    out = df.copy()

    if "ad_cost" not in out.columns:
        out["ad_cost"] = np.nan
    out["ad_cost"] = pd.to_numeric(out["ad_cost"], errors="coerce")
    out["ad_cost_log"] = np.log1p(out["ad_cost"].clip(lower=0))

    if "promo_code" in out.columns:
        out["promo_code_flag"] = (
            out["promo_code"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "1.0", "yes", "t"])
            .astype(int)
        )
    else:
        out["promo_code_flag"] = 0

    if "timestamp" in out.columns:
        ts = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
        out["hour"] = ts.dt.hour.fillna(0).astype(int)
        out["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(int)
    else:
        out["hour"] = 0
        out["dayofweek"] = 0

    return out


def chronological_split(df, train_frac=0.70, val_frac=0.15):
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def build_segment_baselines(df, prior_strength=25):
    global_rate = float(df[TARGET].mean()) if len(df) else 0.05
    baselines = {}

    for keys in BASELINE_KEY_SETS:
        if keys:
            grp = df.groupby(list(keys))[TARGET].agg(conv="sum", n="count").reset_index()
        else:
            grp = pd.DataFrame({"conv": [float(df[TARGET].sum())], "n": [int(len(df))]})

        for _, r in grp.iterrows():
            key = tuple(str(r[k]) for k in keys) if keys else ()
            n = float(r["n"])
            conv = float(r["conv"])
            rate = (conv + prior_strength * global_rate) / (n + prior_strength) if n + prior_strength > 0 else global_rate
            baselines[key] = {"rate": float(rate), "n": int(n), "conv": int(conv)}

    return baselines, global_rate


def build_segment_revenue_baselines(df, prior_strength=20):
    global_mean = float(df[VALUE_TARGET].mean()) if len(df) else 100.0
    rev = {}

    for keys in BASELINE_KEY_SETS:
        if keys:
            grp = df.groupby(list(keys))[VALUE_TARGET].agg(total="sum", n="count").reset_index()
        else:
            grp = pd.DataFrame({"total": [float(df[VALUE_TARGET].sum())], "n": [int(len(df))]})

        for _, r in grp.iterrows():
            key = tuple(str(r[k]) for k in keys) if keys else ()
            n = float(r["n"])
            total = float(r["total"])
            mean = (total + prior_strength * global_mean) / (n + prior_strength) if n + prior_strength > 0 else global_mean
            rev[key] = {"value": float(max(mean, 1.0)), "n": int(n)}

    return rev, global_mean


def make_preprocessor():
    num = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    return ColumnTransformer(
        [
            ("num", num, NUMERIC_FEATURES),
            ("cat", cat, CAT_FEATURES),
        ],
        remainder="drop",
    )


def safe_binary_metrics(y_true, prob):
    y_true = as_int_array(y_true)
    prob = np.clip(as_float_array(prob), 0.0, 1.0)

    out = {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else 0.0,
        "auc": np.nan,
        "pr_auc": np.nan,
        "brier": np.nan,
        "logloss": np.nan,
    }

    if len(y_true) == 0 or len(prob) == 0 or len(y_true) != len(prob):
        return out

    if len(np.unique(y_true)) < 2:
        out["brier"] = float(brier_score_loss(y_true, prob))
        return out

    out["auc"] = float(roc_auc_score(y_true, prob))
    out["pr_auc"] = float(average_precision_score(y_true, prob))
    out["brier"] = float(brier_score_loss(y_true, prob))
    out["logloss"] = float(log_loss(y_true, prob, labels=[0, 1]))

    return out


def train_and_evaluate(train, val, test):
    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET].astype(int)

    X_val = val[FEATURE_COLUMNS] if len(val) else None
    y_val = val[TARGET].astype(int) if len(val) else np.array([], dtype=int)

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET].astype(int)

    lr = Pipeline(
        [
            ("prep", make_preprocessor()),
            ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
        ]
    )

    lr.fit(X_train, y_train)

    lr_val_prob = lr.predict_proba(X_val)[:, 1] if X_val is not None and len(X_val) > 0 else np.array([])
    lr_test_prob = lr.predict_proba(X_test)[:, 1]

    models = {"logistic_regression": lr}
    metrics = {
        "logistic_regression": {
            "validation": safe_binary_metrics(y_val, lr_val_prob) if len(lr_val_prob) > 0 else {},
            "test": safe_binary_metrics(y_test, lr_test_prob),
        }
    }

    if HistGradientBoostingClassifier is not None:
        try:
            gb = Pipeline(
                [
                    ("prep", make_preprocessor()),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_iter=180,
                            learning_rate=0.08,
                            max_leaf_nodes=31,
                            l2_regularization=1.0,
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            )

            gb.fit(X_train, y_train)

            gb_val_prob = gb.predict_proba(X_val)[:, 1] if X_val is not None and len(X_val) > 0 else np.array([])
            gb_test_prob = gb.predict_proba(X_test)[:, 1]

            models["gradient_boosting"] = gb
            metrics["gradient_boosting"] = {
                "validation": safe_binary_metrics(y_val, gb_val_prob) if len(gb_val_prob) > 0 else {},
                "test": safe_binary_metrics(y_test, gb_test_prob),
            }
        except Exception:
            pass

    def selection_score(metric_block):
        d = metric_block.get("validation") or metric_block.get("test") or {}
        for k in ["pr_auc", "auc"]:
            v = d.get(k, np.nan)
            if np.isfinite(v):
                return float(v)
        return -1.0

    chosen = "logistic_regression"
    if "gradient_boosting" in models:
        if selection_score(metrics["gradient_boosting"]) >= selection_score(metrics["logistic_regression"]):
            chosen = "gradient_boosting"

    return models, metrics, chosen


def extract_lr_coefficients(lr_pipe):
    try:
        prep = lr_pipe.named_steps["prep"]
        clf = lr_pipe.named_steps["clf"]
        names = list(prep.get_feature_names_out())
        coef = clf.coef_[0]
        df = pd.DataFrame({"feature": names, "coefficient": coef})
        df["abs"] = df["coefficient"].abs()
        return df.sort_values("abs", ascending=False).reset_index(drop=True)
    except Exception:
        return None


def fallback_probability(row, baselines, global_rate):
    for keys in BASELINE_KEY_SETS:
        key = safe_tuple(row, keys)
        info = baselines.get(key)
        min_n = 10 if keys else 1
        if info and info["n"] >= min_n:
            source = "baseline_" + "_".join(keys) if keys else "global_baseline"
            return info["rate"], source, info["n"]
    return global_rate, "global_fallback", 0


def fallback_revenue(row, revenue_baselines, global_revenue):
    for keys in BASELINE_KEY_SETS:
        key = safe_tuple(row, keys)
        info = revenue_baselines.get(key)
        min_n = 10 if keys else 1
        if info and info["n"] >= min_n:
            source = "revenue_baseline_" + "_".join(keys) if keys else "global_revenue_baseline"
            return info["value"], source, info["n"]
    return global_revenue, "global_revenue_fallback", 0


def score_batch(df, models, model_name, baselines, revenue_baselines, global_rate, global_revenue, global_median_cost):
    out = df.copy()

    if out.empty:
        all_cols = FEATURE_COLUMNS + [TARGET, VALUE_TARGET, "ad_cost", "timestamp"] + SCORED_OUTPUT_COLUMNS
        for c in all_cols:
            if c not in out.columns:
                out[c] = pd.Series(dtype="object")
        return out

    for c in FEATURE_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan

    if TARGET not in out.columns:
        out[TARGET] = 0
    if VALUE_TARGET not in out.columns:
        out[VALUE_TARGET] = np.nan
    if "ad_cost" not in out.columns:
        out["ad_cost"] = np.nan
    if "timestamp" not in out.columns:
        out["timestamp"] = pd.NaT

    out["input_quality"] = "ok"

    missing_crit = out[["channel", "device", "region", "campaign", "product_interest", "timestamp"]].isna().any(axis=1)
    invalid_cat = (
        (~out["channel"].isin(CHANNELS))
        | (~out["device"].isin(DEVICES))
        | (~out["region"].isin(REGIONS))
        | (~out["product_interest"].isin(PRODUCTS))
    )
    bad_range = (
        (out["form_completion"].notna() & ((out["form_completion"] < 0) | (out["form_completion"] > 1)))
        | (out["landing_seconds"].notna() & (out["landing_seconds"] < 0))
        | (out["ad_cost"].notna() & (out["ad_cost"] < 0))
        | (out["revenue_if_converted"].notna() & (out["revenue_if_converted"] < 0))
    )

    now_utc = pd.Timestamp.now(tz="UTC")
    ts = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    future = ts > now_utc + pd.Timedelta(hours=1)
    old = ts < now_utc - pd.Timedelta(days=365)

    if "lead_id" in out.columns:
        dup = out.duplicated(subset=["lead_id"], keep="first")
    else:
        dup = pd.Series(False, index=out.index)

    out.loc[bad_range, "input_quality"] = "low"
    out.loc[missing_crit, "input_quality"] = "low"
    out.loc[invalid_cat, "input_quality"] = "invalid"
    out.loc[future, "input_quality"] = "invalid"
    out.loc[old & (out["input_quality"] == "ok"), "input_quality"] = "low"
    out.loc[dup & (out["input_quality"] == "ok"), "input_quality"] = "low"

    eligible = ~out["input_quality"].isin(["low", "invalid"])

    probs = np.full(len(out), np.nan, dtype=float)
    sources = np.array(["model"] * len(out), dtype=object)
    unc = np.full(len(out), np.nan, dtype=float)

    if eligible.any() and model_name in models:
        try:
            X = out.loc[eligible, FEATURE_COLUMNS]
            p = models[model_name].predict_proba(X)[:, 1]
            probs[np.asarray(eligible)] = p
        except Exception:
            pass

    rev_expected = []
    cost_used = []

    for i, (_, row) in enumerate(out.iterrows()):
        p = probs[i]
        source = sources[i]
        n_eff = 100

        if not np.isfinite(p) or row["input_quality"] in ["low", "invalid"]:
            fb, source, n = fallback_probability(row, baselines, global_rate)
            p = fb
            n_eff = max(n, 1)
        else:
            _, _, n = fallback_probability(row, baselines, global_rate)
            n_eff = max(n, 50)

        penalty = {"ok": 0.0, "low": 0.08, "invalid": 0.15}.get(row["input_quality"], 0.0)
        se = math.sqrt(max(float(p) * (1.0 - float(p)), 1e-6) / max(float(n_eff), 1.0))
        u = min(0.49, 2.0 * se + penalty)

        probs[i] = float(np.clip(p, 0.001, 0.999))
        sources[i] = source
        unc[i] = float(u)

        rv = row.get(VALUE_TARGET, np.nan)
        if not finite_number(rv) or float(rv) <= 0 or float(rv) > 100000:
            rv, _, _ = fallback_revenue(row, revenue_baselines, global_revenue)
        rev_expected.append(float(rv))

        cost = row.get("ad_cost", np.nan)
        if not finite_number(cost) or float(cost) < 0 or float(cost) > 100000:
            cost = global_median_cost
        cost_used.append(float(max(cost, 0.01)))

    out["conversion_probability"] = probs
    out["probability_uncertainty"] = unc
    out["score_source"] = sources
    out["expected_order_value"] = rev_expected
    out["expected_revenue"] = out["conversion_probability"] * out["expected_order_value"]
    out["attributed_cost"] = cost_used
    out["expected_roi"] = (out["expected_revenue"] - out["attributed_cost"]) / out["attributed_cost"]
    out["recommendation"] = np.where(
        out["expected_roi"] > 0.25,
        "increase_bid_or_budget",
        np.where(out["expected_roi"] > 0.0, "hold", "reduce_or_exclude"),
    )

    return out


def generate_live_batch(tick, size, rng):
    size = int(max(1, size))
    base_time = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=1)
    offsets = np.sort(rng.uniform(0, 2, size))
    ts = [base_time + pd.Timedelta(seconds=float(o)) for o in offsets]

    channel_raw = rng.choice(CHANNELS, size=size, p=[0.31, 0.24, 0.19, 0.16, 0.10]).astype(object)
    campaign = np.array([rng.choice(CAMPAIGN_BY_CHANNEL[c]) for c in channel_raw], dtype=object)
    device_raw = rng.choice(DEVICES, size=size, p=[0.42, 0.50, 0.08]).astype(object)
    region_raw = rng.choice(REGIONS, size=size, p=[0.38, 0.29, 0.22, 0.11]).astype(object)
    product_raw = rng.choice(PRODUCTS, size=size, p=[0.38, 0.34, 0.18, 0.10]).astype(object)

    landing = np.clip(
        rng.gamma(shape=2.0, scale=18.0, size=size) + np.where(channel_raw == "email", 12.0, 0.0),
        0.0,
        None,
    )
    form = np.clip(
        rng.beta(2.2, 3.0, size=size) + np.where(channel_raw == "search", 0.08, 0.0),
        0.0,
        1.0,
    )
    prior = rng.poisson(1.2 + np.where(channel_raw == "email", 0.8, 0.0))
    email_opens = rng.poisson(0.7 + np.where(channel_raw == "email", 1.3, 0.0))
    page_views = rng.poisson(2.1 + np.where(channel_raw == "search", 1.1, 0.0))
    promo = rng.random(size) < np.where(channel_raw == "affiliate", 0.25, 0.08)

    cost_mult = np.where(
        channel_raw == "search",
        1.45,
        np.where(
            channel_raw == "social",
            1.05,
            np.where(
                channel_raw == "display",
                0.75,
                np.where(channel_raw == "email", 0.35, 0.90),
            ),
        ),
    )
    ad_cost = np.exp(rng.normal(loc=math.log(1.8), scale=0.75, size=size)) * cost_mult

    channel = channel_raw.copy()
    device = device_raw.copy()

    miss_ch = rng.random(size) < 0.04
    channel[miss_ch] = np.nan
    campaign[miss_ch] = "unknown"

    miss_device = rng.random(size) < 0.02
    device[miss_device] = np.nan

    neg_cost = rng.random(size) < 0.03
    ad_cost = ad_cost.copy()
    ad_cost[neg_cost] = -np.abs(ad_cost[neg_cost])

    bad_form = rng.random(size) < 0.03
    form = form.copy()
    if bad_form.sum() > 0:
        form[bad_form] = 1.0 + rng.random(int(bad_form.sum())) * 0.5

    land_miss = rng.random(size) < 0.05
    landing = landing.copy()
    landing[land_miss] = np.nan

    base = np.array([BASE_RATE.get(c, 0.05) for c in channel_raw])
    dev_mult = np.array([DEVICE_MULT.get(d, 1.0) for d in device_raw])
    reg_mult = np.array([REGION_MULT.get(r, 1.0) for r in region_raw])
    prod_mult = np.array([PRODUCT_MULT.get(p, 1.0) for p in product_raw])

    landing_median = 20.0
    form_median = 0.45

    z_landing = (np.nan_to_num(landing, nan=landing_median) - landing_median) / 15.0
    z_form = (np.nan_to_num(form, nan=form_median) - form_median) / 0.20
    z_prior = (prior - 1.5) / 1.2
    z_email = (email_opens - 1.0) / 1.3
    z_page = (page_views - 2.5) / 1.6

    base_logit = np.log(base / (1.0 - base))
    logit = (
        base_logit
        + np.log(np.clip(dev_mult * reg_mult * prod_mult, 0.20, 5.0))
        + 0.55 * z_landing
        + 0.95 * z_form
        + 0.35 * z_prior
        + 0.25 * z_email
        + 0.30 * z_page
        + 0.18 * promo.astype(int)
        + rng.normal(0.0, 0.35, size)
    )
    logit = np.clip(logit, -8.0, 8.0)
    p_true = sigmoid(logit)
    converted = (rng.random(size) < p_true).astype(int)

    aov_scale = np.array([AOV_BASE.get(p, 100.0) for p in product_raw], dtype=float) / 2.0
    revenue = rng.gamma(shape=2.0, scale=aov_scale)
    revenue = np.clip(revenue, 10.0, 5000.0)

    lead_ids = [f"L-{tick:05d}-{i:03d}" for i in range(size)]
    if size > 1 and rng.random() < 0.25:
        lead_ids[-1] = lead_ids[0]

    df = pd.DataFrame(
        {
            "lead_id": lead_ids,
            "timestamp": ts,
            "channel": channel,
            "campaign": campaign,
            "device": device,
            "region": region_raw,
            "product_interest": product_raw,
            "ad_cost": ad_cost,
            "landing_seconds": landing,
            "form_completion": form,
            "prior_sessions": prior,
            "email_opens": email_opens,
            "page_views": page_views,
            "promo_code": promo,
            "revenue_if_converted": revenue,
            "converted": converted,
        }
    )

    return add_derived_features(df)


def create_video_writer():
    temp_dir = Path(tempfile.gettempdir())
    candidates = [
        ("customer_conversion_engine_demo.mp4", "avc1", "video/mp4"),
        ("customer_conversion_engine_demo.mp4", "mp4v", "video/mp4"),
        ("customer_conversion_engine_demo.webm", "VP80", "video/webm"),
        ("customer_conversion_engine_demo.avi", "MJPG", "video/x-msvideo"),
    ]

    for fname, cc, mime in candidates:
        path = temp_dir / fname
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass

        fourcc = cv2.VideoWriter_fourcc(*cc)
        writer = cv2.VideoWriter(str(path), fourcc, 10, (1280, 720))
        if writer.isOpened():
            return writer, path, mime

    return None, None, None


def compute_kpis(df):
    if df is None or df.empty:
        return {
            "leads": 0,
            "observed_conversions": 0,
            "predicted_conversions": 0.0,
            "observed_revenue": 0.0,
            "expected_revenue": 0.0,
            "cost": 0.0,
            "roas": 0.0,
            "avg_prob": 0.0,
            "calibration_gap": 0.0,
        }

    leads = int(len(df))
    conv = col_series(df, TARGET, 0)
    prob = col_series(df, "conversion_probability", 0)
    aov = col_series(df, "expected_order_value", 0)
    exp_rev = col_series(df, "expected_revenue", 0)
    cost = col_series(df, "attributed_cost", 0)

    observed_conversions = float(conv.sum())
    predicted_conversions = float(prob.sum())
    observed_revenue = float((conv * aov).sum())
    expected_revenue = float(exp_rev.sum())
    total_cost = float(cost.sum())
    roas = observed_revenue / total_cost if total_cost > 0 else 0.0
    avg_prob = float(prob.mean()) if leads else 0.0
    calibration_gap = abs(predicted_conversions - observed_conversions) / leads if leads else 0.0

    return {
        "leads": leads,
        "observed_conversions": observed_conversions,
        "predicted_conversions": predicted_conversions,
        "observed_revenue": observed_revenue,
        "expected_revenue": expected_revenue,
        "cost": total_cost,
        "roas": roas,
        "avg_prob": avg_prob,
        "calibration_gap": calibration_gap,
    }


def top_segments(df, n=7):
    if df is None or df.empty or "expected_revenue" not in df.columns:
        return []

    tmp = df.copy()
    tmp["segment"] = tmp["channel"].astype(str) + " | " + tmp["campaign"].astype(str)
    agg = (
        tmp.groupby("segment", as_index=False)["expected_revenue"]
        .sum()
        .sort_values("expected_revenue", ascending=False)
        .head(n)
    )
    return [(str(r["segment"]), safe_float(r["expected_revenue"], 0.0)) for _, r in agg.iterrows()]


def draw_frame(tick, kpi, segment_bars, note=""):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (24, 24, 30)

    def put(text, org, scale=0.60, color=(230, 230, 230), thickness=1):
        cv2.putText(frame, str(text), org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 0), (1280, 78), (40, 44, 60), -1)
    put("Customer Conversion Probability Engine - Live Synthetic Demo", (24, 38), 0.82, (245, 245, 245), 2)
    put(
        f"Tick {tick} | {pd.Timestamp.now(tz='UTC').strftime('%H:%M:%S')} UTC | Model {MODEL_VERSION}",
        (24, 64),
        0.52,
        (185, 185, 195),
        1,
    )

    cards = [
        ("Live leads", f"{int(kpi['leads']):,}"),
        ("Observed conv.", f"{kpi['observed_conversions']:.0f}"),
        ("Predicted conv.", f"{kpi['predicted_conversions']:.1f}"),
        ("Expected revenue", f"${kpi['expected_revenue']:,.0f}"),
        ("Simulated ROAS", f"{kpi['roas']:.2f}"),
    ]

    for idx, (label, value) in enumerate(cards):
        x = 24 + idx * 248
        y = 98
        w = 228
        h = 92
        cv2.rectangle(frame, (x, y), (x + w, y + h), (34, 38, 52), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 76, 96), 1)
        put(label, (x + 14, y + 28), 0.52, (180, 185, 200), 1)
        put(value, (x + 14, y + 66), 0.78, (90, 220, 160), 2)

    cv2.rectangle(frame, (24, 214), (770, 690), (28, 30, 40), -1)
    put("Top segments by expected revenue, last 500 leads", (42, 246), 0.58, (220, 220, 230), 1)

    if segment_bars:
        maxval = max([v for _, v in segment_bars] + [1.0])
        for i, (seg, val) in enumerate(segment_bars[:7]):
            y = 292 + i * 54
            barw = int((safe_float(val, 0.0) / maxval) * 430)
            color = (80, 200, 120) if val >= 0 else (80, 80, 220)
            cv2.rectangle(frame, (42, y), (42 + max(barw, 2), y + 24), color, -1)
            put(seg[:34], (42, y - 8), 0.46, (210, 210, 220), 1)
            put(f"${val:,.0f}", (48 + max(barw, 2), y + 18), 0.46, (235, 235, 235), 1)
    else:
        put("Waiting for simulated leads...", (42, 320), 0.60, (180, 180, 190), 1)

    cv2.rectangle(frame, (790, 214), (1256, 690), (28, 30, 40), -1)
    put("Decision signals", (810, 246), 0.58, (220, 220, 230), 1)

    lines = [
        f"Average probability: {kpi['avg_prob']:.3f}",
        f"Calibration gap: {kpi['calibration_gap']:.3f}",
        f"Attributed cost: ${kpi['cost']:,.0f}",
        f"Observed revenue: ${kpi['observed_revenue']:,.0f}",
        "Fallback enabled for invalid rows",
        "Human approval required for spend changes",
        "Synthetic data only",
    ]

    for i, line in enumerate(lines):
        put(line, (810, 292 + i * 38), 0.50, (205, 210, 220), 1)

    put(note, (24, 708), 0.45, (150, 155, 170), 1)
    return frame


def add_event(event_type, detail, entity="system"):
    if "events" not in st.session_state or st.session_state.events is None:
        st.session_state.events = pd.DataFrame(
            columns=["event_time", "event_type", "entity", "detail", "actor", "model_version"]
        )

    row = pd.DataFrame(
        [
            {
                "event_time": pd.Timestamp.now(tz="UTC").isoformat(),
                "event_type": event_type,
                "entity": entity,
                "detail": detail,
                "actor": "demo_user",
                "model_version": MODEL_VERSION,
            }
        ]
    )

    st.session_state.events = pd.concat([st.session_state.events, row], ignore_index=True)
    if len(st.session_state.events) > 2000:
        st.session_state.events = st.session_state.events.tail(2000).reset_index(drop=True)


def append_simulation_ticks(ticks, writer=None):
    for _ in range(int(ticks)):
        batch = generate_live_batch(
            st.session_state.tick,
            int(st.session_state.batch_size),
            st.session_state.rng,
        )

        scored = score_batch(
            batch,
            st.session_state.models,
            st.session_state.selected_model,
            st.session_state.baselines,
            st.session_state.revenue_baselines,
            st.session_state.global_rate,
            st.session_state.global_revenue,
            st.session_state.global_median_cost,
        )

        st.session_state.live_df = pd.concat([st.session_state.live_df, scored], ignore_index=True)
        if len(st.session_state.live_df) > 5000:
            st.session_state.live_df = st.session_state.live_df.tail(5000).reset_index(drop=True)

        st.session_state.tick += 1

        mean_p = float(scored["conversion_probability"].mean()) if len(scored) else 0.0
        add_event(
            "simulation_tick",
            f"Processed {len(scored)} synthetic leads; mean p={mean_p:.3f}",
            entity=f"tick_{st.session_state.tick}",
        )

        if writer is not None:
            kpi = compute_kpis(st.session_state.live_df.tail(500))
            bars = top_segments(st.session_state.live_df.tail(500))
            frame = draw_frame(
                st.session_state.tick,
                kpi,
                bars,
                note="Synthetic live simulation | no real customer data | deterministic fallback active",
            )
            writer.write(frame)


def record_auto_demo(ticks=45):
    writer, path, mime = create_video_writer()

    progress_slot = st.empty()
    status_slot = st.empty()

    progress_slot.progress(0.0)
    status_slot.write("Preparing synthetic live demo...")

    for i in range(int(ticks)):
        append_simulation_ticks(1, writer=writer)
        progress_slot.progress((i + 1) / float(ticks))
        status_slot.write(
            f"Recording live demo frame {i + 1}/{ticks}; "
            f"tick={st.session_state.tick}; rows={len(st.session_state.live_df)}"
        )
        time.sleep(0.02)

    if writer is not None:
        writer.release()

    progress_slot.progress(1.0)
    status_slot.write("Live demo recording finished." if path is not None else "Video codec unavailable; dashboard continues.")
    time.sleep(0.2)

    try:
        progress_slot.empty()
        status_slot.empty()
    except Exception:
        pass

    if path is not None and path.exists() and path.stat().st_size > 5000:
        return path, mime

    return None, None


def make_bad_row(lead_id, **kwargs):
    row = {
        "lead_id": lead_id,
        "timestamp": pd.Timestamp.now(tz="UTC"),
        "channel": "search",
        "campaign": "brand_core",
        "device": "desktop",
        "region": "NA",
        "product_interest": "growth",
        "ad_cost": 2.0,
        "landing_seconds": 25.0,
        "form_completion": 0.60,
        "prior_sessions": 1,
        "email_opens": 0,
        "page_views": 3,
        "promo_code": False,
        "revenue_if_converted": 260.0,
        "converted": 0,
    }
    row.update(kwargs)
    return row


def inject_failure_batch():
    rows = [
        make_bad_row("BAD-MISSING-CHANNEL", channel=np.nan),
        make_bad_row("BAD-INVALID-DEVICE", device="smart_fridge"),
        make_bad_row("BAD-NEGATIVE-COST", ad_cost=-4.25),
        make_bad_row("BAD-FORM-RANGE", form_completion=1.35),
        make_bad_row("BAD-FUTURE-TIME", timestamp=pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=2)),
        make_bad_row("BAD-DUPLICATE-ID"),
    ]
    rows.append(make_bad_row("BAD-DUPLICATE-ID", landing_seconds=11.0))

    df = pd.DataFrame(rows)
    df = add_derived_features(df)

    scored = score_batch(
        df,
        st.session_state.models,
        st.session_state.selected_model,
        st.session_state.baselines,
        st.session_state.revenue_baselines,
        st.session_state.global_rate,
        st.session_state.global_revenue,
        st.session_state.global_median_cost,
    )

    st.session_state.live_df = pd.concat([st.session_state.live_df, scored], ignore_index=True)
    add_event("failure_injection", f"Injected {len(scored)} malformed synthetic rows for fallback testing")


def reset_live():
    st.session_state.live_df = pd.DataFrame()
    st.session_state.tick = 0
    add_event("reset_live", "Live synthetic stream reset")


def logistic_contributions(row, lr_pipe, top_n=8):
    try:
        X = pd.DataFrame([row])
        for c in FEATURE_COLUMNS:
            if c not in X.columns:
                X[c] = np.nan

        Xt = lr_pipe.named_steps["prep"].transform(X[FEATURE_COLUMNS])
        coef = lr_pipe.named_steps["clf"].coef_[0]
        names = lr_pipe.named_steps["prep"].get_feature_names_out()

        contrib = pd.DataFrame(
            {
                "feature": names,
                "contribution": np.asarray(Xt).ravel() * coef,
            }
        )
        contrib["abs"] = contrib["contribution"].abs()
        return contrib.sort_values("abs", ascending=False).head(top_n)
    except Exception:
        return None


def allocate_budget(df, total_budget, min_roi=0.0, cap_share=0.35):
    total_budget = float(max(0.0, total_budget))

    if df is None or df.empty or "expected_revenue" not in df.columns:
        return pd.DataFrame(
            columns=[
                "segment",
                "current_spend",
                "current_share_spend",
                "expected_revenue",
                "expected_revenue_per_dollar",
                "recommended_spend",
                "delta",
                "leads",
                "avg_prob",
            ]
        )

    tmp = df.copy()
    tmp["segment"] = tmp["channel"].astype(str) + " | " + tmp["campaign"].astype(str)

    agg = (
        tmp.groupby("segment", as_index=False)
        .agg(
            current_spend=("attributed_cost", "sum"),
            expected_revenue=("expected_revenue", "sum"),
            leads=("lead_id", "count"),
            avg_prob=("conversion_probability", "mean"),
        )
    )

    agg["current_spend"] = agg["current_spend"].fillna(0).clip(lower=0)
    agg["expected_revenue"] = agg["expected_revenue"].fillna(0).clip(lower=0)
    agg["expected_revenue_per_dollar"] = agg["expected_revenue"] / agg["current_spend"].clip(lower=1.0)
    agg["incremental_value"] = agg["expected_revenue"] - agg["current_spend"]

    total_current = float(agg["current_spend"].sum())
    if total_current > 0:
        agg["current_share_spend"] = agg["current_spend"] / total_current * total_budget
    else:
        agg["current_share_spend"] = total_budget / max(len(agg), 1)

    agg["recommended_spend"] = 0.0
    eligible = agg["expected_revenue_per_dollar"] >= float(min_roi)

    if total_budget > 0 and eligible.any():
        weights = agg.loc[eligible, "incremental_value"].clip(lower=0) + 1e-6
        if float(weights.sum()) <= 1e-5:
            weights = agg.loc[eligible, "expected_revenue_per_dollar"].clip(lower=0) + 1e-6

        alloc = total_budget * weights / float(weights.sum())
        cap = cap_share * total_budget

        for _ in range(12):
            over = alloc > cap
            if not over.any():
                break

            excess = float(alloc[over].sum()) - cap * int(over.sum())
            alloc[over] = cap

            under = ~over
            if not under.any():
                break

            under_weights = weights[under]
            if float(under_weights.sum()) <= 1e-8:
                alloc[under] += excess / int(under.sum())
            else:
                alloc[under] += excess * under_weights / float(under_weights.sum())

        agg.loc[eligible, "recommended_spend"] = alloc.values

    agg["delta"] = agg["recommended_spend"] - agg["current_share_spend"]
    return agg.sort_values("recommended_spend", ascending=False).reset_index(drop=True)


def run_acceptance_checks():
    checks = []

    def add(name, passed, detail=""):
        checks.append({"Check": name, "Pass": bool(passed), "Detail": str(detail)})

    hist = st.session_state.get("historical_clean", pd.DataFrame())
    add("Synthetic history loaded", len(hist) >= 1000, f"{len(hist)} usable rows")

    quar = st.session_state.get("quarantine", pd.DataFrame())
    add("Duplicate/malformed records quarantined", len(quar) > 0, f"{len(quar)} quarantined rows")

    live = st.session_state.get("live_df", pd.DataFrame())
    add("Live simulation produced scored rows", len(live) > 0, f"{len(live)} live rows")

    if len(live) > 0 and "conversion_probability" in live.columns:
        add(
            "Probabilities are bounded [0, 1]",
            bool(((live["conversion_probability"] >= 0) & (live["conversion_probability"] <= 1)).all()),
            "",
        )
        add(
            "Uncertainty is non-negative",
            bool((live["probability_uncertainty"] >= 0).all()),
            "",
        )
    else:
        add("Probabilities are bounded [0, 1]", False, "No live rows")
        add("Uncertainty is non-negative", False, "No live rows")

    try:
        bad = pd.DataFrame([make_bad_row("CHECK-BAD", channel=np.nan)])
        bad = add_derived_features(bad)
        scored = score_batch(
            bad,
            st.session_state.models,
            st.session_state.selected_model,
            st.session_state.baselines,
            st.session_state.revenue_baselines,
            st.session_state.global_rate,
            st.session_state.global_revenue,
            st.session_state.global_median_cost,
        )
        src = str(scored["score_source"].iloc[0])
        add("Invalid input triggers deterministic fallback", "baseline" in src or "fallback" in src, src)
    except Exception as exc:
        add("Invalid input triggers deterministic fallback", False, str(exc))

    try:
        hs = st.session_state.get("historical_scored", pd.DataFrame())
        alloc = allocate_budget(hs, 10000.0, min_roi=0.0, cap_share=0.35)
        rec_total = float(alloc["recommended_spend"].sum()) if not alloc.empty else 0.0
        max_seg = float(alloc["recommended_spend"].max()) if not alloc.empty else 0.0
        add("Budget allocation does not exceed target", rec_total <= 10000.0 + 1.0, f"recommended ${rec_total:,.2f}")
        add("Per-segment cap enforced", max_seg <= 3500.0 + 1.0, f"max segment ${max_seg:,.2f}")
    except Exception as exc:
        add("Budget allocation does not exceed target", False, str(exc))
        add("Per-segment cap enforced", False, str(exc))

    vp = st.session_state.get("video_path")
    video_ok = vp is not None and Path(vp).exists() and Path(vp).stat().st_size > 5000
    add("Live demo video file generated", video_ok, str(vp))

    events = st.session_state.get("events", pd.DataFrame())
    add("Audit log has events", len(events) > 0, f"{len(events)} events")

    return pd.DataFrame(checks)


def initialize_session():
    st.session_state.rng = np.random.default_rng(RANDOM_SEED)
    st.session_state.tick = 0
    st.session_state.live_df = pd.DataFrame()
    st.session_state.events = pd.DataFrame(
        columns=["event_time", "event_type", "entity", "detail", "actor", "model_version"]
    )
    st.session_state.sim_running = False
    st.session_state.auto_ticks_remaining = 0
    st.session_state.batch_size = 25
    st.session_state.speed_delay = 0.25
    st.session_state.next_budget = 25000.0
    st.session_state.roi_hurdle = 0.0

    raw = generate_historical()
    clean, quar, dq = validate_dataframe(raw, name="historical_synthetic")
    clean = add_derived_features(clean)

    baselines, global_rate = build_segment_baselines(clean)
    revenue_baselines, global_revenue = build_segment_revenue_baselines(clean)
    global_median_cost = float(clean["ad_cost"].median()) if clean["ad_cost"].notna().any() else 2.0

    train, val, test = chronological_split(clean)
    models, metrics, chosen = train_and_evaluate(train, val, test)
    lr_coeff = extract_lr_coefficients(models["logistic_regression"])

    st.session_state.raw = raw
    st.session_state.historical_clean = clean
    st.session_state.quarantine = quar
    st.session_state.dq_summary = dq
    st.session_state.baselines = baselines
    st.session_state.revenue_baselines = revenue_baselines
    st.session_state.global_rate = global_rate
    st.session_state.global_revenue = global_revenue
    st.session_state.global_median_cost = global_median_cost
    st.session_state.models = models
    st.session_state.metrics = metrics
    st.session_state.chosen_model = chosen
    st.session_state.selected_model = chosen
    st.session_state.previous_selected_model = chosen
    st.session_state.lr_coeff = lr_coeff

    historical_scored = score_batch(
        clean,
        models,
        chosen,
        baselines,
        revenue_baselines,
        global_rate,
        global_revenue,
        global_median_cost,
    )
    test_scored = score_batch(
        test,
        models,
        chosen,
        baselines,
        revenue_baselines,
        global_rate,
        global_revenue,
        global_median_cost,
    )

    st.session_state.historical_scored = historical_scored
    st.session_state.test_scored = test_scored

    add_event(
        "bootstrap",
        f"Loaded {len(clean)} synthetic usable rows; quarantined {len(quar)}; trained model={chosen}",
    )

    video_path, video_mime = record_auto_demo(ticks=45)
    st.session_state.video_path = video_path
    st.session_state.video_mime = video_mime
    st.session_state.auto_ticks_remaining = 12
    st.session_state.sim_running = True
    st.session_state.bootstrapped = True


def main():
    if not st.session_state.get("bootstrapped", False):
        with st.spinner("Loading synthetic demo, training engine, and recording live preview..."):
            initialize_session()

    st.sidebar.title("⚙️ Engine controls")
    st.sidebar.caption(f"Model: {MODEL_VERSION} | Synthetic data only | No API keys")

    c1, c2 = st.sidebar.columns(2)
    if c1.button("▶ Start", use_container_width=True):
        st.session_state.sim_running = True
        add_event("control", "Live simulation started")

    if c2.button("⏸ Pause", use_container_width=True):
        st.session_state.sim_running = False
        add_event("control", "Live simulation paused")

    if st.sidebar.button("Step 10 ticks", use_container_width=True):
        append_simulation_ticks(10)
        add_event("control", "Manual step: 10 ticks")

    st.sidebar.slider(
        "Batch size per tick",
        5,
        100,
        value=int(st.session_state.get("batch_size", 25)),
        key="batch_size",
    )
    st.sidebar.slider(
        "Rerun delay seconds",
        0.05,
        1.00,
        value=float(st.session_state.get("speed_delay", 0.25)),
        step=0.05,
        key="speed_delay",
    )

    model_options = list(st.session_state.models.keys())
    if not model_options:
        model_options = ["logistic_regression"]

    if st.session_state.selected_model not in model_options:
        st.session_state.selected_model = model_options[0]

    st.sidebar.selectbox(
        "Scoring model",
        model_options,
        key="selected_model",
    )

    st.sidebar.slider(
        "ROI hurdle for allocation",
        -0.50,
        1.00,
        value=float(st.session_state.get("roi_hurdle", 0.0)),
        step=0.05,
        key="roi_hurdle",
    )
    st.sidebar.number_input(
        "Next period budget USD",
        min_value=0.0,
        value=float(st.session_state.get("next_budget", 25000.0)),
        step=1000.0,
        key="next_budget",
    )

    if st.sidebar.button("Inject failure batch", use_container_width=True):
        inject_failure_batch()

    if st.sidebar.button("Reset live stream", use_container_width=True):
        reset_live()

    if st.sidebar.button("Regenerate synthetic demo", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.sidebar.success("Prototype guardrail: recommendations are sandbox-only and require human approval.")

    if st.session_state.get("previous_selected_model") != st.session_state.selected_model:
        st.session_state.historical_scored = score_batch(
            st.session_state.historical_clean,
            st.session_state.models,
            st.session_state.selected_model,
            st.session_state.baselines,
            st.session_state.revenue_baselines,
            st.session_state.global_rate,
            st.session_state.global_revenue,
            st.session_state.global_median_cost,
        )
        if not st.session_state.live_df.empty:
            st.session_state.live_df = score_batch(
                st.session_state.live_df,
                st.session_state.models,
                st.session_state.selected_model,
                st.session_state.baselines,
                st.session_state.revenue_baselines,
                st.session_state.global_rate,
                st.session_state.global_revenue,
                st.session_state.global_median_cost,
            )
        st.session_state.previous_selected_model = st.session_state.selected_model
        add_event("model_switch", f"Scoring model switched to {st.session_state.selected_model}")

    st.title("📈 Customer Conversion Probability Engine")
    st.caption(
        "AI Advertising Intelligence and Revenue Optimization. This demo uses clearly labeled synthetic data, "
        "deterministic fallbacks, uncertainty bands, audit history, and sandbox-only budget recommendations."
    )

    live_df = st.session_state.live_df
    live_kpis = compute_kpis(live_df)
    recent_kpis = compute_kpis(live_df.tail(500) if not live_df.empty else live_df)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Live leads", f"{live_kpis['leads']:,}")
    m2.metric("Recent leads", f"{recent_kpis['leads']:,}")
    m3.metric("Predicted conversions", f"{live_kpis['predicted_conversions']:.1f}")
    m4.metric("Expected revenue", f"${live_kpis['expected_revenue']:,.0f}")
    m5.metric("Simulated ROAS", f"{live_kpis['roas']:.2f}")
    m6.metric("Chosen model", st.session_state.selected_model.replace("_", " "))

    if st.session_state.video_path is not None:
        vp = Path(st.session_state.video_path)
        if vp.exists():
            st.subheader("🎥 Generated live demo video")
            mime = st.session_state.get("video_mime") or "video/mp4"

            try:
                st.video(str(vp))
            except Exception:
                try:
                    st.video(vp.read_bytes(), formats=[mime])
                except Exception:
                    st.info("Inline video player unavailable in this environment. Use the download button below.")

            st.download_button(
                "Download demo video",
                data=vp.read_bytes(),
                file_name=vp.name,
                mime=mime,
            )
    else:
        st.warning("Video codec was unavailable in this environment. The live dashboard and simulation still run.")

    tab_demo, tab_analytics, tab_model, tab_opt, tab_data, tab_value, tab_audit = st.tabs(
        ["Live demo", "Analytics", "Model", "Optimization", "Data contract", "Value calculator", "Audit & acceptance"]
    )

    with tab_demo:
        st.subheader("Live conversion scoring and revenue decisions")
        st.caption(
            "Observed values are simulated ground-truth labels in the synthetic demo. Predicted values are model outputs. "
            "Invalid or low-quality rows use deterministic fallbacks and receive higher uncertainty."
        )

        display_cols = [
            "lead_id",
            "timestamp",
            "channel",
            "campaign",
            "device",
            "region",
            "conversion_probability",
            "probability_uncertainty",
            "score_source",
            "expected_order_value",
            "expected_revenue",
            "attributed_cost",
            "expected_roi",
            "recommendation",
            "input_quality",
        ]

        table_df = live_df.copy()
        for col in display_cols:
            if col not in table_df.columns:
                table_df[col] = np.nan

        st.dataframe(table_df[display_cols].tail(25), use_container_width=True)

        if not live_df.empty and "lead_id" in live_df.columns:
            st.subheader("Evidence inspector")
            lead_ids = [x for x in pd.unique(live_df["lead_id"]) if pd.notna(x)][::-1]
            if lead_ids:
                selected_lead = st.selectbox("Select live lead", lead_ids, key="inspect_lead")
                matches = live_df[live_df["lead_id"] == selected_lead]
                if not matches.empty:
                    row = matches.iloc[0]

                    e1, e2, e3 = st.columns(3)
                    e1.metric(
                        "Conversion probability",
                        f"{safe_float(row.get('conversion_probability'), 0.0):.3f}",
                        f"±{safe_float(row.get('probability_uncertainty'), 0.0):.3f}",
                    )
                    e2.metric("Expected revenue", f"${safe_float(row.get('expected_revenue'), 0.0):,.2f}")
                    e3.metric("Recommendation", str(row.get("recommendation", "n/a")).replace("_", " "))

                    st.write(
                        f"Score source: **{row.get('score_source', 'n/a')}** | "
                        f"Input quality: **{row.get('input_quality', 'n/a')}**"
                    )

                    fb, fb_src, fb_n = fallback_probability(
                        row,
                        st.session_state.baselines,
                        st.session_state.global_rate,
                    )
                    st.write(f"Deterministic segment baseline: **{fb:.3f}** from `{fb_src}` with n={fb_n}")

                    contrib = logistic_contributions(row, st.session_state.models["logistic_regression"])
                    if contrib is not None and not contrib.empty:
                        st.plotly_chart(
                            px.bar(
                                contrib,
                                x="contribution",
                                y="feature",
                                orientation="h",
                                title="Top logistic evidence contributions, synthetic model explanation",
                            ),
                            use_container_width=True,
                        )

    with tab_analytics:
        st.subheader("Historical and test-set analytics")
        hist = st.session_state.historical_scored.copy()
        test = st.session_state.test_scored.copy()

        if not hist.empty:
            hist["date"] = hist["timestamp"].dt.date
            daily = (
                hist.groupby("date")
                .agg(
                    observed=(TARGET, "mean"),
                    predicted=("conversion_probability", "mean"),
                    expected_revenue=("expected_revenue", "sum"),
                    attributed_cost=("attributed_cost", "sum"),
                )
                .reset_index()
            )

            st.plotly_chart(
                px.line(
                    daily,
                    x="date",
                    y=["observed", "predicted"],
                    title="Observed versus predicted conversion rate by day",
                ),
                use_container_width=True,
            )

            channel = (
                hist.groupby("channel", as_index=False)
                .agg(
                    leads=("lead_id", "count"),
                    expected_revenue=("expected_revenue", "sum"),
                    attributed_cost=("attributed_cost", "sum"),
                    avg_prob=("conversion_probability", "mean"),
                )
            )
            channel["roas"] = channel["expected_revenue"] / channel["attributed_cost"].clip(lower=1.0)

            st.plotly_chart(
                px.bar(
                    channel,
                    x="channel",
                    y=["expected_revenue", "attributed_cost"],
                    barmode="group",
                    title="Expected revenue versus attributed cost by channel",
                ),
                use_container_width=True,
            )

            st.plotly_chart(
                px.histogram(
                    hist,
                    x="conversion_probability",
                    nbins=40,
                    title="Predicted conversion probability distribution",
                ),
                use_container_width=True,
            )

        if not test.empty:
            frac_pos, mean_pred = calibration_curve_manual(
                test[TARGET].astype(int),
                test["conversion_probability"],
                n_bins=10,
            )
            if len(frac_pos) > 0:
                cal = pd.DataFrame({"mean_predicted": mean_pred, "observed": frac_pos})
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=cal["mean_predicted"],
                        y=cal["observed"],
                        mode="lines+markers",
                        name="Held-out model",
                    )
                )
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration"))
                fig.update_layout(title="Calibration on chronological held-out test set")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough scored test rows to build a calibration curve.")

        if not hist.empty:
            segment = (
                hist.groupby(["channel", "device", "region"], as_index=False)
                .agg(
                    leads=("lead_id", "count"),
                    avg_prob=("conversion_probability", "mean"),
                    observed_conv=(TARGET, "mean"),
                    expected_revenue=("expected_revenue", "sum"),
                    attributed_cost=("attributed_cost", "sum"),
                )
                .sort_values("expected_revenue", ascending=False)
            )
            st.subheader("Segment performance")
            st.dataframe(segment.head(75), use_container_width=True)

    with tab_model:
        st.subheader("Model evaluation and transparency")
        st.write(f"Chronologically chosen model: **{st.session_state.chosen_model}**")
        st.write("Active scoring model: **{}**".format(st.session_state.selected_model))

        rows = []
        for model_name, splits in st.session_state.metrics.items():
            for split, vals in splits.items():
                if isinstance(vals, dict):
                    for metric, value in vals.items():
                        rows.append(
                            {
                                "Model": model_name,
                                "Split": split,
                                "Metric": metric,
                                "Value": value,
                            }
                        )

        metrics_table = pd.DataFrame(rows)
        st.dataframe(metrics_table, use_container_width=True)

        st.markdown(
            """
            **Training versus inference contract**
            - Training uses historical synthetic leads only.
            - Validation and test sets are later in time to reduce leakage.
            - Inference scores new live leads using only pre-conversion signals.
            - No post-conversion fields are used as features.
            - Deterministic segment baselines are always available as fallback.
            """
        )

        if st.session_state.lr_coeff is not None and not st.session_state.lr_coeff.empty:
            st.subheader("Logistic baseline coefficients")
            st.plotly_chart(
                px.bar(
                    st.session_state.lr_coeff.head(20),
                    x="coefficient",
                    y="feature",
                    orientation="h",
                    title="Top 20 logistic coefficients, synthetic data",
                ),
                use_container_width=True,
            )

    with tab_opt:
        st.subheader("Sandbox budget and bid optimization")
        alloc = allocate_budget(
            st.session_state.historical_scored,
            float(st.session_state.next_budget),
            min_roi=float(st.session_state.roi_hurdle),
            cap_share=0.35,
        )

        st.dataframe(alloc, use_container_width=True)

        if not alloc.empty:
            st.plotly_chart(
                px.bar(
                    alloc,
                    x="segment",
                    y=["current_share_spend", "recommended_spend"],
                    barmode="group",
                    title="Current share versus recommended sandbox spend",
                ),
                use_container_width=True,
            )

            if float(alloc["recommended_spend"].sum()) <= 0.01:
                st.warning("No segment clears the ROI hurdle. Engine recommends holding budget and running new tests.")

        approve = st.checkbox("I approve applying this recommendation in a sandbox environment only.")
        if st.button("Record human approval"):
            if not approve:
                st.warning("Approval checkbox is required before recording a human-approved sandbox action.")
            else:
                alloc_total = float(alloc["recommended_spend"].sum()) if not alloc.empty and "recommended_spend" in alloc.columns else 0.0
                add_event(
                    "human_approval",
                    f"Approved sandbox budget plan total ${alloc_total:,.2f}",
                )
                st.success("Approval recorded in audit log. No external ad platform is modified.")

    with tab_data:
        st.subheader("Data contract")
        st.dataframe(pd.DataFrame(DATA_DICTIONARY), use_container_width=True)

        st.subheader("Example synthetic record")
        if not st.session_state.historical_clean.empty:
            sample = st.session_state.historical_clean.iloc[0].to_dict()
        else:
            sample = {}
        st.code(json.dumps(sample, indent=2, default=str), language="json")

        st.subheader("Quality summary")
        st.dataframe(st.session_state.dq_summary, use_container_width=True)

        st.subheader("Quarantine sample")
        quar = st.session_state.quarantine
        if not quar.empty:
            st.dataframe(quar[["lead_id", "timestamp", "channel", "device", "quality_flags"]].head(25), use_container_width=True)
        else:
            st.info("No quarantined rows in the current synthetic batch.")

        st.subheader("Failure cases and acceptance behavior")
        st.dataframe(pd.DataFrame(FAILURE_CASES), use_container_width=True)

        if not live_df.empty and "input_quality" in live_df.columns:
            bad_live = live_df[live_df["input_quality"] != "ok"]
            st.subheader("Injected or detected live failure rows")
            if not bad_live.empty:
                show_cols = [
                    "lead_id",
                    "channel",
                    "device",
                    "ad_cost",
                    "form_completion",
                    "conversion_probability",
                    "probability_uncertainty",
                    "score_source",
                    "input_quality",
                ]
                bad_show = bad_live.copy()
                for col in show_cols:
                    if col not in bad_show.columns:
                        bad_show[col] = np.nan
                st.dataframe(bad_show[show_cols].tail(25), use_container_width=True)
            else:
                st.info("No invalid live rows currently in buffer. Use Inject failure batch to test fallbacks.")

    with tab_value:
        st.subheader("Transparent business value calculator")
        st.caption("This calculator uses user-entered assumptions. It does not fabricate customer results or guarantee commercial outcomes.")

        vc1, vc2, vc3, vc4 = st.columns(4)
        monthly_leads = vc1.number_input("Monthly qualified leads", min_value=1000, max_value=10000000, value=120000, step=1000)
        base_cr = vc2.slider("Current conversion rate", 0.001, 0.500, value=0.045, step=0.001)
        aov = vc3.number_input("Average order value USD", min_value=10.0, max_value=5000.0, value=260.0, step=10.0)
        spend = vc4.number_input("Monthly ad spend USD", min_value=1000.0, max_value=10000000.0, value=900000.0, step=10000.0)

        uplift = st.slider("Assumed relative improvement in targeting or budget efficiency", 0.0, 0.25, value=0.06, step=0.01)
        haircut = st.slider("Confidence haircut", 0.0, 1.0, value=0.50, step=0.05)
        engine_cost = st.number_input("Annualized engine operating cost USD", min_value=0.0, max_value=10000000.0, value=120000.0, step=10000.0)

        baseline_conversions = monthly_leads * base_cr
        baseline_revenue = baseline_conversions * aov
        current_cac = spend / baseline_conversions if baseline_conversions > 0 else 0.0
        incremental_revenue = baseline_revenue * uplift * haircut
        monthly_net = incremental_revenue - engine_cost / 12.0
        payback = engine_cost / (incremental_revenue * 12.0) if incremental_revenue > 0 and monthly_net > 0 else None

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Baseline monthly revenue", f"${baseline_revenue:,.0f}")
        g2.metric("Current CAC assumption", f"${current_cac:,.2f}")
        g3.metric("Assumption-based incremental monthly revenue", f"${incremental_revenue:,.0f}")
        g4.metric("Net monthly value after engine cost", f"${monthly_net:,.0f}")

        st.metric("Simple payback months", f"{payback:.1f}" if payback is not None else "n/a")

        sensitivity = pd.DataFrame(
            [
                {
                    "Uplift assumption": u,
                    "Incremental monthly revenue": baseline_revenue * u * haircut,
                    "Net monthly value": baseline_revenue * u * haircut - engine_cost / 12.0,
                }
                for u in [0.0, 0.03, 0.06, 0.10, 0.15]
            ]
        )
        st.dataframe(sensitivity, use_container_width=True)

    with tab_audit:
        st.subheader("Audit history")
        events = st.session_state.events.copy()
        if not events.empty:
            events = events.sort_values("event_time", ascending=False)
        st.dataframe(events.tail(100), use_container_width=True)

        st.subheader("Acceptance checks")
        checks = run_acceptance_checks()
        st.dataframe(checks, use_container_width=True)

        st.subheader("Prototype versus production enterprise service")
        st.markdown(
            """
            **Useful prototype in this app**
            - Runs locally on CPU with synthetic data.
            - Demonstrates scoring, fallback, uncertainty, audit, optimization, and explanation.
            - Validates data contracts and failure handling.
            - Produces a live demo video and interactive dashboard.

            **Production enterprise service would additionally require**
            - Authenticated multi-tenant access and role-based approvals.
            - Real licensed ad-platform and CRM data connectors.
            - Causal uplift experimentation, not just correlation-based scoring.
            - Model registry, champion/challenger rollout, drift monitors, and rollback.
            - Privacy review, retention policy enforcement, and legal approval for personal data.
            - Serving SLAs, queueing, idempotent ingestion, disaster recovery, and cost governance.
            """
        )

        st.subheader("Known limitations")
        st.markdown(
            """
            - All data is synthetic and does not represent any real customer.
            - Conversion probability is predictive, not causal. Budget recommendations require experimentation.
            - Uncertainty is an approximate band, not a formal prediction interval.
            - Video generation depends on available OpenCV codecs; the dashboard remains functional if video encoding fails.
            """
        )

    if st.session_state.auto_ticks_remaining > 0:
        chunk = min(3, int(st.session_state.auto_ticks_remaining))
        append_simulation_ticks(chunk)
        st.session_state.auto_ticks_remaining -= chunk
        time.sleep(0.10)
        st.rerun()
    elif st.session_state.sim_running:
        append_simulation_ticks(2)
        time.sleep(float(st.session_state.speed_delay))
        st.rerun()


main()