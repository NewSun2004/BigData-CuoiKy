import io
import re
import math
import hashlib
from datetime import date, time, datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import s3fs

# =========================
# CONFIG
# =========================
MINIO_ENDPOINT = "http://52.62.39.221:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "admin123"

BUCKET = "vpandas.bucket"
MODEL_KEY = "sklearn_gbt_model.joblib"

st.set_page_config(page_title="Model QA - Flight Price", layout="wide")

# =========================
# MODEL LOAD
# =========================
@st.cache_resource
def load_model_cached(endpoint, access, secret, bucket, key):
    fs = s3fs.S3FileSystem(
        key=access,
        secret=secret,
        client_kwargs={"endpoint_url": endpoint},
    )
    with fs.open(f"s3://{bucket}/{key}", "rb") as f:
        return joblib.load(io.BytesIO(f.read()))

# =========================
# FEATURE ENGINEERING (your logic)
# =========================
def parse_duration_to_minutes(s: str) -> int:
    s = (s or "").lower().strip()
    h = 0
    m = 0
    mh = re.search(r"(\d+)\s*h", s)
    mm = re.search(r"(\d+)\s*m", s)
    if mh:
        h = int(mh.group(1))
    if mm:
        m = int(mm.group(1))
    if not mh and not mm:
        try:
            return int(float(s))
        except:
            return 0
    return h * 60 + m

def stable_hash_int(s: str) -> int:
    h = hashlib.md5((s or "").encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def stable_hash01(s: str) -> float:
    return (stable_hash_int(s) % 10_000) / 10_000.0

def stable_hash_range(s: str, low: float, high: float) -> float:
    return low + (high - low) * stable_hash01(s)

def raw_to_7features(raw: dict) -> np.ndarray:
    d: date = raw["Date_of_Journey"]
    dep: time = raw["Dep_Time"]
    dur_min = parse_duration_to_minutes(raw["Duration"])

    airline_h = stable_hash_range("air:"+raw["Airline"], -3.0, 3.0)
    source_h  = stable_hash_range("src:"+raw["Source"], -2.5, 2.5)
    dest_h    = stable_hash_range("dst:"+raw["Destination"], -2.5, 2.5)
    stops_h   = stable_hash_range("stp:"+raw["Total_Stops"], 0.0, 4.0)

    dep_float = dep.hour + dep.minute / 60.0
    dur_scaled = math.log1p(max(dur_min, 0))  # log(1+minutes)

    mix1 = math.sin((airline_h + source_h * 1.3 + dest_h * 0.7) * 2.1)
    mix2 = math.cos((stops_h + dep_float * 0.15 + dur_scaled) * 1.7)

    f0 = airline_h
    f1 = source_h
    f2 = dest_h
    f3 = stops_h
    f4 = dep_float + mix1 * 0.8
    f5 = dur_scaled + mix2 * 0.6
    f6 = (d.day + d.month * 0.3) + (mix1 - mix2) * 0.5

    return np.array([[f0, f1, f2, f3, f4, f5, f6]], dtype=float)

FEATURE_NAMES = ["airline_h", "source_h", "dest_h", "stops_h", "dep_mix", "dur_mix", "date_mix"]

# =========================
# UI CONSTANTS
# =========================
AIRLINES = [
    "IndiGo", "Air India", "Jet Airways", "SpiceJet", "GoAir", "Vistara",
    "Multiple carriers", "Multiple carriers Premium economy",
    "Trujet", "Air Asia", "Jet Airways Business", "Vistara Premium economy"
]
CITIES = ["Banglore", "Kolkata", "Delhi", "Chennai", "Mumbai", "Cochin", "Hyderabad", "New Delhi"]
STOPS = ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"]

DURATION_PATTERN = re.compile(r"^\s*(\d+\s*h)?\s*(\d+\s*m)?\s*$", re.IGNORECASE)

def validate_raw(raw: dict) -> list[str]:
    issues = []
    dur = (raw.get("Duration") or "").strip()
    if not dur:
        issues.append("Duration đang trống.")
    elif not DURATION_PATTERN.match(dur) and not dur.replace(".", "", 1).isdigit():
        issues.append("Duration format chưa đúng. Ví dụ: `2h 50m` hoặc `170` (phút).")

    if raw.get("Source") == raw.get("Destination"):
        issues.append("Source và Destination đang giống nhau (case lạ, nên kiểm tra).")

    dur_min = parse_duration_to_minutes(dur)
    if dur_min <= 0:
        issues.append("Duration phút <= 0 (bất thường).")
    return issues

def init_state():
    st.session_state.setdefault("Airline", "IndiGo" if "IndiGo" in AIRLINES else AIRLINES[0])
    st.session_state.setdefault("Source", "Banglore" if "Banglore" in CITIES else CITIES[0])
    st.session_state.setdefault("Destination", "New Delhi" if "New Delhi" in CITIES else CITIES[-1])
    st.session_state.setdefault("Total_Stops", "non-stop" if "non-stop" in STOPS else STOPS[0])
    st.session_state.setdefault("Date_of_Journey", date(2019, 3, 24))
    st.session_state.setdefault("Dep_Time", time(22, 20))
    st.session_state.setdefault("Duration", "2h 50m")

def reset_state():
    for k in ["Airline", "Source", "Destination", "Total_Stops", "Date_of_Journey", "Dep_Time", "Duration"]:
        if k in st.session_state:
            del st.session_state[k]
    init_state()
    st.rerun()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Model Settings")
with st.sidebar:
    st.caption("Dùng để demo kiểm thử & đánh giá mô hình.")
    endpoint = st.text_input("MinIO endpoint", value=MINIO_ENDPOINT)
    access = st.text_input("Access key", value=MINIO_ACCESS_KEY, type="password")
    secret = st.text_input("Secret key", value=MINIO_SECRET_KEY, type="password")
    bucket = st.text_input("Bucket", value=BUCKET)
    model_key = st.text_input("Model key", value=MODEL_KEY)

    colA, colB = st.columns(2)
    with colA:
        reload_btn = st.button("🔄 Reload model", use_container_width=True)
    with colB:
        st.button("🧹 Reset form", use_container_width=True, on_click=reset_state)

# load model
try:
    if reload_btn:
        load_model_cached.clear()
    model = load_model_cached(endpoint, access, secret, bucket, model_key)
    st.sidebar.success("Model loaded OK ✅")
except Exception as e:
    st.sidebar.error("Load model failed ❌")
    st.sidebar.exception(e)
    st.stop()

# =========================
# HEADER
# =========================
st.title("✈️ Predict Flight Price")
st.markdown(
    """
Mục tiêu: **kiểm thử** (input → feature → predict).
"""
)

init_state()

tab1 = st.tabs(["🧪 Single Prediction"])[0]

# =========================
# TAB 1: SINGLE
# =========================
with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Inputs")
        st.selectbox("Airline", AIRLINES, key="Airline")
        st.selectbox("Source", CITIES, key="Source")
        st.selectbox("Destination", CITIES, key="Destination")
        st.selectbox("Total_Stops", STOPS, key="Total_Stops")

    with right:
        st.subheader("Time")
        st.date_input("Date_of_Journey", key="Date_of_Journey")
        st.time_input("Dep_Time", key="Dep_Time")
        st.text_input("Duration", key="Duration", help="Ví dụ: 2h 50m hoặc nhập số phút như 170")

    raw = {
        "Airline": st.session_state["Airline"],
        "Source": st.session_state["Source"],
        "Destination": st.session_state["Destination"],
        "Total_Stops": st.session_state["Total_Stops"],
        "Date_of_Journey": st.session_state["Date_of_Journey"],
        "Dep_Time": st.session_state["Dep_Time"],
        "Duration": st.session_state["Duration"],
    }

    issues = validate_raw(raw)
    if issues:
        for msg in issues:
            st.warning(msg)

    c1, c2, c3 = st.columns([1, 1, 2], gap="large")
    with c1:
        do_pred = st.button("🚀 Predict", use_container_width=True, disabled=len(issues) > 0)
    with c2:
        do_sanity = st.button("🧠 Sanity check", use_container_width=True, disabled=len(issues) > 0)

    if do_pred:
        try:
            x = raw_to_7features(raw)
            pred = float(model.predict(x)[0])

            st.success("Predict OK ✅")
            st.metric("Giá dự đoán", f"{pred:,.4f}")

            with st.expander("🔎 Debug: raw input"):
                st.json({
                    **raw,
                    "Date_of_Journey": str(raw["Date_of_Journey"]),
                    "Dep_Time": raw["Dep_Time"].strftime("%H:%M"),
                    "Duration_minutes": parse_duration_to_minutes(raw["Duration"]),
                })

            with st.expander("🧾 Debug: engineered features (7)"):
                feat_df = pd.DataFrame(x, columns=FEATURE_NAMES)
                st.dataframe(feat_df, use_container_width=True)

        except Exception as e:
            st.error("Predict lỗi ❌")
            st.exception(e)

    if do_sanity:
        try:
            base_x = raw_to_7features(raw)
            base_pred = float(model.predict(base_x)[0])

            # counterfactual: increase duration by +60m
            dur_min = parse_duration_to_minutes(raw["Duration"])
            raw2 = dict(raw)
            raw2["Duration"] = f"{dur_min + 60}m"
            x2 = raw_to_7features(raw2)
            pred2 = float(model.predict(x2)[0])

            # counterfactual: change stops (if possible)
            raw3 = dict(raw)
            raw3["Total_Stops"] = "1 stop" if raw["Total_Stops"] == "non-stop" else "non-stop"
            x3 = raw_to_7features(raw3)
            pred3 = float(model.predict(x3)[0])

            st.info("Sanity checks (quick)")

            df = pd.DataFrame([
                {"Scenario": "Base", "Pred": base_pred},
                {"Scenario": "Duration + 60m", "Pred": pred2},
                {"Scenario": f"Stops → {raw3['Total_Stops']}", "Pred": pred3},
            ])
            st.dataframe(df, use_container_width=True)

            st.caption("Mục đích: phát hiện model phản ứng “ngược đời” quá mức (demo QA).")

        except Exception as e:
            st.error("Sanity check lỗi ❌")
            st.exception(e)
