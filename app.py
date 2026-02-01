import os
import streamlit as st
import pandas as pd
import numpy as np
import time

# PySpark Imports
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import StructType, StructField, ArrayType, DoubleType

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Flight Price Predictor | Spark ML + MinIO",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CONFIGURATION MANAGEMENT ---
def cfg(key: str, default: str = "") -> str:
    """Helper to get config from Streamlit secrets or Environment variables"""
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

# Default Configs (Editable in Sidebar)
DEFAULT_MINIO_ENDPOINT = cfg("MINIO_ENDPOINT", "http://52.62.39.221:9000")
DEFAULT_ACCESS_KEY = cfg("MINIO_ACCESS_KEY", "admin")
DEFAULT_SECRET_KEY = cfg("MINIO_SECRET_KEY", "admin123")
DEFAULT_MODEL_PATH = cfg("MODEL_PATH", "s3a://vpandas.bucket/best_pipeline")

HADOOP_VER = "3.3.4"
PACKAGES = [
    f"org.apache.hadoop:hadoop-aws:{HADOOP_VER}",
    "com.amazonaws:aws-java-sdk-bundle:1.12.262"
]

# --- 3. SPARK SESSION MANAGER ---
@st.cache_resource(show_spinner="Initializing Spark Engine...")
def get_spark(endpoint, access_key, secret_key):
    try:
        spark = (
            SparkSession.builder
            .appName("Streamlit_Spark_MinIO")
            .master("local[2]") # Use 2 cores for slightly better responsiveness
            .config("spark.driver.memory", "1g") # Increased memory slightly
            .config("spark.jars.packages", ",".join(PACKAGES))
            .config("spark.hadoop.fs.s3a.endpoint", endpoint)
            .config("spark.hadoop.fs.s3a.access.key", access_key)
            .config("spark.hadoop.fs.s3a.secret.key", secret_key)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false" if "https" not in endpoint else "true")
            # Speed up S3A initial connection
            .config("spark.hadoop.fs.s3a.fast.upload", "true") 
            .getOrCreate()
        )
        # Quiet logs
        spark.sparkContext.setLogLevel("ERROR")
        return spark
    except Exception as e:
        st.error(f"❌ Critical Error: Could not start Spark.\n\n{e}")
        return None

@st.cache_resource(show_spinner="Loading Model from MinIO...")
def load_model(_spark, model_path):
    try:
        model = PipelineModel.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model from `{model_path}`.")
        with st.expander("See Error Details"):
            st.code(str(e))
        return None

def get_expected_features(model):
    """Attempt to extract the number of features the model expects."""
    try:
        # Check the last stage (usually the estimator)
        last_stage = model.stages[-1]
        if hasattr(last_stage, "numFeatures"):
            return int(last_stage.numFeatures)
        if hasattr(last_stage, "getNumFeatures"):
            return int(last_stage.getNumFeatures())
        
        # If pipeline, check if VectorAssembler is present (often near the end of transformers)
        for stage in reversed(model.stages):
            if "VectorAssembler" in str(type(stage)):
                # VectorAssembler doesn't explicitly store 'output size' easily without metadata,
                # but we can try to guess or just return None.
                pass
    except:
        pass
    return None

# --- 4. SIDEBAR SETUP ---
with st.sidebar:
    st.image("https://spark.apache.org/images/spark-logo-trademark.png", width=150)
    st.header("⚙️ System Config")
    
    with st.expander("MinIO Connection", expanded=False):
        minio_ep = st.text_input("Endpoint", DEFAULT_MINIO_ENDPOINT)
        minio_ak = st.text_input("Access Key", DEFAULT_ACCESS_KEY, type="password")
        minio_sk = st.text_input("Secret Key", DEFAULT_SECRET_KEY, type="password")
    
    model_path_input = st.text_input("Model Path (S3A)", DEFAULT_MODEL_PATH)
    
    st.markdown("---")
    st.caption(f"Spark Version: {HADOOP_VER}")
    
    if st.button("♻️ Reset / Reload App"):
        st.cache_resource.clear()
        st.rerun()

# --- 5. MAIN APP LOGIC ---

st.title("✈️ Flight Price Predictor")
st.markdown("Predict flight ticket prices using a **Spark ML Pipeline** loaded directly from **MinIO Object Storage**.")

# Initialize Spark & Model
spark = get_spark(minio_ep, minio_ak, minio_sk)

if spark:
    model = load_model(spark, model_path_input)
    
    if model:
        # --- INPUT SECTION ---
        col_main, col_viz = st.columns([1, 2], gap="large")
        
        with col_main:
            st.subheader("📝 Flight Details")
            with st.container(border=True):
                distance_km = st.slider("Distance (km)", 0.0, 15000.0, 800.0, 50.0)
                days_to_dep = st.slider("Days to Departure", 0, 365, 30)
                dep_hour = st.slider("Departure Hour", 0, 23, 9)
                stops = st.radio("Stops", [0, 1, 2, 3], horizontal=True, index=0)

            with st.expander("🛠 Advanced: Feature Vector"):
                st.info("Override inputs manually. Expects comma-separated floats.")
                manual_features = st.text_area("Raw Features", height=100)
                
            predict_btn = st.button("🚀 Calculate Price", type="primary")

        # --- PREDICTION LOGIC ---
        if predict_btn:
            with st.spinner("Spark is crunching numbers..."):
                start_time = time.time()
                
                # 1. Prepare Feature Vector
                num_features = get_expected_features(model)
                
                if manual_features.strip():
                    try:
                        base_feats = [float(x.strip()) for x in manual_features.split(",") if x.strip()]
                    except ValueError:
                        st.error("Invalid format in Advanced Features.")
                        st.stop()
                else:
                    base_feats = [float(distance_km), float(days_to_dep), float(dep_hour), float(stops)]

                # 2. Padding/Truncating if model expects specific size
                if num_features and len(base_feats) != num_features:
                    if len(base_feats) < num_features:
                        base_feats += [0.0] * (num_features - len(base_feats))
                    else:
                        base_feats = base_feats[:num_features]
                
                # 3. Create Variance Scenarios (Sensitivity Analysis)
                # We create 5 small variations to show model stability/confidence
                variants = 5
                perturbations = np.linspace(0.95, 1.05, variants) # +/- 5% variation on inputs
                
                data_rows = []
                for p in perturbations:
                    # Perturb distance slightly to create scenarios
                    scenario_feats = base_feats.copy()
                    scenario_feats[0] = scenario_feats[0] * p 
                    data_rows.append((p, scenario_feats))

                # 4. Construct Spark DataFrame Efficiently
                # Define schema to avoid inference overhead/errors
                schema = StructType([
                    StructField("multiplier", DoubleType(), False),
                    StructField("features_list", ArrayType(DoubleType()), False)
                ])
                
                df_input = spark.createDataFrame(data_rows, schema)

                # Convert Array<Double> to VectorUDT (Required for Spark ML)
                to_vector = F.udf(lambda x: Vectors.dense(x), VectorUDT())
                df_ready = df_input.withColumn("features", to_vector("features_list"))
                
                # 5. Predict
                predictions = model.transform(df_ready).select("multiplier", "prediction")
                
                # Collect to Pandas for Visualization
                pdf = predictions.toPandas()
                
                end_time = time.time()
                elapsed = end_time - start_time

            # --- RESULTS SECTION ---
            with col_viz:
                st.subheader("📊 Prediction Analysis")
                
                # Main Result (using the central prediction where multiplier ~= 1.0)
                # Find the row closest to multiplier 1.0
                center_idx = (np.abs(pdf['multiplier'] - 1.0)).argmin()
                predicted_price = pdf.iloc[center_idx]['prediction']
                
                # Confidence / Range
                min_price = pdf['prediction'].min()
                max_price = pdf['prediction'].max()
                
                # Metrics Row
                m1, m2, m3 = st.columns(3)
                m1.metric("Estimated Price", f"${predicted_price:,.2f}")
                m2.metric("Range (±5% input)", f"${min_price:,.0f} - ${max_price:,.0f}")
                m3.metric("Calculation Time", f"{elapsed:.2f}s")
                
                st.markdown("#### Sensitivity Analysis")
                st.caption("How price changes if Distance varies by ±5%")
                
                # Create a clean chart data
                chart_data = pdf.rename(columns={"multiplier": "Input Factor", "prediction": "Predicted Price"})
                chart_data["Input Factor"] = ((chart_data["Input Factor"] - 1) * 100).round(1).astype(str) + "%"
                
                st.bar_chart(
                    chart_data,
                    x="Input Factor",
                    y="Predicted Price",
                    color="#4CAF50"
                )
                
                with st.expander("See Raw Data"):
                    st.dataframe(pdf.style.format({"multiplier": "{:.2f}", "prediction": "{:,.2f}"}))

    else:
        st.warning("⚠️ Model not loaded. Please check MinIO connection in the sidebar.")
else:
    st.info("💡 waiting for Spark initialization...")
