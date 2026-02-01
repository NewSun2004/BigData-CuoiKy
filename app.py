import os
import streamlit as st
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import Row

st.set_page_config(page_title="MinIO + Spark ML Demo", layout="wide")
st.title("Demo dự đoán giá vé máy bay bằng Spark ML (lưu trên MinIO)")

def cfg(key: str, default: str = "") -> str:
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

# Configuration
MINIO_ENDPOINT = cfg("MINIO_ENDPOINT", "http://52.62.39.221:9000")
MINIO_ACCESS_KEY = cfg("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = cfg("MINIO_SECRET_KEY", "admin123")
MODEL_PATH = cfg("MODEL_PATH", "s3a://vpandas.bucket/best_pipeline")

HADOOP_VER = "3.3.4"
PACKAGES = [
    f"org.apache.hadoop:hadoop-aws:{HADOOP_VER}",
    "com.amazonaws:aws-java-sdk-bundle:1.12.262"
]

@st.cache_resource
def get_spark():
    spark = (
        SparkSession.builder
        .appName("spark_minio_ok")
        .master("local[1]") 
        .config("spark.driver.memory", "512m")
        .config("spark.jars.packages", ",".join(PACKAGES))
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false" if "https" not in MINIO_ENDPOINT else "true")
        .getOrCreate()
    )
    return spark

@st.cache_resource
def load_model(_spark):
    return PipelineModel.load(MODEL_PATH)


def try_get_num_features(model: PipelineModel):
    try:
        last = model.stages[-1]
        if hasattr(last, "numFeatures"):
            return int(last.numFeatures)
        if hasattr(last, "getNumFeatures"):
            return int(last.getNumFeatures())
    except Exception:
        pass
    return None

with st.sidebar:
    st.header("Cấu hình")
    st.write("MODEL_PATH:", MODEL_PATH)
    st.write("MINIO_ENDPOINT:", MINIO_ENDPOINT)

    if st.button("🔄 Reload model"):
        st.cache_resource.clear()
        st.rerun()

try:
    spark = get_spark()
    model = load_model(spark)

    st.subheader("Thông tin model")
    stage_names = [type(s).__name__ for s in model.stages]
    st.write("Stages:", stage_names)

    num_features = try_get_num_features(model)
    if num_features:
        st.info(f"Model kỳ vọng **{num_features} features**.")
    else:
        st.warning("Không đọc được numFeatures. Ứng dụng sẽ cung cấp 4 input mặc định, hoặc bạn có thể dùng chế độ nâng cao để nhập trực tiếp features_list.")

    st.subheader("Nhập thông tin chuyến bay (3-4 inputs) để dự đoán giá vé")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        distance_km = st.number_input("Distance (km)", min_value=0.0, value=800.0, step=10.0, format="%.1f")
    with col2:
        days_to_departure = st.number_input("Days to departure", min_value=0, value=30, step=1)
    with col3:
        dep_hour = st.number_input("Departure hour (0-23)", min_value=0, max_value=23, value=9, step=1)
    with col4:
        stops = st.number_input("Number of stops", min_value=0, value=0, step=1)

    st.markdown("---")
    st.write("Nếu model của bạn dùng feature khác hoặc số lượng features khác, bật chế độ nâng cao để nhập `features_list` trực tiếp (comma-separated).")
    advanced = st.expander("Chế độ nâng cao: nhập features_list raw")
    with advanced:
        features_text = st.text_area("features_list (comma-separated numbers)", value="", height=120)
        st.write("Ví dụ: 800, 30, 9, 0")

    variants = st.selectbox("Số lượng kết quả dự đoán (biến thể)", options=[4,5], index=1)

    st.subheader("Kết quả dự đoán")

    if st.button("🚀 Predict"):
        if features_text and features_text.strip() != "":
            try:
                base_feats = [float(x) for x in features_text.replace("\n", ",").split(",") if x.strip() != ""]
            except Exception as ex:
                st.error(f"Không thể phân tích features_list: {ex}")
                st.stop()
        else:
            base_feats = [float(distance_km), float(days_to_departure), float(dep_hour), float(stops)]

        if num_features and len(base_feats) != num_features:
            st.warning(f"Số lượng input hiện tại = {len(base_feats)}, model cần {num_features}. Ứng dụng sẽ tự pad/truncate để phù hợp.")
            if len(base_feats) < num_features:
                base_feats = base_feats + [0.0] * (num_features - len(base_feats))
            else:
                base_feats = base_feats[:num_features]

        factors = np.linspace(0.98, 1.02, variants)
        feat_rows = []
        for f in factors:
            row = [float(x) * float(f) for x in base_feats]
            feat_rows.append(row)

        # Build Spark DataFrame from Row objects to avoid pandas internals causing .iteritems errors
        rows = [Row(features_list=fr) for fr in feat_rows]
        df = spark.createDataFrame(rows)

        to_vec = F.udf(lambda xs: Vectors.dense(xs), VectorUDT())
        df = df.withColumn("features", to_vec(F.col("features_list")))

        out = model.transform(df).select("prediction").toPandas()
        preds = out["prediction"].astype(float).tolist()

        mean_pred = float(np.mean(preds))
        std_pred = float(np.std(preds))

        confidence = 0.0
        if mean_pred != 0:
            rel_std = std_pred / (abs(mean_pred) + 1e-9)
            confidence = max(0.0, 1.0 - rel_std)
        confidence_pct = float(np.clip(confidence * 100.0, 0.0, 100.0))

        results_df = pd.DataFrame({
            "variant": list(range(1, len(preds) + 1)),
            "multiplier": [float(x) for x in factors],
            "prediction": preds
        })

        st.success(f"✅ Mean prediction: {mean_pred:,.2f}")
        st.info(f"Độ tin cậy (ước tính): {confidence_pct:.1f}% (dựa trên độ biến thiên của các biến thể)")

        st.write("Các dự đoán cho từng biến thể:" )
        st.dataframe(results_df)
        st.line_chart(results_df.set_index('variant')[['prediction']])

        st.write("Thêm thống kê:")
        st.write({
            "mean": mean_pred,
            "std": std_pred,
            "min": float(np.min(preds)),
            "max": float(np.max(preds)),
            "confidence_pct": confidence_pct
        })

except Exception as e:
    st.error(f"Lỗi khởi tạo hoặc tải model: {e}")