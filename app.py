import os
import streamlit as st
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors, VectorUDT

st.set_page_config(page_title="MinIO + Spark ML Demo", layout="wide")
st.title("Demo dự đoán bằng Spark ML model lưu trên MinIO")

def cfg(key: str, default: str = "") -> str:
    # Check Streamlit secrets first, then environment variables, then default
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
    # Optimization: Use local[1] and limited memory for Streamlit Cloud (1GB RAM limit)
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
        # Handle HTTP vs HTTPS for S3A connection
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false" if "https" not in MINIO_ENDPOINT else "true")
        .getOrCreate()
    )
    return spark

@st.cache_resource
def load_model(_spark): # Pass spark to ensure dependency is cached
    return PipelineModel.load(MODEL_PATH)

def parse_features_list(text: str):
    parts = [p.strip() for p in text.replace("\n", ",").split(",") if p.strip() != ""]
    return [float(x) for x in parts]

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
        st.warning("Không đọc được numFeatures. Bạn phải nhập đúng số feature như lúc train.")

    st.subheader("Nhập features_list để dự đoán")
    default_text = "0, 0, 0" if not num_features else ", ".join(["0"] * min(num_features, 10))
    features_text = st.text_area("features_list (comma-separated numbers)", value=default_text, height=120)

    if st.button("🚀 Predict"):
        feats = parse_features_list(features_text)

        if num_features is not None and len(feats) != num_features:
            st.error(f"Sai số lượng feature: nhập {len(feats)} nhưng model cần {num_features}.")
            st.stop()

        pdf = pd.DataFrame({"features_list": [feats]})
        df = spark.createDataFrame(pdf)

        to_vec = F.udf(lambda xs: Vectors.dense(xs), VectorUDT())
        df = df.withColumn("features", to_vec(F.col("features_list")))

        out = model.transform(df).select("prediction").toPandas()
        pred_value = float(out.loc[0, "prediction"])
        st.success(f"✅ Prediction: {pred_value:,.4f}")

except Exception as e:
    st.error(f"Lỗi khởi tạo hoặc tải model: {e}")
