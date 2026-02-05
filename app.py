import os
import glob
import json
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.pipeline import (
    enrich_dataframe,
    retry_failed_only,
    append_enrichment_to_original_df,
    merge_phone_events_into_original_df,
)

# -------------------------------------------------
# Setup
# -------------------------------------------------
load_dotenv()

UPLOAD_DIR = "data/uploads"
OUTPUT_DIR = "data/outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def phone_progress(df: pd.DataFrame) -> tuple[int, int]:
    if "apollo_person_id" not in df.columns:
        return 0, 0

    total = int(df["apollo_person_id"].notna().sum())
    received = int(df["apollo_phone_number"].notna().sum()) if "apollo_phone_number" in df.columns else 0
    return received, total


def latest_checkpoint_file() -> str | None:
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint_*.json")))
    return files[-1] if files else None


def latest_output_csv() -> str | None:
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "enriched_*.csv")))
    return files[-1] if files else None


st.set_page_config(page_title="Apollo Company Enrichment", layout="wide")
st.title("🔍 Apollo Company Enrichment")
st.caption("Upload → select website column → select row range → enrich → download")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ Apollo Info")

    if os.getenv("APOLLO_API_KEY"):
        st.success("Apollo API key loaded")
    else:
        st.error("APOLLO_API_KEY not found in .env")

    st.markdown("---")
    st.info(
        "ℹ️ **Credits notice**\n\n"
        "Phone number enrichment consumes credits.\n"
        "Check remaining credits in Apollo dashboard."
    )

    st.markdown("---")
    debug_mode = st.checkbox("🧪 Debug mode (prints to terminal)", value=False)

# -------------------------------------------------
# Step 1 — Upload
# -------------------------------------------------
st.header("1️⃣ Upload file")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
if not uploaded_file:
    st.stop()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
upload_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{uploaded_file.name}")

with open(upload_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

original_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

st.subheader("Preview (first 10 rows)")
st.dataframe(original_df.head(10), width="stretch")

total_rows = len(original_df)
st.caption(f"Total rows: **{total_rows}**")

# -------------------------------------------------
# Step 2 — Website column
# -------------------------------------------------
st.header("2️⃣ Select website column")
website_col = st.selectbox("Which column contains company websites?", original_df.columns.tolist())

# -------------------------------------------------
# Step 3 — Row range
# -------------------------------------------------
st.header("3️⃣ Select row range")
start_row = st.number_input("Start row (1-based)", 1, total_rows, 1)
end_row = st.number_input("End row (1-based)", 1, total_rows, total_rows)

if start_row > end_row:
    st.error("Start row must be ≤ end row.")
    st.stop()

# -------------------------------------------------
# Step 3.5 — Phones
# -------------------------------------------------
st.header("3️⃣➕ Phone numbers (optional)")

reveal_phone = st.checkbox("Reveal phone numbers (async, costs credits)", value=False)

webhook_url = None
if reveal_phone:
    st.warning(
        "📞 Phone enrichment is **asynchronous** and consumes credits.\n\n"
        "Webhook + ngrok must be running."
    )
    webhook_url = st.text_input(
        "Webhook URL",
        placeholder="https://xxxx.ngrok.io/apollo/phone-webhook",
    )
    if not webhook_url:
        st.stop()

# -------------------------------------------------
# Step 4 — Run / Resume / Retry
# -------------------------------------------------
st.header("4️⃣ Run enrichment")

colA, colB, colC = st.columns(3)
with colA:
    run_clicked = st.button("🚀 Start enrichment", type="primary")
with colB:
    resume_clicked = st.button("▶️ Resume last run")
with colC:
    retry_clicked = st.button("🔁 Retry failed only")

progress_bar = st.progress(0)
status_text = st.empty()


def progress_cb(pct, msg):
    progress_bar.progress(pct)
    status_text.info(msg)


checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint_{timestamp}.json")
output_path = os.path.join(OUTPUT_DIR, f"enriched_{timestamp}_rows_{start_row}_{end_row}.csv")

enrichment_df = None

# ---- Resume
if resume_clicked:
    ckpt_file = latest_checkpoint_file()
    if not ckpt_file:
        st.warning("No checkpoint found.")
        st.stop()

    with open(ckpt_file, "r") as f:
        ckpt = json.load(f)

    resume_from = int(ckpt.get("last_completed_row", 0)) + 1
    resume_end = int(ckpt.get("end_row", end_row))
    resume_col = ckpt.get("website_column", website_col)

    enrichment_df = enrich_dataframe(
        df=original_df,
        website_column=resume_col,
        start_row=resume_from,
        end_row=resume_end,
        progress_cb=progress_cb,
        checkpoint_path=checkpoint_path,
        reveal_phone_number=reveal_phone,
        webhook_url=webhook_url,
        debug=debug_mode,
    )

# ---- Fresh run
if run_clicked:
    enrichment_df = enrich_dataframe(
        df=original_df,
        website_column=website_col,
        start_row=start_row,
        end_row=end_row,
        progress_cb=progress_cb,
        checkpoint_path=checkpoint_path,
        reveal_phone_number=reveal_phone,
        webhook_url=webhook_url,
        debug=debug_mode,
    )

# ---- Retry failed
if retry_clicked:
    latest_csv = latest_output_csv()
    if not latest_csv:
        st.warning("No previous output CSV found to retry from.")
        st.stop()

    prev = pd.read_csv(latest_csv)
    retry_df = retry_failed_only(
        df_original=original_df,
        website_column=website_col,
        previous_output_df=prev,
        progress_cb=progress_cb,
        checkpoint_path=checkpoint_path,
        reveal_phone_number=reveal_phone,
        webhook_url=webhook_url,
        debug=debug_mode,
    )
    enrichment_df = retry_df

# -------------------------------------------------
# Post-processing
# -------------------------------------------------
if enrichment_df is not None and not enrichment_df.empty:
    merged_df = append_enrichment_to_original_df(original_df, enrichment_df)
    merged_df.to_csv(output_path, index=False)

    st.session_state["last_output_df"] = merged_df
    st.session_state["last_output_path"] = output_path

# -------------------------------------------------
# Step 5 — Results
# -------------------------------------------------
st.header("5️⃣ Results")

last_df = st.session_state.get("last_output_df")
last_path = st.session_state.get("last_output_path")

if last_df is None:
    latest_csv = latest_output_csv()
    if latest_csv:
        last_df = pd.read_csv(latest_csv)
        last_path = latest_csv
        st.session_state["last_output_df"] = last_df
        st.session_state["last_output_path"] = last_path

if last_df is None:
    st.warning("No results available yet.")
    st.stop()

st.success(f"Loaded results from `{last_path}`")

# Phone progress widget
received, total = phone_progress(last_df)
if reveal_phone:
    st.info(f"📞 Phone progress: **{received}/{total}** received")

st.info(
    "📞 **Phone numbers arrive asynchronously**\n\n"
    "If you enabled phone enrichment, numbers may appear minutes later.\n"
    "Click **Refresh phone numbers** to merge newly delivered phones into the file."
)

if st.button("🔄 Refresh phone numbers"):
    refreshed_df = merge_phone_events_into_original_df(last_df)
    refreshed_df.to_csv(last_path, index=False)
    st.session_state["last_output_df"] = refreshed_df
    last_df = refreshed_df
    st.success("Phone numbers merged from webhook events.")

st.dataframe(last_df, width="stretch")

st.subheader("Download merged file")

default_name = os.path.basename(last_path).replace(".csv", "")
custom_name = st.text_input("File name (without extension)", value=default_name)

st.download_button(
    "⬇️ Download CSV",
    data=last_df.to_csv(index=False),
    file_name=f"{custom_name}.csv",
    mime="text/csv",
)

st.info(
    "📁 **Server-side storage**\n\n"
    f"The file is stored on the server at:\n"
    f"`{last_path}`\n\n"
    "Download it to keep a local copy."
)
