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
)

# -------------------------------------------------
# Setup
# -------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Apollo Company Enrichment",
    layout="wide",
)

st.title("🔍 Apollo Company Enrichment")
st.caption(
    "Upload → select website column → select row range → enrich "
    "→ append contact data → download"
)

UPLOAD_DIR = "data/uploads"
OUTPUT_DIR = "data/outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def latest_checkpoint_file() -> str | None:
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint_*.json")))
    return files[-1] if files else None


def latest_output_csv() -> str | None:
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "enriched_*.csv")))
    return files[-1] if files else None


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ Apollo Info")

    api_key = os.getenv("APOLLO_API_KEY")
    if api_key:
        st.success("Apollo API key loaded")
    else:
        st.error("APOLLO_API_KEY not found in .env")

    st.markdown("---")

    st.info(
        "ℹ️ **Credits notice**\n\n"
        "Apollo credit balance is not reliably exposed via API.\n\n"
        "Check remaining credits in:\n"
        "**Apollo → Settings → Billing → Usage**"
    )

    st.markdown("---")

    ckpt = latest_checkpoint_file()
    if ckpt:
        st.caption("Latest checkpoint found:")
        st.code(ckpt)

# -------------------------------------------------
# Step 1 — Upload file
# -------------------------------------------------
st.header("1️⃣ Upload file")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"],
)

if not uploaded_file:
    st.stop()

timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
upload_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{uploaded_file.name}")

with open(upload_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

st.success(f"File saved on server at `{upload_path}`")

# Load dataframe
if uploaded_file.name.endswith(".csv"):
    original_df = pd.read_csv(uploaded_file)
else:
    original_df = pd.read_excel(uploaded_file)

st.subheader("Preview (first 10 rows)")
st.dataframe(original_df.head(10), use_container_width=True)

total_rows = len(original_df)
st.caption(f"Total rows in file: **{total_rows}**")

# -------------------------------------------------
# Step 2 — Select website column
# -------------------------------------------------
st.header("2️⃣ Select website column")

website_col = st.selectbox(
    "Which column contains company websites?",
    original_df.columns.tolist(),
)

# -------------------------------------------------
# Step 3 — Select row range
# -------------------------------------------------
st.header("3️⃣ Select row range")

col1, col2 = st.columns(2)

with col1:
    start_row = st.number_input(
        "Start row (1-based, inclusive)",
        min_value=1,
        max_value=total_rows,
        value=1,
        step=1,
    )

with col2:
    end_row = st.number_input(
        "End row (1-based, inclusive)",
        min_value=1,
        max_value=total_rows,
        value=total_rows,
        step=1,
    )

if start_row > end_row:
    st.error("Start row must be less than or equal to end row.")
    st.stop()

rows_to_process = end_row - start_row + 1

st.caption(
    f"🔢 This run will enrich **rows {start_row} → {end_row}** "
    f"(**{rows_to_process} companies**)"
)

# -------------------------------------------------
# Step 4 — Run / Resume
# -------------------------------------------------
st.header("4️⃣ Run enrichment")

run_clicked = st.button("🚀 Start enrichment", type="primary")
resume_clicked = st.button("▶️ Resume last run from checkpoint")

progress_bar = st.progress(0)
status_text = st.empty()

def progress_cb(pct, msg):
    progress_bar.progress(pct)
    status_text.info(msg)

checkpoint_path = os.path.join(
    OUTPUT_DIR, f"checkpoint_{timestamp}.json"
)

output_path = os.path.join(
    OUTPUT_DIR,
    f"enriched_{timestamp}_rows_{start_row}_{end_row}.csv",
)

enrichment_df = None
merged_df = None

# -------------------------------------------------
# Resume logic
# -------------------------------------------------
if resume_clicked:
    ckpt_file = latest_checkpoint_file()
    if not ckpt_file:
        st.warning("No checkpoint found to resume.")
        st.stop()

    with open(ckpt_file, "r", encoding="utf-8") as f:
        ckpt = json.load(f)

    resume_from = int(ckpt.get("last_completed_row", 0)) + 1
    resume_end = int(ckpt.get("end_row", end_row))

    if resume_from > resume_end:
        st.success("Checkpoint indicates the run is already complete.")
        st.stop()

    st.info(
        f"Resuming from row **{resume_from} → {resume_end}** "
        f"using checkpoint `{ckpt_file}`"
    )

    with st.spinner("Resuming enrichment…"):
        enrichment_df = enrich_dataframe(
            df=original_df,
            website_column=ckpt.get("website_column", website_col),
            start_row=resume_from,
            end_row=resume_end,
            progress_cb=progress_cb,
            checkpoint_path=checkpoint_path,
        )

# -------------------------------------------------
# Fresh run
# -------------------------------------------------
if run_clicked:
    with st.spinner("Enriching companies…"):
        enrichment_df = enrich_dataframe(
            df=original_df,
            website_column=website_col,
            start_row=start_row,
            end_row=end_row,
            progress_cb=progress_cb,
            checkpoint_path=checkpoint_path,
        )

# -------------------------------------------------
# Post-processing (merge + save)
# -------------------------------------------------
if enrichment_df is not None:
    progress_bar.progress(1.0)
    status_text.success("Enrichment completed")

    merged_df = append_enrichment_to_original_df(
        original_df,
        enrichment_df,
    )

    merged_df.to_csv(output_path, index=False)

    st.success(
        "✅ Enrichment appended to original file and saved at:\n"
        f"`{output_path}`"
    )

    st.session_state["last_output_df"] = merged_df
    st.session_state["last_enrichment_df"] = enrichment_df
    st.session_state["last_output_path"] = output_path

# -------------------------------------------------
# Step 5 — Results + Retry failed only
# -------------------------------------------------
st.header("5️⃣ Results")

last_df = st.session_state.get("last_output_df")
last_path = st.session_state.get("last_output_path")

if last_df is None:
    latest_csv = latest_output_csv()
    if latest_csv:
        try:
            last_df = pd.read_csv(latest_csv)
            last_path = latest_csv
            st.session_state["last_output_df"] = last_df
            st.session_state["last_output_path"] = last_path
        except Exception:
            last_df = None

if last_df is not None:
    st.success(f"Loaded results from `{last_path}`")
    st.dataframe(last_df, use_container_width=True)

    # Download
    st.subheader("Download merged file")

    default_name = os.path.basename(last_path).replace(".csv", "")
    custom_name = st.text_input(
        "File name (without extension)",
        value=default_name,
    )

    st.download_button(
        "⬇️ Download CSV",
        data=last_df.to_csv(index=False),
        file_name=f"{custom_name}.csv",
        mime="text/csv",
    )

    st.info(
        "📁 **Server-side storage**\n\n"
        f"The merged file is stored on the machine running this app at:\n"
        f"`{last_path}`\n\n"
        "If this app runs remotely, download the file to keep a local copy."
    )

    # Retry failed only
    st.subheader("Retry failed only")

    retry_mask = last_df["apollo_status"].isin(
        ["error", "no_enrich", "no_match"]
    )

    failed_count = int(retry_mask.sum())

    st.caption(f"Rows eligible for retry: **{failed_count}**")

    if failed_count > 0:
        retry_clicked = st.button("🔁 Retry failed rows only")

        if retry_clicked:
            retry_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            retry_checkpoint = os.path.join(
                OUTPUT_DIR, f"checkpoint_retry_{retry_timestamp}.json"
            )
            retry_output = os.path.join(
                OUTPUT_DIR, f"enriched_retry_{retry_timestamp}.csv"
            )

            progress_bar = st.progress(0)
            status_text = st.empty()

            def retry_progress(pct, msg):
                progress_bar.progress(pct)
                status_text.info(msg)

            with st.spinner("Retrying failed rows…"):
                retried_df = retry_failed_only(
                    df_original=original_df,
                    website_column=website_col,
                    previous_output_df=last_df,
                    progress_cb=retry_progress,
                    checkpoint_path=retry_checkpoint,
                )

            merged_retry_df = append_enrichment_to_original_df(
                original_df,
                retried_df,
            )

            merged_retry_df.to_csv(retry_output, index=False)

            progress_bar.progress(1.0)
            status_text.success("Retry completed")

            st.success(
                "🔁 Retry results merged and saved at:\n"
                f"`{retry_output}`"
            )

            st.session_state["last_output_df"] = merged_retry_df
            st.session_state["last_output_path"] = retry_output

else:
    st.warning("No results available yet. Run enrichment first.")