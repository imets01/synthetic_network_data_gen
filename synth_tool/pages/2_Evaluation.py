import streamlit as st
import pandas as pd
import pydeck as pdk
from urllib.error import URLError

from evaluation import (
    evaluate_synthetic_data
)

# Page config
st.set_page_config(layout="wide", page_title="Synthetic Data Evaluation")
st.title("Synthetic Data Evaluation")

# =========================================================================
# Section 1: Data Upload
# =========================================================================
st.header("1. Upload Data for Evaluation")

col1, col2 = st.columns(2)

original_df = None
synthetic_df = None

with col1:
    st.subheader("Original Data")
    original_file = st.file_uploader("Upload original data CSV", type=['csv'], key="original_upload")
    if original_file is not None:
        original_df = pd.read_csv(original_file)
        st.success(f"Uploaded {len(original_df):,} original samples.")
        st.write(f"Shape: {original_df.shape}")
        st.dataframe(original_df.head(), use_container_width=True)

with col2:
    st.subheader("Synthetic Data")
    synthetic_file = st.file_uploader("Upload synthetic data CSV", type=['csv'], key="synthetic_upload")
    if synthetic_file is not None:
        synthetic_df = pd.read_csv(synthetic_file)
        st.success(f"Uploaded {len(synthetic_df):,} synthetic samples.")
        st.write(f"Shape: {synthetic_df.shape}")
        st.dataframe(synthetic_df.head(), use_container_width=True)

# Check if both datasets are loaded
if original_df is None or synthetic_df is None:
    st.info("Please upload both original and synthetic data files to continue.")
    st.stop()

# =========================================================================
# Section 2: Data Validation
# =========================================================================
st.header("2. Data Validation")

# Check if columns match
if set(original_df.columns) != set(synthetic_df.columns):
    st.error("❌ Column mismatch between original and synthetic data!")
    st.write("**Original columns:**", list(original_df.columns))
    st.write("**Synthetic columns:**", list(synthetic_df.columns))
    st.warning("Please ensure both datasets have the same columns.")
    st.stop()
else:
    st.success("✅ Column names match between datasets.")

# Check data types consistency
original_dtypes = original_df.dtypes
synthetic_dtypes = synthetic_df.dtypes

dtype_mismatches = []
for col in original_df.columns:
    if original_dtypes[col] != synthetic_dtypes[col]:
        dtype_mismatches.append((col, original_dtypes[col], synthetic_dtypes[col]))

if dtype_mismatches:
    st.warning("⚠️ Data type mismatches detected:")
    for col, orig_dtype, synth_dtype in dtype_mismatches:
        st.write(f"- **{col}**: Original ({orig_dtype}) vs Synthetic ({synth_dtype})")
    st.info("The evaluation will attempt to handle type conversions automatically.")
else:
    st.success("✅ Data types are consistent between datasets.")

# =========================================================================
# Section 3: Run Evaluation
# =========================================================================
st.header("3. Run Evaluation")

if st.button("Run Privacy & Utility Evaluation", type="primary"):
    with st.spinner("Running evaluation... This may take a few minutes."):
        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(percent):
                progress_bar.progress(percent)
                status_text.text(f"Progress: {percent}%")

            # Run evaluation
            results = evaluate_synthetic_data(original_df, synthetic_df, progress_callback)

            # Store results in session state
            st.session_state.evaluation_results = results

            progress_bar.empty()
            status_text.empty()
            st.success("Evaluation completed!")

        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            st.exception(e)

# =========================================================================
# Section 4: Display Results
# =========================================================================
if 'evaluation_results' in st.session_state:
    results = st.session_state.evaluation_results

    st.header("4. Evaluation Results")

    tab1, tab2 = st.tabs(["Privacy Metrics", "Utility Metrics"])

    with tab1:
        st.subheader("Privacy Evaluation Results")
        if 'privacy' in results:
            privacy_results = results['privacy']
            if privacy_results:
                for metric_name, metric_results in privacy_results.items():
                    with st.expander(f"📊 {metric_name}", expanded=True):
                        if isinstance(metric_results, dict):
                            for key, value in metric_results.items():
                                if isinstance(value, (int, float)):
                                    st.metric(key, f"{value:.4f}")
                                else:
                                    st.write(f"**{key}:** {value}")
                        else:
                            st.write(metric_results)
            else:
                st.info("No privacy results available.")
        else:
            st.info("No privacy evaluation results found.")

    with tab2:
        st.subheader("Utility Evaluation Results")
        if 'utility' in results:
            utility_results = results['utility']
            if utility_results:
                for metric_name, metric_results in utility_results.items():
                    with st.expander(f"📈 {metric_name}", expanded=True):
                        if isinstance(metric_results, dict):
                            for key, value in metric_results.items():
                                if isinstance(value, (int, float)):
                                    st.metric(key, f"{value:.4f}")
                                else:
                                    st.write(f"**{key}:** {value}")
                        else:
                            st.write(metric_results)
            else:
                st.info("No utility results available.")
        else:
            st.info("No utility evaluation results found.")

    # =========================================================================
    # Section 5: Export Results
    # =========================================================================
    st.header("5. Export Results")

    if st.button("Export Evaluation Results as JSON"):
        import json
        results_json = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="Download JSON",
            data=results_json,
            file_name="evaluation_results.json",
            mime="application/json"
        )

