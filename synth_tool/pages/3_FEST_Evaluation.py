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
    st.warning("⚠️ Column mismatch detected between original and synthetic data.")
    st.write("**Original columns:**", list(original_df.columns))
    st.write("**Synthetic columns:**", list(synthetic_df.columns))
    
    # Check if synthetic columns are a subset of original columns
    missing_in_original = set(synthetic_df.columns) - set(original_df.columns)
    missing_in_synthetic = set(original_df.columns) - set(synthetic_df.columns)
    
    if missing_in_original:
        st.error(f"❌ Synthetic data contains columns not in original data: {missing_in_original}")
        st.stop()
    
    if missing_in_synthetic:
        st.info(f"ℹ️ Synthetic data contains a subset of original columns. {len(missing_in_synthetic)} columns from original will be excluded from evaluation.")
        st.write(f"**Excluded columns:** {list(missing_in_synthetic)}")
        # Filter original data to match synthetic columns
        original_df = original_df[synthetic_df.columns]
        st.success(f"✅ Using {len(original_df.columns)} common columns for evaluation.")
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

# Check for missing values
original_missing = original_df.isnull().sum()
synthetic_missing = synthetic_df.isnull().sum()

if original_missing.any() or synthetic_missing.any():
    st.warning("⚠️ Missing values detected in the data:")
    col1_missing, col2_missing = st.columns(2)
    
    with col1_missing:
        if original_missing.any():
            st.write("**Original data missing values:**")
            missing_cols = original_missing[original_missing > 0]
            for col, count in missing_cols.items():
                st.write(f"- {col}: {count} ({count/len(original_df)*100:.1f}%)")
    
    with col2_missing:
        if synthetic_missing.any():
            st.write("**Synthetic data missing values:**")
            missing_cols = synthetic_missing[synthetic_missing > 0]
            for col, count in missing_cols.items():
                st.write(f"- {col}: {count} ({count/len(synthetic_df)*100:.1f}%)")
    
    st.info("ℹ️ Rows with missing values will be automatically removed during evaluation.")
else:
    st.success("✅ No missing values detected.")

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

    # Define interpretation logic based on the FEST Framework
    METRIC_GUIDE = {
        # --- Privacy Metrics ---
        "DCR": {
            "full_name": "Distance to Closest Record",
            "ideal": "Range 0.0 to infinity (Avoid 0)",
            "desc": "Distance to the nearest real record. 0.0 means a synthetic record is a clone of a real one."
        },
        "NNAdversarialAccuracy": {
            "full_name": "Nearest-Neighbor Adversarial Accuracy",
            "ideal": "Target: ~0.5",
            "desc": "0.5 = Indistinguishable from real data. < 0.5 = Overfitting. > 0.5 = Easily distinguishable (Low Utility)."
        },
        "NNDR": {
            "full_name": "Nearest Neighbor Distance Ratio",
            "ideal": "Range 0.0 - 1.0",
            "desc": "Evaluates relative distances. Lower values indicate isolated points; higher values indicate dense areas."
        },
        # "DiSCO": {
        #     "full_name": "Disclosive Mechanisms",
        #     "ideal": "Lower is Better (Target: 0.0)",
        #     "desc": "Proportion of records that reveal sensitive attributes of real individuals."
        # },
        # "repU": {
        #     "full_name": "Replicated Uniques",
        #     "ideal": "Lower is Better (Target: 0.0)",
        #     "desc": "Percentage of unique real records that were exactly replicated in the synthetic data."
        # },
        # "Singling Out Risk": {
        #     "full_name": "Singling Out Risk",
        #     "ideal": "Lower is Better",
        #     "desc": "Probability that an attacker can isolate a specific individual."
        # },
        # "Linkability Risk": {
        #     "full_name": "Linkability Risk",
        #     "ideal": "Lower is Better",
        #     "desc": "Risk of linking two separate datasets to identify an individual."
        # },
        # "Inference Risk": {
        #     "full_name": "Inference Risk",
        #     "ideal": "Lower is Better",
        #     "desc": "Risk of deducing sensitive attributes using auxiliary information."
        # },

        # --- Statistical Utility Metrics ---
        "Wasserstein": {
            "full_name": "Wasserstein Distance",
            "ideal": "Lower is Better (Target: 0.0)",
            "desc": "Cost to move synthetic distribution to match real distribution. 0 means identical."
        },
        "KS": {
            "full_name": "KS Similarity Score",
            "ideal": "Higher is Better (Max: 1.0)",
            "desc": "Similarity of cumulative distributions. 1.0 means distributions are identical."
        },
        "Pearson & Spearman Correlation": {
            "full_name": "Pearson & Spearman Correlation",
            "ideal": "Higher is Better (Max: 1.0)",
            "desc": "Measures if column correlations in synthetic data match real data."
        },
        "MI Mutual Information": {
            "full_name": "Mutual Information Score",
            "ideal": "Higher is Better (Max: 1.0)",
            "desc": "Measures if dependencies between variables are preserved."
        },
        "JS": {
            "full_name": "Jensen-Shannon Similarity",
            "ideal": "Higher is Better (Max: 1.0)",
            "desc": "Similarity of probability distributions (1 - distance)."
        },
    }

    def get_metric_display_name(metric_name):
        if metric_name in METRIC_GUIDE:
            return METRIC_GUIDE[metric_name].get('full_name', metric_name)
        for guide_key, guide_info in METRIC_GUIDE.items():
            if guide_key.lower() in metric_name.lower():
                return guide_info.get('full_name', metric_name)        
        return metric_name
    
    def display_smart_metric(key, value):
        info = METRIC_GUIDE.get(key)
        if not info:
            for guide_key, guide_info in METRIC_GUIDE.items():
                if guide_key in key:
                    info = guide_info
                    break
        
        if isinstance(value, (int, float)):
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric(label=key, value=f"{value:.4f}")
            with col_b:
                if info:
                    st.caption(f"**Goal:** {info['ideal']}")
                    st.info(f"{info['desc']}", icon="ℹ️")
                else:
                    st.caption("No specific guide available.")
        else:
            st.write(f"**{key}:** {value}")
            if info:
                 st.caption(f"*{info['desc']}*")

    tab1, tab2 = st.tabs(["Privacy Metrics", "Utility Metrics"])

    with tab1:
        st.subheader("🛡️ Privacy Evaluation")
        if 'privacy' in results and results['privacy']:
            for metric_name, metric_data in results['privacy'].items():
                display_name = get_metric_display_name(metric_name)
                with st.expander(display_name, expanded=True):
                    if isinstance(metric_data, dict):
                        for sub_key, sub_val in metric_data.items():
                            display_smart_metric(sub_key, sub_val)
                            st.divider()
                    else:
                        display_smart_metric(metric_name, metric_data)
        else:
            st.info("No privacy results available.")

    with tab2:
        st.subheader("📈 Utility Evaluation")
        if 'utility' in results and results['utility']:
            for metric_name, metric_data in results['utility'].items():
                display_name = get_metric_display_name(metric_name)
                with st.expander(display_name, expanded=True):
                    if isinstance(metric_data, dict):
                        for sub_key, sub_val in metric_data.items():
                             display_smart_metric(sub_key, sub_val)
                             st.divider()
                    else:
                        display_smart_metric(metric_name, metric_data)
        else:
            st.info("No utility results available.")

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

