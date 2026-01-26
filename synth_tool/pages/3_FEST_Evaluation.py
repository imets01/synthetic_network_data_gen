import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from urllib.error import URLError
import matplotlib.pyplot as plt

from evaluation import (
    evaluate_synthetic_data,
    run_ml_utility_evaluation,
    run_low_level_ml_utility_lstm,
    run_sequence_level_privacy_evaluation,
    IDEAL_HIGH_LEVEL_FEATURES,
    LOW_LEVEL_FEATURE_COLUMNS,
    EXCLUDED_COLUMNS
)

# Page config
st.set_page_config(layout="wide", page_title="Synthetic Data Evaluation")
st.title("Synthetic Data Evaluation")

# =========================================================================
# Section 1: Evaluation Configuration
# =========================================================================
st.header("1. Evaluation Configuration")

st.markdown("""
Select the type of evaluation based on your dataset:
- **High Level**: Uses predefined QUIC protocol features for ML utility evaluation
- **Low Level**: Uses all numeric features for ML utility evaluation
""")

# Evaluation type selection
eval_type = st.radio(
    "Select Evaluation Type",
    options=["High Level", "Low Level"],
    horizontal=True,
    help="High Level: Uses predefined QUIC protocol features. Low Level: Uses all numeric features."
)

evaluation_type = "high_level" if eval_type == "High Level" else "low_level"

# Show expected features for high-level evaluation
if evaluation_type == "high_level":
    with st.expander("Expected High-Level Features", expanded=False):
        st.write("The following features are expected for high-level ML utility evaluation:")
        for i, feature in enumerate(IDEAL_HIGH_LEVEL_FEATURES, 1):
            st.write(f"{i}. `{feature}`")
else:
    with st.expander("Expected Low-Level Features", expanded=False):
        st.write("The following features are used for low-level ML utility evaluation (LSTM-based):")
        for i, feature in enumerate(LOW_LEVEL_FEATURE_COLUMNS, 1):
            st.write(f"{i}. `{feature}`")
        st.info("Note: `capture_id` column is required for sequence grouping but excluded from features.")

# Show excluded columns info
st.info(f"The following columns are always excluded from evaluation: `{', '.join(EXCLUDED_COLUMNS)}`")

# =========================================================================
# Section 2: Data Upload
# =========================================================================
st.header("2. Upload Data for Evaluation")

original_df = None
synthetic_df = None
original_high_level_df = None
synthetic_high_level_df = None

if evaluation_type == "low_level":
    st.markdown("""
    **For low-level evaluation, you need to upload:**
    1. Low-level packet data (sequences with `capture_id`)
    2. High-level data (to get `implementation` labels for each `capture_id`)
    """)
    
    # Low-level data upload
    st.subheader("Low-Level Data (Packet Sequences)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Low-Level Data**")
        original_file = st.file_uploader("Upload original low-level CSV", type=['csv'], key="original_low_upload")
        if original_file is not None:
            original_df = pd.read_csv(original_file)
            st.success(f"Uploaded {len(original_df):,} rows.")
            st.write(f"Shape: {original_df.shape}")
            if 'capture_id' in original_df.columns:
                st.write(f"Unique captures: {original_df['capture_id'].nunique()}")
            st.dataframe(original_df.head(), use_container_width=True)
    
    with col2:
        st.markdown("**Synthetic Low-Level Data**")
        synthetic_file = st.file_uploader("Upload synthetic low-level CSV", type=['csv'], key="synthetic_low_upload")
        if synthetic_file is not None:
            synthetic_df = pd.read_csv(synthetic_file)
            st.success(f"Uploaded {len(synthetic_df):,} rows.")
            st.write(f"Shape: {synthetic_df.shape}")
            if 'capture_id' in synthetic_df.columns:
                st.write(f"Unique captures: {synthetic_df['capture_id'].nunique()}")
            st.dataframe(synthetic_df.head(), use_container_width=True)
    
    # High-level data upload for implementation labels
    st.subheader("High-Level Data (Implementation Labels)")
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**Original High-Level Data**")
        original_high_file = st.file_uploader("Upload original high-level CSV", type=['csv'], key="original_high_upload")
        if original_high_file is not None:
            original_high_level_df = pd.read_csv(original_high_file)
            st.success(f"Uploaded {len(original_high_level_df):,} samples.")
            if 'implementation' in original_high_level_df.columns:
                st.write(f"Implementations: {original_high_level_df['implementation'].value_counts().to_dict()}")
            st.dataframe(original_high_level_df.head(), use_container_width=True)
    
    with col4:
        st.markdown("**Synthetic High-Level Data**")
        synthetic_high_file = st.file_uploader("Upload synthetic high-level CSV", type=['csv'], key="synthetic_high_upload")
        if synthetic_high_file is not None:
            synthetic_high_level_df = pd.read_csv(synthetic_high_file)
            st.success(f"Uploaded {len(synthetic_high_level_df):,} samples.")
            if 'implementation' in synthetic_high_level_df.columns:
                st.write(f"Implementations: {synthetic_high_level_df['implementation'].value_counts().to_dict()}")
            st.dataframe(synthetic_high_level_df.head(), use_container_width=True)

else:
    # High-level evaluation - original upload flow
    col1, col2 = st.columns(2)
    
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

# Check if required datasets are loaded
if evaluation_type == "low_level":
    if original_df is None or synthetic_df is None:
        st.info("Please upload both original and synthetic low-level data files to continue.")
        st.stop()
    if original_high_level_df is None or synthetic_high_level_df is None:
        st.info("Please upload both original and synthetic high-level data files (for implementation labels) to continue.")
        st.stop()
else:
    if original_df is None or synthetic_df is None:
        st.info("Please upload both original and synthetic data files to continue.")
        st.stop()

# Target column selection for ML utility
st.subheader("ML Utility Configuration")

if evaluation_type == "low_level":
    # For low-level, target comes from high-level data
    possible_targets = original_high_level_df.select_dtypes(include=['object']).columns.tolist()
    if 'implementation' in possible_targets:
        default_idx = possible_targets.index('implementation')
    else:
        default_idx = 0
    
    if possible_targets:
        target_column = st.selectbox(
            "Select Target Column (from High-Level Data)",
            options=possible_targets,
            index=default_idx,
            help="The column containing implementation labels. Must be present in high-level data."
        )
    else:
        target_column = "implementation"
        st.info("Using 'implementation' as target column.")

else:
    # For high-level, target is in the main data
    possible_targets = original_df.select_dtypes(include=['object']).columns.tolist()
    if 'implementation' in possible_targets:
        default_idx = possible_targets.index('implementation')
    else:
        default_idx = 0

    if possible_targets:
        target_column = st.selectbox(
            "Select Target Column (Classification Label)",
            options=possible_targets,
            index=default_idx,
            help="The column to predict for ML utility evaluation. For QUIC data, this is typically 'implementation'."
        )
    else:
        target_column = st.text_input(
            "Enter Target Column Name",
            value="implementation",
            help="The column to predict for ML utility evaluation."
        )

# =========================================================================
# Section 3: Data Validation
# =========================================================================
st.header("3. Data Validation")

# For low-level, show implementation mapping info
if evaluation_type == "low_level":
    st.subheader("Implementation Mapping")
    
    # Get file_id column name from high-level data
    orig_id_col = 'file_id' if 'file_id' in original_high_level_df.columns else 'capture_id'
    synth_id_col = 'file_id' if 'file_id' in synthetic_high_level_df.columns else 'capture_id'
    
    # Get capture_ids from low-level data
    orig_low_captures = set(original_df['capture_id'].unique())
    synth_low_captures = set(synthetic_df['capture_id'].unique())
    
    # Get capture_ids from high-level data
    orig_high_captures = set(original_high_level_df[orig_id_col].unique())
    synth_high_captures = set(synthetic_high_level_df[synth_id_col].unique())
    
    # Find matching captures
    orig_matched = orig_low_captures & orig_high_captures
    synth_matched = synth_low_captures & synth_high_captures
    
    col1_map, col2_map = st.columns(2)
    
    with col1_map:
        st.markdown("**Original Data Mapping**")
        st.write(f"Low-level captures: {len(orig_low_captures)}")
        st.write(f"High-level entries: {len(orig_high_captures)}")
        st.write(f"Matched: {len(orig_matched)}")
        
        if len(orig_matched) < len(orig_low_captures):
            unmatched = len(orig_low_captures) - len(orig_matched)
            st.warning(f"{unmatched} captures without implementation label")
        else:
            st.success("All captures have implementation labels")
        
        # Show implementation distribution
        if target_column in original_high_level_df.columns:
            matched_impls = original_high_level_df[original_high_level_df[orig_id_col].isin(orig_matched)][target_column].value_counts()
            st.write("Implementation distribution:")
            for impl, count in matched_impls.items():
                st.write(f"  - {impl}: {count}")
    
    with col2_map:
        st.markdown("**Synthetic Data Mapping**")
        st.write(f"Low-level captures: {len(synth_low_captures)}")
        st.write(f"High-level entries: {len(synth_high_captures)}")
        st.write(f"Matched: {len(synth_matched)}")
        
        if len(synth_matched) < len(synth_low_captures):
            unmatched = len(synth_low_captures) - len(synth_matched)
            st.warning(f"{unmatched} captures without implementation label")
        else:
            st.success("All captures have implementation labels")
        
        # Show implementation distribution
        if target_column in synthetic_high_level_df.columns:
            matched_impls = synthetic_high_level_df[synthetic_high_level_df[synth_id_col].isin(synth_matched)][target_column].value_counts()
            st.write("Implementation distribution:")
            for impl, count in matched_impls.items():
                st.write(f"  - {impl}: {count}")

# Remove excluded columns for validation display
original_cols_clean = set(original_df.columns) - set(EXCLUDED_COLUMNS)
synthetic_cols_clean = set(synthetic_df.columns) - set(EXCLUDED_COLUMNS)

# Find common columns
common_columns = original_cols_clean & synthetic_cols_clean
only_in_original = original_cols_clean - synthetic_cols_clean
only_in_synthetic = synthetic_cols_clean - original_cols_clean

# Check if columns match
if only_in_original or only_in_synthetic:
    
    with st.expander("View Column Details", expanded=False):
        col1_cols, col2_cols = st.columns(2)
        
        with col1_cols:
            st.write(f"**Common columns ({len(common_columns)}):**")
            if common_columns:
                for col in sorted(common_columns):
                    st.write(f"  `{col}`")
        
        with col2_cols:
            if only_in_original:
                st.write(f"**Only in original data ({len(only_in_original)}):**")
                for col in sorted(only_in_original):
                    st.write(f"  `{col}`")
            
            if only_in_synthetic:
                st.write(f"**Only in synthetic data ({len(only_in_synthetic)}):**")
                for col in sorted(only_in_synthetic):
                    st.write(f"  `{col}`")
    
    st.success(f"Evaluation will use **{len(common_columns)} common columns** (excluding ID columns).")
else:
    st.success(f"Column names match between datasets ({len(common_columns)} columns, excluding IDs).")

# Check data types consistency (only for common columns)
common_cols_list = list(common_columns)
if common_cols_list:
    original_dtypes = original_df[common_cols_list].dtypes
    synthetic_dtypes = synthetic_df[common_cols_list].dtypes

    dtype_mismatches = []
    for col in common_cols_list:
        if original_dtypes[col] != synthetic_dtypes[col]:
            dtype_mismatches.append((col, original_dtypes[col], synthetic_dtypes[col]))

    if dtype_mismatches:
        st.warning("Data type mismatches detected:")
        for col, orig_dtype, synth_dtype in dtype_mismatches:
            st.write(f"- **{col}**: Original ({orig_dtype}) vs Synthetic ({synth_dtype})")
        st.info("The evaluation will attempt to handle type conversions automatically.")
    else:
        st.success("Data types are consistent between datasets.")

# Check for missing values
original_missing = original_df.isnull().sum()
synthetic_missing = synthetic_df.isnull().sum()

if original_missing.any() or synthetic_missing.any():
    st.warning("Missing values detected in the data:")
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
    
    st.info("Rows with missing values will be automatically removed during evaluation.")
else:
    st.success("No missing values detected.")

# =========================================================================
# Section 4: Run Evaluation
# =========================================================================
st.header("4. Run Evaluation")

if st.button("Run Privacy & Utility Evaluation", type="primary"):
    with st.spinner("Running evaluation... This may take a few minutes."):
        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            if evaluation_type == "low_level":
                # For low-level data, run sequence-level privacy evaluation
                # (aggregates sequences, then computes privacy metrics)
                st.info("For low-level packet data, privacy is evaluated at the **sequence level** "
                       "(each capture is aggregated into statistical features).")
                
                progress_bar.progress(5)
                status_text.text("Running sequence-level privacy evaluation...")
                
                # Run sequence-level privacy
                privacy_results = run_sequence_level_privacy_evaluation(
                    original_low_df=original_df,
                    synthetic_low_df=synthetic_df,
                    feature_columns=LOW_LEVEL_FEATURE_COLUMNS
                )
                
                if 'error' in privacy_results:
                    st.warning(f"Privacy evaluation issue: {privacy_results['error']}")
                    results = {'privacy': {}, 'utility': {}}
                else:
                    # Extract metadata and remove from results display
                    metadata = privacy_results.pop('_metadata', {})
                    results = {'privacy': privacy_results, 'utility': {}}
                    st.caption(f"Evaluated {metadata.get('num_original_sequences', 'N/A')} original and "
                              f"{metadata.get('num_synthetic_sequences', 'N/A')} synthetic sequences")
                
                progress_bar.progress(30)
                status_text.text("Running LSTM ML Utility evaluation...")
                
                def lstm_progress_callback(epoch, total_epochs, train_loss, train_acc, val_loss, val_acc):
                    if total_epochs > 0:
                        percent = 30 + int((epoch / total_epochs) * 70)
                    else:
                        percent = 30
                    progress_bar.progress(min(percent, 100))
                    status_text.text(f"LSTM Training: Epoch {epoch}/{total_epochs} - Val Acc: {val_acc:.4f}")
                
                ml_results = run_low_level_ml_utility_lstm(
                    synthetic_low_df=synthetic_df, 
                    real_low_df=original_df,
                    synthetic_high_df=synthetic_high_level_df,
                    real_high_df=original_high_level_df,
                    target_column=target_column,
                    epochs=5,
                    batch_size=32,
                    progress_callback=lstm_progress_callback
                )
            else:
                # High-level evaluation - run full privacy/statistical + ML utility
                def progress_callback(percent):
                    # Scale to 0-66% for privacy/utility evaluation
                    scaled_percent = int(percent * 0.66)
                    progress_bar.progress(scaled_percent)
                    status_text.text(f"Privacy & Statistical Utility: {percent}%")

                # Run privacy and statistical utility evaluation
                status_text.text("Running Privacy & Statistical Utility evaluation...")
                results = evaluate_synthetic_data(original_df, synthetic_df, progress_callback)

                # Run ML utility evaluation
                progress_bar.progress(70)
                status_text.text("Running ML Utility evaluation...")
                
                ml_results = run_ml_utility_evaluation(
                    synthetic_df, 
                    original_df, 
                    evaluation_type=evaluation_type,
                    target_column=target_column
                )
            
            progress_bar.progress(100)

            # Store results in session state
            st.session_state.evaluation_results = results
            st.session_state.ml_utility_results = ml_results

            progress_bar.empty()
            status_text.empty()
            st.success("Evaluation completed!")

        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            st.exception(e)



# =========================================================================
# Section 5: Display Results
# =========================================================================
if 'evaluation_results' in st.session_state:
    results = st.session_state.evaluation_results

    st.header("5. Evaluation Results")

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
                    st.info(f"{info['desc']}")
                else:
                    st.caption("No specific guide available.")
        else:
            st.write(f"**{key}:** {value}")
            if info:
                 st.caption(f"*{info['desc']}*")

    # Create tabs including ML Utility
    tab1, tab2, tab3 = st.tabs(["Privacy Metrics", "Statistical Utility", "ML Utility"])

    with tab1:
        st.subheader("Privacy Evaluation")
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
        st.subheader("Statistical Utility Evaluation")
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

    with tab3:
        st.subheader("ML Utility Evaluation")
        
        if 'ml_utility_results' in st.session_state:
            ml_results = st.session_state.ml_utility_results
            
            if 'error' in ml_results:
                st.error(f"Evaluation Error: {ml_results['error']}")
                if 'available_columns' in ml_results:
                    st.write("Available columns:", ml_results['available_columns'])
            else:
                # Check if this is LSTM-based evaluation
                is_lstm = ml_results.get('evaluation_type') == 'low_level_lstm'
                
                if is_lstm:
                    st.markdown("""
                    This evaluates ML utility using a **Bidirectional LSTM with Attention** trained on synthetic 
                    sequence data and tested on real data (TSTR - Train Synthetic Test Real).
                    High accuracy indicates synthetic data preserves sequential patterns needed for ML tasks.
                    """)
                else:
                    st.markdown("""
                    This evaluates Machine Learning utility by training a Random Forest classifier on synthetic data 
                    and testing on real data. High accuracy indicates synthetic data preserves patterns needed for ML tasks.
                    """)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Accuracy",
                        value=f"{ml_results['accuracy']:.4f}",
                        help="Classification accuracy on real data using model trained on synthetic data"
                    )
                
                with col2:
                    if is_lstm:
                        st.metric(
                            label="Training Sequences",
                            value=f"{ml_results.get('num_training_sequences', 'N/A'):,}",
                            help="Number of synthetic sequences used for training"
                        )
                    else:
                        st.metric(
                            label="Training Samples",
                            value=f"{ml_results['num_training_samples']:,}",
                            help="Number of synthetic samples used for training"
                        )
                
                with col3:
                    if is_lstm:
                        st.metric(
                            label="Test Sequences",
                            value=f"{ml_results.get('num_test_sequences', 'N/A'):,}",
                            help="Number of real sequences used for testing"
                        )
                    else:
                        st.metric(
                            label="Test Samples",
                            value=f"{ml_results['num_test_samples']:,}",
                            help="Number of real samples used for testing"
                        )
                
                # Show device info for LSTM
                if is_lstm:
                    st.caption(f"Device: {ml_results.get('device', 'N/A')}")
                
                # Training history plot for LSTM
                if is_lstm and 'training_history' in ml_results:
                    with st.expander("Training History", expanded=True):
                        history = ml_results['training_history']
                        
                        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
                        
                        # Loss plot
                        axes[0].plot(history['train_loss'], label='Train Loss', marker='o', markersize=4)
                        axes[0].plot(history['val_loss'], label='Val Loss', marker='o', markersize=4)
                        axes[0].set_xlabel('Epoch')
                        axes[0].set_ylabel('Loss')
                        axes[0].set_title('Training and Validation Loss')
                        axes[0].legend()
                        axes[0].grid(True, alpha=0.3)
                        
                        # Accuracy plot
                        axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o', markersize=4)
                        axes[1].plot(history['val_acc'], label='Val Accuracy', marker='o', markersize=4)
                        axes[1].set_xlabel('Epoch')
                        axes[1].set_ylabel('Accuracy')
                        axes[1].set_title('Training and Validation Accuracy')
                        axes[1].legend()
                        axes[1].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                
                # Detailed results in expanders
                with st.expander("Classification Report", expanded=True):
                    report = ml_results['classification_report']
                    
                    # Convert to DataFrame for display
                    report_df_data = []
                    for class_name, metrics in report.items():
                        if isinstance(metrics, dict):
                            report_df_data.append({
                                'Class': class_name,
                                'Precision': metrics.get('precision', 0),
                                'Recall': metrics.get('recall', 0),
                                'F1-Score': metrics.get('f1-score', 0),
                                'Support': metrics.get('support', 0)
                            })
                    
                    if report_df_data:
                        report_df = pd.DataFrame(report_df_data)
                        st.dataframe(report_df, use_container_width=True)
                
                with st.expander("Confusion Matrix", expanded=True):
                    cm = np.array(ml_results['confusion_matrix'])
                    class_labels = ml_results['class_labels']
                    
                    # Create confusion matrix plot
                    fig, ax = plt.subplots(figsize=(5, 4))
                    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                    ax.figure.colorbar(im, ax=ax)
                    
                    # Set ticks and labels
                    ax.set(xticks=np.arange(cm.shape[1]),
                           yticks=np.arange(cm.shape[0]),
                           xticklabels=class_labels,
                           yticklabels=class_labels,
                           ylabel='True label',
                           xlabel='Predicted label')
                    
                    # Rotate x labels for better readability
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                    
                    # Add text annotations
                    thresh = cm.max() / 2.
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax.text(j, i, format(cm[i, j], 'd'),
                                    ha="center", va="center",
                                    color="white" if cm[i, j] > thresh else "black")
                    
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with st.expander("Feature Importance", expanded=True):
                    if 'feature_importance' in ml_results:
                        st.markdown("Features most important for identifying different implementations:")
                        
                        # Sort features by importance
                        importance_dict = ml_results['feature_importance']
                        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                        
                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(5, 3))
                        features = [item[0] for item in sorted_importance]
                        importances = [item[1] for item in sorted_importance]
                        
                        y_pos = np.arange(len(features))
                        ax.barh(y_pos, importances, color='skyblue', align='center')
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(features)
                        ax.invert_yaxis()  # Top feature at top
                        ax.set_xlabel('Importance Score')
                        ax.set_title('Feature Importance for Classification')
                        ax.grid(axis='x', linestyle='--', alpha=0.6)
                        
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Also show as table
                        importance_df = pd.DataFrame(sorted_importance, columns=['Feature', 'Importance'])
                        importance_df['Importance'] = importance_df['Importance'].apply(lambda x: f"{x:.4f}")
                        st.dataframe(importance_df, use_container_width=True)
                    else:
                        # LSTM doesn't have direct feature importance
                        st.info("Feature importance is not available for LSTM-based evaluation. "
                               "LSTM models learn complex sequential patterns that don't translate to individual feature importance scores.")
                        st.write("**Features used for sequence classification:**")
                        for f in ml_results.get('features_used', []):
                            st.write(f"- `{f}`")
                
                # Show features info
                with st.expander("Features Information", expanded=False):
                    eval_type_display = ml_results.get('evaluation_type', 'unknown').replace('_', ' ').title()
                    st.write(f"**Evaluation Type:** {eval_type_display}")
                    
                    features_used = ml_results.get('features_used', [])
                    st.write(f"**Features Used ({len(features_used)}):**")
                    for f in features_used:
                        st.write(f"- `{f}`")
                    
                    missing_features = ml_results.get('missing_features', [])
                    if missing_features:
                        st.write(f"\n**Missing Features ({len(missing_features)}):**")
                        for f in missing_features:
                            st.write(f"- `{f}`")
        else:
            st.info("No ML utility results available.")

    # =========================================================================
    # Section 6: Export Results
    # =========================================================================
    st.header("6. Export Results")

    if st.button("Export All Results as JSON"):
        import json
        # Combine all results
        all_results = {
            'privacy': results.get('privacy', {}),
            'statistical_utility': results.get('utility', {}),
            'ml_utility': st.session_state.get('ml_utility_results', {})
        }
        results_json = json.dumps(all_results, indent=2, default=str)
        st.download_button(
            label="Download JSON",
            data=results_json,
            file_name="evaluation_results.json",
            mime="application/json"
        )

