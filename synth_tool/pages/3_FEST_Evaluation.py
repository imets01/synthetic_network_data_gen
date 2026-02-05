import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from evaluation import (
    evaluate_synthetic_data,
    run_ml_utility_evaluation,
    run_low_level_ml_utility_lstm,
    run_sequence_level_privacy_evaluation,
    run_low_level_statistical_similarity,
    run_low_level_structural_similarity,
    IDEAL_HIGH_LEVEL_FEATURES,
    LOW_LEVEL_FEATURE_COLUMNS,
    EXCLUDED_COLUMNS
)

st.set_page_config(layout="wide", page_title="Synthetic Data Evaluation")
st.title("Synthetic Data Evaluation")

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
        st.write("The following features are used for low-level statistical and structural similarity evaluation:")
        for i, feature in enumerate(LOW_LEVEL_FEATURE_COLUMNS, 1):
            st.write(f"{i}. `{feature}`")
        st.info("Note: `capture_id` column is required for sequence grouping but excluded from features.")

# Show excluded columns info
st.info(f"The following columns are always excluded from evaluation: `{', '.join(EXCLUDED_COLUMNS)}`")

st.header("2. Upload Data for Evaluation")

original_df = None
synthetic_df = None

if evaluation_type == "low_level":
    st.markdown("""
    **For low-level evaluation, upload:**
    - Low-level packet data (sequences with `capture_id`)
    
    The evaluation includes:
    - **Privacy**: Sequence-level privacy metrics (DCR, NNDR, etc.)
    - **Statistical Similarity**: CDF plots comparing distributions of each feature
    - **Structural Similarity**: t-SNE visualization of real vs synthetic sequences
    """)
    
    # Low-level data upload
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
else:
    if original_df is None or synthetic_df is None:
        st.info("Please upload both original and synthetic data files to continue.")
        st.stop()

# Target column selection for ML utility (only for high-level)
if evaluation_type == "high_level":
    st.subheader("ML Utility Configuration")
    
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
else:
    target_column = None

st.header("3. Data Validation")

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
                status_text.text("Computing statistical similarity (CDF)...")
                
                # Run statistical similarity (CDF)
                stat_results = run_low_level_statistical_similarity(
                    original_df, 
                    synthetic_df, 
                    feature_columns=LOW_LEVEL_FEATURE_COLUMNS
                )
                
                progress_bar.progress(60)
                status_text.text("Computing structural similarity (t-SNE)...")
                
                # Store data for t-SNE (will be computed on-demand with user-selected sample size)
                # Store results
                ml_results = {
                    'statistical_similarity': stat_results,
                    'structural_similarity': {},  # Will be computed on-demand
                    'evaluation_type': 'low_level_visual'
                }
                
                # Store dataframes for on-demand t-SNE computation
                st.session_state.original_low_df = original_df
                st.session_state.synthetic_low_df = synthetic_df
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

if 'evaluation_results' in st.session_state:
    results = st.session_state.evaluation_results

    st.header("5. Evaluation Results")

    METRIC_GUIDE = {
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

    # Create tabs based on evaluation type
    if evaluation_type == "low_level":
        tab1, tab2, tab3 = st.tabs(["Privacy Metrics", "Statistical Similarity (CDF)", "Structural Similarity (t-SNE)"])
    else:
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
        
        # Check if this is low-level visual evaluation
        if 'ml_utility_results' in st.session_state and st.session_state.ml_utility_results.get('evaluation_type') == 'low_level_visual':
            # Show CDF plots for low-level
            ml_results = st.session_state.ml_utility_results
            stat_results = ml_results.get('statistical_similarity', {})
            
            if 'error' in stat_results:
                st.error(f"Statistical Similarity Error: {stat_results['error']}")
            else:
                st.markdown("""
                **Cumulative Distribution Functions (CDF)** compare the distribution of each feature between 
                real and synthetic data. Similar CDF curves indicate that the synthetic data preserves the 
                statistical properties of the original data.
                """)
                
                cdf_data = stat_results.get('cdf_data', {})
                ks_stats = stat_results.get('ks_statistics', {})
                features = stat_results.get('features_analyzed', [])
                
                if cdf_data:
                    # Show KS statistics summary
                    with st.expander("KS Test Summary", expanded=True):
                        ks_summary = []
                        for feat in features:
                            if feat in ks_stats and 'statistic' in ks_stats[feat]:
                                ks_summary.append({
                                    'Feature': feat,
                                    'KS Statistic': f"{ks_stats[feat]['statistic']:.4f}",
                                    'Similarity': f"{ks_stats[feat]['similarity']:.4f}",
                                    'p-value': f"{ks_stats[feat]['p_value']:.4f}"
                                })
                        if ks_summary:
                            ks_df = pd.DataFrame(ks_summary)
                            st.dataframe(ks_df, use_container_width=True)
                            
                            # Average similarity score
                            avg_similarity = np.mean([ks_stats[f]['similarity'] for f in features if f in ks_stats and 'similarity' in ks_stats[f]])
                            st.metric("Average KS Similarity", f"{avg_similarity:.4f}", 
                                     help="1.0 = identical distributions, 0.0 = completely different")
                    
                    # CDF Plots
                    st.subheader("CDF Plots by Feature")
                    
                    # Create grid of CDF plots
                    num_features = len(features)
                    cols_per_row = 3
                    
                    for i in range(0, num_features, cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            feat_idx = i + j
                            if feat_idx < num_features:
                                feat = features[feat_idx]
                                if feat in cdf_data:
                                    with col:
                                        fig, ax = plt.subplots(figsize=(4, 3))
                                        
                                        data = cdf_data[feat]
                                        ax.plot(data['real_values'], data['real_cdf'], 
                                               label='Real', color='blue', alpha=0.7)
                                        ax.plot(data['synthetic_values'], data['synthetic_cdf'], 
                                               label='Synthetic', color='red', alpha=0.7)
                                        
                                        ax.set_xlabel('Value')
                                        ax.set_ylabel('CDF')
                                        ax.set_title(feat, fontsize=10)
                                        ax.legend(fontsize=8)
                                        ax.grid(True, alpha=0.3)
                                        
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close()
                else:
                    st.info("No CDF data available.")
        elif 'utility' in results and results['utility']:
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
        # Check if this is low-level visual evaluation (t-SNE)
        if 'ml_utility_results' in st.session_state and st.session_state.ml_utility_results.get('evaluation_type') == 'low_level_visual':
            st.subheader("Structural Similarity (t-SNE)")
            
            st.markdown("""
            **t-SNE (t-Distributed Stochastic Neighbor Embedding)** projects high-dimensional sequence data 
            into 2D for visualization. If synthetic data overlaps well with real data, it indicates 
            that the model has learned the underlying structure of the sequences.
            """)
            
            # Check if we have the dataframes stored
            if 'original_low_df' in st.session_state and 'synthetic_low_df' in st.session_state:
                orig_df = st.session_state.original_low_df
                synth_df = st.session_state.synthetic_low_df
                
                # Get total sequence counts
                total_real = orig_df['capture_id'].nunique() if 'capture_id' in orig_df.columns else len(orig_df)
                total_synth = synth_df['capture_id'].nunique() if 'capture_id' in synth_df.columns else len(synth_df)
                
                # Compute t-SNE if not already cached
                if 'tsne_results_all' not in st.session_state:
                    with st.spinner(f"Computing t-SNE on all {total_real + total_synth:,} sequences..."):
                        struct_results = run_low_level_structural_similarity(
                            orig_df, 
                            synth_df, 
                            feature_columns=LOW_LEVEL_FEATURE_COLUMNS,
                            max_sequences=None,  # Use all sequences
                            perplexity=30
                        )
                        st.session_state.tsne_results_all = struct_results
                
                struct_results = st.session_state.tsne_results_all
                
                if 'error' in struct_results:
                    st.error(f"t-SNE Error: {struct_results['error']}")
                else:
                    real_tsne = struct_results.get('real_tsne', [])
                    synth_tsne = struct_results.get('synthetic_tsne', [])
                    
                    if real_tsne and synth_tsne:
                        # Info metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Real Sequences", struct_results.get('num_real_sequences', 0))
                        with col2:
                            st.metric("Synthetic Sequences", struct_results.get('num_synthetic_sequences', 0))
                        with col3:
                            st.metric("Max Sequence Length", struct_results.get('max_seq_len', 0))
                        
                        # t-SNE Plot (smaller size)
                        real_arr = np.array(real_tsne)
                        synth_arr = np.array(synth_tsne)
                        
                        fig, ax = plt.subplots(figsize=(5, 4))
                        
                        ax.scatter(real_arr[:, 0], real_arr[:, 1], 
                                  c='blue', alpha=0.5, label='Real', s=15)
                        ax.scatter(synth_arr[:, 0], synth_arr[:, 1], 
                                  c='red', alpha=0.5, label='Synthetic', s=15)
                        
                        ax.set_xlabel('t-SNE Dimension 1')
                        ax.set_ylabel('t-SNE Dimension 2')
                        ax.set_title('t-SNE: Real vs Synthetic Sequences')
                        ax.legend(fontsize=8)
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Features used
                        with st.expander("Features Used", expanded=False):
                            features_used = struct_results.get('features_used', [])
                            st.write(f"**{len(features_used)} features used:**")
                            for f in features_used:
                                st.write(f"- `{f}`")
                    else:
                        st.info("No t-SNE data available.")
            else:
                st.warning("No data available. Please run the evaluation first.")
        else:
            # Original ML Utility display for high-level
            st.subheader("ML Utility Evaluation")
        
            if 'ml_utility_results' in st.session_state:
                ml_results = st.session_state.ml_utility_results
                
                if 'error' in ml_results:
                    st.error(f"Evaluation Error: {ml_results['error']}")
                    if 'available_columns' in ml_results:
                        st.write("Available columns:", ml_results['available_columns'])
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
                        st.metric(
                            label="Training Samples",
                            value=f"{ml_results['num_training_samples']:,}",
                            help="Number of synthetic samples used for training"
                        )
                    
                    with col3:
                        st.metric(
                            label="Test Samples",
                            value=f"{ml_results['num_test_samples']:,}",
                            help="Number of real samples used for testing"
                        )
                    
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
                            st.info("Feature importance is not available.")
                    
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

    st.header("6. Export Results")

    if st.button("Export All Results as JSON"):
        import json
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