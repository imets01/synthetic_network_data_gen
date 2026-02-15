import streamlit as st
import pandas as pd


DEFAULT_TARGET = 'connection_duration'
DEFAULT_FEATURES = [
    'implementation', 'retry_occurred', 'version_negotiation_occurred', 'migration_type',
    'handshake_duration', 'time_to_migration', 'migration_duration',
    'packets_before_migration', 'total_bidi_streams_client_init',
    'total_udi_streams_client_init',
    'path_validation_initiated',
    'connection_close_type',
    'bytes_sent_client', 'bytes_sent_server', 'first_path_validation_response_latency', 
    'packets_sent_client', 'packets_sent_server',
]
DEFAULT_CATEGORICAL = [
    'implementation', 'migration_type',
    'path_validation_initiated', 'connection_close_type',
]
DEFAULT_BOOLEAN = [
    'version_negotiation_occurred', 'retry_occurred',
]
DEFAULT_NUMERICAL = [
    'connection_duration', 'handshake_duration', 'time_to_migration', 'migration_duration',
    'first_path_validation_response_latency', 'packets_sent_client', 'packets_sent_server',
]


def init_session_state():
    """Initialize all session state variables with defaults."""
    defaults = {
        'uploaded': False,
        'configured': False,
        'generated': False,
        'synthetic_df': None,
        'synthetic_df_raw': None, 
        'original_df': None,
        'handler': None,
        'training_log': [],
        'selected_model': None,
        'model_loaded': False,
        'loaded_synthesizer': None,
        'tabddpm_model_path': None,
        'tabddpm_config': None,
        'tabddpm_config_content': None,
        'tabddpm_model_uploaded': None,
        'uploaded_data_files': {},
        'post_processing_config': {'enabled': True},
        'post_processing_log': [],
        'violations_before': None,
        'violations_after': None,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def reset_generation_state():
    """Reset generation-related state when new file is uploaded."""
    st.session_state.configured = False
    st.session_state.generated = False
    st.session_state.synthetic_df = None
    st.session_state.handler = None


def render_file_upload():
    """Render file upload section and handle uploaded file."""
    st.write("## 1. Upload Real Network Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None and not st.session_state.uploaded:
        st.session_state.original_df = pd.read_csv(uploaded_file)
        st.session_state.uploaded = True
        reset_generation_state()
    
    return st.session_state.uploaded


def render_data_preview():
    """Show uploaded data info and preview."""
    df = st.session_state.original_df
    st.success(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")
    
    with st.expander("Preview Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)


def render_model_selection():
    """Render model selection radio and return if CTGAN is selected."""
    st.write("## 2. Select Model")
    model_choice = st.radio(
        "Choose generation model",
        ["CTGAN (Fast, good quality)", "TabDDPM (Slower, higher quality)"],
        index=0
    )
    return "CTGAN" in model_choice


def render_training_mode_selection(is_ctgan: bool):
    """Render training mode selection based on model type."""
    st.write("## 3. Training Mode")
    
    if is_ctgan:
        return st.radio(
            "Choose training mode",
            ["Train new model", "Upload pre-trained CTGAN model (.pkl)"],
            index=0
        )
    else:
        return st.radio(
            "Choose training mode",
            [
                "Quick tune (2 trials, ~5 min) - for development",
                "Full tune (50 trials, ~hours) - for production",
                "Use pre-tuned config (.toml) - skip tuning, train with known good params",
                "Upload pre-trained model (.toml + .pt) - generate immediately"
            ],
            index=0
        )


def render_ctgan_column_config(df: pd.DataFrame):
    """Render column configuration for CTGAN and return selections."""
    all_columns = list(df.columns)
    
    st.info("CTGAN generates all columns together - no target column needed")
    
    # Columns to include
    default_include = [c for c in DEFAULT_FEATURES + [DEFAULT_TARGET] if c in all_columns]
    if not default_include:
        default_include = all_columns[:10]
    
    include_cols = st.multiselect(
        "Columns to Include",
        options=all_columns,
        default=default_include
    )
    
    # Numerical columns
    default_num = [c for c in DEFAULT_NUMERICAL if c in include_cols]
    if not default_num:
        default_num = [c for c in include_cols if df[c].dtype in ['float64', 'float32']]
    
    num_cols = st.multiselect(
        "Numerical Columns (continuous values)",
        options=include_cols,
        default=default_num
    )
    
    # Boolean columns
    default_bool = [c for c in DEFAULT_BOOLEAN if c in include_cols]
    if not default_bool:
        default_bool = [c for c in include_cols 
                        if df[c].nunique() == 2 and set(df[c].dropna().unique()).issubset({0, 1, True, False})]
    
    bool_cols = st.multiselect(
        "Boolean Columns (0/1 or True/False)",
        options=[c for c in include_cols if c not in num_cols],
        default=default_bool
    )
    
    # Categorical columns
    remaining_cols = [c for c in include_cols if c not in num_cols and c not in bool_cols]
    default_cat = [c for c in DEFAULT_CATEGORICAL if c in remaining_cols]
    if not default_cat:
        default_cat = remaining_cols
    
    cat_cols = st.multiselect(
        "Categorical Columns",
        options=remaining_cols,
        default=default_cat
    )
    
    return {
        'include_cols': include_cols,
        'num_cols': num_cols,
        'bool_cols': bool_cols,
        'cat_cols': cat_cols,
        'target_col': None,
        'feature_cols': include_cols,
    }


def render_tabddpm_column_config(df: pd.DataFrame):
    """Render column configuration for TabDDPM and return selections."""
    all_columns = list(df.columns)
    
    # Target column selection
    default_idx = all_columns.index(DEFAULT_TARGET) if DEFAULT_TARGET in all_columns else (len(all_columns)-1 if all_columns else 0)
    
    target_col = st.selectbox(
        "Target Column (what to predict)",
        options=all_columns,
        index=default_idx
    )
    
    # Feature columns selection
    available_features = [c for c in all_columns if c != target_col]
    default_feature_selection = [c for c in DEFAULT_FEATURES if c in available_features]
    if not default_feature_selection:
        default_feature_selection = available_features[:10]
    
    feature_cols = st.multiselect(
        "Feature Columns to Keep",
        options=available_features,
        default=default_feature_selection
    )
    
    # Categorical columns selection
    default_cat_selection = [c for c in DEFAULT_CATEGORICAL + DEFAULT_BOOLEAN if c in feature_cols]
    if not default_cat_selection:
        default_cat_selection = [c for c in feature_cols 
                                  if df[c].dtype == 'object' or df[c].nunique() <= 10][:5]
    
    cat_cols = st.multiselect(
        "Categorical Columns (from features)",
        options=feature_cols,
        default=default_cat_selection
    )
    
    return {
        'include_cols': feature_cols,
        'num_cols': [],
        'bool_cols': [],
        'cat_cols': cat_cols,
        'target_col': target_col,
        'feature_cols': feature_cols,
    }


def render_metadata_summary(summary: dict):
    """Display metadata schema summary in columns."""
    st.write("### Metadata Schema")
    
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    with meta_col1:
        st.metric("Numerical", len(summary.get('numerical_float', [])))
        if summary.get('numerical_float'):
            st.write(", ".join(summary['numerical_float']))
    with meta_col2:
        st.metric("Categorical", len(summary.get('categorical', [])))
        if summary.get('categorical'):
            st.write(", ".join(summary['categorical']))
    with meta_col3:
        st.metric("Boolean", len(summary.get('boolean', [])))
        if summary.get('boolean'):
            st.write(", ".join(summary['boolean']))


def render_download_buttons(synthetic_df: pd.DataFrame):
    """Render download buttons for synthetic data and model."""
    import pickle
    import io
    
    # Check if we have both raw and processed versions
    has_raw = st.session_state.get('synthetic_df_raw') is not None
    
    if has_raw:
        dl_col1, dl_col2, dl_col3 = st.columns(3)
    else:
        dl_col1, dl_col2 = st.columns(2)
        dl_col3 = None
    
    with dl_col1:
        csv = synthetic_df.to_csv(index=False).encode('utf-8')
        label = "📥 Download Synthetic Data (CSV)" if not has_raw else "📥 Download Post-Processed Data (CSV)"
        st.download_button(
            label=label,
            data=csv,
            file_name='synthetic_network_data.csv' if not has_raw else 'synthetic_network_data_postprocessed.csv',
            mime='text/csv',
        )
    
    if has_raw and dl_col3:
        with dl_col3:
            raw_csv = st.session_state.synthetic_df_raw.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Raw Data (CSV)",
                data=raw_csv,
                file_name='synthetic_network_data_raw.csv',
                mime='text/csv',
            )
    
    with dl_col2:
        handler = st.session_state.handler
        loaded_synth = st.session_state.loaded_synthesizer
        
        # Case 1: Newly trained CTGAN model
        if st.session_state.selected_model == "ctgan" and handler is not None and hasattr(handler, 'synthesizer') and handler.synthesizer is not None:
            model_buffer = io.BytesIO()
            pickle.dump(handler.synthesizer, model_buffer)
            model_buffer.seek(0)
            st.download_button(
                label="📥 Download Trained Model (PKL)",
                data=model_buffer,
                file_name='ctgan_model.pkl',
                mime='application/octet-stream',
            )
        # Case 2: Loaded model
        elif st.session_state.model_loaded and loaded_synth is not None:
            model_buffer = io.BytesIO()
            pickle.dump(loaded_synth, model_buffer)
            model_buffer.seek(0)
            st.download_button(
                label="📥 Re-download Model (PKL)",
                data=model_buffer,
                file_name='ctgan_model.pkl',
                mime='application/octet-stream',
            )
        elif st.session_state.selected_model == "tabddpm":
            st.info("TabDDPM model files are saved in the exp folder")


def render_training_log():
    """Render training log in an expander."""
    if st.session_state.training_log:
        with st.expander("Training Log", expanded=False):
            st.code('\n'.join(st.session_state.training_log))


def render_post_processing_config():
    st.write("### Post-Processing Options")
    st.info("""
Post-processing applies domain-specific constraints to ensure generated data is valid:
- **Non-negativity**: Clips negative values to 0
- **Duration constraints**: Ensures sub-durations ≤ connection_duration  
- **Combined duration**: Ensures handshake + migration ≤ connection_duration
- **Integer columns**: Rounds count/byte columns to integers
- **Logical constraints**: Ensures temporal ordering (e.g., migration after handshake)
- **Migration endpoints**: Adds realistic IP/port addresses for QUIC migration
    """)
    
    enable_postprocessing = st.checkbox(
        "Enable post-processing",
        value=True,
        help="Apply domain-specific constraints to the generated data"
    )
    
    if not enable_postprocessing:
        return {'enabled': False}
    
    with st.expander("Advanced Post-Processing Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            duration_method = st.selectbox(
                "Duration Constraint Method",
                options=['clip', 'scale', 'drop'],
                index=0,
                help=(
                    "**clip**: Cap values at connection_duration\n"
                    "**scale**: Scale values proportionally\n"
                    "**drop**: Remove violating rows"
                )
            )
            
            fix_integers = st.checkbox(
                "Round integer columns",
                value=True,
                help="Round count/byte columns to nearest integer"
            )
        
        with col2:
            combined_method = st.selectbox(
                "Combined Duration Method",
                options=['scale', 'prioritize_handshake', 'drop'],
                index=0,
                help=(
                    "**scale**: Scale both handshake and migration proportionally\n"
                    "**prioritize_handshake**: Keep handshake, reduce migration\n"
                    "**drop**: Remove violating rows"
                )
            )
            
            fix_logical = st.checkbox(
                "Apply logical constraints",
                value=True,
                help="Ensure temporal ordering (e.g., migration after handshake)"
            )
        
        st.write("**Migration Endpoints**")
        add_endpoints = st.checkbox(
            "Add migration endpoint addresses",
            value=True,
            help="Generate realistic IP/port addresses for QUIC connection migration"
        )
    
    return {
        'enabled': True,
        'duration_method': duration_method,
        'combined_method': combined_method,
        'fix_integers': fix_integers,
        'fix_logical': fix_logical,
        'add_endpoints': add_endpoints
    }


def render_violation_analysis(violations: dict):
    from post_processing import format_violations_report
    
    summary = violations.get('summary', {})
    has_violations = summary.get('has_violations', False)
    
    if has_violations:
        st.warning(f"Found {summary.get('total_violation_types', 0)} types of constraint violations")
        
        with st.expander("View Violation Details", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Non-Negativity**")
                if violations.get('non_negative'):
                    for col, info in violations['non_negative'].items():
                        st.write(f"• {col}: {info['count']} ({info['percentage']:.1f}%)")
                else:
                    st.write("✓ No violations")
            
            with col2:
                st.write("**Duration Constraints**")
                if violations.get('duration_constraints'):
                    for col, info in violations['duration_constraints'].items():
                        st.write(f"• {col}: {info['count']} ({info['percentage']:.1f}%)")
                else:
                    st.write("✓ No violations")
            
            with col3:
                st.write("**Combined Duration**")
                if violations.get('combined_duration'):
                    for constraint, info in violations['combined_duration'].items():
                        st.write(f"• {info['count']} rows ({info['percentage']:.1f}%)")
                else:
                    st.write("✓ No violations")
    else:
        st.success("✓ No constraint violations found")


def render_post_processing_log(log: list):
    if log:
        with st.expander("Post-Processing Log", expanded=False):
            st.code('\n'.join(log))
