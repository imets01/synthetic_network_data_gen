"""
Synthetic Network Data Generator - Streamlit App

A streamlined application for generating synthetic network data using CTGAN or TabDDPM.
"""
import streamlit as st
import pandas as pd
import toml
import traceback

# Local imports
from ui_components import (
    init_session_state,
    render_file_upload,
    render_data_preview,
    render_model_selection,
    render_training_mode_selection,
    render_ctgan_column_config,
    render_tabddpm_column_config,
    render_metadata_summary,
    render_download_buttons,
    render_training_log,
    render_post_processing_config,
    render_violation_analysis,
    render_post_processing_log,
)
from training import (
    train_ctgan,
    train_tabddpm,
    generate_from_loaded_ctgan,
    generate_from_pretrained_tabddpm,
    configure_ctgan_handler,
    configure_tabddpm_handler,
)
from visualizations import render_all_comparisons
from post_processing import apply_post_processing, analyze_violations

# Page config
st.set_page_config(layout="wide", page_title="Synthetic Network Data Generator", page_icon="📊",)
st.title("Synthetic Network Data Generator")

# Initialize session state
init_session_state()


# =========================================================================
# Section 1: File Upload
# =========================================================================
if not render_file_upload():
    st.stop()

render_data_preview()
df = st.session_state.original_df


# =========================================================================
# Section 2: Model Selection
# =========================================================================
is_ctgan = render_model_selection()


# =========================================================================
# Section 3: Training Mode
# =========================================================================
training_mode = render_training_mode_selection(is_ctgan)


# =========================================================================
# Section 3b: Pre-trained Model Upload (CTGAN)
# =========================================================================
if training_mode == "Upload pre-trained CTGAN model (.pkl)" and is_ctgan:
    uploaded_model = st.file_uploader("Upload trained CTGAN model file", type=['pkl'])
    
    if uploaded_model is not None:
        try:
            import pickle
            import torch
            import io
            
            # Read the uploaded file into bytes
            model_bytes = uploaded_model.read()
            
            # Custom unpickler to handle CUDA tensors on CPU-only machines
            class CPUUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == 'torch.storage' and name == '_load_from_bytes':
                        return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
                    return super().find_class(module, name)
            
            synthesizer = CPUUnpickler(io.BytesIO(model_bytes)).load()
            st.session_state.loaded_synthesizer = synthesizer
            st.session_state.model_loaded = True
            st.session_state.selected_model = "ctgan"
            st.success("CTGAN model loaded successfully!")
            st.info(f"Model type: {type(synthesizer).__name__}")
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            st.code(traceback.format_exc())
    
    # Generation from loaded model
    if st.session_state.model_loaded and st.session_state.loaded_synthesizer is not None:
        st.write("## 4. Generate Synthetic Data")
        num_samples = st.number_input(
            "Number of samples to generate", 
            min_value=100, max_value=100000, value=len(df)
        )
        
        if st.button("Generate from Loaded Model"):
            with st.spinner(f"Generating {num_samples} synthetic samples..."):
                try:
                    synthetic_df = generate_from_loaded_ctgan(
                        st.session_state.loaded_synthesizer, num_samples
                    )
                    st.session_state.synthetic_df = synthetic_df
                    st.session_state.generated = True
                    st.success(f"Generated {len(synthetic_df):,} synthetic samples!")
                except Exception as e:
                    st.error(f"Generation failed: {str(e)}")
                    st.code(traceback.format_exc())


# =========================================================================
# Section 3c: Pre-trained Model Upload (TabDDPM)
# =========================================================================
elif "Upload pre-trained model" in training_mode and not is_ctgan:
    st.info("""
**Upload config.toml, model.pt, and training data files to generate immediately (no training)**

You need to upload all files from a previous training run.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_config = st.file_uploader("Upload config.toml (required)", type=['toml'], key='config_upload')
    with col2:
        uploaded_model = st.file_uploader("Upload model.pt (required)", type=['pt'], key='model_upload')
    
    # Data file uploaders
    st.write("**Training Data Files** (from `data/custom/` folder)")
    st.caption("These ensure the model's feature dimensions match.")
    
    data_col1, data_col2, data_col3 = st.columns(3)
    with data_col1:
        uploaded_info = st.file_uploader("info.json", type=['json'], key='info_upload')
        uploaded_X_num_train = st.file_uploader("X_num_train.npy", type=['npy'], key='xnum_train_upload')
    with data_col2:
        uploaded_X_cat_train = st.file_uploader("X_cat_train.npy", type=['npy'], key='xcat_train_upload')
        uploaded_y_train = st.file_uploader("y_train.npy", type=['npy'], key='y_train_upload')
    with data_col3:
        uploaded_column_config = st.file_uploader("column_config.json (optional)", type=['json'], key='colconfig_upload')
    
    # Store uploaded files
    if uploaded_info is not None:
        st.session_state.uploaded_data_files['info.json'] = uploaded_info.getvalue()
    if uploaded_X_num_train is not None:
        st.session_state.uploaded_data_files['X_num_train.npy'] = uploaded_X_num_train.getvalue()
    if uploaded_X_cat_train is not None:
        st.session_state.uploaded_data_files['X_cat_train.npy'] = uploaded_X_cat_train.getvalue()
    if uploaded_y_train is not None:
        st.session_state.uploaded_data_files['y_train.npy'] = uploaded_y_train.getvalue()
    if uploaded_column_config is not None:
        st.session_state.uploaded_data_files['column_config.json'] = uploaded_column_config.getvalue()
    
    # Status indicators
    required_files = ['info.json', 'X_num_train.npy', 'X_cat_train.npy', 'y_train.npy']
    uploaded_count = sum(1 for f in required_files if f in st.session_state.uploaded_data_files)
    
    if uploaded_count == len(required_files):
        st.success(f"✓ All {len(required_files)} required data files uploaded")
    else:
        st.warning(f"⚠ Data files: {uploaded_count}/{len(required_files)} required files uploaded")
    
    if uploaded_model is not None:
        st.session_state.tabddpm_model_uploaded = uploaded_model
        st.success("✓ Model uploaded")
    else:
        st.warning("⚠ Model.pt required")
    
    if uploaded_config is not None:
        try:
            config_content = uploaded_config.read().decode('utf-8')
            config = toml.loads(config_content)
            st.session_state.tabddpm_config = config
            st.session_state.tabddpm_config_content = config_content
            st.success("✓ Config loaded successfully!")
            
            with st.expander("View Config", expanded=False):
                st.json(config)
        except Exception as e:
            st.error(f"Failed to load config: {str(e)}")
            st.code(traceback.format_exc())
    else:
        st.warning("⚠ Config.toml required")
    
    # Generation section
    data_ready = all(f in st.session_state.uploaded_data_files for f in required_files)
    
    if st.session_state.tabddpm_model_uploaded and st.session_state.tabddpm_config:
        if not data_ready:
            st.error("⚠ Please upload all required training data files")
        else:
            st.write("## 4. Generate Synthetic Data")
            st.info("✓ Ready to generate - using uploaded model, config, and data files")
            
            num_samples = st.number_input(
                "Number of samples to generate", 
                min_value=100, max_value=100000, value=5000,
                key="tabddpm_pretrained_samples"
            )
            
            if st.button("Generate Synthetic Data", key="generate_pretrained"):
                st.session_state.training_log = []
                log_placeholder = st.empty()
                
                with st.spinner("Generating synthetic data..."):
                    try:
                        synthetic_df = generate_from_pretrained_tabddpm(num_samples, log_placeholder)
                        if synthetic_df is not None:
                            st.session_state.synthetic_df = synthetic_df
                            st.session_state.generated = True
                            st.success(f"Generated {len(synthetic_df):,} synthetic samples!")
                    except Exception as e:
                        st.error(f"Generation failed: {str(e)}")
                        st.code(traceback.format_exc())


# =========================================================================
# Section 3d: Pre-tuned Config Upload (TabDDPM)
# =========================================================================
elif "Use pre-tuned config (.toml)" in training_mode and not is_ctgan:
    st.info("""
**Upload config.toml to train with tuned hyperparameters**

This skips hyperparameter search and trains using known-good parameters.
    """)
    
    uploaded_config = st.file_uploader("Upload config.toml", type=['toml'], key='config_upload')
    
    if uploaded_config is not None:
        try:
            config_content = uploaded_config.read().decode('utf-8')
            config = toml.loads(config_content)
            st.session_state.tabddpm_config = config
            st.session_state.tabddpm_config_content = config_content
            st.success("✓ Config loaded successfully!")
            
            with st.expander("View Config", expanded=False):
                st.json(config)
        except Exception as e:
            st.error(f"Failed to load config: {str(e)}")
            st.code(traceback.format_exc())


# =========================================================================
# Section 4: Column Configuration (for training modes)
# =========================================================================
show_column_config = (
    training_mode == "Train new model" or 
    "Quick tune" in training_mode or 
    "Full tune" in training_mode or 
    "Use pre-tuned config (.toml)" in training_mode
)

if show_column_config:
    st.write("## 4. Configure Columns")
    
    if is_ctgan:
        config = render_ctgan_column_config(df)
    else:
        config = render_tabddpm_column_config(df)
    
    if st.button("Configure & Preprocess"):
        if not config['feature_cols']:
            st.error("Please select at least one feature column")
        else:
            st.session_state.selected_model = "ctgan" if is_ctgan else "tabddpm"
            
            with st.spinner("Initializing and preprocessing..."):
                try:
                    if is_ctgan:
                        handler = configure_ctgan_handler(df, config)
                        st.session_state.handler = handler
                        st.session_state.configured = True
                        st.success(f"Configured! {len(config['include_cols'])} columns ready for CTGAN training")
                        render_metadata_summary(handler.get_metadata_summary())
                    else:
                        handler, result = configure_tabddpm_handler(df, config)
                        st.session_state.handler = handler
                        st.session_state.configured = True
                        st.success(f"Preprocessed! Train: {result['train']}, Val: {result['val']}, Test: {result['test']}")
                except Exception as e:
                    st.error(f"Configuration failed: {str(e)}")
                    st.code(traceback.format_exc())


# =========================================================================
# Section 5: Generate Synthetic Data (after configuration)
# =========================================================================
if st.session_state.configured:
    st.write("## 5. Generate Synthetic Data")
    
    handler = st.session_state.handler
    is_ctgan_model = st.session_state.selected_model == "ctgan"
    
    if is_ctgan_model:
        st.info(f"Ready to train CTGAN | Columns: {len(handler.included_cols)}")
        
        with st.expander("CTGAN Parameters", expanded=False):
            epochs = st.slider("Epochs", min_value=100, max_value=3000, value=700, step=100)
            batch_size = st.selectbox("Batch Size", [250, 500, 1000], index=0)
            num_samples = st.number_input("Number of samples to generate", 
                                         min_value=100, max_value=100000, value=len(df))
    else:
        st.info(f"Ready to train TabDDPM | Num features: {len(handler.num_features)} | Cat features: {len(handler.cat_features)}")
        
        if "Quick" in training_mode:
            st.write("**Mode:** Quick tune (2 trials)")
        elif "Full" in training_mode:
            st.write("**Mode:** Full tune (50 trials)")
        elif "pre-tuned" in training_mode:
            st.write("**Mode:** Using pre-tuned config")
        
        num_samples = st.number_input("Number of samples to generate", 
                                     min_value=100, max_value=100000, value=len(df),
                                     key="tabddpm_num_samples")
    
    if st.button("Start Generation"):
        st.session_state.training_log = []
        log_placeholder = st.empty()
        
        with st.spinner("Training model..."):
            try:
                if is_ctgan_model:
                    synthetic_df = train_ctgan(handler, epochs, batch_size, num_samples)
                else:
                    synthetic_df = train_tabddpm(handler, training_mode, num_samples, log_placeholder)
                
                if synthetic_df is not None:
                    st.session_state.synthetic_df = synthetic_df
                    st.session_state.generated = True
                    st.success(f"Generated {len(synthetic_df):,} synthetic samples!")
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
                st.code(traceback.format_exc())


# =========================================================================
# Section 6: Results & Post-Processing
# =========================================================================
if st.session_state.generated and st.session_state.synthetic_df is not None:
    st.write("## 6. Results & Post-Processing")
    
    synthetic_df = st.session_state.synthetic_df
    
    if 'connection_duration' in synthetic_df.columns:
        conn_dur_col = 'connection_duration'
    elif 'Target' in synthetic_df.columns:
        conn_dur_col = 'Target'
    else:
        conn_dur_col = 'connection_duration'
    
    st.write("### Constraint Violation Analysis (Before Post-Processing)")
    violations_before = analyze_violations(synthetic_df, conn_dur_col)
    render_violation_analysis(violations_before)
    
    st.write("---")
    pp_config = render_post_processing_config()
    st.session_state.post_processing_config = pp_config
    
    if pp_config.get('enabled', False):
        if st.button("Apply Post-Processing"):
            with st.spinner("Applying post-processing rules..."):
                processed_df, pp_log = apply_post_processing(
                    synthetic_df,
                    connection_duration_col=conn_dur_col,
                    duration_method=pp_config.get('duration_method', 'clip'),
                    combined_duration_method=pp_config.get('combined_method', 'scale'),
                    fix_integers=pp_config.get('fix_integers', True),
                    fix_logical=pp_config.get('fix_logical', True)
                )
                
                st.session_state.synthetic_df_raw = synthetic_df.copy()
                st.session_state.synthetic_df = processed_df
                st.session_state.post_processing_log = pp_log
                
                violations_after = analyze_violations(processed_df, conn_dur_col)
                st.session_state.violations_after = violations_after
                
                st.success("Post-processing complete!")
                st.rerun()
    
    if st.session_state.post_processing_log:
        render_post_processing_log(st.session_state.post_processing_log)
        
        if st.session_state.violations_after:
            st.write("### Constraint Violation Analysis (After Post-Processing)")
            render_violation_analysis(st.session_state.violations_after)
    
    st.write("### Generated Synthetic Data")
    st.dataframe(st.session_state.synthetic_df.head(20), use_container_width=True)
    st.write(f"**Total rows:** {len(st.session_state.synthetic_df):,}")
    
    render_download_buttons(st.session_state.synthetic_df)


# =========================================================================
# Section 7: Visualizations
# =========================================================================
st.write("## 7. Data Visualizations")

if st.session_state.generated and st.session_state.synthetic_df is not None:
    render_all_comparisons(st.session_state.original_df, st.session_state.synthetic_df)
elif st.session_state.uploaded:
    st.info("Configure and generate synthetic data to view comparisons")
else:
    st.info("Upload data to get started")

render_training_log()
