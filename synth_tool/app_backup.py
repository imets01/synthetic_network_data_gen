import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import toml

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set page config to wide mode
st.set_page_config(layout="wide", page_title="Synthetic Network Data Generator")

st.title("Synthetic Network Data Generator")

# Initialize session state
if 'uploaded' not in st.session_state:
    st.session_state.uploaded = False
if 'configured' not in st.session_state:
    st.session_state.configured = False
if 'generated' not in st.session_state:
    st.session_state.generated = False
if 'synthetic_df' not in st.session_state:
    st.session_state.synthetic_df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'handler' not in st.session_state:
    st.session_state.handler = None
if 'training_log' not in st.session_state:
    st.session_state.training_log = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'loaded_synthesizer' not in st.session_state:
    st.session_state.loaded_synthesizer = None
if 'tabddpm_model_path' not in st.session_state:
    st.session_state.tabddpm_model_path = None
if 'tabddpm_config' not in st.session_state:
    st.session_state.tabddpm_config = None
if 'tabddpm_config_content' not in st.session_state:
    st.session_state.tabddpm_config_content = None
if 'tabddpm_model_uploaded' not in st.session_state:
    st.session_state.tabddpm_model_uploaded = None

st.write("## 1. Upload Real Network Data")

uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None and not st.session_state.uploaded:
    st.session_state.original_df = pd.read_csv(uploaded_file)
    st.session_state.uploaded = True
    st.session_state.configured = False
    st.session_state.generated = False

# Show uploaded data info
if st.session_state.uploaded and st.session_state.original_df is not None:
    df = st.session_state.original_df
    st.success(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")
    
    with st.expander("Preview Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Model selection
    st.write("## 2. Select Model")
    model_choice = st.radio(
        "Choose generation model",
        ["CTGAN (Fast, good quality)", "TabDDPM (Slower, higher quality)"],
        index=0
    )
    
    is_ctgan_selected = "CTGAN" in model_choice
    
    # Option to upload pre-trained model
    st.write("## 3. Training Mode")
    
    if is_ctgan_selected:
        training_mode = st.radio(
            "Choose training mode",
            ["Train new model", "Upload pre-trained CTGAN model (.pkl)"],
            index=0
        )
    else:
        training_mode = st.radio(
            "Choose training mode",
            [
                "Quick tune (2 trials, ~5 min) - for development",
                "Full tune (50 trials, ~hours) - for production",
                "Use pre-tuned config (.toml) - skip tuning, train with known good params",
                "Upload pre-trained model (.toml + .pt) - generate immediately"
            ],
            index=0
        )
    
    # CTGAN model upload
    if training_mode == "Upload pre-trained CTGAN model (.pkl)" and is_ctgan_selected:
        uploaded_model = st.file_uploader("Upload trained CTGAN model file", type=['pkl'])
        
        if uploaded_model is not None:
            try:
                import pickle
                from sdv.single_table import CTGANSynthesizer
                
                # Load the model
                synthesizer = pickle.load(uploaded_model)
                st.session_state.loaded_synthesizer = synthesizer
                st.session_state.model_loaded = True
                st.session_state.selected_model = "ctgan"
                st.success("CTGAN model loaded successfully!")
                
                # Show model info
                st.info(f"Model type: {type(synthesizer).__name__}")
                
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        # Sampling section for loaded CTGAN model
        if st.session_state.model_loaded and st.session_state.loaded_synthesizer is not None:
            st.write("## 4. Generate Synthetic Data")
            num_samples = st.number_input(
                "Number of samples to generate", 
                min_value=100, 
                max_value=100000, 
                value=len(df)
            )
            
            if st.button("Generate from Loaded Model"):
                with st.spinner(f"Generating {num_samples} synthetic samples..."):
                    try:
                        synthetic_df = st.session_state.loaded_synthesizer.sample(num_rows=num_samples)
                        st.session_state.synthetic_df = synthetic_df
                        st.session_state.generated = True
                        st.success(f"Generated {len(synthetic_df):,} synthetic samples!")
                    except Exception as e:
                        st.error(f"Generation failed: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # TabDDPM pre-tuned config upload
    elif "Upload pre-trained model" in training_mode and not is_ctgan_selected:
        st.info("""
**Upload config.toml, model.pt, and training data files to generate immediately (no training)**

You need to upload all files from a previous training run. The data files ensure feature dimensions match the model.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            uploaded_config = st.file_uploader("Upload config.toml (required)", type=['toml'], key='config_upload')
        with col2:
            uploaded_model = st.file_uploader("Upload model.pt (required)", type=['pt'], key='model_upload')
        
        # Add data file uploaders
        st.write("**Training Data Files** (from `data/custom/` folder)")
        st.caption("These ensure the model's feature dimensions match. Upload at minimum: info.json and the train .npy files.")
        
        data_col1, data_col2, data_col3 = st.columns(3)
        with data_col1:
            uploaded_info = st.file_uploader("info.json", type=['json'], key='info_upload')
            uploaded_X_num_train = st.file_uploader("X_num_train.npy", type=['npy'], key='xnum_train_upload')
        with data_col2:
            uploaded_X_cat_train = st.file_uploader("X_cat_train.npy", type=['npy'], key='xcat_train_upload')
            uploaded_y_train = st.file_uploader("y_train.npy", type=['npy'], key='y_train_upload')
        with data_col3:
            uploaded_column_config = st.file_uploader("column_config.json (optional)", type=['json'], key='colconfig_upload')
        
        # Store uploaded data files in session state
        if 'uploaded_data_files' not in st.session_state:
            st.session_state.uploaded_data_files = {}
        
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
        
        # Check which files are uploaded
        required_data_files = ['info.json', 'X_num_train.npy', 'X_cat_train.npy', 'y_train.npy']
        uploaded_data_count = sum(1 for f in required_data_files if f in st.session_state.uploaded_data_files)
        
        if uploaded_data_count == len(required_data_files):
            st.success(f"✓ All {len(required_data_files)} required data files uploaded")
        else:
            st.warning(f"⚠ Data files: {uploaded_data_count}/{len(required_data_files)} required files uploaded")
        
        if uploaded_model is not None:
            st.session_state.tabddpm_model_uploaded = uploaded_model
            st.success("✓ Model uploaded")
        else:
            st.warning("⚠ Model.pt required for this mode")
        
        if uploaded_config is not None:
            try:
                config_content = uploaded_config.read().decode('utf-8')
                config = toml.loads(config_content)
                
                st.session_state.tabddpm_config = config
                st.session_state.tabddpm_config_content = config_content
                st.success("✓ Config loaded successfully!")
                
                # Show config
                with st.expander("View Config", expanded=False):
                    st.json(config)
                
            except Exception as e:
                st.error(f"Failed to load config: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("⚠ Config.toml required for this mode")
        
        # If BOTH model and config are uploaded, skip to generation
        # Also check if data files are uploaded
        data_files_ready = all(f in st.session_state.get('uploaded_data_files', {}) 
                               for f in ['info.json', 'X_num_train.npy', 'X_cat_train.npy', 'y_train.npy'])
        
        if st.session_state.tabddpm_model_uploaded is not None and st.session_state.tabddpm_config is not None:
            if not data_files_ready:
                st.error("⚠ Please upload the required training data files (info.json, X_num_train.npy, X_cat_train.npy, y_train.npy)")
            else:
                st.write("## 4. Generate Synthetic Data")
                st.info("✓ Ready to generate - using uploaded model, config, and data files")
                
                num_samples = st.number_input(
                    "Number of samples to generate", 
                    min_value=100, 
                    max_value=100000, 
                    value=5000,
                    key="tabddpm_pretrained_samples"
                )
                
                
                if st.button("Generate Synthetic Data", key="generate_pretrained"):
                    st.session_state.training_log = []
                    log_placeholder = st.empty()
                    
                    with st.spinner("Generating synthetic data..."):
                        try:
                            import subprocess
                            
                            # Use the .conda environment
                            conda_env = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.conda'))
                            lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tab-ddpm', 'tab-ddpm-main'))
                            pipeline_script = os.path.join(lib_path, 'scripts', 'pipeline.py')
                            
                            # Save uploaded data files to data/custom/ FIRST
                            data_dir = os.path.join(lib_path, 'data', 'custom')
                            os.makedirs(data_dir, exist_ok=True)
                            
                            for filename, content in st.session_state.uploaded_data_files.items():
                                file_path = os.path.join(data_dir, filename)
                                with open(file_path, 'wb') as f:
                                    f.write(content)
                            st.session_state.training_log.append(f"Saved {len(st.session_state.uploaded_data_files)} data files to {data_dir}")
                            
                            # Also create dummy val/test files if not uploaded (copy from train)
                            for split in ['val', 'test']:
                                for prefix in ['X_num', 'X_cat', 'y']:
                                    src_file = f'{prefix}_train.npy'
                                    dst_file = f'{prefix}_{split}.npy'
                                    if dst_file not in st.session_state.uploaded_data_files and src_file in st.session_state.uploaded_data_files:
                                        dst_path = os.path.join(data_dir, dst_file)
                                        with open(dst_path, 'wb') as f:
                                            f.write(st.session_state.uploaded_data_files[src_file])
      
                            config = st.session_state.tabddpm_config.copy()
                            config['parent_dir'] = 'exp/custom'
                            config['real_data_path'] = 'data/custom/'  # Now safe - data files match the model
                            
                            import torch
                            config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                            if 'sample' in config:
                                config['sample']['num_samples'] = num_samples
                            # Save config
                            config_path = os.path.join(lib_path, 'exp', 'custom', 'config.toml')
                            os.makedirs(os.path.dirname(config_path), exist_ok=True)
                            with open(config_path, 'w') as f:
                                toml.dump(config, f)
                            # Save uploaded model
                            model_path = os.path.join(lib_path, 'exp', 'custom', 'model.pt')
                            with open(model_path, 'wb') as f:
                                f.write(st.session_state.tabddpm_model_uploaded.getvalue())
                            
                            cmd = [
                                'conda', 'run', '-p', conda_env,
                                'python', '-u',
                                pipeline_script,
                                '--config', config_path,
                                '--sample'
                            ]
                            
                            env = os.environ.copy()
                            env['PYTHONPATH'] = lib_path
                            env['PYTHONUNBUFFERED'] = '1'
                            
                            st.write(f"Running: `{' '.join(cmd)}`")
                            
                            process = subprocess.Popen(
                                cmd,
                                cwd=lib_path,
                                env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                bufsize=1,
                                encoding='utf-8',
                                errors='replace'
                            )
                            
                            import time
                            last_update = time.time()
                            for line in iter(process.stdout.readline, ''):
                                if line.strip():
                                    st.session_state.training_log.append(line.strip())
                                if time.time() - last_update > 0.5:
                                    log_placeholder.code('\n'.join(st.session_state.training_log[-15:]))
                                    last_update = time.time()
                            
                            process.wait()
                            
                            if process.returncode == 0:
                                # Load the generated synthetic data
                                import json
                                
                                output_dir = os.path.join(lib_path, 'exp', 'custom')
                                data_dir = os.path.join(lib_path, 'data', 'custom')
                                
                                # Load generated data FIRST to get actual dimensions
                                X_num = np.load(os.path.join(output_dir, 'X_num_train.npy'))
                                X_cat = np.load(os.path.join(output_dir, 'X_cat_train.npy'), allow_pickle=True)
                                y = np.load(os.path.join(output_dir, 'y_train.npy'))
                                
                                n_num = X_num.shape[1] if X_num is not None and len(X_num.shape) > 1 else 0
                                n_cat = X_cat.shape[1] if X_cat is not None and len(X_cat.shape) > 1 else 0
                                total_features = n_num + n_cat
                                
                                st.info(f"Generated data shape: {n_num} numerical + {n_cat} categorical = {total_features} features")
                                
                                # Try to load column config - prefer uploaded one, then check disk
                                feature_names = None
                                target_col = None
                                
                                # Check if user uploaded column_config.json
                                if 'column_config.json' in st.session_state.get('uploaded_data_files', {}):
                                    try:
                                        col_config = json.loads(st.session_state.uploaded_data_files['column_config.json'].decode('utf-8'))
                                        num_cols = col_config.get('numerical_columns', [])
                                        cat_cols = col_config.get('categorical_columns', [])
                                        target_col = col_config.get('target_column', None)
                                        
                                        # Remove target from numerical columns if present (target is stored separately in y)
                                        if target_col and target_col in num_cols:
                                            num_cols = [c for c in num_cols if c != target_col]
                                        if target_col and target_col in cat_cols:
                                            cat_cols = [c for c in cat_cols if c != target_col]
                                        
                                        st.info(f"Uploaded column_config: {len(num_cols)} numerical + {len(cat_cols)} categorical (target: {target_col})")
                                        # Only use if dimensions match
                                        if len(num_cols) == n_num and len(cat_cols) == n_cat:
                                            feature_names = num_cols + cat_cols
                                            st.success("✓ Column names loaded from uploaded config")
                                        else:
                                            st.warning(f"Column config mismatch! Config has {len(num_cols)} num + {len(cat_cols)} cat, but data has {n_num} num + {n_cat} cat. Using generic names.")
                                    except Exception as e:
                                        st.warning(f"Could not parse uploaded column_config.json: {e}")
                                
                                # Fallback: use generic names based on actual data dimensions
                                if feature_names is None:
                                    num_names = [f'num_feature_{i}' for i in range(n_num)]
                                    cat_names = [f'cat_feature_{i}' for i in range(n_cat)]
                                    feature_names = num_names + cat_names
                                    target_col = 'target'
                                
                                # Combine into DataFrame
                                X = np.concatenate([X_num, X_cat], axis=1)
                                synthetic_df = pd.DataFrame(X, columns=feature_names)
                                synthetic_df[target_col] = y.flatten()
                                
                                st.session_state.synthetic_df = synthetic_df
                                st.session_state.generated = True
                                st.success(f"Generated {len(synthetic_df):,} synthetic samples!")
                            else:
                                st.error(f"Generation failed with code {process.returncode}")
                                st.code('\n'.join(st.session_state.training_log[-20:]))
                        
                        except Exception as e:
                            st.error(f"Generation failed: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
    
    # TabDDPM config-only upload (will train, then generate)
    elif "Use pre-tuned config (.toml)" in training_mode and not is_ctgan_selected:
        st.info("""
**Upload config.toml to train with tuned hyperparameters**

This skips the hyperparameter search and trains using known-good parameters.
You'll still need to select columns and the model will train before generating.
        """)
        
        uploaded_config = st.file_uploader("Upload config.toml", type=['toml'], key='config_upload')
        
        if uploaded_config is not None:
            try:
                config_content = uploaded_config.read().decode('utf-8')
                config = toml.loads(config_content)
                
                st.session_state.tabddpm_config = config
                st.session_state.tabddpm_config_content = config_content
                st.success("✓ Config loaded successfully!")
                
                # Show config
                with st.expander("View Config", expanded=False):
                    st.json(config)
                
            except Exception as e:
                st.error(f"Failed to load config: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Show column configuration for: CTGAN training, TabDDPM tuning, or TabDDPM config-only
    # NOT for pre-trained model mode - features are already fixed in the model
    show_column_config = (
        training_mode == "Train new model" or 
        "Quick tune" in training_mode or 
        "Full tune" in training_mode or 
        "Use pre-tuned config (.toml)" in training_mode
    )
    
    if show_column_config:
        # Training new model - configure columns
        st.write("## 4. Configure Columns")
        
        all_columns = list(df.columns)
        
        # Default settings from notebook
        DEFAULT_TARGET = 'connection_duration'
        DEFAULT_FEATURES = [
            'implementation', 'retry_occurred', 'version_negotiation_occurred', 'migration_type',
            'handshake_duration', 'time_to_migration', 'migration_duration',
            'packets_before_migration', 'total_bidi_streams_client_init',
            'total_udi_streams_client_init',
            'path_validation_initiated',
            'connection_close_type',
            'bytes_sent_client', 'bytes_sent_server', 'first_path_validation_response_latency', 'packets_sent_client',
            'packets_sent_server',
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
        
        if is_ctgan_selected:
            # CTGAN: Select columns to include (no target needed)
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
            
            # Numerical columns (float)
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
                default_bool = [c for c in include_cols if df[c].nunique() == 2 and set(df[c].dropna().unique()).issubset({0, 1, True, False})]
            
            bool_cols = st.multiselect(
                "Boolean Columns (0/1 or True/False)",
                options=[c for c in include_cols if c not in num_cols],
                default=default_bool
            )
            
            # Categorical columns (remaining)
            remaining_cols = [c for c in include_cols if c not in num_cols and c not in bool_cols]
            default_cat = [c for c in DEFAULT_CATEGORICAL if c in remaining_cols]
            if not default_cat:
                default_cat = remaining_cols
            
            cat_cols = st.multiselect(
                "Categorical Columns",
                options=remaining_cols,
                default=default_cat
            )
            
            # Store for later
            target_col = None
            feature_cols = include_cols
            
        else:
            # TabDDPM: Needs target column
            # Target column selection - default to connection_duration if available
            default_idx = all_columns.index(DEFAULT_TARGET) if DEFAULT_TARGET in all_columns else (len(all_columns)-1 if all_columns else 0)
            
            target_col = st.selectbox(
                "Target Column (what to predict)",
                options=all_columns,
                index=default_idx
            )
            
            # Feature columns selection - default to notebook features if available
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
                default_cat_selection = [c for c in feature_cols if df[c].dtype == 'object' or df[c].nunique() <= 10][:5]
            
            cat_cols = st.multiselect(
                "Categorical Columns (from features)",
                options=feature_cols,
                default=default_cat_selection
            )
            
            # Not used for TabDDPM but define for consistency
            bool_cols = []
            num_cols = []
            include_cols = feature_cols
        
        if st.button("Configure & Preprocess"):
            if not feature_cols:
                st.error("Please select at least one feature column")
            else:
                is_ctgan = "CTGAN" in model_choice
                st.session_state.selected_model = "ctgan" if is_ctgan else "tabddpm"
                
                if is_ctgan:
                    # CTGAN configuration
                    with st.spinner("Initializing CTGAN and preprocessing..."):
                        try:
                            from wrappers.ctgan_wrapper import CTGANHandler
                            
                            handler = CTGANHandler()
                            st.session_state.handler = handler
                            
                            # Load data with selected columns (wrapper handles negative value removal)
                            handler.load_data(
                                df, 
                                include_cols=include_cols,
                                remove_negative_rows=True,
                                numerical_cols=num_cols if num_cols else None
                            )
                            
                            # Show warning if rows were removed
                            if hasattr(handler, 'removed_negative_rows') and handler.removed_negative_rows > 0:
                                st.warning(f"Removed {handler.removed_negative_rows} rows with negative values in numerical columns")
                            
                            handler.detect_metadata()
                            
                            handler.configure_metadata(
                                numerical_float=num_cols if num_cols else None,
                                categorical=cat_cols if cat_cols else None,
                                boolean=bool_cols if bool_cols else None
                            )
                            
                            st.session_state.configured = True
                            st.success(f"Configured! {len(include_cols)} columns ready for CTGAN training")
                            
                            # Show metadata summary as a table
                            st.write("### Metadata Schema")
                            summary = handler.get_metadata_summary()
                            
                            meta_col1, meta_col2, meta_col3 = st.columns(3)
                            with meta_col1:
                                st.metric("Numerical", len(summary['numerical_float']))
                                if summary['numerical_float']:
                                    st.write(", ".join(summary['numerical_float']))
                            with meta_col2:
                                st.metric("Categorical", len(summary['categorical']))
                                if summary['categorical']:
                                    st.write(", ".join(summary['categorical']))
                            with meta_col3:
                                st.metric("Boolean", len(summary['boolean']))
                                if summary['boolean']:
                                    st.write(", ".join(summary['boolean']))
                            
                        except Exception as e:
                            st.error(f"Configuration failed: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    # TabDDPM configuration
                    with st.spinner("Initializing TabDDPM and preprocessing..."):
                        try:
                            from wrappers.tabddpm_wrapper import TabDDPMHandler
                            
                            handler = TabDDPMHandler()
                            st.session_state.handler = handler
                            
                            # Preprocess
                            result = handler.preprocess(
                                df,
                                target_col=target_col,
                                cat_cols=cat_cols if cat_cols else None,
                                columns_to_keep=feature_cols
                            )
                            
                            # Create config with quick settings
                            handler.create_config(
                                steps=200,          # Quick test
                                num_timesteps=50,   # Quick test
                                d_layers=[128, 128],
                                batch_size=256,
                                num_samples=min(result['train'], 500)
                            )
                            
                            st.session_state.configured = True
                            st.success(f"Preprocessed! Train: {result['train']}, Val: {result['val']}, Test: {result['test']}")
                            
                        except Exception as e:
                            st.error(f"Configuration failed: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
        
        # Show configuration status
        if st.session_state.configured:
            st.write("## 5. Generate Synthetic Data")
            
            handler = st.session_state.handler
            is_ctgan = st.session_state.selected_model == "ctgan"
            
            if is_ctgan:
                st.info(f"Ready to train CTGAN | Columns: {len(handler.included_cols)}")
                
                # CTGAN parameters
                with st.expander("CTGAN Parameters", expanded=False):
                    epochs = st.slider("Epochs", min_value=100, max_value=3000, value=700, step=100)
                    batch_size = st.selectbox("Batch Size", [250, 500, 1000], index=0)
                    num_samples = st.number_input("Number of samples to generate", min_value=100, max_value=100000, value=len(df))
                
            else:
                st.info(f"Ready to train TabDDPM | Num features: {len(handler.num_features)} | Cat features: {len(handler.cat_features)}")
                
                # Show which mode was selected in step 3
                if "Quick" in training_mode:
                    st.write("**Mode:** Quick tune (2 trials)")
                elif "Full" in training_mode:
                    st.write("**Mode:** Full tune (50 trials)")
                elif "pre-tuned" in training_mode:
                    st.write("**Mode:** Using pre-tuned config")
                
                num_samples = st.number_input(
                    "Number of samples to generate", 
                    min_value=100, 
                    max_value=100000, 
                    value=len(df),
                    key="tabddpm_num_samples"
                )
            
            if st.button("Start Generation"):
                st.session_state.training_log = []
                log_placeholder = st.empty()
                
                if is_ctgan:
                    # CTGAN training
                    with st.spinner("Training CTGAN model..."):
                        try:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.text("Creating synthesizer...")
                            handler.create_synthesizer(
                                epochs=epochs,
                                batch_size=batch_size
                            )
                            progress_bar.progress(10)
                            
                            status_text.text(f"Training CTGAN for {epochs} epochs...")
                            handler.fit()
                            progress_bar.progress(80)
                            
                            status_text.text(f"Generating {num_samples} synthetic samples...")
                            synthetic_df = handler.sample(num_rows=num_samples)
                            progress_bar.progress(100)
                            
                            st.session_state.synthetic_df = synthetic_df
                            st.session_state.generated = True
                            status_text.text("")
                            st.success(f"Generated {len(synthetic_df):,} synthetic samples!")
                            
                        except Exception as e:
                            st.error(f"Generation failed: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    # TabDDPM training
                    with st.spinner("Training TabDDPM model..."):
                        try:
                            import subprocess
                            
                            # Use the .conda environment
                            conda_env = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.conda'))
                            
                            if "Quick" in training_mode:
                                # Quick tune
                                tune_script = os.path.join(handler.lib_path, 'scripts', 'tune_ddpm_quick.py')
                                cmd = [
                                    'conda', 'run', '-p', conda_env,
                                    'python', '-u',
                                    tune_script,
                                    'custom',
                                    str(handler.train_size),
                                    'synthetic',
                                    'catboost',
                                    'ddpm_tune',
                                    '--n_trials', '2'
                                ]
                            elif "Full" in training_mode:
                                # Full tune
                                tune_script = os.path.join(handler.lib_path, 'scripts', 'tune_ddpm.py')
                                cmd = [
                                    'conda', 'run', '-p', conda_env,
                                    'python', '-u',
                                    tune_script,
                                    'custom',
                                    str(handler.train_size),
                                    'synthetic',
                                    'catboost',
                                    'ddpm_tune',
                                    '--eval_seeds'
                                ]
                            elif "pre-tuned" in training_mode:
                                # Use pre-tuned config - run pipeline directly
                                pipeline_script = os.path.join(handler.lib_path, 'scripts', 'pipeline.py')
                                
                                # Update config with current dataset info
                                config = st.session_state.tabddpm_config.copy()
                                
                                # Update parent_dir and data path
                                config['parent_dir'] = 'exp/custom'
                                config['real_data_path'] = 'data/custom/'  # Data files are in data/custom/
                                
                                # Update device - use GPU if available, otherwise CPU
                                import torch
                                config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                                
                                # Update feature counts
                                config['num_numerical_features'] = len(handler.num_features)
                                
                                # Update model input dimension (num + cat features)
                                total_features = len(handler.num_features) + len(handler.cat_features)
                                if 'model_params' in config:
                                    config['model_params']['d_in'] = total_features
                                
                                # Update sample size if needed
                                if 'sample' in config:
                                    config['sample']['num_samples'] = num_samples
                                
                                # Save the updated config to the expected location
                                config_path = os.path.join(handler.lib_path, 'exp', 'custom', 'config.toml')
                                with open(config_path, 'w') as f:
                                    toml.dump(config, f)
                                
                                # Check if user uploaded a trained model
                                skip_training = st.session_state.tabddpm_model_uploaded is not None
                                
                                if skip_training:
                                    # Save uploaded model to expected location
                                    model_dir = os.path.join(handler.lib_path, 'exp', 'custom')
                                    os.makedirs(model_dir, exist_ok=True)
                                    model_path = os.path.join(model_dir, 'model.pt')
                                    
                                    with open(model_path, 'wb') as f:
                                        f.write(st.session_state.tabddpm_model_uploaded.getvalue())
                                    
                                    st.info("Using uploaded model - skipping training")
                                    
                                    cmd = [
                                        'conda', 'run', '-p', conda_env,
                                        'python', '-u',
                                        pipeline_script,
                                        '--config', config_path,
                                        '--sample'  # Only sample, no training
                                    ]
                                else:
                                    st.info("No model uploaded - will train then generate")
                                    
                                    cmd = [
                                        'conda', 'run', '-p', conda_env,
                                        'python', '-u',  # Unbuffered output
                                        pipeline_script,
                                        '--config', config_path,
                                        '--train', '--sample'
                                    ]
                            
                            # Set up environment
                            env = os.environ.copy()
                            env['PYTHONPATH'] = handler.lib_path
                            env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
                            
                            st.write(f"Running: `{' '.join(cmd)}`")
                            st.info("Training started... This may take several minutes. Watch the log below for progress.")
                            
                            # Run the tuning process
                            process = subprocess.Popen(
                                cmd,
                                cwd=handler.lib_path,
                                env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                bufsize=1,
                                encoding='utf-8',
                                errors='replace'
                            )
                            
                            # Stream output with more frequent updates
                            import time
                            last_update = time.time()
                            for line in iter(process.stdout.readline, ''):
                                if line.strip():  # Only log non-empty lines
                                    st.session_state.training_log.append(line.strip())
                                # Update display every 0.5 seconds or every 5 lines
                                if time.time() - last_update > 0.5 or len(st.session_state.training_log) % 5 == 0:
                                    log_placeholder.code('\n'.join(st.session_state.training_log[-15:]))
                                    last_update = time.time()
                            
                            process.wait()
                            
                            if process.returncode == 0:
                                # Load synthetic data
                                best_dir = os.path.join(handler.lib_path, 'exp', 'custom', 'ddpm_tune_best')
                                
                                X_num = np.load(os.path.join(best_dir, 'X_num_train.npy'))
                                X_cat = np.load(os.path.join(best_dir, 'X_cat_train.npy'), allow_pickle=True)
                                y = np.load(os.path.join(best_dir, 'y_train.npy'))
                                
                                X = np.concatenate([X_num, X_cat], axis=1)
                                feature_names = handler.num_features + handler.cat_features
                                
                                synthetic_df = pd.DataFrame(X, columns=feature_names)
                                
                                # Only add target column if it's not already in feature_names
                                if handler.target_col not in feature_names:
                                    synthetic_df[handler.target_col] = y.flatten()
                                
                                st.session_state.synthetic_df = synthetic_df
                                st.session_state.generated = True
                                st.success(f"Generated {len(synthetic_df)} synthetic samples!")
                            else:
                                st.error(f"Training failed with code {process.returncode}")
                                st.code('\n'.join(st.session_state.training_log[-20:]))
                                
                        except Exception as e:
                            st.error(f"Generation failed: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

# Show generated synthetic data
if st.session_state.generated and st.session_state.synthetic_df is not None:
    st.write("### Generated Synthetic Data")
    
    synthetic_df = st.session_state.synthetic_df
    st.dataframe(synthetic_df.head(20), use_container_width=True)
    st.write(f"**Total rows generated:** {len(synthetic_df)}")
    
    # Download buttons
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        csv = synthetic_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Synthetic Data (CSV)",
            data=csv,
            file_name='synthetic_network_data.csv',
            mime='text/csv',
        )
    with dl_col2:
        # Download model button
        import pickle
        import io
        
        handler = st.session_state.handler
        loaded_synth = st.session_state.loaded_synthesizer
        
        # Case 1: Newly trained CTGAN model
        if st.session_state.selected_model == "ctgan" and handler is not None and handler.synthesizer is not None:
            model_buffer = io.BytesIO()
            pickle.dump(handler.synthesizer, model_buffer)
            model_buffer.seek(0)
            st.download_button(
                label="📥 Download Trained Model (PKL)",
                data=model_buffer,
                file_name='ctgan_model.pkl',
                mime='application/octet-stream',
            )
        # Case 2: Loaded model (already a synthesizer)
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

# =========================================================================
# Data Visualizations Section
# =========================================================================
st.write("## Data Visualizations")

if st.session_state.generated and st.session_state.synthetic_df is not None:
    synthetic_df = st.session_state.synthetic_df
    original_df = st.session_state.original_df
    
    # Statistics comparison
    st.write("### 📊 Statistics Comparison")
    
    numeric_cols = synthetic_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create stats comparison table
    stats_data = []
    for col in numeric_cols:
        if col in original_df.columns:
            orig_col = pd.to_numeric(original_df[col], errors='coerce').dropna()
            synth_col = pd.to_numeric(synthetic_df[col], errors='coerce').dropna()
            
            stats_data.append({
                'Column': col,
                'Orig Mean': f"{orig_col.mean():.2f}",
                'Synth Mean': f"{synth_col.mean():.2f}",
                'Orig Std': f"{orig_col.std():.2f}",
                'Synth Std': f"{synth_col.std():.2f}",
                'Orig Min': f"{orig_col.min():.2f}",
                'Synth Min': f"{synth_col.min():.2f}",
                'Orig Max': f"{orig_col.max():.2f}",
                'Synth Max': f"{synth_col.max():.2f}",
            })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Categorical columns comparison
    cat_cols = synthetic_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        st.write("### 📋 Categorical Distribution")
        for col in cat_cols[:3]:  # Show first 3 categorical columns
            if col in original_df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{col} (Original)**")
                    orig_counts = original_df[col].value_counts().head(10)
                    st.bar_chart(orig_counts)
                with col2:
                    st.write(f"**{col} (Synthetic)**")
                    synth_counts = synthetic_df[col].value_counts().head(10)
                    st.bar_chart(synth_counts)
    
    # Distribution histograms
    st.write("### 📈 Distribution Comparison")
    
    # Let user select which column to view
    selected_col = st.selectbox("Select column to visualize", options=numeric_cols)
    
    if selected_col and selected_col in original_df.columns:
        fig = go.Figure()
        
        orig_data = pd.to_numeric(original_df[selected_col], errors='coerce').dropna()
        synth_data = pd.to_numeric(synthetic_df[selected_col], errors='coerce').dropna()
        
        fig.add_trace(go.Histogram(
            x=orig_data,
            name='Original',
            opacity=0.6,
            nbinsx=50,
            marker_color='blue'
        ))
        
        fig.add_trace(go.Histogram(
            x=synth_data,
            name='Synthetic',
            opacity=0.6,
            nbinsx=50,
            marker_color='orange'
        ))
        
        fig.update_layout(
            barmode='overlay',
            height=400,
            title=f"Distribution: {selected_col}",
            xaxis_title=selected_col,
            yaxis_title="Count",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plot comparison
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=orig_data, name='Original', marker_color='blue'))
        fig_box.add_trace(go.Box(y=synth_data, name='Synthetic', marker_color='orange'))
        fig_box.update_layout(
            height=300,
            title=f"Box Plot: {selected_col}",
            yaxis_title=selected_col
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Correlation heatmap comparison
    st.write("### 🔗 Correlation Comparison")
    
    corr_cols = numeric_cols[:8]  # Limit to 8 columns for readability
    if len(corr_cols) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Data**")
            orig_numeric = original_df[corr_cols].apply(pd.to_numeric, errors='coerce').dropna()
            if len(orig_numeric) > 0:
                orig_corr = orig_numeric.corr()
                fig_corr_orig = px.imshow(
                    orig_corr,
                    labels=dict(color="Correlation"),
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    aspect='auto'
                )
                fig_corr_orig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_corr_orig, use_container_width=True)
        
        with col2:
            st.write("**Synthetic Data**")
            synth_numeric = synthetic_df[corr_cols].apply(pd.to_numeric, errors='coerce').dropna()
            if len(synth_numeric) > 0:
                synth_corr = synth_numeric.corr()
                fig_corr_synth = px.imshow(
                    synth_corr,
                    labels=dict(color="Correlation"),
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    aspect='auto'
                )
                fig_corr_synth.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_corr_synth, use_container_width=True)
    
    # Scatter plot for relationship exploration
    st.write("### 🔍 Relationship Explorer")
    scatter_col1, scatter_col2 = st.columns(2)
    with scatter_col1:
        x_col = st.selectbox("X-axis", options=numeric_cols, index=0, key="scatter_x")
    with scatter_col2:
        y_col = st.selectbox("Y-axis", options=numeric_cols, index=min(1, len(numeric_cols)-1), key="scatter_y")
    
    if x_col and y_col and x_col in original_df.columns and y_col in original_df.columns:
        fig_scatter = go.Figure()
        
        # Sample if too many points
        max_points = 1000
        orig_sample = original_df[[x_col, y_col]].dropna().sample(n=min(max_points, len(original_df)), random_state=42)
        synth_sample = synthetic_df[[x_col, y_col]].apply(pd.to_numeric, errors='coerce').dropna()
        if len(synth_sample) > max_points:
            synth_sample = synth_sample.sample(n=max_points, random_state=42)
        
        fig_scatter.add_trace(go.Scatter(
            x=orig_sample[x_col], y=orig_sample[y_col],
            mode='markers', name='Original',
            marker=dict(color='blue', opacity=0.5, size=5)
        ))
        fig_scatter.add_trace(go.Scatter(
            x=synth_sample[x_col], y=synth_sample[y_col],
            mode='markers', name='Synthetic',
            marker=dict(color='orange', opacity=0.5, size=5)
        ))
        
        fig_scatter.update_layout(
            height=400,
            title=f"{x_col} vs {y_col}",
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
elif st.session_state.uploaded:
    st.info("Configure and generate synthetic data to view comparisons")
else:
    st.info("Upload data to get started")

# Training log section
if st.session_state.training_log:
    with st.expander("Training Log", expanded=False):
        st.code('\n'.join(st.session_state.training_log))