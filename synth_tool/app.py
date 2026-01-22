import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

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

# Create two columns for layout
col_left, col_right = st.columns([1, 1])

with col_left:
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
        
        st.write("## 2. Configure Columns")
        
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
            'bytes_sent_client', 'bytes_sent_server',
        ]
        DEFAULT_CATEGORICAL = [
            'implementation', 'version_negotiation_occurred', 'retry_occurred', 'migration_type',
            'path_validation_initiated', 'connection_close_type',
        ]
        
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
            default_feature_selection = available_features[:10]  # Fallback to first 10
        
        feature_cols = st.multiselect(
            "Feature Columns to Keep",
            options=available_features,
            default=default_feature_selection
        )
        
        # Categorical columns selection - default to notebook categorical if available
        default_cat_selection = [c for c in DEFAULT_CATEGORICAL if c in feature_cols]
        if not default_cat_selection:
            default_cat_selection = [c for c in feature_cols if df[c].dtype == 'object' or df[c].nunique() <= 10][:5]
        
        cat_cols = st.multiselect(
            "Categorical Columns (from features)",
            options=feature_cols,
            default=default_cat_selection
        )
        
        if st.button("Configure & Preprocess"):
            if not feature_cols:
                st.error("Please select at least one feature column")
            else:
                with st.spinner("Initializing TabDDPM and preprocessing..."):
                    try:
                        from src.tabddpm_wrapper import TabDDPMHandler
                        
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
            st.write("## 3. Generate Synthetic Data")
            
            handler = st.session_state.handler
            st.info(f"Ready to train | Num features: {len(handler.num_features)} | Cat features: {len(handler.cat_features)}")
            
            generation_mode = st.radio(
                "Generation Mode",
                ["Quick Test (2 trials, ~5 min)", "Full Tuning (50 trials, ~hours)"],
                index=0
            )
            
            if st.button("Start Generation"):
                st.session_state.training_log = []
                log_placeholder = st.empty()
                
                with st.spinner("Training TabDDPM model..."):
                    try:
                        import subprocess
                        
                        # Use the .conda environment
                        conda_env = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.conda'))
                        
                        if "Quick" in generation_mode:
                            # Quick tune
                            tune_script = os.path.join(handler.lib_path, 'scripts', 'tune_ddpm_quick.py')
                            cmd = [
                                'conda', 'run', '-p', conda_env, '--no-capture-output',
                                'python',
                                tune_script,
                                'custom',
                                str(handler.train_size),
                                'synthetic',
                                'catboost',
                                'ddpm_tune',
                                '--n_trials', '2'
                            ]
                        else:
                            # Full tune
                            tune_script = os.path.join(handler.lib_path, 'scripts', 'tune_ddpm.py')
                            cmd = [
                                'conda', 'run', '-p', conda_env, '--no-capture-output',
                                'python',
                                tune_script,
                                'custom',
                                str(handler.train_size),
                                'synthetic',
                                'catboost',
                                'ddpm_tune',
                                '--eval_seeds'
                            ]
                        
                        # Set up environment
                        env = os.environ.copy()
                        env['PYTHONPATH'] = handler.lib_path
                        
                        st.write(f"Running: `{' '.join(cmd)}`")
                        
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
                        
                        # Stream output
                        for line in iter(process.stdout.readline, ''):
                            st.session_state.training_log.append(line.strip())
                            # Show last 10 lines
                            log_placeholder.code('\n'.join(st.session_state.training_log[-10:]))
                        
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
        
        # Download button
        csv = synthetic_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Synthetic Data as CSV",
            data=csv,
            file_name='synthetic_network_data.csv',
            mime='text/csv',
        )

with col_right:
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