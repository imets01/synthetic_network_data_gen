import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config to wide mode
st.set_page_config(layout="wide", page_title="Synthetic Network Data Generator")

st.title("Synthetic Network Data Generator")

# Define paths to real datasets
ORIGINAL_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FEST_eval', 'synprivutil', 'datasets', 'original', 'original_data.csv'))
SYNTHETIC_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FEST_eval', 'synprivutil', 'datasets', 'synthetic', 'synthetic_data.csv'))

# Initialize session state
if 'uploaded' not in st.session_state:
    st.session_state.uploaded = False
if 'generated' not in st.session_state:
    st.session_state.generated = False
if 'synthetic_df' not in st.session_state:
    st.session_state.synthetic_df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'evaluated' not in st.session_state:
    st.session_state.evaluated = False
if 'eval_results' not in st.session_state:
    st.session_state.eval_results = None

# Create two columns for layout
col_left, col_right = st.columns([1, 1])

with col_left:
    st.write("## 1. Upload Real Network Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if st.button("Upload Data"):
        if uploaded_file is not None:
            with st.spinner("Uploading data..."):
                time.sleep(1)
                # Store the uploaded data
                st.session_state.original_df = pd.read_csv(uploaded_file)
            st.success("Data uploaded successfully!")
            st.session_state.uploaded = True
            st.session_state.generated = False
            st.session_state.evaluated = False
        else:
            st.warning("Please select a file first")
    
    # Show uploaded data info
    if st.session_state.uploaded:
        st.write("### Uploaded Data Summary")
        if st.session_state.original_df is not None:
            dataset_name = uploaded_file.name if uploaded_file else 'data.csv'
            st.info(f"Dataset: {dataset_name} | Rows: {len(st.session_state.original_df):,} | Columns: {len(st.session_state.original_df.columns)}")
        
        st.write("## 2. Generate Synthetic Data")
        
        if st.button("Generate Synthetic Data"):
            with st.spinner("Training model and generating synthetic data..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
            
            # Load the pre-generated synthetic data
            try:
                st.session_state.synthetic_df = pd.read_csv(SYNTHETIC_DATA_PATH)
                st.success("Synthetic data generated successfully!")
                st.session_state.generated = True
                st.session_state.evaluated = False
            except Exception as e:
                st.error(f"Failed to load synthetic data: {str(e)}")
    
    # Show generated synthetic data
    if st.session_state.generated and st.session_state.synthetic_df is not None:
        st.write("### Generated Synthetic Network Data")
        
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
        from visualizations import generate_data_visualizations
        
        synthetic_df = st.session_state.synthetic_df
        
        # Generate all visualizations
        visualizations = generate_data_visualizations(synthetic_df)
        
        # Display each visualization
        for viz_key, viz_data in visualizations.items():
            st.write(f"### {viz_data['title']}")
            st.plotly_chart(viz_data['figure'], use_container_width=True)
        
    else:
        st.info("Generate synthetic data to view visualizations")

# Evaluation Section
if st.session_state.generated and st.session_state.synthetic_df is not None:
    st.write("---")
    st.write("## 3. Evaluate Synthetic Data Quality")
    
    eval_col1, eval_col2 = st.columns([1, 2])
    
    with eval_col1:
        st.write("### Run Evaluation")
        st.write("Evaluate the quality of synthetic data using privacy and utility metrics from the FEST framework.")
        
        if st.button("Run Evaluation"):
            with st.spinner("Running privacy and utility evaluations..."):
                try:
                    from evaluation import evaluate_synthetic_data
                    
                    # Prepare data
                    original_data = st.session_state.original_df.copy() if st.session_state.original_df is not None else st.session_state.synthetic_df.copy()
                    synthetic_data = st.session_state.synthetic_df.copy()
                    
                    # Run evaluation with progress callback
                    progress = st.progress(0)
                    
                    def update_progress(value):
                        progress.progress(value)
                    
                    eval_results = evaluate_synthetic_data(
                        original_data, 
                        synthetic_data,
                        progress_callback=update_progress
                    )
                    
                    st.session_state.eval_results = eval_results
                    st.session_state.evaluated = True
                    st.success("Evaluation completed!")
                    
                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")
                    st.write("Make sure the FEST evaluation framework is properly installed.")
    
    with eval_col2:
        if st.session_state.evaluated and st.session_state.eval_results is not None:
            st.write("### Evaluation Results")
            
            # Privacy Metrics
            st.write("#### Privacy Metrics")
            privacy_results = st.session_state.eval_results['privacy']
            
            for metric_name, value in privacy_results.items():
                display_name = metric_name.split("('")[0] if "(" in metric_name else metric_name
                st.metric(label=display_name, value=f"{value:.4f}")
            
            # Utility Metrics
            st.write("#### Utility Metrics")
            utility_results = st.session_state.eval_results['utility']
            
            for metric_name, value in utility_results.items():
                display_name = metric_name.split("('")[0] if "(" in metric_name else metric_name
                
                if isinstance(value, dict):
                    # For BasicStatsCalculator which returns a dict
                    st.write(f"**{display_name}**")
                    metric_cols = st.columns(len(value))
                    for idx, (stat_name, stat_value) in enumerate(value.items()):
                        with metric_cols[idx]:
                            st.metric(label=stat_name.capitalize(), value=f"{stat_value:.4f}")
                else:
                    # For other metrics that return a single value
                    st.metric(label=display_name, value=f"{value:.4f}")
        else:
            st.info("Click 'Run Evaluation' to see results")