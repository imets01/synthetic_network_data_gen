import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Get columns that contain numeric data, even if stored as object dtype.
    
    This handles cases where numeric data is loaded from .npy files
    and ends up as object dtype.
    """
    numeric_cols = []
    for col in df.columns:
        # First check if already numeric dtype
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            # Try to convert and check if most values are numeric
            try:
                converted = pd.to_numeric(df[col], errors='coerce')
                # If more than 50% of non-null values are numeric, consider it numeric
                valid_ratio = converted.notna().sum() / max(df[col].notna().sum(), 1)
                if valid_ratio > 0.5:
                    numeric_cols.append(col)
            except:
                pass
    return numeric_cols


def get_categorical_columns(df: pd.DataFrame, numeric_cols: list = None) -> list:
    """
    Get categorical columns (non-numeric columns).
    """
    if numeric_cols is None:
        numeric_cols = get_numeric_columns(df)
    return [col for col in df.columns if col not in numeric_cols]


def render_statistics_comparison(original_df: pd.DataFrame, synthetic_df: pd.DataFrame):
    """Render statistics comparison table between original and synthetic data."""
    st.write("### 📊 Statistics Comparison")
    
    numeric_cols = get_numeric_columns(synthetic_df)
    
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


def render_categorical_comparison(original_df: pd.DataFrame, synthetic_df: pd.DataFrame, max_cols: int = 3):
    """Render categorical distribution comparison."""
    numeric_cols = get_numeric_columns(synthetic_df)
    cat_cols = get_categorical_columns(synthetic_df, numeric_cols)
    
    if not cat_cols:
        return
    
    st.write("### 📋 Categorical Distribution")
    
    for col in cat_cols[:max_cols]:
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


def render_distribution_comparison(original_df: pd.DataFrame, synthetic_df: pd.DataFrame):
    """Render distribution comparison with histogram and box plot."""
    st.write("### 📈 Distribution Comparison")
    
    numeric_cols = get_numeric_columns(synthetic_df)
    
    if not numeric_cols:
        st.info("No numeric columns to visualize")
        return
    
    selected_col = st.selectbox("Select column to visualize", options=numeric_cols)
    
    if selected_col and selected_col in original_df.columns:
        orig_data = pd.to_numeric(original_df[selected_col], errors='coerce').dropna()
        synth_data = pd.to_numeric(synthetic_df[selected_col], errors='coerce').dropna()
        
        # Histogram
        fig = go.Figure()
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
        
        # Box plot
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=orig_data, name='Original', marker_color='blue'))
        fig_box.add_trace(go.Box(y=synth_data, name='Synthetic', marker_color='orange'))
        fig_box.update_layout(
            height=300,
            title=f"Box Plot: {selected_col}",
            yaxis_title=selected_col
        )
        st.plotly_chart(fig_box, use_container_width=True)


def render_correlation_comparison(original_df: pd.DataFrame, synthetic_df: pd.DataFrame, max_cols: int = 8):
    """Render correlation heatmap comparison."""
    st.write("### 🔗 Correlation Comparison")
    
    numeric_cols = get_numeric_columns(synthetic_df)
    corr_cols = [c for c in numeric_cols[:max_cols] if c in original_df.columns]
    
    if len(corr_cols) < 2:
        st.info("Need at least 2 numeric columns for correlation comparison")
        return
    
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


def render_scatter_comparison(original_df: pd.DataFrame, synthetic_df: pd.DataFrame, max_points: int = 1000):
    """Render scatter plot comparison for relationship exploration."""
    st.write("### 🔍 Relationship Explorer")
    
    numeric_cols = get_numeric_columns(synthetic_df)
    
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns for scatter plot")
        return
    
    scatter_col1, scatter_col2 = st.columns(2)
    with scatter_col1:
        x_col = st.selectbox("X-axis", options=numeric_cols, index=0, key="scatter_x")
    with scatter_col2:
        y_col = st.selectbox("Y-axis", options=numeric_cols, index=min(1, len(numeric_cols)-1), key="scatter_y")
    
    if x_col and y_col and x_col in original_df.columns and y_col in original_df.columns:
        fig_scatter = go.Figure()
        
        # Sample if too many points
        orig_sample = original_df[[x_col, y_col]].dropna()
        if len(orig_sample) > max_points:
            orig_sample = orig_sample.sample(n=max_points, random_state=42)
        
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


def render_all_comparisons(original_df: pd.DataFrame, synthetic_df: pd.DataFrame):
    """Render all comparison visualizations."""
    render_statistics_comparison(original_df, synthetic_df)
    render_categorical_comparison(original_df, synthetic_df)
    render_distribution_comparison(original_df, synthetic_df)
    render_correlation_comparison(original_df, synthetic_df)
    render_scatter_comparison(original_df, synthetic_df)
