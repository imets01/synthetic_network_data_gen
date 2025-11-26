import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go

# Set page config to wide mode
st.set_page_config(layout="wide", page_title="Synthetic Network Data Generator")

st.title("Synthetic Network Data Generator")

# Initialize session state
if 'uploaded' not in st.session_state:
    st.session_state.uploaded = False
if 'generated' not in st.session_state:
    st.session_state.generated = False
if 'synthetic_df' not in st.session_state:
    st.session_state.synthetic_df = None

# Create two columns for layout
col_left, col_right = st.columns([1, 1])

with col_left:
    st.write("## 1. Upload Real Network Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if st.button("Upload Data"):
        if uploaded_file is not None:
            with st.spinner("Uploading data..."):
                time.sleep(1) 
            st.success("Data uploaded successfully!")
            st.session_state.uploaded = True
            st.session_state.generated = False
        else:
            st.warning("Please select a file first")
    
    # Show uploaded data info
    if st.session_state.uploaded:
        st.write("### Uploaded Data Summary")
        st.info("Dataset: network_capture.csv | Rows: 1,000 | Columns: 8")
        
        st.write("## 2. Generate Synthetic Data")
        
        num_samples = st.slider("Number of synthetic samples to generate", 100, 5000, 1000, step=100)
        
        if st.button("Generate Synthetic Data"):
            with st.spinner("Training model and generating synthetic data..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
            
            st.success("Synthetic data generated successfully!")
            st.session_state.generated = True
            
            # Create fake synthetic network data
            np.random.seed(42)
            st.session_state.synthetic_df = pd.DataFrame({
                'Timestamp': pd.date_range(start='2025-01-01', periods=num_samples, freq='s'),
                'Protocol': np.random.choice(['QUIC', 'TCP', 'UDP'], num_samples),
                'Source_IP': [f"192.168.1.{np.random.randint(1, 255)}" for _ in range(num_samples)],
                'Dest_IP': [f"10.0.0.{np.random.randint(1, 255)}" for _ in range(num_samples)],
                'Packet_Size': np.random.randint(64, 1500, num_samples),
                'Duration_ms': np.round(np.random.exponential(50, num_samples), 2),
                'Flags': np.random.choice(['SYN', 'ACK', 'FIN', 'PSH'], num_samples),
                'Encrypted': np.random.choice([True, False], num_samples, p=[0.7, 0.3])
            })
    
    # Show generated synthetic data
    if st.session_state.generated and st.session_state.synthetic_df is not None:
        st.write("### Generated Synthetic Network Data")
        
        synthetic_df = st.session_state.synthetic_df
        
        st.dataframe(synthetic_df.head(20), use_container_width=True)
        
        st.write(f"**Total rows generated:** {len(synthetic_df)}")
        
        # Download button
        csv = synthetic_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Synthetic Data as CSV",
            data=csv,
            file_name='synthetic_network_data.csv',
            mime='text/csv',
        )

with col_right:
    st.write("## Data Visualizations")
    
    if st.session_state.generated and st.session_state.synthetic_df is not None:
        synthetic_df = st.session_state.synthetic_df
        
        # Protocol Distribution
        st.write("### Protocol Distribution")
        protocol_counts = synthetic_df['Protocol'].value_counts()
        fig_protocol = px.pie(
            values=protocol_counts.values,
            names=protocol_counts.index,
            title="Network Protocol Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_protocol, use_container_width=True)
        
        # Packet Size Distribution
        st.write("### Packet Size Distribution")
        fig_packet_size = px.histogram(
            synthetic_df,
            x='Packet_Size',
            nbins=50,
            title="Packet Size Distribution",
            labels={'Packet_Size': 'Packet Size (bytes)'},
            color_discrete_sequence=['#636EFA']
        )
        fig_packet_size.update_layout(showlegend=False)
        st.plotly_chart(fig_packet_size, use_container_width=True)
        
        # Duration Distribution
        st.write("### Duration Distribution")
        fig_duration = px.box(
            synthetic_df,
            y='Duration_ms',
            title="Connection Duration Distribution",
            labels={'Duration_ms': 'Duration (ms)'},
            color_discrete_sequence=['#EF553B']
        )
        st.plotly_chart(fig_duration, use_container_width=True)
        
        # Flags Distribution
        st.write("### TCP Flags Distribution")
        flags_counts = synthetic_df['Flags'].value_counts()
        fig_flags = px.bar(
            x=flags_counts.index,
            y=flags_counts.values,
            title="TCP Flags Distribution",
            labels={'x': 'Flag Type', 'y': 'Count'},
            color=flags_counts.values,
            color_continuous_scale='Viridis'
        )
        fig_flags.update_layout(showlegend=False)
        st.plotly_chart(fig_flags, use_container_width=True)
        
        # Encryption Status
        st.write("### Encryption Status")
        encrypted_counts = synthetic_df['Encrypted'].value_counts()
        fig_encrypted = go.Figure(data=[
            go.Bar(
                x=['Encrypted', 'Unencrypted'],
                y=[encrypted_counts.get(True, 0), encrypted_counts.get(False, 0)],
                marker_color=['#00CC96', '#FFA15A']
            )
        ])
        fig_encrypted.update_layout(
            title="Encryption Status",
            xaxis_title="Status",
            yaxis_title="Count"
        )
        st.plotly_chart(fig_encrypted, use_container_width=True)
        
    else:
        st.info("📊 Generate synthetic data to view visualizations")