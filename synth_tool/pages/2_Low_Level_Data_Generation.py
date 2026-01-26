"""
Low-Level Synthetic Data Generation with WGAN-GP

Generates synthetic packet-level network data using a Wasserstein GAN with Gradient Penalty.
Supports both loading pre-trained models and training new models from scratch.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import torch
import tempfile
import io
import zipfile
import joblib
import json
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from wrappers.wgan_wrapper import WGANHandler, Generator

# =============================================================================
# Constants
# =============================================================================
DEFAULT_SEQ_LENGTH = 40
CONNECTION_CLOSE_COL = 'count_connection_close'

CONDITION_COLUMNS = [
            'implementation', 'connection_duration',  'version_negotiation_occurred', 'retry_occurred', 'migration_type',
            #'first_path_validation_response_latency', #'path_validation_initiated',
              'packets_sent_client', 
            'packets_sent_server', 'handshake_duration', 'time_to_migration', 'migration_duration', # Corrected
            'packets_before_migration', #'total_bidi_streams_client_init',
            #'total_udi_streams_client_init',
            # 'connection_close_type'
]

CATEGORICAL_COLUMNS = ['migration_type', 'implementation']


# =============================================================================
# Helper Functions
# =============================================================================
def init_session_state():
    """Initialize all session state variables with defaults."""
    defaults = {
        'wgan_handler': None,
        'wgan_data_loaded': False,
        'wgan_model_trained': False,
        'wgan_generated_data': None,
        'wgan_mode': None,
        'model_loaded': False,
        'high_level_df': None,
        'generator': None,
        'sequence_scaler': None,
        'condition_scaler': None,
        'sequence_columns': None,
        'device': None,
        'latent_dim': None,
        'wgan_capture_ids': None,  # Store capture_ids for generated sequences
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def truncate_at_connection_close(sequences: np.ndarray, sequence_columns: list) -> list:
    """
    Truncate each sequence at the first connection close packet.
    
    Args:
        sequences: Generated sequences array of shape (n_samples, seq_len, n_features)
        sequence_columns: List of column names
    
    Returns:
        List of truncated sequences (variable length)
    """
    conn_close_idx = None
    if CONNECTION_CLOSE_COL in sequence_columns:
        conn_close_idx = sequence_columns.index(CONNECTION_CLOSE_COL)
    
    truncated = []
    for seq in sequences:
        if conn_close_idx is not None:
            # Find first frame where count_connection_close > 0
            close_indices = np.where(seq[:, conn_close_idx] > 0.5)[0]
            if len(close_indices) > 0:
                # Cut at first connection close (inclusive)
                truncated.append(seq[:close_indices[0] + 1])
            else:
                truncated.append(seq)
        else:
            truncated.append(seq)
    
    return truncated


def prepare_conditions(high_level_df: pd.DataFrame) -> np.ndarray:
    """
    Prepare condition vectors from high-level DataFrame.
    
    Args:
        high_level_df: DataFrame with high-level features
    
    Returns:
        Condition array ready for scaling
    """
    condition_df = high_level_df.copy()
    
    # Add missing columns with 0
    for col in CONDITION_COLUMNS:
        if col not in condition_df.columns:
            condition_df[col] = 0
    
    # Keep only required columns
    available_cols = [c for c in CONDITION_COLUMNS if c in condition_df.columns]
    condition_df = condition_df[available_cols].copy()
    
    # Handle categorical columns
    for col in CATEGORICAL_COLUMNS:
        if col in condition_df.columns and condition_df[col].dtype == 'object':
            condition_df[col] = condition_df[col].astype('category')
    
    # One-hot encode
    categorical_cols = condition_df.select_dtypes(include=['category']).columns
    condition_df = pd.get_dummies(condition_df, columns=categorical_cols, prefix=categorical_cols)
    
    return condition_df.values.astype(np.float32)


def sequences_to_dataframe(sequences: list, sequence_columns: list, capture_ids: list = None) -> pd.DataFrame:
    """
    Convert list of sequences to a flat DataFrame.
    
    Args:
        sequences: List of sequence arrays
        sequence_columns: Column names for features
        capture_ids: Optional list of capture_ids (one per sequence) to use instead of sequence_id
    
    Returns:
        DataFrame with capture_id (or sequence_id), frame_number, and feature columns
    """
    rows = []
    for seq_idx, seq in enumerate(sequences):
        # Use provided capture_id if available, otherwise use sequence index
        if capture_ids is not None and seq_idx < len(capture_ids):
            cid = capture_ids[seq_idx]
        else:
            cid = seq_idx
        
        for frame_idx, frame in enumerate(seq):
            row = {'capture_id': cid, 'frame_number': frame_idx + 1}
            for k, col in enumerate(sequence_columns):
                value = frame[k]
                # Round to integer for all columns except delta_time
                if col != 'delta_time':
                    value = int(round(value))
                row[col] = value
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Ensure proper dtypes - integers for all except delta_time
    for col in sequence_columns:
        if col in df.columns and col != 'delta_time':
            df[col] = df[col].astype(int)
    
    return df


def render_download_button(sequences: list, sequence_columns: list, key: str, capture_ids: list = None):
    """Render download button for generated sequences."""
    try:
        df = sequences_to_dataframe(sequences, sequence_columns, capture_ids=capture_ids)
        st.download_button(
            label="📥 Download CSV",
            data=df.to_csv(index=False),
            file_name="synthetic_low_level_sequences.csv",
            mime="text/csv",
            key=key
        )
    except Exception as e:
        st.error(f"Error preparing data for download: {str(e)}")


# =============================================================================
# Page Setup
# =============================================================================
st.set_page_config(layout="wide", page_title="Low-Level Synthetic Data Generation")
st.title("Low-Level Synthetic Data Generation with WGAN-GP")

init_session_state()

# ============================================================================
# Step 1: Choose Mode
# ============================================================================
st.header("1. Choose Mode")

mode = st.radio(
    "What would you like to do?",
    ["Load Pre-trained Model", "Train New Model"],
    horizontal=True,
    key="mode_selector"
)

st.session_state.wgan_mode = mode

# ============================================================================
# Mode A: Load Pre-trained Model
# ============================================================================
if mode == "Load Pre-trained Model":
    st.header("2. Upload Required Files")
    
    st.markdown("""
    - **High-Level Features CSV**: Contains conditions for generation (one sequence generated per row)
    - **Model ZIP**: Contains generator.pth, sequence_scaler.gz, condition_scaler.gz, sequence_columns.json, model_params.json
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("High-Level Features CSV")
        high_level_file = st.file_uploader(
            "Upload high-level features CSV",
            type=['csv'],
            key='high_level_csv_load',
            help="CSV file containing high-level features for conditioning"
        )
        
        if high_level_file is not None:
            st.success(f"✓ Uploaded: {high_level_file.name}")
    
    with col2:
        st.subheader("Model Files (ZIP)")
        model_zip_file = st.file_uploader(
            "Upload model ZIP file",
            type=['zip'],
            key='model_zip',
            help="ZIP containing generator.pth, scalers, sequence_columns.json, and model_params.json"
        )
        
        if model_zip_file is not None:
            st.success(f"✓ Uploaded: {model_zip_file.name}")
    
    # Load button
    if st.button("Load Model", type="primary", disabled=not (high_level_file and model_zip_file)):
        with st.spinner("Loading model and data..."):
            try:
                # Load high-level CSV
                st.session_state.high_level_df = pd.read_csv(high_level_file)
                st.session_state.high_level_df = st.session_state.high_level_df.replace([np.inf, -np.inf], np.nan).dropna(how='any')
                
                # Extract model files from ZIP
                model_zip_bytes = io.BytesIO(model_zip_file.read())
                
                with zipfile.ZipFile(model_zip_bytes, 'r') as zf:
                    # Load model parameters
                    if 'model_params.json' in zf.namelist():
                        with zf.open('model_params.json') as f:
                            model_params = json.load(f)
                        latent_dim = model_params.get('latent_dim', 20)
                        hidden_dim = model_params.get('hidden_dim', 256)
                        num_layers_gen = model_params.get('num_layers_gen', 2)
                    else:
                        # Fallback defaults if no params file
                        st.warning("No model_params.json found in ZIP, using defaults")
                        latent_dim = 20
                        hidden_dim = 256
                        num_layers_gen = 2
                    
                    # Load scalers
                    with zf.open('sequence_scaler.gz') as f:
                        sequence_scaler = joblib.load(io.BytesIO(f.read()))
                    
                    with zf.open('condition_scaler.gz') as f:
                        condition_scaler = joblib.load(io.BytesIO(f.read()))
                    
                    # Load sequence columns
                    with zf.open('sequence_columns.json') as f:
                        sequence_columns = json.load(f)
                    
                    # Load generator weights
                    with zf.open('generator.pth') as f:
                        generator_state_dict = torch.load(io.BytesIO(f.read()), map_location='cpu')
                
                # Get dimensions from scalers
                sequence_feature_dim = sequence_scaler.n_features_in_
                condition_dim = condition_scaler.n_features_in_
                
                # Set device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Create generator with matching architecture
                generator = Generator(
                    latent_dim=latent_dim,
                    condition_dim=condition_dim,
                    sequence_feature_dim=sequence_feature_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers_gen
                )
                generator.load_state_dict(generator_state_dict)
                generator.to(device)
                generator.eval()
                
                # Store everything in session state
                st.session_state.generator = generator
                st.session_state.sequence_scaler = sequence_scaler
                st.session_state.condition_scaler = condition_scaler
                st.session_state.sequence_columns = sequence_columns
                st.session_state.device = device
                st.session_state.latent_dim = latent_dim
                st.session_state.model_loaded = True
                
                st.success("✓ Model loaded successfully!")
                
                # Show info
                col_i1, col_i2, col_i3 = st.columns(3)
                with col_i1:
                    st.metric("High-Level Rows", len(st.session_state.high_level_df))
                with col_i2:
                    st.metric("Condition Dimension", condition_dim)
                with col_i3:
                    st.metric("Sequence Features", sequence_feature_dim)
                
                st.markdown(f"**Model Parameters:** Latent={latent_dim}, Hidden={hidden_dim}, Layers={num_layers_gen}")
                st.info(f"Using device: **{str(device).upper()}**")
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.exception(e)
    
    # ============================================================================
    # Generate Synthetic Data (Load Mode)
    # ============================================================================
    if st.session_state.model_loaded:
        st.header("3. Generate Synthetic Sequences")
        
        st.markdown("**One sequence will be generated for each row in the high-level CSV** (using each row as a condition).")
        st.markdown("Each generated sequence will have the same `capture_id` as the `file_id` from the high-level row used as condition.")
        
        # Show high-level data preview
        with st.expander("Preview High-Level Data"):
            st.dataframe(st.session_state.high_level_df.head(20), use_container_width=True)
        
        num_samples = len(st.session_state.high_level_df)
        st.info(f"Will generate **{num_samples}** sequences (one per high-level row, cut at first connection close)")
        
        if st.button("Generate Samples", type="primary", key="gen_samples_btn"):
            with st.spinner(f"Generating {num_samples} synthetic sequences..."):
                try:
                    # Get capture_ids from high-level data (file_id column)
                    if 'file_id' in st.session_state.high_level_df.columns:
                        capture_ids = st.session_state.high_level_df['file_id'].tolist()
                    else:
                        # Fallback to index if no file_id column
                        capture_ids = list(range(num_samples))
                        st.warning("No 'file_id' column found. Using sequential IDs.")
                    
                    # Prepare conditions using helper function
                    all_conditions = prepare_conditions(st.session_state.high_level_df)
                    
                    # Scale conditions
                    scaled_conditions = st.session_state.condition_scaler.transform(all_conditions)
                    conditions_tensor = torch.FloatTensor(scaled_conditions).to(st.session_state.device)
                    
                    # Generate sequences
                    st.session_state.generator.eval()
                    with torch.no_grad():
                        noise = torch.randn(num_samples, st.session_state.latent_dim, device=st.session_state.device)
                        generated_scaled = st.session_state.generator(noise, conditions_tensor, DEFAULT_SEQ_LENGTH)
                    
                    # Convert to numpy and inverse transform
                    generated_np = generated_scaled.cpu().numpy()
                    generated_unscaled = np.zeros_like(generated_np)
                    for i in range(generated_np.shape[0]):
                        generated_unscaled[i] = st.session_state.sequence_scaler.inverse_transform(generated_np[i])
                    
                    # Truncate at connection close using helper function
                    truncated_sequences = truncate_at_connection_close(
                        generated_unscaled, 
                        st.session_state.sequence_columns
                    )
                    
                    st.session_state.wgan_generated_data = truncated_sequences
                    st.session_state.wgan_capture_ids = capture_ids  # Store capture_ids
                    
                    # Display results
                    seq_lengths = [len(s) for s in truncated_sequences]
                    st.success(f"✓ Generated {len(truncated_sequences)} sequences!")
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Sequences", len(truncated_sequences))
                    with col_stat2:
                        st.metric("Avg Length", f"{np.mean(seq_lengths):.1f}")
                    with col_stat3:
                        st.metric("Features", truncated_sequences[0].shape[1] if truncated_sequences else 0)
                    
                except Exception as e:
                    st.error(f"Error generating samples: {str(e)}")
                    st.exception(e)
        
        # Export section
        if st.session_state.wgan_generated_data is not None:
            st.header("4. Export Generated Data")
            
            # Preview generated data
            with st.expander("Preview Generated Data"):
                preview_df = sequences_to_dataframe(
                    st.session_state.wgan_generated_data[:5],  # First 5 sequences
                    st.session_state.sequence_columns,
                    capture_ids=st.session_state.wgan_capture_ids[:5] if st.session_state.wgan_capture_ids else None
                )
                st.dataframe(preview_df.head(50), use_container_width=True)
            
            render_download_button(
                st.session_state.wgan_generated_data,
                st.session_state.sequence_columns,
                capture_ids=st.session_state.wgan_capture_ids,
                key="download_csv_load"
            )

# ============================================================================
# Mode B: Train New Model
# ============================================================================
elif mode == "Train New Model":
    st.header("2. Upload Training Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("High-Level Features CSV")
        high_level_file = st.file_uploader(
            "Upload high-level features CSV",
            type=['csv'],
            key='high_level_csv_train',
            help="CSV file containing high-level features with file_id column"
        )
        
        if high_level_file is not None:
            st.success(f"✓ Uploaded: {high_level_file.name}")
    
    with col2:
        st.subheader("Low-Level Sequences ZIP")
        low_level_file = st.file_uploader(
            "Upload low-level sequences ZIP",
            type=['zip'],
            key='low_level_zip_train',
            help="ZIP file containing individual CSV files for each sequence"
        )
        
        if low_level_file is not None:
            st.success(f"✓ Uploaded: {low_level_file.name}")
    
    # Data loading configuration
    with st.expander("⚙️ Data Loading Options"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            batch_size = st.number_input(
                "Batch Size",
                min_value=16,
                max_value=512,
                value=128,
                step=16,
                help="Number of sequences per training batch"
            )
            
            remove_negatives = st.checkbox(
                "Remove sequences with negative values",
                value=True,
                help="Filter out sequences containing negative values"
            )
        
        with col_b:
            default_workers = 0 if os.name == 'nt' else 4
            num_workers = st.number_input(
                "Number of Data Loader Workers",
                min_value=0,
                max_value=16,
                value=default_workers,
                step=1,
                help="Number of parallel workers for data loading (0 recommended on Windows)"
            )
            
            folder_in_zip = st.text_input(
                "Folder name in ZIP",
                value="separate_low_level_files",
                help="Name of the folder inside ZIP file containing CSVs"
            )
    
    # Load data button
    if st.button("Load Data", type="primary", disabled=not (high_level_file and low_level_file)):
        with st.spinner("Loading data..."):
            try:
                # Create temp files
                temp_high = tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False)
                temp_high.write(high_level_file.getbuffer())
                temp_high.close()
                high_level_path = temp_high.name
                
                temp_low = tempfile.NamedTemporaryFile(mode='wb', suffix='.zip', delete=False)
                temp_low.write(low_level_file.getbuffer())
                temp_low.close()
                low_level_zip_path = temp_low.name
                
                # Initialize handler
                device = "cuda" if torch.cuda.is_available() else "cpu"
                st.session_state.wgan_handler = WGANHandler(device=device)
                
                # Load data
                stats = st.session_state.wgan_handler.load_data(
                    high_level_csv=high_level_path,
                    low_level_zip_path=low_level_zip_path,
                    folder_name_in_zip=folder_in_zip,
                    remove_negative_rows=remove_negatives,
                    batch_size=batch_size,
                    num_workers=num_workers
                )
                
                st.session_state.wgan_data_loaded = True
                st.session_state.data_stats = stats
                
                st.success("✓ Data loaded successfully!")
                
                # Display stats
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Number of Sequences", f"{stats['num_sequences']:,}")
                with col_s2:
                    st.metric("Condition Dimension", stats['condition_dim'])
                with col_s3:
                    st.metric("Sequence Feature Dimension", stats['sequence_feature_dim'])
                
                st.info(f"Using device: **{device.upper()}**")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.exception(e)
    
    # ============================================================================
    # Model Configuration and Training
    # ============================================================================
    if st.session_state.wgan_data_loaded:
        st.header("3. Configure and Train WGAN-GP")
        
        st.subheader("Model Architecture")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("**Generator Settings**")
            latent_dim = st.number_input(
                "Latent Dimension",
                min_value=10,
                max_value=200,
                value=20,
                step=10,
                help="Dimension of random noise vector",
                key="train_latent_dim"
            )
            
            hidden_dim = st.number_input(
                "Hidden Dimension",
                min_value=64,
                max_value=1024,
                value=256,
                step=64,
                help="Size of LSTM hidden layers",
                key="train_hidden_dim"
            )
            
            num_layers_gen = st.number_input(
                "Generator LSTM Layers",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                key="train_num_layers_gen"
            )
        
        with col_m2:
            st.markdown("**Critic Settings**")
            num_layers_critic = st.number_input(
                "Critic LSTM Layers",
                min_value=1,
                max_value=5,
                value=1,
                step=1,
                key="train_num_layers_critic"
            )
            
            critic_iterations = st.number_input(
                "Critic Iterations",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                help="Number of critic updates per generator update"
            )
            
            lambda_gp = st.number_input(
                "Lambda GP",
                min_value=1.0,
                max_value=50.0,
                value=10.0,
                step=1.0,
                help="Gradient penalty coefficient"
            )
        
        st.subheader("Training Hyperparameters")
        
        col_h1, col_h2 = st.columns(2)
        
        with col_h1:
            epochs = st.number_input(
                "Number of Epochs",
                min_value=1,
                max_value=10000,
                value=2000,
                step=100
            )
            
            g_lr = st.number_input(
                "Generator Learning Rate",
                min_value=1e-6,
                max_value=1e-2,
                value=5e-5,
                format="%.6f"
            )
            
            weight_decay = st.number_input(
                "Weight Decay",
                min_value=0.0,
                max_value=1e-3,
                value=1e-6,
                format="%.6f"
            )
        
        with col_h2:
            d_lr = st.number_input(
                "Critic Learning Rate",
                min_value=1e-6,
                max_value=1e-2,
                value=5e-5,
                format="%.6f"
            )
            
            max_grad_norm = st.number_input(
                "Max Gradient Norm",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Maximum norm for gradient clipping"
            )
        
        if st.button("Create Models", key="create_models_btn"):
            with st.spinner("Creating models..."):
                try:
                    st.session_state.wgan_handler.create_models(
                        latent_dim=latent_dim,
                        hidden_dim=hidden_dim,
                        num_layers_generator=num_layers_gen,
                        num_layers_critic=num_layers_critic,
                        g_lr=g_lr,
                        d_lr=d_lr,
                        weight_decay=weight_decay
                    )
                    st.success("✓ Models created successfully!")
                    
                except Exception as e:
                    st.error(f"Error creating models: {str(e)}")
                    st.exception(e)
        
        st.divider()
        
        st.subheader("Train WGAN-GP Model")
        
        if st.session_state.wgan_handler and st.session_state.wgan_handler.generator is not None:
            if st.session_state.wgan_model_trained:
                st.success("✓ Model trained")
            
            if st.button("Start Training", type="primary", key="start_training_btn"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(epoch, g_loss, d_loss):
                    progress = epoch / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch}/{epochs} - G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")
                
                try:
                    st.session_state.wgan_handler.train(
                        epochs=epochs,
                        critic_iterations=critic_iterations,
                        lambda_gp=lambda_gp,
                        max_grad_norm=max_grad_norm,
                        progress_callback=progress_callback
                    )
                    
                    st.session_state.wgan_model_trained = True
                    progress_bar.progress(1.0)
                    status_text.text("Training complete!")
                    st.success("✓ Training completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    st.exception(e)
        else:
            st.warning("⚠️ Please create models first")
    
    # ============================================================================
    # Generate and Save (Train Mode)
    # ============================================================================
    if st.session_state.wgan_model_trained:
        st.header("4. Generate Synthetic Sequences")
        
        num_samples = st.number_input(
            "Number of Samples to Generate",
            min_value=1,
            max_value=100000,
            value=1000,
            step=100,
            key="train_num_samples"
        )
        
        st.info("Sequences will be cut at first connection close packet")
        st.markdown("Each generated sequence will have the same `capture_id` as the `file_id` from the condition vector sampled from training data.")
        
        if st.button("Generate Samples", type="primary", key="train_gen_samples"):
            with st.spinner(f"Generating {num_samples} synthetic sequences..."):
                try:
                    # Generate sequences with sampled indices to track capture_ids
                    generated, sampled_indices = st.session_state.wgan_handler.generate_samples(
                        num_samples=num_samples,
                        seq_len=DEFAULT_SEQ_LENGTH,
                        return_numpy=True,
                        return_sampled_indices=True
                    )
                    
                    # Get capture_ids from the sampled indices
                    if sampled_indices is not None:
                        capture_ids = st.session_state.wgan_handler.get_file_ids_from_indices(sampled_indices)
                    else:
                        capture_ids = list(range(num_samples))
                        st.warning("Could not track condition indices. Using sequential IDs.")
                    
                    # Get sequence columns from dataset
                    seq_cols = []
                    if st.session_state.wgan_handler.dataset:
                        seq_cols = getattr(st.session_state.wgan_handler.dataset, 'sequence_columns', [])
                    
                    # Truncate at connection close using helper function
                    truncated_sequences = truncate_at_connection_close(generated, seq_cols)
                    st.session_state.wgan_generated_data = truncated_sequences
                    st.session_state.wgan_capture_ids = capture_ids  # Store capture_ids
                    
                    # Display results
                    seq_lengths = [len(s) for s in truncated_sequences]
                    st.success(f"✓ Generated {len(truncated_sequences)} sequences!")
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Sequences", len(truncated_sequences))
                    with col_stat2:
                        st.metric("Avg Length", f"{np.mean(seq_lengths):.1f}")
                    with col_stat3:
                        st.metric("Features", truncated_sequences[0].shape[1] if truncated_sequences else 0)
                    
                except Exception as e:
                    st.error(f"Error generating samples: {str(e)}")
                    st.exception(e)
        
        # Save Model
        st.header("5. Download Model")
        
        try:
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Model parameters
                model_params = {
                    "latent_dim": latent_dim,
                    "hidden_dim": hidden_dim,
                    "num_layers_gen": num_layers_gen
                }
                zip_file.writestr("model_params.json", json.dumps(model_params, indent=2))
                
                # Generator & Critic
                buffer = io.BytesIO()
                torch.save(st.session_state.wgan_handler.generator.state_dict(), buffer)
                zip_file.writestr("generator.pth", buffer.getvalue())
                
                buffer = io.BytesIO()
                torch.save(st.session_state.wgan_handler.critic.state_dict(), buffer)
                zip_file.writestr("critic.pth", buffer.getvalue())
                
                # Scalers and columns
                if st.session_state.wgan_handler.dataset:
                    buffer = io.BytesIO()
                    joblib.dump(st.session_state.wgan_handler.dataset.sequence_scaler, buffer)
                    zip_file.writestr("sequence_scaler.gz", buffer.getvalue())
                    
                    buffer = io.BytesIO()
                    joblib.dump(st.session_state.wgan_handler.dataset.condition_scaler, buffer)
                    zip_file.writestr("condition_scaler.gz", buffer.getvalue())
                    
                    if hasattr(st.session_state.wgan_handler.dataset, 'sequence_columns'):
                        columns_json = json.dumps(st.session_state.wgan_handler.dataset.sequence_columns, indent=2)
                        zip_file.writestr("sequence_columns.json", columns_json)
            
            zip_buffer.seek(0)
            
            st.download_button(
                label="📥 Download WGAN Model (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="wgan_model.zip",
                mime="application/zip",
                key="download_model_train"
            )
            
        except Exception as e:
            st.error(f"Error preparing model files: {str(e)}")
        
        # Export generated data
        if st.session_state.wgan_generated_data is not None:
            st.header("6. Export Generated Data")
            
            # Get sequence columns
            seq_cols = []
            if st.session_state.wgan_handler.dataset:
                seq_cols = getattr(st.session_state.wgan_handler.dataset, 'sequence_columns', [])
            
            if seq_cols:
                # Preview generated data
                with st.expander("Preview Generated Data"):
                    preview_df = sequences_to_dataframe(
                        st.session_state.wgan_generated_data[:5],
                        seq_cols,
                        capture_ids=st.session_state.wgan_capture_ids[:5] if st.session_state.wgan_capture_ids else None
                    )
                    st.dataframe(preview_df.head(50), use_container_width=True)
                
                render_download_button(
                    st.session_state.wgan_generated_data,
                    seq_cols,
                    capture_ids=st.session_state.wgan_capture_ids,
                    key="download_csv_train"
                )
            else:
                st.warning("Unable to export - sequence columns not available")