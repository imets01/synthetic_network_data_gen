# Synthetic QUIC Network Data Generator

A tool for generating synthetic QUIC network traffic data using state-of-the-art generative models. This project implements a hierarchical approach for creating realistic network traffic fingerprints with support for QUIC connection migration features.

> **Note:** This project was developed with the assistance of [GitHub Copilot](https://github.com/features/copilot).

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start with Example Data](#quick-start-with-example-data)
- [Important: FEST Evaluation Framework](#important-fest-evaluation-framework)
- [Experiment Notebooks](#experiment-notebooks)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Using the Streamlit Tool](#using-the-streamlit-tool)
- [Data Generation Workflow](#data-generation-workflow)
- [Model Wrappers](#model-wrappers)
- [Evaluation Framework](#evaluation-framework)
- [Capture Generation](#capture-generation)
- [Features Reference](#features-reference)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project provides a hierarchical approach to synthetic network data generation:

1. **Stage 1: High-Level Features** - Generate flow-level metadata (connection duration, packet counts, migration info, etc.) using **CTGAN** or **TabDDPM** models
2. **Stage 2: Low-Level Features** - Generate packet-level sequential data using **WGAN-GP**, conditioned on the high-level features
3. **Stage 3: Post-Processing** - Apply protocol-specific rules to ensure QUIC protocol correctness

The toolkit includes:
- **Streamlit Web Interface** for easy model training and data generation
- **FEST Evaluation Framework** for comparing synthetic vs. original data quality
- **Modular Model Wrappers** for CTGAN, TabDDPM, and WGAN-GP
- **Capture Generation Scripts** for creating real QUIC traffic captures

---

## Project Structure

```
synthetic_network_data_gen/
├── synth_tool/                     # Main Streamlit application
│   ├── 1_High_Level_Data_Generation.py
│   ├── pages/                      # Additional app pages
│   ├── wrappers/                   # Model wrapper classes (CTGAN, TabDDPM, WGAN)
│   ├── example_data/               # Pre-trained models & sample datasets
│   ├── tab-ddpm/                   # TabDDPM implementation
│   ├── evaluation.py, training.py, post_processing.py, ...
│   └── requirements.txt
│
├── captures/                       # Network capture utilities & scripts
│   ├── capture_generation_scripts/ # QUIC capture generation scripts
│   ├── captures_json/              # Parsed JSON captures
│   ├── keylog_files/               # TLS keylog files
│   └── pcap_files/                 # Raw PCAP files
│
├── high_level_features/            # High-level feature experiments
│   ├── dataset/                    # Training datasets
│   ├── CTGAN/                      # CTGAN experiments (includes archiv/)
│   └── tabddpm/                    # TabDDPM experiments
│
├── low_level_features/             # Low-level packet feature experiments
│   ├── dataset/                    # Packet-level datasets
│   ├── WGAN/                       # WGAN-GP experiments (includes archiv/)
│   ├── RGAN/                       # RGAN experiments
│   ├── PARSynthesizer/             # PAR model experiments
│   └── baseline_model/             # Baseline comparisons
│
├── FEST_eval/                      # Evaluation framework (not in repo)
│   └── synprivutil/                # Required for evaluation page
│
├── test_gan_data/                  # Legacy test data from experimentations
├── features.md                     # Feature documentation
└── README.md
```

> **Note:** Model folders (e.g., `CTGAN/`, `WGAN/`) contain an `archiv/` subfolder with initial experimentations and exploratory notebooks. The `test_gan_data/` folder contains previous versions of datasets used during development.

---

## Quick Start with Example Data

After completing the [Installation](#installation), try the app with pre-trained models in `synth_tool/example_data/`:

| Folder | Contents | Usage |
|--------|----------|-------|
| `ctgan/` | Pre-trained CTGAN model (`ctgan_model.pkl`) | Load in High-Level page for instant generation |
| `tabddpm/` | Pre-trained TabDDPM model (`model.pt` + `config.toml`) | Load in High-Level page |
| `wgan/` | Pre-trained WGAN-GP model (`wgan_model.zip`) + sample low-level data | Load in Low-Level page |
| `all_captures_dataset.csv` | Sample high-level dataset | Use as original data for training or evaluation |

**Try It:**
1. Complete installation steps 1-5
2. In the app, select "Load Pre-trained Model"
3. Upload `example_data/ctgan/ctgan_model.pkl`
4. Generate synthetic samples

---

## Important: FEST Evaluation Framework

The **FEST Evaluation** page requires the `synprivutil` library, which is **not included in this repository**.

To enable the evaluation functionality, clone the synprivutil repository:

```bash
cd FEST_eval
git clone https://github.com/Karo2222/synprivutil.git
```

After cloning, the directory structure should be:
```
FEST_eval/
└── synprivutil/
    └── privacy_utility_framework/
        └── privacy_utility_framework/
            └── metrics/
```

Without `synprivutil`, the evaluation page will not function. All other features (training, generation, post-processing) work independently.

---

## Experiment Notebooks

Model experiments are organized in separate folders within `high_level_features/` and `low_level_features/`:

- **High-Level Features** (`high_level_features/`)
  - `CTGAN/` - CTGAN training notebooks and generated data
  - `tabddpm/` - TabDDPM experiments
  - Dataset generation and post-processing notebooks

- **Low-Level Features** (`low_level_features/`)
  - `WGAN/` - WGAN-GP training and post-processing notebooks
  - `RGAN/` - RGAN alternative experiments
  - `PARSynthesizer/` - PAR model experiments
  - `baseline_model/` - Baseline script

Each model folder includes an `archiv/` subfolder containing initial exploratory notebooks and early experiments.

---

## Requirements

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- Windows/Linux/macOS

### Python Dependencies

Core dependencies are listed in `synth_tool/requirements.txt`:

```
streamlit>=1.20.0
pandas>=1.3.4
numpy>=1.21.4
scikit-learn>=1.0.2
plotly>=5.0.0
matplotlib>=3.5.0
torch>=1.10.0
sdv>=0.6.0
rdt>=1.0.0
tomli>=1.2.2
tomli-w>=0.4.0
tqdm>=4.60.0
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd synthetic_network_data_gen
```

### 2. Verify Python Version

Ensure you have Python 3.8 or higher:

```bash
python --version
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 4. Install Dependencies

```bash
cd synth_tool
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run 1_High_Level_Data_Generation.py
```

This launches the web interface at `http://localhost:8501`.

### 6. (Optional) Install TabDDPM

For TabDDPM model support, install additional dependencies:

```bash
cd tab-ddpm/tab-ddpm-main
pip install -r requirements.txt
```

---

## Quick Start

After installation, the app runs at `http://localhost:8501` with three pages:
1. **High-Level Data Generation** - Train/use CTGAN or TabDDPM models
2. **Low-Level Data Generation** - Train/use WGAN-GP for packet sequences
3. **FEST Evaluation** - Evaluate synthetic data quality

---

## Using the Streamlit Tool

### Page 1: High-Level Data Generation

Generate synthetic flow-level features using CTGAN or TabDDPM.

#### Workflow:
1. **Upload Data**: Upload your original high-level features CSV
2. **Select Model**: Choose between CTGAN or TabDDPM
3. **Training Mode**:
   - Train from scratch with custom hyperparameters
   - Upload a pre-trained model (`.pkl` for CTGAN, `.pt` + `config.toml` for TabDDPM)
4. **Configure Columns**: Set numerical, categorical, boolean, and ID columns
5. **Train/Generate**: Start training or generate samples from loaded model
6. **Download**: Export synthetic data as CSV

#### CTGAN Configuration Options:
- Embedding dimension
- Generator/Discriminator dimensions
- Batch size
- Number of epochs
- Constraints (inequalities, fixed combinations, ranges)

#### TabDDPM Configuration Options:
- Target column selection
- Categorical vs numerical feature detection
- Test/validation split ratios
- Number of training iterations

### Page 2: Low-Level Data Generation

Generate synthetic packet sequences using WGAN-GP conditioned on high-level features.

#### Workflow:
1. **Select Mode**: Train new model or load pre-trained
2. **Upload Data**:
   - High-level features CSV (conditions)
   - Low-level packet data (ZIP of individual capture CSVs or merged CSV)
3. **Configure Training**: Set epochs, batch size, learning rates
4. **Train**: WGAN-GP training with gradient penalty
5. **Generate**: Create sequences conditioned on high-level data
6. **Post-Process**: Truncate at connection close, apply rules
7. **Download**: Export generated sequences

#### WGAN-GP Parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 20 | Noise vector dimension |
| `hidden_dim` | 256 | LSTM hidden size |
| `batch_size` | 128 | Training batch size |
| `epochs` | 2000 | Number of training epochs |
| `critic_iterations` | 5 | Critic updates per generator update |
| `lambda_gp` | 10 | Gradient penalty coefficient |

### Page 3: FEST Evaluation

Evaluate synthetic data quality using multiple metrics.

#### Evaluation Types:
- **High-Level Evaluation**: Uses predefined QUIC protocol features
- **Low-Level Evaluation**: Uses LSTM-based sequence classification

#### Metrics Provided:
- **Basic Statistics**: Mean, std, min, max comparisons
- **Distribution Similarity**: KL divergence, Wasserstein distance
- **ML Utility**: Classification accuracy (Random Forest for high-level, LSTM for low-level)
- **Privacy Metrics**: Nearest-neighbor distance ratio

---

## Data Generation Workflow

### Hierarchical Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: High-Level Features                 │
│                                                                 │
│  Original Data  ──>  CTGAN/TabDDPM  ──>  Synthetic Metadata    │
│  (flow metadata)        Training         (conditioning vector)  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: Low-Level Features                  │
│                                                                 │
│  Synthetic Metadata  ──>  WGAN-GP  ──>  Packet Sequences       │
│  (conditions)             Generation    (per-packet features)   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 3: Post-Processing                     │
│                                                                 │
│  Rule-based corrections (protocol validity, CIDs, timestamps)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model Wrappers

### CTGANHandler

```python
from wrappers.ctgan_wrapper import CTGANHandler

# Initialize
handler = CTGANHandler()

# Load and configure data
handler.load_data(df, include_cols=['col1', 'col2', ...])
handler.detect_metadata()

# Set column types
handler.set_numerical_float_columns(['duration', 'bytes_sent'])
handler.set_categorical_columns(['migration_type'])
handler.set_boolean_columns(['retry_occurred'])

# Train
handler.train(epochs=500, batch_size=250)

# Generate
synthetic_df = handler.sample(num_samples=1000)

# Save/Load
handler.save_model('model.pkl')
handler.load_model('model.pkl')
```

### TabDDPMHandler

```python
from wrappers.tabddpm_wrapper import TabDDPMHandler

# Initialize
handler = TabDDPMHandler()

# Preprocess data
handler.preprocess(
    df,
    target_col='implementation',
    cat_cols=['migration_type', 'connection_close_type'],
    columns_to_keep=['col1', 'col2', ...]
)

# Train
handler.train(num_iterations=5000)

# Generate
synthetic_df = handler.sample(num_samples=1000)
```

### WGANHandler

```python
from wrappers.wgan_wrapper import WGANHandler

# Initialize and load data
handler = WGANHandler()
handler.load_data(
    high_level_csv='high_level.csv',
    low_level_dir='low_level_sequences/'
)

# Train
handler.train(epochs=2000, batch_size=128)

# Generate sequences
sequences = handler.generate(conditions_df, seq_length=40)

# Save/Load
handler.save_model('wgan_model.zip')
handler.load_model('wgan_model.zip')
```

---

## Evaluation Framework

### Running Evaluations

```python
from evaluation import (
    evaluate_synthetic_data,
    run_ml_utility_evaluation,
    run_low_level_ml_utility_lstm
)

# Basic evaluation (statistics, distributions)
results = evaluate_synthetic_data(original_df, synthetic_df)

# ML Utility for high-level data
ml_results = run_ml_utility_evaluation(
    original_df, 
    synthetic_df,
    target_col='implementation'
)

# LSTM-based evaluation for low-level sequences
lstm_results = run_low_level_ml_utility_lstm(
    original_df, 
    synthetic_df,
    original_high_level_df,
    synthetic_high_level_df
)
```

## Capture Generation

### Generating QUIC Captures (Windows)

```powershell
cd captures/capture_generation_scripts
.\generate_captures_windows.ps1
```

### Converting PCAP to JSON

```powershell
.\convert_pcap_to_json.ps1
```

### Using aioquic for Custom Captures

```python
cd captures/capture_generation_scripts/aioquic_code

# Generate certificates
python generate_cert.py

# Run server
python server_combined.py

# Run client (in separate terminal)
python client_combined.py
```

---

## Features Reference

See [features.md](features.md) for detailed documentation on:

### High-Level Features (52 features)
- Connection metadata (IPs, ports, timestamps)
- Traffic statistics (bytes sent, packets sent)
- QUIC-specific features (migration type, handshake duration)
- Stream statistics (bidirectional/unidirectional)
- Frame counts (ACK, CRYPTO, PATH_CHALLENGE, etc.)

### Low-Level Features (24 features per packet)
- Timing (`delta_time`)
- Size (`packet_length`, `stream_length`)
- Direction (`packet_direction`)
- QUIC packet types (Initial, Handshake, 1-RTT, etc.)
- Frame types (ACK, Padding, Connection Close, etc.)
- HTTP/3 stream information

---

## Post-Processing Rules

After generation, apply protocol-specific rules:

1. **Timestamp Ordering**: Ensure monotonic packet timestamps
2. **CID Consistency**: Validate connection ID sequences
3. **Frame Constraints**: Enforce valid frame combinations
4. **Negative Values**: Ensure non-negative numeric values
5. **Sequence Truncation**: Cut at connection close events

See notebooks in `high_level_features/` and `low_level_features/` for detailed post-processing examples.

---

## Acknowledgments

- **GitHub Copilot**: This project was developed with the assistance of [GitHub Copilot](https://github.com/features/copilot), an AI pair programmer that helped accelerate development and implementation of the codebase.
- **FEST Framework**: The evaluation module uses the privacy-utility evaluation framework for computing privacy and utility metrics on synthetic data.
