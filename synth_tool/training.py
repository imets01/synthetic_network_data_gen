import os
import subprocess
import numpy as np
import pandas as pd
import toml
import streamlit as st


def get_conda_env_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.conda'))


def get_tabddpm_lib_path():
    """Get the path to the tab-ddpm library."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), 'tab-ddpm', 'tab-ddpm-main'))


def run_subprocess_with_logging(cmd: list, cwd: str, env: dict, log_placeholder) -> int:
    import time
    
    st.write(f"Running: `{' '.join(cmd)}`")
    
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding='utf-8',
        errors='replace'
    )
    
    last_update = time.time()
    for line in iter(process.stdout.readline, ''):
        if line.strip():
            st.session_state.training_log.append(line.strip())
        if time.time() - last_update > 0.5:
            log_placeholder.code('\n'.join(st.session_state.training_log[-15:]))
            last_update = time.time()
    
    process.wait()
    return process.returncode


def load_tabddpm_output(output_dir: str, data_dir: str, handler=None, uploaded_data_files: dict = None):
    import json
    
    # Load generated data
    X_num = np.load(os.path.join(output_dir, 'X_num_train.npy'))
    X_cat = np.load(os.path.join(output_dir, 'X_cat_train.npy'), allow_pickle=True)
    y = np.load(os.path.join(output_dir, 'y_train.npy'))
    
    n_num = X_num.shape[1] if X_num is not None and len(X_num.shape) > 1 else 0
    n_cat = X_cat.shape[1] if X_cat is not None and len(X_cat.shape) > 1 else 0
    
    st.info(f"Generated data shape: {n_num} numerical + {n_cat} categorical = {n_num + n_cat} features")
    
    feature_names = None
    target_col = None
    
    # Try to get column names from handler
    if handler is not None:
        feature_names = handler.num_features + handler.cat_features
        target_col = handler.target_col
    
    # Try to get column names from uploaded config
    elif uploaded_data_files and 'column_config.json' in uploaded_data_files:
        try:
            col_config = json.loads(uploaded_data_files['column_config.json'].decode('utf-8'))
            num_cols = col_config.get('numerical_columns', [])
            cat_cols = col_config.get('categorical_columns', [])
            target_col = col_config.get('target_column', None)
            
            # Remove target from columns if present
            if target_col and target_col in num_cols:
                num_cols = [c for c in num_cols if c != target_col]
            if target_col and target_col in cat_cols:
                cat_cols = [c for c in cat_cols if c != target_col]
            
            st.info(f"Uploaded column_config: {len(num_cols)} numerical + {len(cat_cols)} categorical (target: {target_col})")
            
            if len(num_cols) == n_num and len(cat_cols) == n_cat:
                feature_names = num_cols + cat_cols
                st.success("✓ Column names loaded from uploaded config")
            else:
                st.warning(f"Column config mismatch! Config has {len(num_cols)} num + {len(cat_cols)} cat, "
                          f"but data has {n_num} num + {n_cat} cat. Using generic names.")
        except Exception as e:
            st.warning(f"Could not parse uploaded column_config.json: {e}")
    
    # Fallback to generic names
    if feature_names is None:
        feature_names = [f'num_feature_{i}' for i in range(n_num)] + [f'cat_feature_{i}' for i in range(n_cat)]
        target_col = 'target'
    
    # Combine into DataFrame
    X = np.concatenate([X_num, X_cat], axis=1)
    synthetic_df = pd.DataFrame(X, columns=feature_names)
    
    # Only add target column if not already in features
    if target_col not in feature_names:
        synthetic_df[target_col] = y.flatten()
    
    return synthetic_df


def save_tabddpm_data_files(data_dir: str, uploaded_data_files: dict):
    """Save uploaded data files to the data directory."""
    os.makedirs(data_dir, exist_ok=True)
    
    for filename, content in uploaded_data_files.items():
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(content)
    
    st.session_state.training_log.append(f"Saved {len(uploaded_data_files)} data files to {data_dir}")
    
    # Create dummy val/test files if not uploaded
    for split in ['val', 'test']:
        for prefix in ['X_num', 'X_cat', 'y']:
            src_file = f'{prefix}_train.npy'
            dst_file = f'{prefix}_{split}.npy'
            if dst_file not in uploaded_data_files and src_file in uploaded_data_files:
                dst_path = os.path.join(data_dir, dst_file)
                with open(dst_path, 'wb') as f:
                    f.write(uploaded_data_files[src_file])


def train_ctgan(handler, epochs: int, batch_size: int, num_samples: int) -> pd.DataFrame:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Creating synthesizer...")
    handler.create_synthesizer(epochs=epochs, batch_size=batch_size)
    progress_bar.progress(10)
    
    status_text.text(f"Training CTGAN for {epochs} epochs...")
    handler.fit()
    progress_bar.progress(80)
    
    status_text.text(f"Generating {num_samples} synthetic samples...")
    synthetic_df = handler.sample(num_rows=num_samples)
    progress_bar.progress(100)
    
    status_text.text("")
    return synthetic_df


def generate_from_loaded_ctgan(synthesizer, num_samples: int) -> pd.DataFrame:
    """Generate synthetic data from a loaded CTGAN synthesizer."""
    return synthesizer.sample(num_rows=num_samples)


def train_tabddpm(handler, training_mode: str, num_samples: int, log_placeholder) -> pd.DataFrame:
    import torch
    
    conda_env = get_conda_env_path()
    lib_path = handler.lib_path
    
    # Build command based on training mode
    if "Quick" in training_mode:
        tune_script = os.path.join(lib_path, 'scripts', 'tune_ddpm_quick.py')
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
        tune_script = os.path.join(lib_path, 'scripts', 'tune_ddpm.py')
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
        pipeline_script = os.path.join(lib_path, 'scripts', 'pipeline.py')
        
        # Update config
        config = st.session_state.tabddpm_config.copy()
        config['parent_dir'] = 'exp/custom'
        config['real_data_path'] = 'data/custom/'
        config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        config['num_numerical_features'] = len(handler.num_features)
        
        total_features = len(handler.num_features) + len(handler.cat_features)
        if 'model_params' in config:
            config['model_params']['d_in'] = total_features
        
        if 'sample' in config:
            config['sample']['num_samples'] = num_samples
        
        # Save config
        config_path = os.path.join(lib_path, 'exp', 'custom', 'config.toml')
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            toml.dump(config, f)
        
        # Check if user uploaded a trained model
        skip_training = st.session_state.tabddpm_model_uploaded is not None
        
        if skip_training:
            model_dir = os.path.join(lib_path, 'exp', 'custom')
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
                '--sample'
            ]
        else:
            st.info("No model uploaded - will train then generate")
            cmd = [
                'conda', 'run', '-p', conda_env,
                'python', '-u',
                pipeline_script,
                '--config', config_path,
                '--train', '--sample'
            ]
    else:
        st.error(f"Unknown training mode: {training_mode}")
        return None
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = lib_path
    env['PYTHONUNBUFFERED'] = '1'
    
    st.info("Training started... This may take several minutes. Watch the log below for progress.")
    
    # Run subprocess
    return_code = run_subprocess_with_logging(cmd, lib_path, env, log_placeholder)
    
    if return_code == 0:
        best_dir = os.path.join(lib_path, 'exp', 'custom', 'ddpm_tune_best')
        return load_tabddpm_output(best_dir, os.path.join(lib_path, 'data', 'custom'), handler)
    else:
        st.error(f"Training failed with code {return_code}")
        st.code('\n'.join(st.session_state.training_log[-20:]))
        return None


def generate_from_pretrained_tabddpm(num_samples: int, log_placeholder) -> pd.DataFrame:
    import torch
    
    conda_env = get_conda_env_path()
    lib_path = get_tabddpm_lib_path()
    pipeline_script = os.path.join(lib_path, 'scripts', 'pipeline.py')
    
    # Save uploaded data files
    data_dir = os.path.join(lib_path, 'data', 'custom')
    save_tabddpm_data_files(data_dir, st.session_state.uploaded_data_files)
    
    # Update and save config
    config = st.session_state.tabddpm_config.copy()
    config['parent_dir'] = 'exp/custom'
    config['real_data_path'] = 'data/custom/'
    config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    if 'sample' in config:
        config['sample']['num_samples'] = num_samples
    
    config_path = os.path.join(lib_path, 'exp', 'custom', 'config.toml')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        toml.dump(config, f)
    
    # Save uploaded model
    model_path = os.path.join(lib_path, 'exp', 'custom', 'model.pt')
    with open(model_path, 'wb') as f:
        f.write(st.session_state.tabddpm_model_uploaded.getvalue())
    
    # Build command
    cmd = [
        'conda', 'run', '-p', conda_env,
        'python', '-u',
        pipeline_script,
        '--config', config_path,
        '--sample'
    ]
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = lib_path
    env['PYTHONUNBUFFERED'] = '1'
    
    # Run subprocess
    return_code = run_subprocess_with_logging(cmd, lib_path, env, log_placeholder)
    
    if return_code == 0:
        output_dir = os.path.join(lib_path, 'exp', 'custom')
        return load_tabddpm_output(
            output_dir, 
            data_dir, 
            uploaded_data_files=st.session_state.uploaded_data_files
        )
    else:
        st.error(f"Generation failed with code {return_code}")
        st.code('\n'.join(st.session_state.training_log[-20:]))
        return None


def configure_ctgan_handler(df: pd.DataFrame, config: dict):
    from wrappers.ctgan_wrapper import CTGANHandler
    
    handler = CTGANHandler()
    
    handler.load_data(
        df,
        include_cols=config['include_cols'],
        remove_negative_rows=True,
        numerical_cols=config['num_cols'] if config['num_cols'] else None
    )
    
    if hasattr(handler, 'removed_negative_rows') and handler.removed_negative_rows > 0:
        st.warning(f"Removed {handler.removed_negative_rows} rows with negative values in numerical columns")
    
    handler.detect_metadata()
    
    handler.configure_metadata(
        numerical_float=config['num_cols'] if config['num_cols'] else None,
        categorical=config['cat_cols'] if config['cat_cols'] else None,
        boolean=config['bool_cols'] if config['bool_cols'] else None
    )
    
    return handler


def configure_tabddpm_handler(df: pd.DataFrame, config: dict):
    from wrappers.tabddpm_wrapper import TabDDPMHandler
    
    handler = TabDDPMHandler()
    
    result = handler.preprocess(
        df,
        target_col=config['target_col'],
        cat_cols=config['cat_cols'] if config['cat_cols'] else None,
        columns_to_keep=config['feature_cols']
    )
    
    handler.create_config(
        steps=200,
        num_timesteps=50,
        d_layers=[128, 128],
        batch_size=256,
        num_samples=min(result['train'], 500)
    )
    
    return handler, result
