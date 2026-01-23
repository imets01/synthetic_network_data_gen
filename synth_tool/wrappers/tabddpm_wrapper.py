import pandas as pd
import numpy as np
import json
import os
import sys
import shutil
import tomli
import tomli_w
import torch
from typing import Optional, List, Dict, Any, Callable
from sklearn.model_selection import train_test_split


class TabDDPMHandler:
    
    def __init__(
        self, 
        lib_path: Optional[str] = None,
        device: str = "auto"
    ):
        # Set up paths relative to this module's location
        module_dir = os.path.dirname(os.path.abspath(__file__))
        synth_tool_dir = os.path.dirname(module_dir)
        
        if lib_path is None:
            lib_path = os.path.join(synth_tool_dir, "tab-ddpm", "tab-ddpm-main")
            
        self.lib_path = os.path.abspath(lib_path)
        
        # Data and exp directories inside lib_path (where tune scripts expect them)
        # Use "custom" as the dataset name for user-uploaded data
        self.ds_name = "custom"
        self.data_dir = os.path.join(self.lib_path, "data", self.ds_name)
        self.exp_dir = os.path.join(self.lib_path, "exp", self.ds_name)
        
        # Set device
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.exp_dir, exist_ok=True)

        # State storage
        self.feature_names: List[str] = []
        self.cat_features: List[str] = []
        self.num_features: List[str] = []
        self.target_col: str = ""
        self.train_size: int = 0
        
        # Add lib path to sys.path for imports
        if self.lib_path not in sys.path:
            sys.path.insert(0, self.lib_path)
        
        # Store config for later use
        self._config: Optional[Dict[str, Any]] = None
        
    def select_columns(
        self,
        df: pd.DataFrame,
        columns_to_keep: List[str],
        target_col: str
    ) -> pd.DataFrame:

        print("Selecting columns...")
        
        # Validate that all requested columns exist
        all_cols = columns_to_keep + [target_col]
        missing_cols = [c for c in all_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        
        # Keep only the specified columns
        df_selected = df[all_cols].copy()
        
        print(f"  Kept {len(columns_to_keep)} feature columns + target '{target_col}'")
        print(f"  Columns: {columns_to_keep}")
        
        return df_selected
    
    def identify_categorical_columns(
        self,
        df: pd.DataFrame,
        cat_cols: Optional[List[str]] = None,
        auto_detect: bool = True,
        max_unique_for_auto: int = 20
    ) -> List[str]:

        print("Identifying categorical columns...")
        
        if cat_cols is not None:
            # Validate that all specified categorical columns exist
            missing = [c for c in cat_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Categorical columns not found in DataFrame: {missing}")
            print(f"  Using specified categorical columns: {cat_cols}")
            return cat_cols
        
        if auto_detect:
            # Auto-detect: object/category types, or numeric with few unique values
            detected = []
            for col in df.columns:
                if df[col].dtype in ['object', 'category']:
                    detected.append(col)
                elif df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                    n_unique = df[col].nunique()
                    if n_unique <= max_unique_for_auto:
                        detected.append(col)
            print(f"  Auto-detected categorical columns: {detected}")
            return detected
        
        return []
    
    def preprocess(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        cat_cols: Optional[List[str]] = None,
        columns_to_keep: Optional[List[str]] = None,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int = 42,
        remove_negative: bool = True,
        non_negative_cols: Optional[List[str]] = None
    ) -> Dict[str, int]:

        print("Preprocessing data...")
        df = df.copy().dropna()
        self.target_col = target_col
        
        # Step 1: Keep only necessary columns if specified
        if columns_to_keep is not None:
            df = self.select_columns(df, columns_to_keep, target_col)
        
        # Step 2: Remove negative values if requested
        if remove_negative:
            if non_negative_cols is None:
                # Default: check all numeric columns
                non_negative_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            before_clean = len(df)
            for col in non_negative_cols:
                if col in df.columns:
                    df = df[df[col] >= 0]
            after_clean = len(df)
            if before_clean != after_clean:
                print(f"  Removed {before_clean - after_clean} rows with negative values")
        
        # Step 3: Identify categorical columns (use the dedicated method)
        self.cat_features = self.identify_categorical_columns(
            df, 
            cat_cols=cat_cols, 
            auto_detect=(cat_cols is None)
        )
        # Remove target from categorical if present
        if target_col in self.cat_features:
            self.cat_features.remove(target_col)
            
        self.num_features = [c for c in df.columns if c not in self.cat_features and c != target_col]
        self.feature_names = self.num_features + self.cat_features + [target_col]
        
        print(f"  Numerical features ({len(self.num_features)}): {self.num_features}")
        print(f"  Categorical features ({len(self.cat_features)}): {self.cat_features}")

        # Binary conversion logic from notebook (convert 0/1 to Yes/No for categoricals)
        for col in self.cat_features:
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1}):
                print(f"  Converting binary column '{col}' to Yes/No")
                df[col] = df[col].map({0: "No", 1: "Yes"}).astype("object")
            else:
                df[col] = df[col].astype("object")

        # Split Data
        train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=random_state)
        
        self.train_size = len(train_df)
        print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Helper to save splits
        def save_split(df_split: pd.DataFrame, name: str):
            X_num = df_split[self.num_features].values.astype(np.float32)
            X_cat = df_split[self.cat_features].astype(str).values
            y = df_split[target_col].values.astype(np.float32).reshape(-1, 1)
            
            np.save(os.path.join(self.data_dir, f'X_num_{name}.npy'), X_num)
            np.save(os.path.join(self.data_dir, f'X_cat_{name}.npy'), X_cat)
            np.save(os.path.join(self.data_dir, f'y_{name}.npy'), y)

        save_split(train_df, 'train')
        save_split(val_df, 'val')
        save_split(test_df, 'test')

        # Create info.json required by TabDDPM
        info_dict = {
            "name": "CustomData",
            "id": "custom-default",
            "task_type": "regression",
            "n_num_features": len(self.num_features),
            "n_cat_features": len(self.cat_features),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df)
        }
        
        with open(os.path.join(self.data_dir, 'info.json'), 'w') as f:
            json.dump(info_dict, f, indent=4)
        
        # Save column config for reference
        column_config = {
            "target_column": target_col,
            "numerical_columns": self.num_features,
            "categorical_columns": self.cat_features,
            "feature_names": self.feature_names
        }
        with open(os.path.join(self.data_dir, 'column_config.json'), 'w') as f:
            json.dump(column_config, f, indent=4)
            
        print("Preprocessing complete.")
        return {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    
    def load_data(
        self, 
        split: str = "train"
    ) -> pd.DataFrame:

        print(f"Loading {split} data...")
        
        # Load info.json to get metadata
        info_path = os.path.join(self.data_dir, 'info.json')
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        # Load column config
        config_path = os.path.join(self.data_dir, 'column_config.json')
        with open(config_path, 'r') as f:
            col_config = json.load(f)
        
        # Update instance state from loaded config
        self.num_features = col_config['numerical_columns']
        self.cat_features = col_config['categorical_columns']
        self.feature_names = col_config['feature_names']
        self.target_col = col_config['target_column']
        
        # Load npy files
        X_num = np.load(os.path.join(self.data_dir, f'X_num_{split}.npy'))
        X_cat = np.load(os.path.join(self.data_dir, f'X_cat_{split}.npy'), allow_pickle=True)
        y = np.load(os.path.join(self.data_dir, f'y_{split}.npy'))
        
        # Combine numerical and categorical features
        X = np.concatenate([X_num, X_cat], axis=1)
        
        # Build DataFrame using column names
        feature_names = self.num_features + self.cat_features
        df = pd.DataFrame(X, columns=feature_names)
        df[self.target_col] = y.flatten()
        
        print(f"  Loaded {len(df)} rows with {len(feature_names)} features")
        print(f"  Info: {info}")
        
        return df
    
    def create_config(
        self,
        model_type: str = "mlp",
        num_timesteps: int = 1000,
        steps: int = 20000,
        lr: float = 0.002,
        batch_size: int = 256,
        d_layers: List[int] = None,
        dropout: float = 0.1,
        scheduler: str = "cosine",
        is_y_cond: bool = False,
        num_samples: Optional[int] = None
    ) -> str:

        print("Creating config.toml...")
        
        if d_layers is None:
            d_layers = [512, 512]
        
        if num_samples is None:
            num_samples = self.train_size
        
        n_num = len(self.num_features)
        n_cat = len(self.cat_features)
        d_in = n_num + n_cat
        
        # Build config dictionary
        # Use relative paths since tune scripts run from lib_path
        config = {
            "seed": 0,
            "parent_dir": f"exp/{self.ds_name}",
            "real_data_path": f"data/{self.ds_name}",
            "num_numerical_features": n_num,
            "model_type": model_type,
            "device": self.device,
            "model_params": {
                "d_in": d_in,
                "num_classes": 0,
                "is_y_cond": is_y_cond,
                "rtdl_params": {
                    "d_layers": d_layers,
                    "dropout": dropout
                }
            },
            "diffusion_params": {
                "num_timesteps": num_timesteps,
                "gaussian_loss_type": "mse",
                "scheduler": scheduler
            },
            "train": {
                "main": {
                    "steps": steps,
                    "lr": lr,
                    "weight_decay": 1e-04,
                    "batch_size": batch_size
                },
                "T": {
                    "seed": 0,
                    "normalization": "quantile",
                    "num_nan_policy": "__none__",
                    "cat_nan_policy": "__none__",
                    "cat_min_frequency": "__none__",
                    "cat_encoding": "__none__",
                    "y_policy": "default"
                }
            },
            "sample": {
                "num_samples": num_samples,
                "batch_size": batch_size,
                "seed": 0
            },
            "eval": {
                "type": {
                    "eval_model": "catboost",
                    "eval_type": "synthetic"
                },
                "T": {
                    "seed": 0,
                    "normalization": "__none__",
                    "num_nan_policy": "__none__",
                    "cat_nan_policy": "__none__",
                    "cat_min_frequency": "__none__",
                    "cat_encoding": "__none__",
                    "y_policy": "default"
                }
            }
        }
        
        # Store config for later use
        self._config = config
        
        # Write config.toml
        config_path = os.path.join(self.exp_dir, 'config.toml')
        with open(config_path, 'wb') as f:
            tomli_w.dump(config, f)
        
        print(f"  Config written to: {config_path}")
        print(f"  num_numerical_features = {n_num}")
        print(f"  d_in = {d_in} ({n_num} numerical + {n_cat} categorical)")
        print(f"  num_timesteps = {num_timesteps}")
        print(f"  steps = {steps}")
        print(f"  d_layers = {d_layers}")
        print(f"  lr = {lr}")
        print(f"  num_samples = {num_samples}")
        
        return config_path

    def tune(
        self,
        ds_name: str = "custom",
        eval_type: str = "synthetic",
        eval_model: str = "catboost",
        prefix: str = "ddpm_tune",
        eval_seeds: bool = True,
        python_cmd: str = "python"
    ) -> str:
        
        import subprocess
        
        print(f"Starting hyperparameter tuning...")
        print(f"  Dataset: {ds_name}")
        print(f"  Train size: {self.train_size}")
        print(f"  Eval type: {eval_type}")
        print(f"  Eval model: {eval_model}")
        
        tune_script = os.path.join(self.lib_path, 'scripts', 'tune_ddpm.py')
        
        cmd = [
            python_cmd,
            tune_script,
            ds_name,
            str(self.train_size),
            eval_type,
            eval_model,
            prefix
        ]
        
        if eval_seeds:
            cmd.append('--eval_seeds')
        
        print(f"  Running: {' '.join(cmd)}")
        
        # Run tuning from the lib_path directory
        subprocess.run(cmd, check=True, cwd=self.lib_path)
        
        # Best model directory
        best_dir = os.path.join(self.exp_dir, f'{prefix}_best')
        self._best_dir = best_dir
        
        print(f"Tuning complete! Best model saved to: {best_dir}")
        
        return best_dir

    def tune_quick(
        self,
        ds_name: str = "custom",
        eval_type: str = "synthetic",
        eval_model: str = "catboost",
        prefix: str = "ddpm_tune",
        n_trials: int = 2,
        python_cmd: str = "python"
    ) -> str:
        """
        Run quick hyperparameter tuning for testing (2 trials, 200 steps).
        
        Args:
            ds_name: Dataset name
            eval_type: 'synthetic' or 'merged'
            eval_model: 'catboost' or 'mlp'
            prefix: Prefix for output directory
            n_trials: Number of Optuna trials (default 2 for quick test)
            python_cmd: Python command to use
            
        Returns:
            Path to best model directory
        """
        import subprocess
        
        print(f"Starting QUICK hyperparameter tuning ({n_trials} trials)...")
        print(f"  Dataset: {ds_name}")
        print(f"  Train size: {self.train_size}")
        
        tune_script = os.path.join(self.lib_path, 'scripts', 'tune_ddpm_quick.py')
        
        cmd = [
            python_cmd,
            tune_script,
            ds_name,
            str(self.train_size),
            eval_type,
            eval_model,
            prefix,
            '--n_trials', str(n_trials)
        ]
        
        print(f"  Running: {' '.join(cmd)}")
        
        env = os.environ.copy()
        env['PYTHONPATH'] = self.lib_path
        
        subprocess.run(cmd, check=True, cwd=self.lib_path, env=env)
        
        best_dir = os.path.join(self.lib_path, 'exp', ds_name, f'{prefix}_best')
        self._best_dir = best_dir
        
        print(f"Quick tuning complete! Best model: {best_dir}")
        
        return best_dir