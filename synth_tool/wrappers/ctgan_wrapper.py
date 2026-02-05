import pandas as pd
import numpy as np
import torch
from typing import Optional, List, Dict, Any, Tuple, Union
from sdv.metadata import Metadata
from sdv.cag import Inequality, FixedCombinations, Range
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot


# Default CTGAN hyperparameters (tuned values from Optuna optimization)
DEFAULT_CTGAN_PARAMS = {
    'embedding_dim': 128,
    'generator_dim': (256, 512),
    'discriminator_dim': (384, 512),
    'batch_size': 250,
    'epochs': 700,
    'verbose': True,
    'cuda': True,
}


class CTGANHandler:
    def __init__(self):
        # State storage
        self.df: Optional[pd.DataFrame] = None
        self.metadata: Optional[Metadata] = None
        self.synthesizer = None
        
        # Column type tracking
        self.numerical_cols_float: List[str] = []
        self.numerical_cols_int: List[str] = []
        self.categorical_cols: List[str] = []
        self.boolean_cols: List[str] = []
        self.id_cols: List[str] = []
        
        # Columns to include in training
        self.included_cols: List[str] = []
        
        # Constraints
        self.constraints: List[Any] = []
        
        # Stats
        self.removed_negative_rows: int = 0
        
    def load_data(
        self,
        df: pd.DataFrame,
        include_cols: Optional[List[str]] = None,
        remove_negative_rows: bool = True,
        numerical_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        
        print("Loading data...")
        
        if include_cols:
            # Validate that all requested columns exist
            missing = [c for c in include_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Columns not found in DataFrame: {missing}")
            
            self.included_cols = include_cols
            self.df = df[include_cols].copy()
            print(f"  Kept {len(include_cols)} columns")
        else:
            self.df = df.copy()
        
        # Remove rows with negative values in numerical columns
        if remove_negative_rows:
            original_len = len(self.df)
            # Determine which columns to check for negatives
            if numerical_cols:
                cols_to_check = [c for c in numerical_cols if c in self.df.columns]
            else:
                # Auto-detect numerical columns
                cols_to_check = self.df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
            
            if cols_to_check:
                for col in cols_to_check:
                    self.df = self.df[self.df[col] >= 0]
                removed_rows = original_len - len(self.df)
                if removed_rows > 0:
                    print(f"  Removed {removed_rows} rows with negative values in numerical columns")
                    self.removed_negative_rows = removed_rows
        
        print(f"  Loaded: {len(self.df):,} rows, {len(self.df.columns)} columns")
        return self.df
    
    def detect_metadata(self) -> Metadata:

        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("Detecting metadata from DataFrame...")
        self.metadata = Metadata.detect_from_dataframe(self.df)
        print("  Metadata detected successfully")
        return self.metadata
    
    def _update_columns(
        self,
        columns: List[str],
        sdtype: str,
        computer_representation: Optional[str] = None
    ) -> None:

        if self.metadata is None:
            raise ValueError("No metadata available. Call detect_metadata() first.")
        
        for col in columns:
            if col not in self.df.columns:
                print(f"  Warning: Column '{col}' not found in data, skipping")
                continue
                
            self.metadata.update_column(column_name=col, sdtype=sdtype)
            
            if computer_representation is not None:
                self.metadata.update_column(
                    column_name=col,
                    computer_representation=computer_representation
                )
    
    def set_numerical_float_columns(self, columns: List[str]) -> 'CTGANHandler':

        print(f"Setting {len(columns)} columns as numerical (Float)...")
        self.numerical_cols_float = columns
        self._update_columns(columns, 'numerical', 'Float')
        return self
    
    def set_numerical_int_columns(self, columns: List[str]) -> 'CTGANHandler':

        print(f"Setting {len(columns)} columns as numerical (Int64)...")
        self.numerical_cols_int = columns
        self._update_columns(columns, 'numerical', 'Int64')
        return self
    
    def set_categorical_columns(self, columns: List[str]) -> 'CTGANHandler':

        print(f"Setting {len(columns)} columns as categorical...")
        self.categorical_cols = columns
        self._update_columns(columns, 'categorical')
        return self
    
    def set_boolean_columns(self, columns: List[str]) -> 'CTGANHandler':
 
        print(f"Setting {len(columns)} columns as boolean...")
        self.boolean_cols = columns
        self._update_columns(columns, 'boolean')
        return self
    
    def set_id_columns(self, columns: List[str]) -> 'CTGANHandler':

        print(f"Setting {len(columns)} columns as ID...")
        self.id_cols = columns
        self._update_columns(columns, 'id')
        return self
    
    def configure_metadata(
        self,
        numerical_float: Optional[List[str]] = None,
        numerical_int: Optional[List[str]] = None,
        categorical: Optional[List[str]] = None,
        boolean: Optional[List[str]] = None,
        id_cols: Optional[List[str]] = None
    ) -> Metadata:

        if self.metadata is None:
            self.detect_metadata()
        
        print("Configuring metadata...")
        
        if numerical_float:
            self.set_numerical_float_columns(numerical_float)
        if numerical_int:
            self.set_numerical_int_columns(numerical_int)
        if categorical:
            self.set_categorical_columns(categorical)
        if boolean:
            self.set_boolean_columns(boolean)
        if id_cols:
            self.set_id_columns(id_cols)
        
        print("Metadata configuration complete!")
        return self.metadata
    
    def get_metadata_summary(self) -> Dict[str, Any]:

        if self.metadata is None:
            raise ValueError("No metadata available. Call detect_metadata() first.")
        
        # Get column info from metadata
        columns_info = self.metadata.to_dict().get('columns', {})
        
        summary = {
            'total_columns': len(columns_info),
            'numerical_float': self.numerical_cols_float,
            'numerical_int': self.numerical_cols_int,
            'categorical': self.categorical_cols,
            'boolean': self.boolean_cols,
            'id': self.id_cols,
            'columns_detail': columns_info
        }
        
        return summary
    
    def visualize_metadata(self):
 
        if self.metadata is None:
            raise ValueError("No metadata available. Call detect_metadata() first.")
        
        return self.metadata.visualize()
    
    def prepare_data_for_training(self) -> pd.DataFrame:

        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("Preparing data for training...")
        df_prepared = self.df.copy()
        
        # Convert boolean columns from 0/1 to bool
        for col in self.boolean_cols:
            if col in df_prepared.columns:
                df_prepared[col] = df_prepared[col].astype(bool)
                print(f"  Converted '{col}' to boolean")
        
        print("Data preparation complete!")
        return df_prepared
    
    def add_inequality_constraint(
        self,
        low_column: str,
        high_column: str,
        strict: bool = True
    ) -> 'CTGANHandler':
  
        constraint = Inequality(
            low_column_name=low_column,
            high_column_name=high_column,
            strict_boundaries=strict
        )
        self.constraints.append(constraint)
        print(f"  Added inequality constraint: {low_column} < {high_column}")
        return self
    
    def add_range_constraint(
        self,
        column_name: str,
        low_value: float,
        high_value: float,
        strict: bool = False
    ) -> 'CTGANHandler':
 
        constraint = Range(
            column_name=column_name,
            low_value=low_value,
            high_value=high_value,
            strict_boundaries=strict
        )
        self.constraints.append(constraint)
        print(f"  Added range constraint: {low_value} <= {column_name} <= {high_value}")
        return self
    
    def add_fixed_combinations_constraint(
        self,
        column_names: List[str]
    ) -> 'CTGANHandler':

        constraint = FixedCombinations(column_names=column_names)
        self.constraints.append(constraint)
        print(f"  Added fixed combinations constraint for: {column_names}")
        return self
    
    def clear_constraints(self) -> 'CTGANHandler':
        self.constraints = []
        print("  Cleared all constraints")
        return self
    
    def get_constraints(self) -> List[Any]:
        return self.constraints
    
    def get_constraints_summary(self) -> List[Dict[str, Any]]:
        summary = []
        for i, c in enumerate(self.constraints):
            constraint_info = {
                'index': i,
                'type': type(c).__name__,
            }
            if isinstance(c, Inequality):
                constraint_info['low_column'] = c._low_column_name
                constraint_info['high_column'] = c._high_column_name
            elif isinstance(c, Range):
                constraint_info['column'] = c._column_name
                constraint_info['low_value'] = c._low_value
                constraint_info['high_value'] = c._high_value
            elif isinstance(c, FixedCombinations):
                constraint_info['columns'] = c._column_names
            summary.append(constraint_info)
        return summary
    
    def create_synthesizer(
        self,
        embedding_dim: Optional[int] = None,
        generator_dim: Optional[Tuple[int, int]] = None,
        discriminator_dim: Optional[Tuple[int, int]] = None,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        verbose: Optional[bool] = None,
        cuda: Optional[bool] = None,
    ) -> CTGANSynthesizer:

        if self.metadata is None:
            raise ValueError("No metadata available. Call detect_metadata() first.")
        
        # Build params from defaults, overriding with any provided values
        params = DEFAULT_CTGAN_PARAMS.copy()
        
        if embedding_dim is not None:
            params['embedding_dim'] = embedding_dim
        if generator_dim is not None:
            params['generator_dim'] = generator_dim
        if discriminator_dim is not None:
            params['discriminator_dim'] = discriminator_dim
        if batch_size is not None:
            params['batch_size'] = batch_size
        if epochs is not None:
            params['epochs'] = epochs
        if verbose is not None:
            params['verbose'] = verbose
        if cuda is not None:
            params['cuda'] = cuda
        else:
            # Auto-detect GPU availability
            params['cuda'] = torch.cuda.is_available()
        
        print("Creating CTGANSynthesizer...")
        print(f"  embedding_dim: {params['embedding_dim']}")
        print(f"  generator_dim: {params['generator_dim']}")
        print(f"  discriminator_dim: {params['discriminator_dim']}")
        print(f"  batch_size: {params['batch_size']}")
        print(f"  epochs: {params['epochs']}")
        print(f"  cuda: {params['cuda']} (GPU {'available' if torch.cuda.is_available() else 'not available'})")
        
        self.synthesizer = CTGANSynthesizer(
            metadata=self.metadata,
            embedding_dim=params['embedding_dim'],
            generator_dim=params['generator_dim'],
            discriminator_dim=params['discriminator_dim'],
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            verbose=params['verbose'],
            cuda=params['cuda'],
        )
        
        # Store params for reference
        self._training_params = params
        
        # Add constraints if any
        if self.constraints:
            print(f"  Adding {len(self.constraints)} constraints...")
            self.synthesizer.add_constraints(constraints=self.constraints)
        
        print("Synthesizer created successfully!")
        return self.synthesizer
    
    def fit(
        self,
        df: Optional[pd.DataFrame] = None,
        **synthesizer_kwargs
    ) -> 'CTGANHandler':

        # Create synthesizer if not already created
        if self.synthesizer is None:
            self.create_synthesizer(**synthesizer_kwargs)
        
        # Prepare training data
        if df is not None:
            training_data = df
        else:
            training_data = self.prepare_data_for_training()
        
        print(f"Training CTGAN on {len(training_data):,} rows...")
        self.synthesizer.fit(training_data)
        print("Training complete!")
        
        return self
    
    def get_training_params(self) -> Dict[str, Any]:
        if not hasattr(self, '_training_params'):
            return DEFAULT_CTGAN_PARAMS.copy()
        return self._training_params.copy()
    
    def save_model(self, filepath: str) -> None:
        if self.synthesizer is None:
            raise ValueError("No synthesizer to save. Train the model first.")
        
        self.synthesizer.save(filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> 'CTGANHandler':
        self.synthesizer = CTGANSynthesizer.load(filepath)
        print(f"Model loaded from: {filepath}")
        return self
    
    def sample(self, num_rows: int) -> pd.DataFrame:
        if self.synthesizer is None:
            raise ValueError("No trained synthesizer. Train the model first.")
        
        print(f"Generating {num_rows:,} synthetic samples...")
        self.synthetic_data = self.synthesizer.sample(num_rows=num_rows)
        print(f"Generated {len(self.synthetic_data):,} samples successfully!")
        
        return self.synthetic_data
    
    def save_synthetic_data(self, filepath: str) -> None:
        """
        Save generated synthetic data to a CSV file.
        
        Args:
            filepath: Path to save the CSV file
        """
        if not hasattr(self, 'synthetic_data') or self.synthetic_data is None:
            raise ValueError("No synthetic data to save. Call sample() first.")
        
        self.synthetic_data.to_csv(filepath, index=False)
        print(f"Synthetic data saved to: {filepath}")
    
    def run_diagnostic(
        self,
        real_data: Optional[pd.DataFrame] = None,
        synthetic_data: Optional[pd.DataFrame] = None
    ) -> Any:

        if self.metadata is None:
            raise ValueError("No metadata available. Call detect_metadata() first.")
        
        if real_data is None:
            real_data = self.prepare_data_for_training()
        if synthetic_data is None:
            if not hasattr(self, 'synthetic_data') or self.synthetic_data is None:
                raise ValueError("No synthetic data. Call sample() first.")
            synthetic_data = self.synthetic_data
        
        print("Running diagnostic checks...")
        diagnostic = run_diagnostic(real_data, synthetic_data, self.metadata)
        self._diagnostic = diagnostic
        return diagnostic
    
    def evaluate_quality(
        self,
        real_data: Optional[pd.DataFrame] = None,
        synthetic_data: Optional[pd.DataFrame] = None
    ) -> Any:

        if self.metadata is None:
            raise ValueError("No metadata available. Call detect_metadata() first.")
        
        if real_data is None:
            real_data = self.prepare_data_for_training()
        if synthetic_data is None:
            if not hasattr(self, 'synthetic_data') or self.synthetic_data is None:
                raise ValueError("No synthetic data. Call sample() first.")
            synthetic_data = self.synthetic_data
        
        print("Evaluating quality...")
        quality_report = evaluate_quality(real_data, synthetic_data, self.metadata)
        self._quality_report = quality_report
        print(f"Overall quality score: {quality_report.get_score():.4f}")
        return quality_report
    
    def get_column_shapes_visualization(self):
        if not hasattr(self, '_quality_report'):
            raise ValueError("No quality report. Call evaluate_quality() first.")
        return self._quality_report.get_visualization(property_name='Column Shapes')
    
    def get_column_shapes_details(self) -> pd.DataFrame:
        if not hasattr(self, '_quality_report'):
            raise ValueError("No quality report. Call evaluate_quality() first.")
        return self._quality_report.get_details(property_name='Column Shapes')
    
    def get_column_pair_trends_visualization(self):
        if not hasattr(self, '_quality_report'):
            raise ValueError("No quality report. Call evaluate_quality() first.")
        return self._quality_report.get_visualization(property_name='Column Pair Trends')
    
    def get_column_pair_trends_details(self) -> pd.DataFrame:
        if not hasattr(self, '_quality_report'):
            raise ValueError("No quality report. Call evaluate_quality() first.")
        return self._quality_report.get_details(property_name='Column Pair Trends')
    
    def plot_column(
        self,
        column_name: str,
        real_data: Optional[pd.DataFrame] = None,
        synthetic_data: Optional[pd.DataFrame] = None,
        plot_type: Optional[str] = None
    ):

        if self.metadata is None:
            raise ValueError("No metadata available. Call detect_metadata() first.")
        
        if real_data is None:
            real_data = self.prepare_data_for_training()
        if synthetic_data is None:
            if not hasattr(self, 'synthetic_data') or self.synthetic_data is None:
                raise ValueError("No synthetic data. Call sample() first.")
            synthetic_data = self.synthetic_data
        
        fig = get_column_plot(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=self.metadata,
            column_name=column_name,
            plot_type=plot_type
        )
        return fig
    
    def get_quality_score(self) -> float:
        if not hasattr(self, '_quality_report'):
            raise ValueError("No quality report. Call evaluate_quality() first.")
        return self._quality_report.get_score()
