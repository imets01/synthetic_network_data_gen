import sys
import os
import pandas as pd


def prepare_data_for_evaluation(original_df, synthetic_df):
    # Remove file_id column if present (not needed for evaluation)
    columns_to_remove = ['file_id']
    
    for col in columns_to_remove:
        if col in original_df.columns:
            original_df = original_df.drop(columns=[col])
            print(f"Removed '{col}' column from original data")
        if col in synthetic_df.columns:
            synthetic_df = synthetic_df.drop(columns=[col])
            print(f"Removed '{col}' column from synthetic data")
    
    # Ensure both dataframes have the same columns
    # Use only the columns present in synthetic data
    common_columns = list(synthetic_df.columns)
    
    # Check if all synthetic columns exist in original
    missing_in_original = set(common_columns) - set(original_df.columns)
    if missing_in_original:
        raise ValueError(f"Synthetic data contains columns not present in original data: {missing_in_original}")
    
    # Filter original data to have only the columns in synthetic data
    original_data = original_df[common_columns].copy()
    synthetic_data = synthetic_df.copy()
    
    # Handle missing values by dropping rows
    original_missing = original_data.isnull().sum()
    synthetic_missing = synthetic_data.isnull().sum()
    
    original_rows_before = len(original_data)
    synthetic_rows_before = len(synthetic_data)
    
    if original_missing.any() or synthetic_missing.any():
        print(f"Warning: Missing values detected. Removing rows with missing data...")
        if original_missing.any():
            print(f"   Original data: {original_missing[original_missing > 0].to_dict()}")
        if synthetic_missing.any():
            print(f"   Synthetic data: {synthetic_missing[synthetic_missing > 0].to_dict()}")
        
        # Drop rows with any missing values
        original_data = original_data.dropna()
        synthetic_data = synthetic_data.dropna()
        
        original_rows_after = len(original_data)
        synthetic_rows_after = len(synthetic_data)
        
        if original_rows_before != original_rows_after:
            print(f"   Original: Removed {original_rows_before - original_rows_after} rows ({(original_rows_before - original_rows_after)/original_rows_before*100:.1f}%)")
        if synthetic_rows_before != synthetic_rows_after:
            print(f"   Synthetic: Removed {synthetic_rows_before - synthetic_rows_after} rows ({(synthetic_rows_before - synthetic_rows_after)/synthetic_rows_before*100:.1f}%)")
        
        print(f"✓ Missing values removed successfully.")
    
    # Add source identifier
    original_data['source'] = 'real'
    synthetic_data['source'] = 'synthetic'
    
    # Combine datasets
    combined = pd.concat([original_data, synthetic_data], axis=0)
    
    # Identify categorical columns
    categorical_cols = combined.select_dtypes(include=['object']).columns.tolist()
    if 'source' in categorical_cols:
        categorical_cols.remove('source')
    
    # Encode categorical columns
    if categorical_cols:
        combined_encoded = pd.get_dummies(combined, columns=categorical_cols, dtype=int)
    else:
        combined_encoded = combined
    
    # Split back into original and synthetic
    original_encoded = combined_encoded[combined_encoded['source'] == 'real'].drop('source', axis=1)
    synthetic_encoded = combined_encoded[combined_encoded['source'] == 'synthetic'].drop('source', axis=1)
    
    return original_encoded, synthetic_encoded


def run_privacy_evaluation(original_encoded, synthetic_encoded):
    # Add the FEST_eval directory to the path so synprivutil can be imported
    fest_eval_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FEST_eval'))
    if fest_eval_path not in sys.path:
        sys.path.insert(0, fest_eval_path)

    from synprivutil.privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.privacy_metric_manager import PrivacyMetricManager
    from synprivutil.privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.distance.adversarial_accuracy_class import AdversarialAccuracyCalculator
    from synprivutil.privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.distance.dcr_class import DCRCalculator
    from synprivutil.privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics.distance.nndr_class import NNDRCalculator
    
    privman = PrivacyMetricManager()
    privacy_metrics = [
        AdversarialAccuracyCalculator(original_encoded, synthetic_encoded, original_name="Real", synthetic_name="Synthetic"),
        DCRCalculator(original_encoded, synthetic_encoded, original_name="Real", synthetic_name="Synthetic"),
        NNDRCalculator(original_encoded, synthetic_encoded, original_name="Real", synthetic_name="Synthetic"),
    ]
    privman.add_metric(privacy_metrics)
    
    return privman.evaluate_all()


def run_utility_evaluation(original_encoded, synthetic_encoded):
    # Add the FEST_eval directory to the path so synprivutil can be imported
    fest_eval_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FEST_eval'))
    if fest_eval_path not in sys.path:
        sys.path.insert(0, fest_eval_path)

    from synprivutil.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.utility_metric_manager import UtilityMetricManager
    from synprivutil.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.basic_stats import BasicStatsCalculator
    from synprivutil.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.correlation import CorrelationCalculator
    from synprivutil.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.js_similarity import JSCalculator
    from synprivutil.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.ks_test import KSCalculator
    from synprivutil.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.mutual_information import MICalculator
    # from synprivutil.privacy_utility_framework.privacy_utility_framework.metrics.utility_metrics.statistical.wasserstein import WassersteinCalculator
    
    utman = UtilityMetricManager()
    utility_metrics = [
        BasicStatsCalculator(original_encoded, synthetic_encoded, original_name="Real", synthetic_name="Synthetic"),
        CorrelationCalculator(original_encoded, synthetic_encoded, original_name="Real", synthetic_name="Synthetic"),
        JSCalculator(original_encoded, synthetic_encoded, original_name="Real", synthetic_name="Synthetic"),
        KSCalculator(original_encoded, synthetic_encoded, original_name="Real", synthetic_name="Synthetic"),
        MICalculator(original_encoded, synthetic_encoded, original_name="Real", synthetic_name="Synthetic"),
        # WassersteinCalculator(original_encoded, synthetic_encoded, original_name="Real", synthetic_name="
    ]
    utman.add_metric(utility_metrics)
    
    return utman.evaluate_all()


def evaluate_synthetic_data(original_df, synthetic_df, progress_callback=None):
    # Prepare data
    original_encoded, synthetic_encoded = prepare_data_for_evaluation(original_df, synthetic_df)
    
    if progress_callback:
        progress_callback(33)
    
    # Run privacy evaluation
    privacy_results = run_privacy_evaluation(original_encoded, synthetic_encoded)
    
    if progress_callback:
        progress_callback(66)
    
    # Run utility evaluation
    utility_results = run_utility_evaluation(original_encoded, synthetic_encoded)
    
    if progress_callback:
        progress_callback(100)
    
    return {
        'privacy': privacy_results,
        'utility': utility_results
    }
