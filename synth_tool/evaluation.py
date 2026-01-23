import sys
import os
import pandas as pd


def prepare_data_for_evaluation(original_df, synthetic_df):
    original_data = original_df.copy()
    synthetic_data = synthetic_df.copy()
    
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
