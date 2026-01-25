import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# High-level feature definitions for ML utility evaluation
IDEAL_HIGH_LEVEL_FEATURES = [
    'handshake_duration', 'time_to_migration', 'migration_duration', 
    'packets_before_migration', 'total_bidi_streams_client_init', 
    'total_udi_streams_client_init', 'bytes_sent_client', 
    'bytes_sent_server', 'retry_occurred', 'path_validation_initiated',
    'version_negotiation_occurred', 'connection_close_type'
]

HIGH_LEVEL_CATEGORICAL_FEATURES = [
    'retry_occurred', 'path_validation_initiated', 
    'version_negotiation_occurred', 'connection_close_type'
]

# Columns to always exclude from evaluation (they are IDs, not meaningful features)
EXCLUDED_COLUMNS = ['frame_number', 'capture_id', 'file_id']

# Low-level feature columns for ML utility evaluation
LOW_LEVEL_FEATURE_COLUMNS = [
    'delta_time', 'packet_length', 'packet_direction', 'header_form',
    'count_initial', 'count_0rtt', 'count_handshake', 'count_1rtt',
    'count_retry', 'count_vn', 'count_ack', 'count_padding',
    'count_connection_close', 'count_path_challenge', 'count_path_response',
    'count_new_connection_id', 'count_retire_cid', 'count_ping',
    'count_crypto', 'count_handshake_done', 'http3_stream_count',
    'http3_fin_count', 'stream_length', 'stream_type_count'
]


def prepare_data_for_evaluation(original_df, synthetic_df):
    # Remove excluded columns (IDs that shouldn't be compared)
    original_df = original_df.copy()
    synthetic_df = synthetic_df.copy()
    
    for col in EXCLUDED_COLUMNS:
        if col in original_df.columns:
            original_df = original_df.drop(columns=[col])
            print(f"Removed '{col}' column from original data (excluded from evaluation)")
        if col in synthetic_df.columns:
            synthetic_df = synthetic_df.drop(columns=[col])
            print(f"Removed '{col}' column from synthetic data (excluded from evaluation)")
    
    # Find common columns between original and synthetic data
    original_cols = set(original_df.columns)
    synthetic_cols = set(synthetic_df.columns)
    common_columns = list(original_cols & synthetic_cols)
    
    # Report column differences
    only_in_original = original_cols - synthetic_cols
    only_in_synthetic = synthetic_cols - original_cols
    
    if only_in_original:
        print(f"Columns only in original data (excluded): {only_in_original}")
    if only_in_synthetic:
        print(f"Columns only in synthetic data (excluded): {only_in_synthetic}")
    
    if not common_columns:
        raise ValueError("No common columns found between original and synthetic data!")
    
    print(f"Using {len(common_columns)} common columns for evaluation")
    
    # Filter both dataframes to use only common columns
    original_data = original_df[common_columns].copy()
    synthetic_data = synthetic_df[common_columns].copy()
    
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
    # Add the FEST_eval directory and synprivutil to the path so imports work correctly
    fest_eval_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FEST_eval'))
    synprivutil_path = os.path.join(fest_eval_path, 'synprivutil')
    
    if fest_eval_path not in sys.path:
        sys.path.insert(0, fest_eval_path)
    if synprivutil_path not in sys.path:
        sys.path.insert(0, synprivutil_path)

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


def aggregate_sequences_for_privacy(low_level_df, feature_columns, capture_id_col='capture_id'):
    """
    Aggregate sequences into fixed-size representations for privacy evaluation.
    
    For each sequence (capture_id), computes:
    - Mean and std of each feature
    - Sequence length
    
    Args:
        low_level_df: DataFrame with packet-level data
        feature_columns: List of feature columns to aggregate
        capture_id_col: Column containing sequence IDs
    
    Returns:
        DataFrame with one row per sequence, containing aggregated features
    """
    # Get available feature columns
    available_features = [col for col in feature_columns if col in low_level_df.columns]
    
    if not available_features:
        raise ValueError("No feature columns found in data for aggregation")
    
    # Group by capture_id and compute statistics
    agg_dict = {}
    for col in available_features:
        agg_dict[f'{col}_mean'] = (col, 'mean')
        agg_dict[f'{col}_std'] = (col, 'std')
    agg_dict['sequence_length'] = (capture_id_col, 'count')
    
    aggregated = low_level_df.groupby(capture_id_col).agg(**agg_dict).reset_index()
    
    # Fill NaN std values (happens when sequence has only 1 packet) with 0
    std_cols = [col for col in aggregated.columns if col.endswith('_std')]
    aggregated[std_cols] = aggregated[std_cols].fillna(0)
    
    # Drop the capture_id column as we don't want it in the privacy comparison
    aggregated = aggregated.drop(columns=[capture_id_col])
    
    return aggregated


def run_sequence_level_privacy_evaluation(original_low_df, synthetic_low_df, feature_columns=None):
    """
    Run privacy evaluation at the sequence level for low-level data.
    
    Aggregates sequences into fixed-size representations and computes privacy metrics.
    
    Args:
        original_low_df: Original low-level data with capture_id column
        synthetic_low_df: Synthetic low-level data with capture_id column
        feature_columns: List of feature columns (defaults to LOW_LEVEL_FEATURE_COLUMNS)
    
    Returns:
        Dictionary with privacy evaluation results
    """
    if feature_columns is None:
        feature_columns = LOW_LEVEL_FEATURE_COLUMNS
    
    # Check for capture_id column
    if 'capture_id' not in original_low_df.columns:
        return {'error': 'Original data must have capture_id column for sequence-level privacy'}
    if 'capture_id' not in synthetic_low_df.columns:
        return {'error': 'Synthetic data must have capture_id column for sequence-level privacy'}
    
    print("Aggregating sequences for privacy evaluation...")
    
    # Aggregate sequences
    try:
        original_agg = aggregate_sequences_for_privacy(original_low_df, feature_columns)
        synthetic_agg = aggregate_sequences_for_privacy(synthetic_low_df, feature_columns)
    except Exception as e:
        return {'error': f'Failed to aggregate sequences: {str(e)}'}
    
    print(f"  Original: {len(original_agg)} sequences")
    print(f"  Synthetic: {len(synthetic_agg)} sequences")
    
    # Ensure same columns
    common_cols = list(set(original_agg.columns) & set(synthetic_agg.columns))
    original_agg = original_agg[common_cols]
    synthetic_agg = synthetic_agg[common_cols]
    
    # Scale the data
    scaler = MinMaxScaler()
    original_scaled = scaler.fit_transform(original_agg)
    synthetic_scaled = scaler.transform(synthetic_agg)
    
    original_encoded = pd.DataFrame(original_scaled, columns=common_cols)
    synthetic_encoded = pd.DataFrame(synthetic_scaled, columns=common_cols)
    
    # Run privacy evaluation
    print("Computing sequence-level privacy metrics...")
    privacy_results = run_privacy_evaluation(original_encoded, synthetic_encoded)
    
    # Add metadata
    privacy_results['_metadata'] = {
        'evaluation_level': 'sequence',
        'num_original_sequences': len(original_agg),
        'num_synthetic_sequences': len(synthetic_agg),
        'num_features': len(common_cols)
    }
    
    return privacy_results


def run_utility_evaluation(original_encoded, synthetic_encoded):
    # Add the FEST_eval directory and synprivutil to the path so imports work correctly
    fest_eval_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FEST_eval'))
    synprivutil_path = os.path.join(fest_eval_path, 'synprivutil')
    
    if fest_eval_path not in sys.path:
        sys.path.insert(0, fest_eval_path)
    if synprivutil_path not in sys.path:
        sys.path.insert(0, synprivutil_path)

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


def prepare_data_for_ml_utility(training_data, test_data, feature_list, categorical_features, target_column='implementation'):

    training_data = training_data.copy()
    test_data = test_data.copy()

    # Standardizing Data - rename Target to implementation if needed
    if "Target" in training_data.columns:
        training_data = training_data.rename(columns={"Target": target_column})
    
    # Convert Yes/No to 1/0 for certain columns
    cols_to_fix = ['version_negotiation_occurred', 'retry_occurred', 'path_validation_initiated']
    for col in cols_to_fix:
        if col in training_data.columns:
            training_data[col] = training_data[col].replace({'Yes': 1, 'No': 0})
        if col in test_data.columns:
            test_data[col] = test_data[col].replace({'Yes': 1, 'No': 0})
    
    # Clean missing values and reset index
    training_data = training_data.dropna().reset_index(drop=True)
    test_data = test_data.dropna().reset_index(drop=True)

    # Get available features
    available_features = [f for f in feature_list if f in training_data.columns]
    
    return training_data, test_data, available_features


def train_and_evaluate_ml_utility(training_data, test_data, features, categorical_features, target_column='implementation'):

    training_data = training_data.copy()
    test_data = test_data.copy()
    
    # Encode categorical columns
    current_categorical_columns = [c for c in categorical_features if c in features]
    for col in current_categorical_columns:
        le = LabelEncoder()
        le.fit(pd.concat([training_data[col].astype(str), test_data[col].astype(str)]))
        training_data[col] = le.transform(training_data[col].astype(str))
        test_data[col] = le.transform(test_data[col].astype(str))

    # Prepare target variable
    y_train_raw = training_data[target_column]
    if len(y_train_raw.shape) > 1:
        y_train_raw = y_train_raw.iloc[:, 0]
    
    y_test_raw = test_data[target_column]
    if len(y_test_raw.shape) > 1:
        y_test_raw = y_test_raw.iloc[:, 0]

    le_target = LabelEncoder()
    y_train = le_target.fit_transform(y_train_raw.astype(str))
    y_test = le_target.transform(y_test_raw.astype(str))
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(training_data[features], y_train)
    predictions = model.predict(test_data[features])
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=le_target.classes_, 
                                   zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, predictions)
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = dict(zip(features, importance.tolist()))
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_importance,
        'class_labels': le_target.classes_.tolist(),
        'y_true': y_test.tolist(),
        'y_pred': predictions.tolist()
    }


def run_ml_utility_evaluation(synthetic_df, real_df, evaluation_type='high_level', target_column='implementation'):

    if evaluation_type == 'high_level':
        feature_list = IDEAL_HIGH_LEVEL_FEATURES
        categorical_features = HIGH_LEVEL_CATEGORICAL_FEATURES
    else:
        # For low-level, use all numeric columns except target
        feature_list = [col for col in synthetic_df.columns 
                       if col != target_column and synthetic_df[col].dtype in ['int64', 'float64']]
        categorical_features = []
    
    # Prepare data
    training_data, test_data, available_features = prepare_data_for_ml_utility(
        synthetic_df, real_df, feature_list, categorical_features, target_column
    )
    
    if not available_features:
        return {
            'error': 'No valid features found for ML utility evaluation',
            'available_features': [],
            'missing_features': feature_list
        }
    
    # Check if target column exists
    if target_column not in training_data.columns:
        return {
            'error': f'Target column "{target_column}" not found in synthetic data',
            'available_columns': training_data.columns.tolist()
        }
    
    if target_column not in test_data.columns:
        return {
            'error': f'Target column "{target_column}" not found in real data',
            'available_columns': test_data.columns.tolist()
        }
    
    # Run evaluation
    results = train_and_evaluate_ml_utility(
        training_data, test_data, available_features, categorical_features, target_column
    )
    
    # Add metadata
    results['evaluation_type'] = evaluation_type
    results['features_used'] = available_features
    results['missing_features'] = list(set(feature_list) - set(available_features))
    results['num_training_samples'] = len(training_data)
    results['num_test_samples'] = len(test_data)
    
    return results


# =========================================================================
# Low-Level LSTM-based ML Utility Evaluation
# =========================================================================

class LowLevelSequenceDataset(Dataset):
    """
    Dataset class for loading low-level packet sequences.
    Each sequence represents a QUIC connection capture.
    """
    def __init__(self, low_level_df, id_to_implementation, capture_ids, scaler=None, feature_columns=None):
        """
        Args:
            low_level_df: DataFrame with low-level packet features
            id_to_implementation: Dict mapping capture_id to implementation
            capture_ids: List of capture_ids to include in this dataset
            scaler: Pre-fitted scaler (for test set) or None (for train set)
            feature_columns: List of feature columns to use
        """
        self.low_level_df = low_level_df
        self.id_to_implementation = id_to_implementation
        self.capture_ids = capture_ids
        
        # Features to use (exclude frame_number and capture_id)
        if feature_columns is not None:
            self.feature_columns = feature_columns
        else:
            self.feature_columns = LOW_LEVEL_FEATURE_COLUMNS
        
        # Filter to only available columns
        self.feature_columns = [col for col in self.feature_columns if col in low_level_df.columns]
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        all_implementations = list(set(id_to_implementation.values()))
        self.label_encoder.fit(all_implementations)
        
        # Fit or use provided scaler
        if scaler is None:
            self.scaler = MinMaxScaler()
            # Fit scaler on all features for training capture_ids
            train_data = self.low_level_df[self.low_level_df['capture_id'].isin(capture_ids)]
            if len(train_data) > 0 and len(self.feature_columns) > 0:
                self.scaler.fit(train_data[self.feature_columns].values)
        else:
            self.scaler = scaler
    
    def __len__(self):
        return len(self.capture_ids)
    
    def __getitem__(self, idx):
        capture_id = self.capture_ids[idx]
        
        # Get sequence for this capture
        sequence_df = self.low_level_df[self.low_level_df['capture_id'] == capture_id]
        sequence_df = sequence_df[self.feature_columns]
        
        # Scale features
        scaled_sequence = self.scaler.transform(sequence_df.values)
        
        # Get label
        implementation = self.id_to_implementation[capture_id]
        label = self.label_encoder.transform([implementation])[0]
        
        return {
            'sequence': torch.FloatTensor(scaled_sequence),
            'label': torch.LongTensor([label])[0],
            'capture_id': capture_id
        }


def collate_fn_lstm(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads sequences to the maximum length in the batch.
    """
    sequences = [item['sequence'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    # Get sequence lengths before padding
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    # Pad sequences
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    return {
        'sequence': padded_sequences,
        'label': labels,
        'lengths': lengths
    }


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for packet sequences.
    Takes variable-length sequences and predicts implementation.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=4, dropout=0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer for sequence aggregation
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim) - padded sequences
            lengths: (batch_size,) - actual sequence lengths
        """
        batch_size = x.shape[0]
        
        # Pack sequences if lengths provided
        if lengths is not None:
            # Sort by length for packing
            lengths_sorted, sort_idx = lengths.sort(descending=True)
            x_sorted = x[sort_idx]
            
            # Pack padded sequence
            packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
            )
            
            # LSTM forward
            lstm_out, (hn, cn) = self.lstm(packed)
            
            # Unpack
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            
            # Unsort to original order
            _, unsort_idx = sort_idx.sort()
            lstm_out = lstm_out[unsort_idx]
        else:
            lstm_out, (hn, cn) = self.lstm(x)
        
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Classification
        output = self.classifier(context)
        
        return output


def train_lstm_epoch(model, dataloader, optimizer, criterion, device):
    """Train LSTM for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        sequences = batch['sequence'].to(device)
        labels = batch['label'].to(device)
        lengths = batch['lengths']
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sequences, lengths)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(dataloader), correct / total


def evaluate_lstm(model, dataloader, criterion, device):
    """Evaluate LSTM model on dataloader."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            lengths = batch['lengths']
            
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), np.array(all_preds), np.array(all_labels)


def train_lstm_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=15, progress_callback=None):
    """Full training loop with validation."""
    best_val_acc = 0
    best_model_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_lstm_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_preds, val_labels = evaluate_lstm(model, val_loader, criterion, device)
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if progress_callback:
            progress_callback(epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def run_low_level_ml_utility_lstm(synthetic_low_df, real_low_df, 
                                   synthetic_high_df=None, real_high_df=None,
                                   target_column='implementation', 
                                   epochs=10, batch_size=32, 
                                   max_sequences_per_impl=None,
                                   progress_callback=None):
    """
    Run LSTM-based ML utility evaluation for low-level sequential data.
    Trains on synthetic data, tests on real data (TSTR evaluation).
    
    Args:
        synthetic_low_df: Synthetic low-level data with capture_id column
        real_low_df: Real low-level data with capture_id column
        synthetic_high_df: Synthetic high-level data with file_id and implementation columns
        real_high_df: Real high-level data with file_id and implementation columns
        target_column: Column containing implementation labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        max_sequences_per_impl: Max sequences per implementation (for balanced sampling)
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with evaluation results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check for required columns in low-level data
    if 'capture_id' not in synthetic_low_df.columns:
        return {'error': 'Synthetic low-level data must have a "capture_id" column for sequence grouping'}
    if 'capture_id' not in real_low_df.columns:
        return {'error': 'Real low-level data must have a "capture_id" column for sequence grouping'}
    
    # Get feature columns (exclude IDs and target)
    available_features = [col for col in LOW_LEVEL_FEATURE_COLUMNS 
                         if col in synthetic_low_df.columns and col in real_low_df.columns]
    
    if len(available_features) == 0:
        return {'error': 'No common feature columns found between synthetic and real data'}
    
    # Create implementation mapping
    # If high-level data is provided, use it; otherwise try to get from low-level data
    if synthetic_high_df is not None and real_high_df is not None:
        # Use file_id from high-level to map to implementation
        id_col = 'file_id' if 'file_id' in synthetic_high_df.columns else 'capture_id'
        
        if id_col not in synthetic_high_df.columns:
            return {'error': f'High-level data must have "{id_col}" column to map to low-level capture_id'}
        if target_column not in synthetic_high_df.columns:
            return {'error': f'Target column "{target_column}" not found in synthetic high-level data'}
        if target_column not in real_high_df.columns:
            return {'error': f'Target column "{target_column}" not found in real high-level data'}
        
        synthetic_impl_map = dict(zip(synthetic_high_df[id_col], synthetic_high_df[target_column]))
        real_impl_map = dict(zip(real_high_df[id_col], real_high_df[target_column]))
    else:
        # Try to get implementation from low-level data directly
        if target_column not in synthetic_low_df.columns:
            return {'error': f'Target column "{target_column}" not found in synthetic data. Please provide high-level data.'}
        if target_column not in real_low_df.columns:
            return {'error': f'Target column "{target_column}" not found in real data. Please provide high-level data.'}
        
        synthetic_impl_map = synthetic_low_df.groupby('capture_id')[target_column].first().to_dict()
        real_impl_map = real_low_df.groupby('capture_id')[target_column].first().to_dict()
    
    # Get capture_ids that exist in both low-level data and have implementation labels
    synthetic_low_capture_ids = set(synthetic_low_df['capture_id'].unique())
    real_low_capture_ids = set(real_low_df['capture_id'].unique())
    
    synthetic_capture_ids = [cid for cid in synthetic_low_capture_ids if cid in synthetic_impl_map]
    real_capture_ids = [cid for cid in real_low_capture_ids if cid in real_impl_map]
    
    if len(synthetic_capture_ids) == 0:
        return {'error': 'No synthetic capture_ids found with matching implementation labels'}
    if len(real_capture_ids) == 0:
        return {'error': 'No real capture_ids found with matching implementation labels'}
    
    # Get unique classes
    synthetic_implementations = [synthetic_impl_map[cid] for cid in synthetic_capture_ids]
    real_implementations = [real_impl_map[cid] for cid in real_capture_ids]
    all_implementations = list(set(synthetic_implementations + real_implementations))
    num_classes = len(all_implementations)
    
    # Balanced sampling if max_sequences_per_impl is specified
    if max_sequences_per_impl is not None:
        # Sample balanced capture_ids for synthetic data
        synthetic_capture_ids = _balanced_sample_capture_ids(
            synthetic_capture_ids, synthetic_impl_map, all_implementations, max_sequences_per_impl
        )
        # Sample balanced capture_ids for real data
        real_capture_ids = _balanced_sample_capture_ids(
            real_capture_ids, real_impl_map, all_implementations, max_sequences_per_impl
        )
    
    if len(synthetic_capture_ids) < 2:
        return {'error': f'Need at least 2 synthetic captures, found {len(synthetic_capture_ids)}'}
    if len(real_capture_ids) < 2:
        return {'error': f'Need at least 2 real captures, found {len(real_capture_ids)}'}
    
    try:
        # Create synthetic training dataset
        train_dataset = LowLevelSequenceDataset(
            low_level_df=synthetic_low_df,
            id_to_implementation=synthetic_impl_map,
            capture_ids=synthetic_capture_ids,
            scaler=None,
            feature_columns=available_features
        )
        
        # Create real test dataset (using scaler from synthetic)
        test_dataset = LowLevelSequenceDataset(
            low_level_df=real_low_df,
            id_to_implementation=real_impl_map,
            capture_ids=real_capture_ids,
            scaler=train_dataset.scaler,
            feature_columns=available_features
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_lstm)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_lstm)
        
        # Initialize model
        input_dim = len(available_features)
        model = LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=64,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.3
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train
        model, history = train_lstm_model(
            model, train_loader, test_loader, optimizer, criterion, device, 
            epochs=epochs, progress_callback=progress_callback
        )
        
        # Final evaluation
        _, predictions, true_labels = evaluate_lstm(model, test_loader, criterion, device)
        accuracy = accuracy_score(true_labels, predictions)
        
        # Get class names
        class_names = train_dataset.label_encoder.classes_
        
        # Calculate metrics
        report = classification_report(true_labels, predictions, target_names=class_names, 
                                       zero_division=0, output_dict=True)
        cm = confusion_matrix(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_labels': class_names.tolist(),
            'y_true': true_labels.tolist(),
            'y_pred': predictions.tolist(),
            'training_history': history,
            'features_used': available_features,
            'num_training_sequences': len(synthetic_capture_ids),
            'num_test_sequences': len(real_capture_ids),
            'device': str(device),
            'evaluation_type': 'low_level_lstm'
        }
        
    except Exception as e:
        return {'error': f'LSTM evaluation failed: {str(e)}'}


def _balanced_sample_capture_ids(capture_ids, impl_map, all_implementations, max_per_impl):
    """
    Sample capture_ids with balanced representation across implementations.
    
    Args:
        capture_ids: List of capture_ids to sample from
        impl_map: Dict mapping capture_id to implementation
        all_implementations: List of all possible implementations
        max_per_impl: Maximum number of capture_ids per implementation
    
    Returns:
        List of sampled capture_ids
    """
    import random
    
    # Group capture_ids by implementation
    impl_to_captures = {impl: [] for impl in all_implementations}
    for cid in capture_ids:
        if cid in impl_map:
            impl = impl_map[cid]
            if impl in impl_to_captures:
                impl_to_captures[impl].append(cid)
    
    # Sample from each implementation
    sampled_ids = []
    for impl in all_implementations:
        available = impl_to_captures.get(impl, [])
        if len(available) > 0:
            n_sample = min(len(available), max_per_impl)
            sampled = random.sample(available, n_sample)
            sampled_ids.extend(sampled)
    
    return sampled_ids
