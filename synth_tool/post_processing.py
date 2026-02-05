"""
Post-processing rules for high-level synthetic network data.

Applies domain-specific constraints to ensure generated data is valid:
- Non-negativity constraints
- Duration constraints (sub-durations <= connection_duration)
- Integer column rounding
- Logical temporal constraints
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


# =========================================================================
# Column Definitions
# =========================================================================

# Duration columns that should be <= connection_duration
DURATION_COLS = [
    'handshake_duration', 'time_to_migration', 'migration_duration',
    'first_path_validation_response_latency'
]

# Columns that should be integers (counts, bytes, packets)
INTEGER_COLS = [
    'packets_before_migration', 'total_bidi_streams_client_init',
    'total_bidi_streams_server_init', 'total_udi_streams_client_init',
    'total_udi_streams_server_init', 'bytes_sent_client', 'bytes_sent_server',
    'packets_sent_client', 'packets_sent_server', 'quic_packets_sent_client',
    'quic_packets_sent_server', 'new_connection_ids_issued_server',
    'new_connection_ids_issued_client', 'retired_cid_count_client',
    'retired_cid_count_server', 'ack_sent_client', 'ack_sent_server',
    'crypto_sent_client', 'crypto_sent_server', 'handshake_done_client',
    'handshake_done_server', 'path_challenge_sent_client',
    'path_challenge_sent_server', 'path_response_sent_client',
    'path_response_sent_server', 'mtu', 'app_data_bytes_before_migration',
    'padding_bytes_in_validation_pc', 'padding_bytes_in_validation_pr',
    'bytes_bidi_streams_client_init_client_sent',
    'bytes_bidi_streams_client_init_server_sent',
    'bytes_bidi_streams_server_init_client_sent',
    'bytes_bidi_streams_server_init_server_sent',
    'bytes_udi_streams_client_init', 'bytes_udi_streams_server_init',
    'total_client_app_bytes', 'total_server_app_bytes'
]


# =========================================================================
# Violation Analysis
# =========================================================================

def analyze_violations(df: pd.DataFrame, connection_duration_col: str = 'connection_duration') -> Dict:
    violations = {
        'non_negative': {},
        'duration_constraints': {},
        'combined_duration': {},
        'summary': {}
    }
    
    # Check non-negativity
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            violations['non_negative'][col] = {
                'count': int(neg_count),
                'percentage': float(neg_count / len(df) * 100)
            }
    
    # Check duration constraints
    if connection_duration_col in df.columns:
        for col in DURATION_COLS:
            if col in df.columns:
                violation_count = (df[col] > df[connection_duration_col]).sum()
                if violation_count > 0:
                    max_excess = float((df[col] - df[connection_duration_col]).max())
                    violations['duration_constraints'][col] = {
                        'count': int(violation_count),
                        'percentage': float(violation_count / len(df) * 100),
                        'max_excess': max_excess
                    }
    
    # Check combined duration constraint
    if all(c in df.columns for c in ['handshake_duration', 'migration_duration', connection_duration_col]):
        combined = df['handshake_duration'] + df['migration_duration']
        violation_count = (combined > df[connection_duration_col]).sum()
        if violation_count > 0:
            violations['combined_duration']['handshake_plus_migration'] = {
                'count': int(violation_count),
                'percentage': float(violation_count / len(df) * 100)
            }
    
    # Summary
    total_violations = (
        sum(v['count'] for v in violations['non_negative'].values()) +
        sum(v['count'] for v in violations['duration_constraints'].values()) +
        sum(v['count'] for v in violations['combined_duration'].values())
    )
    violations['summary'] = {
        'total_violation_types': (
            len(violations['non_negative']) + 
            len(violations['duration_constraints']) + 
            len(violations['combined_duration'])
        ),
        'has_violations': total_violations > 0
    }
    
    return violations


def format_violations_report(violations: Dict) -> str:
    """Format violations analysis as a readable report string."""
    lines = []
    lines.append("=" * 60)
    lines.append("CONSTRAINT VIOLATION ANALYSIS")
    lines.append("=" * 60)
    
    lines.append("\n1. NON-NEGATIVITY VIOLATIONS:")
    lines.append("-" * 40)
    if violations['non_negative']:
        for col, info in violations['non_negative'].items():
            lines.append(f"  {col}: {info['count']} negative values ({info['percentage']:.2f}%)")
    else:
        lines.append("  No negative values found.")
    
    lines.append("\n2. DURATION CONSTRAINT VIOLATIONS (should be <= connection_duration):")
    lines.append("-" * 40)
    if violations['duration_constraints']:
        for col, info in violations['duration_constraints'].items():
            lines.append(f"  {col}: {info['count']} violations ({info['percentage']:.2f}%)")
            lines.append(f"    Max excess: {info['max_excess']:.4f}")
    else:
        lines.append("  No duration constraint violations found.")
    
    lines.append("\n3. COMBINED DURATION VIOLATIONS:")
    lines.append("-" * 40)
    if violations['combined_duration']:
        for constraint, info in violations['combined_duration'].items():
            lines.append(f"  {constraint}: {info['count']} violations ({info['percentage']:.2f}%)")
    else:
        lines.append("  No combined duration violations found.")
    
    lines.append("\n" + "=" * 60)
    lines.append(f"TOTAL VIOLATION TYPES: {violations['summary']['total_violation_types']}")
    lines.append("=" * 60)
    
    return "\n".join(lines)


# =========================================================================
# Post-Processing Functions
# =========================================================================

def clip_to_non_negative(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Clip all numeric columns (or specified columns) to be >= 0.
    
    Returns:
    --------
    Tuple of (processed DataFrame, list of log messages)
    """
    df = df.copy()
    log = []
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in df.columns:
            before = (df[col] < 0).sum()
            if before > 0:
                df[col] = df[col].clip(lower=0)
                log.append(f"Clipped {before} negative values in '{col}'")
    
    return df, log


def fix_duration_constraints(
    df: pd.DataFrame, 
    connection_duration_col: str = 'connection_duration',
    method: str = 'clip'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fix duration constraints so sub-durations don't exceed connection_duration.
    
    Methods:
    - 'clip': Clip values to connection_duration
    - 'scale': Scale proportionally to fit within connection_duration
    - 'drop': Drop rows with violations
    
    Returns:
    --------
    Tuple of (processed DataFrame, list of log messages)
    """
    df = df.copy()
    log = []
    
    if connection_duration_col not in df.columns:
        log.append(f"Warning: {connection_duration_col} column not found, skipping duration constraints")
        return df, log
    
    log.append(f"Applying duration constraint fix (method: {method})")
    
    if method == 'clip':
        for col in DURATION_COLS:
            if col in df.columns:
                before = (df[col] > df[connection_duration_col]).sum()
                if before > 0:
                    df[col] = df[[col, connection_duration_col]].min(axis=1)
                    log.append(f"  Clipped {before} values in '{col}' to {connection_duration_col}")
    
    elif method == 'scale':
        existing_cols = [c for c in DURATION_COLS if c in df.columns]
        if existing_cols:
            for idx in df.index:
                conn_dur = df.loc[idx, connection_duration_col]
                for col in existing_cols:
                    if df.loc[idx, col] > conn_dur:
                        df.loc[idx, col] = conn_dur * 0.9  # Leave 10% margin
            log.append(f"  Scaled duration columns that exceeded {connection_duration_col}")
    
    elif method == 'drop':
        initial_len = len(df)
        for col in DURATION_COLS:
            if col in df.columns:
                df = df[df[col] <= df[connection_duration_col]]
        dropped = initial_len - len(df)
        if dropped > 0:
            log.append(f"  Dropped {dropped} rows with duration violations")
    
    return df, log


def fix_combined_duration_constraint(
    df: pd.DataFrame,
    connection_duration_col: str = 'connection_duration',
    method: str = 'scale'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Ensure handshake_duration + migration_duration <= connection_duration.
    
    Methods:
    - 'scale': Scale both proportionally
    - 'prioritize_handshake': Keep handshake, reduce migration
    - 'drop': Drop violating rows
    
    Returns:
    --------
    Tuple of (processed DataFrame, list of log messages)
    """
    df = df.copy()
    log = []
    
    if 'handshake_duration' not in df.columns or 'migration_duration' not in df.columns:
        return df, log
    
    if connection_duration_col not in df.columns:
        return df, log
    
    log.append(f"Applying combined duration constraint fix (method: {method})")
    
    combined = df['handshake_duration'] + df['migration_duration']
    violations = combined > df[connection_duration_col]
    violation_count = violations.sum()
    
    if violation_count == 0:
        log.append("  No combined duration violations to fix.")
        return df, log
    
    if method == 'scale':
        for idx in df[violations].index:
            conn_dur = df.loc[idx, connection_duration_col]
            hs_dur = df.loc[idx, 'handshake_duration']
            mig_dur = df.loc[idx, 'migration_duration']
            total = hs_dur + mig_dur
            
            if total > 0:
                scale_factor = conn_dur * 0.95 / total  # Leave 5% margin
                df.loc[idx, 'handshake_duration'] = hs_dur * scale_factor
                df.loc[idx, 'migration_duration'] = mig_dur * scale_factor
        log.append(f"  Scaled {violation_count} rows with combined duration violations")
    
    elif method == 'prioritize_handshake':
        for idx in df[violations].index:
            remaining = df.loc[idx, connection_duration_col] - df.loc[idx, 'handshake_duration']
            df.loc[idx, 'migration_duration'] = max(0, remaining * 0.9)
        log.append(f"  Adjusted migration_duration in {violation_count} rows")
    
    elif method == 'drop':
        df = df[~violations]
        log.append(f"  Dropped {violation_count} rows with combined duration violations")
    
    return df, log


def fix_integer_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Round columns that should be integers to nearest integer.
    
    Returns:
    --------
    Tuple of (processed DataFrame, list of log messages)
    """
    df = df.copy()
    log = []
    
    cols_fixed = []
    for col in INTEGER_COLS:
        if col in df.columns:
            df[col] = df[col].round().astype(int)
            cols_fixed.append(col)
    
    if cols_fixed:
        log.append(f"Rounded {len(cols_fixed)} integer columns")
    
    return df, log


def fix_logical_constraints(
    df: pd.DataFrame,
    connection_duration_col: str = 'connection_duration'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply additional logical constraints.
    
    Returns:
    --------
    Tuple of (processed DataFrame, list of log messages)
    """
    df = df.copy()
    log = []
    
    log.append("Applying additional logical constraints...")
    
    # time_to_migration should be >= handshake_duration (migration starts after handshake)
    # if 'time_to_migration' in df.columns and 'handshake_duration' in df.columns:
    #     violations = df['time_to_migration'] < df['handshake_duration']
    #     if violations.sum() > 0:
    #         df.loc[violations, 'time_to_migration'] = df.loc[violations, 'handshake_duration']
    #         log.append(f"  Adjusted {violations.sum()} rows where time_to_migration < handshake_duration")
    
    # Ensure time_to_migration + migration_duration <= connection_duration
    if all(c in df.columns for c in ['time_to_migration', 'migration_duration', connection_duration_col]):
        combined = df['time_to_migration'] + df['migration_duration']
        violations = combined > df[connection_duration_col]
        if violations.sum() > 0:
            for idx in df[violations].index:
                max_mig = df.loc[idx, connection_duration_col] - df.loc[idx, 'time_to_migration']
                df.loc[idx, 'migration_duration'] = max(0, max_mig * 0.95)
            log.append(f"  Adjusted {violations.sum()} rows where time_to_migration + migration_duration > {connection_duration_col}")
    
    return df, log


# =========================================================================
# Main Post-Processing Function
# =========================================================================

def apply_post_processing(
    df: pd.DataFrame,
    connection_duration_col: str = 'connection_duration',
    duration_method: str = 'clip',
    combined_duration_method: str = 'scale',
    fix_integers: bool = True,
    fix_logical: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply all post-processing rules to the synthetic data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Synthetic data to post-process
    connection_duration_col : str
        Name of the connection duration column (e.g., 'connection_duration' or 'Target')
    duration_method : str
        Method for fixing individual duration constraints ('clip', 'scale', 'drop')
    combined_duration_method : str
        Method for fixing combined duration constraint ('scale', 'prioritize_handshake', 'drop')
    fix_integers : bool
        Whether to round integer columns
    fix_logical : bool
        Whether to apply additional logical constraints
    
    Returns:
    --------
    Tuple of (processed DataFrame, list of log messages)
    """
    all_logs = []
    
    all_logs.append("=" * 60)
    all_logs.append("APPLYING POST-PROCESSING RULES")
    all_logs.append("=" * 60)
    
    # Step 1: Fix non-negative constraints
    all_logs.append("\n1. Fixing non-negative constraints...")
    df, log = clip_to_non_negative(df)
    all_logs.extend(log)
    
    # Step 2: Fix individual duration constraints
    all_logs.append("\n2. Fixing individual duration constraints...")
    df, log = fix_duration_constraints(df, connection_duration_col, method=duration_method)
    all_logs.extend(log)
    
    # Step 3: Fix combined duration constraints
    all_logs.append("\n3. Fixing combined duration constraints...")
    df, log = fix_combined_duration_constraint(df, connection_duration_col, method=combined_duration_method)
    all_logs.extend(log)
    
    # Step 4: Apply additional logical constraints
    if fix_logical:
        all_logs.append("\n4. Fixing additional logical constraints...")
        df, log = fix_logical_constraints(df, connection_duration_col)
        all_logs.extend(log)
    
    # Step 5: Fix integer columns
    if fix_integers:
        all_logs.append("\n5. Fixing integer columns...")
        df, log = fix_integer_columns(df)
        all_logs.extend(log)
    
    all_logs.append("\n" + "=" * 60)
    all_logs.append("POST-PROCESSING COMPLETE")
    all_logs.append("=" * 60)
    
    return df, all_logs
