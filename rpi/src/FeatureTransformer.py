"""
=============================================================================
 FeatureTransformer.py
 Shared feature engineering utility for the Federated IDS pipeline.
=============================================================================
 This module is the single source of truth for all feature engineering
 logic. Both the training pipeline (`train_federated_node.py`) and the
 realtime `InferenceEngine.py` use this same class to guarantee that the exact
 same transformations are applied at training time and at inference time.

 Prevents training/serving skew — a silent bug where the model is trained
 on features computed one way but receives features computed a different
 way at inference time.

 Responsibilities:
   - Define the exact set of CYBER_METADATA columns to drop.
   - Define the exact set of PHYSICAL_METADATA columns to drop.
   - Compute temporal _diff features on key cyber metrics.
   - Maintain a rolling buffer of the previous window's values for diffs.
   - Produce a final, ordered, clean numpy array ready for model inference.
=============================================================================
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Optional


# ---------------------------------------------------------------------------
# Column Definitions — Single Source of Truth
# ---------------------------------------------------------------------------

# Identifier/metadata columns that must be stripped before training or inference.
# These columns encode experiment identity, not attack patterns.
CYBER_METADATA_COLS = [
    "window_id",
    "window_start_time",
    "window_end_time",
    "node_id",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "AttackLabel",
    # --- ABLATED FEATURES (Simulating Perfect Attacker) ---
    "payload_rms_magnitude", 
    "payload_frozen_count", 
    "payload_cross_axis_std", 
    "payload_all_zeros_count",
    "sequence_number_gap", 
    "total_seq_gap", 
    "min_seq_gap", 
    "mean_sequence_increment", 
    "std_sequence_increment",
    "out_of_order_packet_count",
    "max_interarrival_time",
    "min_interarrival_time",
    "anomaly_packet_count",
    "anomaly_packet_rate",
    "timing_jitter_score"
]

PHYSICAL_METADATA_COLS = [
    "Timestamp",
    "Node ID",
    "Label",
]

# Cyber features on which temporal diff is computed.
# NOTE: These were disabled because the ESP32's 2-second sample window and the
# server's 2-second tumbling window run unsynchronised. This causes a permanent
# ±0.5 aliasing sawtooth in packet_rate_diff, which the model learned to
# associate with attack classes, producing false positives on every other window.
# Low importance (ranked 11th and 14th of 29 features) — not worth the noise.
DIFF_FEATURE_COLS: list = []


class FeatureTransformer:
    """
    Stateful feature transformer that maintains a rolling window of the
    previous cyber feature vector so that temporal diff features can be
    computed consistently at both training time and inference time.

    Usage at training time:
        transformer = FeatureTransformer()
        transformed_df = transformer.fit_transform_dataframe(merged_df)

    Usage at inference time (called once per incoming window):
        transformer = FeatureTransformer()   # one instance per node
        feature_vector = transformer.transform_window(cyber_dict, phys_dict)
    """

    def __init__(self):
        # Rolling buffer: stores the previous window's values for diff cols.
        self._prev_values: dict = {col: 0.0 for col in DIFF_FEATURE_COLS}
        self._is_first_window: bool = True

    # ------------------------------------------------------------------
    # Training-time API: transforms a full merged DataFrame at once
    # ------------------------------------------------------------------

    def fit_transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a full merged Cyber+Physical DataFrame as used in
        train_fused_model.py.

        Steps:
          1. Sort by window_start_time so diffs are chronologically meaningful.
          2. Compute _diff features using pandas diff() (efficient vectorized op).
          3. Drop all metadata columns.
          4. Drop rows with NaN values (first row of each experiment after diff).

        Args:
            df: Merged DataFrame containing both cyber and physical columns,
                including 'AttackLabel' and all metadata columns.

        Returns:
            Clean DataFrame with only model-input features + 'AttackLabel'.
        """
        df = df.copy()

        # Sort chronologically within each experiment group so diffs are valid
        if "window_start_time" in df.columns:
            df["window_start_time"] = pd.to_datetime(df["window_start_time"])
            df = df.sort_values("window_start_time").reset_index(drop=True)

        # Compute temporal diffs
        for col in DIFF_FEATURE_COLS:
            if col in df.columns:
                df[f"{col}_diff"] = df[col].diff().fillna(0.0)

        # Preserve AttackLabel before dropping metadata
        labels = df["AttackLabel"].copy() if "AttackLabel" in df.columns else None

        # ------------------------------------------------------------------
        # Derived features — normalise for cross-experiment sniffer variance.
        # DO NOT REMOVE — Node 2 depends on these for FedAvg weight alignment.
        # ------------------------------------------------------------------
        tp = df["total_packets"].replace(0, np.nan)  # avoid divide-by-zero

        # 1. Merged anomaly count: captures duplicate/out-of-order packets
        #    regardless of which raw column the sniffer chose to populate.
        if "duplicate_packet_count" in df.columns and "out_of_order_packet_count" in df.columns:
            df["anomaly_packet_count"] = (
                df["duplicate_packet_count"] + df["out_of_order_packet_count"]
            )
            df["anomaly_packet_rate"] = (df["anomaly_packet_count"] / tp).fillna(0.0)

        # 2. Sequence gap rate — normalise by total packets so SlowDoS and
        #    PacketInjection are distinguishable even when experiment duration
        #    differs between runs.
        if "sequence_number_gap" in df.columns:
            df["sequence_gap_rate"] = (df["sequence_number_gap"] / tp).fillna(0.0)

        # 3. Slowness score — product of inter-arrival time and packet rate.
        #    Should be ~1 for normal flows; deviates strongly for SlowDoS.
        if "mean_interarrival_time" in df.columns and "packet_rate" in df.columns:
            df["slowness_score"] = (
                df["mean_interarrival_time"] * df["packet_rate"]
            ).fillna(0.0)

        # 4. Bytes per packet — compact size signature used by PacketInjection.
        if "total_bytes" in df.columns:
            df["bytes_per_packet"] = (df["total_bytes"] / tp).fillna(0.0)

        # 5. Delay Signature (Interarrival CV)
        if "std_interarrival_time" in df.columns and "mean_interarrival_time" in df.columns:
            df["interarrival_cv"] = (
                df["std_interarrival_time"] / (df["mean_interarrival_time"] + 1e-6)
            ).fillna(0.0)

        # 6. Packet Injection Signature (Sequence Chaos)
        if "std_sequence_increment" in df.columns and "mean_sequence_increment" in df.columns:
            df["sequence_chaos"] = (
                df["std_sequence_increment"] / (df["mean_sequence_increment"] + 1e-6)
            ).fillna(0.0)

        # 7. Payload Efficiency
        if "data_rate" in df.columns and "packet_rate" in df.columns:
            df["payload_efficiency"] = (
                df["data_rate"] / (df["packet_rate"] + 1e-6)
            ).fillna(0.0)

        # 8. Burst-to-Rate Ratio
        if "burst_intensity" in df.columns and "packet_rate" in df.columns:
            df["burst_to_rate_ratio"] = (
                df["burst_intensity"] / (df["packet_rate"] + 1e-6)
            ).fillna(0.0)

        # 9. Partial Packet Rate (SlowDoS primary signal)
        if "partial_packet_count" in df.columns:
            df["partial_packet_rate"] = (df["partial_packet_count"] / tp).fillna(0.0)

        # 10. Invalid Packets Per Second (Injection rate-normalised)
        if "invalid_packet_count" in df.columns and "connection_duration" in df.columns:
            dur = df["connection_duration"].replace(0, np.nan)
            df["invalid_pps"] = (df["invalid_packet_count"] / dur).fillna(0.0)

        # 11. Integrity Score (composite health metric)
        if "crc_failure_rate" in df.columns and "invalid_packet_rate" in df.columns:
            df["integrity_score"] = (
                1.0 - df["crc_failure_rate"] - df["invalid_packet_rate"]
            ).clip(lower=0.0)

        # 12. Effective Throughput (accounts for loss)
        if "data_rate" in df.columns and "packet_loss_rate" in df.columns:
            df["effective_throughput"] = (
                df["data_rate"] * (1.0 - df["packet_loss_rate"])
            ).fillna(0.0)

        # 13. Timing Jitter Score (IAT range; key for Delay attack)
        if "iat_range" in df.columns:
            df["timing_jitter_score"] = df["iat_range"]
        elif "max_interarrival_time" in df.columns and "min_interarrival_time" in df.columns:
            df["timing_jitter_score"] = df["max_interarrival_time"] - df["min_interarrival_time"]

        # 14. Connection Health (reset density)
        if "reconnection_count" in df.columns and "connection_duration" in df.columns:
            df["connection_health"] = (
                df["reconnection_count"] / (df["connection_duration"] + 1.0)
            ).fillna(0.0)

        # 15. Packet Efficiency (valid ratio vs total rate)
        if "valid_packet_rate" in df.columns and "packet_rate" in df.columns:
            df["packet_efficiency"] = (
                df["valid_packet_rate"] / (df["packet_rate"] + 1e-6)
            ).fillna(0.0)
        elif "valid_packet_count" in df.columns:
            df["packet_efficiency"] = (df["valid_packet_count"] / tp).fillna(0.0)

        # 16. Sequence Variance (injection / replay chaos)
        if "std_sequence_increment" in df.columns:
            df["seq_variance"] = (df["std_sequence_increment"] ** 2).fillna(0.0)

        # 17. Loss-to-Duplicate Ratio (distinguishes PacketLoss from DuplicatePacket)
        if "packet_loss_rate" in df.columns and "anomaly_packet_rate" in df.columns:
            df["loss_to_duplicate_ratio"] = (
                df["packet_loss_rate"] / (df["anomaly_packet_rate"] + 1e-6)
            ).fillna(0.0)

        # Drop all metadata columns (ignore missing ones silently)
        cols_to_drop = [c for c in CYBER_METADATA_COLS + PHYSICAL_METADATA_COLS
                        if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        # Re-attach label for downstream balancing / splitting
        if labels is not None:
            df["AttackLabel"] = labels.values

        # Drop any rows with NaN (can occur at group boundaries)
        df = df.dropna().reset_index(drop=True)

        return df

    # ------------------------------------------------------------------
    # Inference-time API: transforms a single incoming window
    # ------------------------------------------------------------------

    def transform_window(
        self,
        cyber_features: dict,
        physical_features: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Transform a single incoming window at inference time.

        This method maintains internal state (_prev_values) to compute
        temporal diffs across consecutive calls, exactly matching the
        pandas diff() behaviour used during training.

        Args:
            cyber_features:    Dict of cyber feature values (as produced by
                               CyberFeatureExtractor.extract()).
            physical_features: Dict of physical feature values (as produced
                               by PhysicalCSVWriter). Pass None if running
                               cyber-only inference.

        Returns:
            1-D numpy float32 array ready to be passed to model.predict().
        """
        combined = {}

        # --- Cyber features ---
        for key, val in cyber_features.items():
            if key not in CYBER_METADATA_COLS:
                combined[key] = float(val) if val is not None else 0.0

        # --- Physical features ---
        if physical_features is not None:
            for key, val in physical_features.items():
                if key not in PHYSICAL_METADATA_COLS:
                    combined[key] = float(val) if val is not None else 0.0

        # --- Temporal diff features ---
        for col in DIFF_FEATURE_COLS:
            current_val = combined.get(col, 0.0)
            if self._is_first_window:
                combined[f"{col}_diff"] = 0.0
            else:
                combined[f"{col}_diff"] = current_val - self._prev_values[col]
            self._prev_values[col] = current_val
            
        self._is_first_window = False

        # --- Derived Features ---
        tp = combined.get("total_packets", 0.0)
        tp_safe = tp if tp != 0.0 else 1e-6
        
        combined["anomaly_packet_count"] = combined.get("duplicate_packet_count", 0.0) + combined.get("out_of_order_packet_count", 0.0)
        combined["anomaly_packet_rate"] = combined["anomaly_packet_count"] / tp_safe
        
        combined["sequence_gap_rate"] = combined.get("sequence_number_gap", 0.0) / tp_safe
        
        combined["slowness_score"] = combined.get("mean_interarrival_time", 0.0) * combined.get("packet_rate", 0.0)
        
        combined["bytes_per_packet"] = combined.get("total_bytes", 0.0) / tp_safe
        
        combined["interarrival_cv"] = combined.get("std_interarrival_time", 0.0) / (combined.get("mean_interarrival_time", 0.0) + 1e-6)
        
        combined["sequence_chaos"] = combined.get("std_sequence_increment", 0.0) / (combined.get("mean_sequence_increment", 0.0) + 1e-6)
        
        combined["payload_efficiency"] = combined.get("data_rate", 0.0) / (combined.get("packet_rate", 0.0) + 1e-6)
        
        combined["burst_to_rate_ratio"] = combined.get("burst_intensity", 0.0) / (combined.get("packet_rate", 0.0) + 1e-6)

        # New derived features (feature expansion)
        combined["partial_packet_rate"] = combined.get("partial_packet_count", 0.0) / tp_safe
        
        dur_safe = combined.get("connection_duration", 0.0) if combined.get("connection_duration", 0.0) != 0.0 else 1e-6
        combined["invalid_pps"] = combined.get("invalid_packet_count", 0.0) / dur_safe
        
        combined["integrity_score"] = max(0.0, 1.0 - combined.get("crc_failure_rate", 0.0) - combined.get("invalid_packet_rate", 0.0))
        
        combined["effective_throughput"] = combined.get("data_rate", 0.0) * (1.0 - combined.get("packet_loss_rate", 0.0))
        
        combined["timing_jitter_score"] = combined.get("iat_range", combined.get("max_interarrival_time", 0.0) - combined.get("min_interarrival_time", 0.0))
        
        combined["connection_health"] = combined.get("reconnection_count", 0.0) / (combined.get("connection_duration", 0.0) + 1.0)
        
        vpr = combined.get("valid_packet_rate", combined.get("valid_packet_count", 0.0) / tp_safe)
        combined["packet_efficiency"] = vpr / (combined.get("packet_rate", 0.0) + 1e-6)
        
        combined["seq_variance"] = combined.get("std_sequence_increment", 0.0) ** 2
        
        combined["loss_to_duplicate_ratio"] = combined.get("packet_loss_rate", 0.0) / (combined.get("anomaly_packet_rate", 0.0) + 1e-6)

        # Build sorted feature vector (alphabetical sort guarantees
        # the same column order as the training DataFrame)
        feature_keys = sorted(k for k in combined if k != "AttackLabel")
        vector = np.array([combined[k] for k in feature_keys], dtype=np.float32)

        return vector, feature_keys

    def reset(self):
        """
        Reset the rolling diff buffer.
        Call this when a new TCP session starts to avoid computing diffs
        across session boundaries.
        """
        self._prev_values = {col: 0.0 for col in DIFF_FEATURE_COLS}
        self._is_first_window = True
