"""
=============================================================================
 InferenceEngine.py
 Real-time attack classification thread for the Federated IDS pipeline.
=============================================================================
 Runs as a background thread inside the Edge Node server. Consumes a
 queue of completed feature windows produced by the WindowManager, applies
 the exact same feature transformations used at training time (via
 FeatureTransformer), and classifies each window using the trained MLP.

 When an attack is detected, it forwards the alert to AlertManager.

 Design principles:
   - STATEFUL: maintains a FeatureTransformer instance to compute diffs
     across consecutive windows, matching the training-time behaviour.
   - FAULT-TOLERANT: if the model file is missing, the thread logs a warning
     and passes all windows through as 'Normal' (fail-open for availability).
   - HOT-RELOADABLE: call reload_model() after receiving a new global model
     from the FL Server to swap the weights without restarting the server.
=============================================================================
"""

import os
import threading
import queue
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Set DEBUG_INFERENCE=1 in environment to print raw feature vectors at inference time
_DEBUG_INFERENCE = True


# Default paths — can be overridden via config
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


class InferenceEngine(threading.Thread):
    """
    Background thread that classifies completed feature windows in real time.

    Args:
        inference_queue: Thread-safe queue that WindowManager pushes windows onto.
                         Each item is a dict: {'cyber': {...}, 'physical': {...},
                                               'window_timestamp': str, 'node_id': int}
        alert_manager:   AlertManager instance to receive attack notifications.
        models_dir:      Directory containing fused_ids_model.pkl, scaler.pkl,
                         label_encoder.pkl, feature_columns.pkl.
        logger:          Logger instance from the EdgeNode framework.
    """

    def __init__(
        self,
        inference_queue: queue.Queue,
        alert_manager,
        models_dir: Path = DEFAULT_MODELS_DIR,
        logger=None,
    ):
        super().__init__(daemon=True, name="InferenceEngine")
        self._queue         = inference_queue
        self._alert_manager = alert_manager
        self._models_dir    = Path(models_dir)
        self._logger        = logger
        self._stop_event    = threading.Event()
        self._lock          = threading.Lock()

        # Telemetry for WebDashboard
        self.last_inference_data = {}

        # Model artifacts — loaded by _load_model()
        self._model          = None
        self._scaler         = None
        self._label_encoder  = None
        self._feature_columns = None
        self._model_ready    = False
        self._local_model    = None

        # Iso Forest artifacts
        self._iso_model          = None
        self._iso_scaler         = None
        self._iso_feature_columns = None
        self._iso_model_ready    = False

        # FeatureTransformer maintains rolling diff state
        # Import here to avoid circular imports at module load time
        from FeatureTransformer import FeatureTransformer
        self._transformer = FeatureTransformer()

        self._load_model()

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Load all model artifacts from disk.
        Silently sets _model_ready=False if any artifact is missing.
        """
        required = {
            "scaler.pkl":           "_scaler",
            "label_encoder.pkl":    "_label_encoder",
            "feature_columns.pkl":  "_feature_columns",
        }

        try:
            for filename, attr in required.items():
                path = self._models_dir / filename
                if not path.exists():
                    self._log(f"[InferenceEngine] Missing artifact: {path}. "
                              "Inference disabled until model is deployed.", level="warning")
                    self._model_ready = False
                    return
                setattr(self, attr, joblib.load(path))

            # --- PyTorch Model Loading ---
            import sys
            import torch
            network_analysis_dir = self._models_dir.parent / "network_analysis"
            if str(network_analysis_dir) not in sys.path:
                sys.path.insert(0, str(network_analysis_dir))
            from mlp_torch import load_model

            model_path = self._models_dir / "fused_ids_model.pth"
            if not model_path.exists():
                self._log(f"[InferenceEngine] Missing PyTorch artifact: {model_path}. Inference disabled.", level="warning")
                self._model_ready = False
                return
            
            self._model = load_model(model_path)
            self._model.eval()

            self._model_ready = True
            self._log(
                f"[InferenceEngine] PyTorch Model loaded. "
                f"Classes: {list(self._label_encoder.classes_)} | "
                f"Features: {len(self._feature_columns)}"
            )
            
            # --- Shadow Testing ---
            local_model_path = self._models_dir / "fused_ids_model_local.pth"
            if local_model_path.exists():
                self._local_model = load_model(local_model_path)
                self._local_model.eval()
                self._log("[InferenceEngine] Shadow model (local PyTorch baseline) loaded for A/B testing.")
            else:
                self._local_model = None

        except Exception as exc:
            self._log(f"[InferenceEngine] Failed to load model: {exc}", level="error")
            self._model_ready = False

        # Load Isolation Forest
        self._iso_model_ready = False
        try:
            iso_model_path = self._models_dir / "iso_forest_model.pkl"
            iso_scaler_path = self._models_dir / "iso_scaler.pkl"
            iso_cols_path = self._models_dir / "iso_physical_columns.pkl"
            
            if iso_model_path.exists() and iso_scaler_path.exists() and iso_cols_path.exists():
                self._iso_model = joblib.load(iso_model_path)
                self._iso_scaler = joblib.load(iso_scaler_path)
                self._iso_feature_columns = joblib.load(iso_cols_path)
                self._iso_model_ready = True
                self._log(f"[InferenceEngine] Isolation Forest loaded. Features: {len(self._iso_feature_columns)}")
            else:
                self._log("[InferenceEngine] Missing Isolation Forest artifacts. Physical anomaly detection disabled.", level="warning")
        except Exception as exc:
            self._log(f"[InferenceEngine] Failed to load Isolation Forest: {exc}", level="error")
    def reload_model(self) -> None:
        """
        Hot-reload the model from disk after the FL Server broadcasts new weights.
        Thread-safe: acquires lock so in-flight predictions complete cleanly.
        """
        self._log("[InferenceEngine] Reloading model from disk...")
        with self._lock:
            self._transformer.reset()
            self._load_model()
        self._log("[InferenceEngine] Model reload complete.")

    # ------------------------------------------------------------------
    # Thread Main Loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._log("[InferenceEngine] Started.")

        while not self._stop_event.is_set():
            try:
                window = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._process_window(window)
            except Exception as exc:
                self._log(f"[InferenceEngine] Error processing window: {exc}", level="error")
            finally:
                self._queue.task_done()

        self._log("[InferenceEngine] Stopped.")

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Window Processing
    # ------------------------------------------------------------------

    def _process_window(self, window: dict) -> None:
        """
        Classify one window and fire an alert if an attack is detected.

        Window dict format:
            {
                'cyber':             dict  (from CyberFeatureExtractor.extract()),
                'physical':          dict  (from PhysicalCSVWriter row, or None),
                'window_timestamp':  str   (ISO timestamp string),
                'node_id':           int,
            }
        """
        if not self._model_ready:
            return

        cyber    = window.get("cyber", {})
        physical = window.get("physical", None)
        ts       = window.get("window_timestamp", "unknown")
        node_id  = window.get("node_id", 0)

        # --- Isolation Forest Physical Anomaly Detection ---
        iso_anomaly_detected = False
        if self._iso_model_ready and physical:
            iso_vals = self._align_iso_features(physical)
            if iso_vals is not None:
                iso_df = pd.DataFrame(iso_vals.reshape(1, -1), columns=self._iso_feature_columns)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    iso_scaled = self._iso_scaler.transform(iso_df)
                
                # Isolation Forest returns -1 for outliers, 1 for inliers
                if self._iso_model.predict(iso_scaled)[0] == -1:
                    iso_anomaly_detected = True
                    self._log("[InferenceEngine] Isolation Forest detected physical anomaly!", level="warning")

        with self._lock:
            # Apply feature engineering (diffs + metadata strip)
            raw_vector, feature_keys = self._transformer.transform_window(
                cyber_features=cyber,
                physical_features=physical,
            )

        # Debug: print raw features going into the models
        if _DEBUG_INFERENCE:
            key_vals = dict(zip(feature_keys, raw_vector))
            interesting = {k: round(float(v), 4) for k, v in key_vals.items()
                           if k in ['total_packets','packet_rate','mean_interarrival_time',
                                    'reconnection_count','connection_duration','sequence_number_gap',
                                    'packet_rate_diff','mean_interarrival_time_diff']}
            print(f"[DEBUG] Live cyber features: {interesting}", flush=True)
            
            if physical:
                iso_keys = self._iso_feature_columns
                iso_vals = self._align_iso_features(physical)
                if iso_vals is not None:
                    phys_dict = dict(zip(iso_keys, iso_vals))
                    phys_interesting = {k: round(float(v), 4) for k, v in phys_dict.items()
                                        if k in ['Standard Deviation X', 'Peak-to-Peak X', 'Mean Z']}
                    print(f"[DEBUG] Live physical features: {phys_interesting}", flush=True)



        # Align to the exact column order the model was trained on
        feature_vector = self._align_features(raw_vector, feature_keys)
        if feature_vector is None:
            return

        # Scale and predict — wrap in DataFrame so column names match the
        # StandardScaler that was fitted on a named DataFrame at training time.
        feature_df = pd.DataFrame(
            feature_vector.reshape(1, -1),
            columns=self._feature_columns,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            scaled = self._scaler.transform(feature_df)
            
        import torch
        tensor_input = torch.tensor(scaled, dtype=torch.float32)
        
        with torch.no_grad():
            logits = self._model(tensor_input)
            proba = torch.softmax(logits, dim=1).numpy()[0]
            
        pred_idx   = int(np.argmax(proba))
        confidence = float(proba[pred_idx])
        label      = self._label_encoder.classes_[pred_idx]
        
        # --- Shadow Testing Comparison ---
        if self._local_model is not None:
            with torch.no_grad():
                local_logits = self._local_model(tensor_input)
                local_proba = torch.softmax(local_logits, dim=1).numpy()[0]
                local_pred_idx = int(np.argmax(local_proba))
                local_label = self._label_encoder.classes_[local_pred_idx]
            
            if local_label != label:
                self._log(f"[ShadowTest] Disagreement! Local predicted {local_label}, Global predicted {label}", level="warning")


        if label != "Normal":
            # Cyber attack detected — fire the cyber label regardless of physical state.
            # (If iso_anomaly is also True, the attacker failed to hide physically;
            #  the cyber label is the most specific classification available.)
            self._alert_manager.alert(
                label=label,
                confidence=confidence,
                window_timestamp=ts,
                node_id=node_id,
            )
        elif node_id not in [1, 2]:
            # --- THE HYBRID OVERRIDE ---
            # The ML model thinks the flow is Normal, but the physical identity is spoofed/corrupted.
            # We deterministically know this is a DataTampering attack that bypassed the ML layer.
            self._alert_manager.alert(
                label="DataTamperingCRCForged",
                confidence=1.0,
                window_timestamp=ts,
                node_id=node_id,
            )
        elif iso_anomaly_detected:
            # Cyber model sees Normal traffic, but physical sensor signals anomaly.
            self._alert_manager.alert(
                label="Physical_Anomaly",
                confidence=1.0,
                window_timestamp=ts,
                node_id=node_id,
            )
        else:
            self._alert_manager.log_normal(window_timestamp=ts, node_id=node_id)
                
        # Store telemetry for the WebDashboard
        self.last_inference_data = {
            "window_timestamp": ts,
            "cyber": cyber,
            "physical": physical,
            "network_label": label,
            "network_confidence": confidence,
            "iso_anomaly": iso_anomaly_detected
        }

    def _align_iso_features(self, physical: dict) -> Optional[np.ndarray]:
        """Align physical dictionary to expected Isolation Forest columns."""
        if not physical or not self._iso_feature_columns:
            return None
        aligned = np.array(
            [float(physical.get(col, 0.0)) for col in self._iso_feature_columns],
            dtype=np.float32,
        )
        return aligned

    def _align_features(
        self,
        raw_vector: np.ndarray,
        feature_keys: list,
    ) -> Optional[np.ndarray]:
        """
        Re-order the feature vector to match the exact column order the
        model was trained on (stored in feature_columns.pkl).

        If a feature is missing from the incoming window, it is filled with 0.0.
        This ensures the engine does not crash if new features are added later.
        """
        if self._feature_columns is None:
            return raw_vector

        key_to_val = dict(zip(feature_keys, raw_vector))
        aligned = np.array(
            [key_to_val.get(col, 0.0) for col in self._feature_columns],
            dtype=np.float32,
        )
        return aligned

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log(self, message: str, level: str = "info") -> None:
        # Map standard level names to the custom Logger's severity strings
        severity_map = {
            "info":    "INFO",
            "warning": "WARNING",
            "error":   "ERROR",
            "debug":   "DEBUG",
        }
        severity = severity_map.get(level.lower(), "INFO")

        if self._logger:
            self._logger.log(message, severity=severity)
        else:
            print(message)
