"""
=============================================================================
 FederatedClient.py
 Edge Node Federated Learning Client.
=============================================================================
 Manages the local node's participation in an FL round.
 
 Triggered manually by the researcher via trigger_fl_round.py after
 a new batch of labelled experiment data is collected.

 Workflow:
   1. Subprocesses `train_federated_node.py` to train a local PyTorch MLP on the 
      node's local `results/` folder.
   2. Extracts the weights (tensors) from the resulting model.
   3. Sends the weights via HTTP POST to the FL Server in the cloud.
   4. Polls the FL Server until the new aggregated global model is ready.
   5. Overwrites the local fused_ids_model.pth with the global weights.
   6. Restarts the InferenceEngine to hot-reload the new global model.
=============================================================================
"""

import os
import sys
import json
import time
import requests
import joblib
import subprocess
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Address of the Oracle Cloud FastAPI server.
# You will need to change this to the public IP of your Oracle VM later.
FL_SERVER_URL = os.environ.get("FL_SERVER_URL", "https://federated-edge-computing.onrender.com")

# Unique identifier for this Edge Node
NODE_ID = os.environ.get("NODE_ID", "edge_node_1")

# Paths
BASE_DIR      = Path(__file__).resolve().parent.parent.parent
MODELS_DIR    = BASE_DIR / "models"
MODEL_PATH    = MODELS_DIR / "fused_ids_model.pth"      # PyTorch checkpoint
TRAIN_SCRIPT  = BASE_DIR / "rpi" / "src" / "train_federated_node.py"

from mlp_torch import FederatedMLP, load_model as _load_torch, apply_weights as _apply_torch, extract_weights as _extract_torch, save_model as _save_torch


class FederatedClient:
    """Manages the full lifecycle of a Federated Learning round on the Edge Node."""

    def __init__(self, server_url: str = FL_SERVER_URL, node_id: str = NODE_ID, 
                 no_kd: bool = False, no_fedprox: bool = False, 
                 no_fedcurv: bool = False, no_freeze: bool = False,
                 no_class_aware: bool = False):
        self.server_url = server_url.rstrip("/")
        self.node_id = node_id
        self.no_kd = no_kd
        self.no_fedprox = no_fedprox
        self.no_fedcurv = no_fedcurv
        self.no_freeze = no_freeze
        self.no_class_aware = no_class_aware

    def trigger_round(self, skip_training: bool = False):
        """Execute a full FL round."""
        print(f"[{self.node_id}] Starting Federated Learning Round")
        
        # 0. Sync Scaler
        if not self._sync_scaler():
            print(f"[{self.node_id}] Failed to sync scaler. Aborting round.")
            return
            
        # 1. Train locally
        if not skip_training:
            if not self._train_local_model():
                return
        else:
            print("  -> Step 1: Skipping local training (using existing model)...")
            
        # 2. Extract local weights
        local_weights = self._extract_weights()
        if not local_weights:
            return
            
        # 3. Submit to FL Server
        round_num = self._submit_weights(local_weights)
        if not round_num:
            return
            
        # 4. Wait for Global Model (Polling)
        global_weights = self._wait_for_global_model(round_num)
        if not global_weights:
            return
            
        # 5. Save Global Model locally & Backup
        self._apply_global_weights(global_weights, round_num)
        
        # 6. Validate Global Model against holdout set
        self._submit_validation(round_num)
        
        print(f"[{self.node_id}] FL Round {round_num} completed successfully. Inference Engine is now using the Global Model.")

    # -----------------------------------------------------------------------
    # Step Implementations
    # -----------------------------------------------------------------------

    def _sync_scaler(self) -> bool:
        """Step 0: Fit local scaler, upload to server, wait for global scaler.
        If the global scaler already exists on the server (from a prior round),
        just download and lock it — skip the expensive re-init phase.
        """
        print("  -> Step 0: Initializing and synchronizing Global Scaler...")
        try:
            # Fast path: global scaler already exists — just download and lock it
            resp = requests.get(f"{self.server_url}/global_scaler", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                scaler_path = MODELS_DIR / "scaler.pkl"
                if scaler_path.exists():
                    scaler = joblib.load(scaler_path)
                else:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    
                scaler.mean_ = np.array(data["scaler_mean"])
                scaler.var_  = np.array(data["scaler_var"])
                scaler.n_samples_seen_ = data["scaler_samples"]
                scale = np.sqrt(scaler.var_)
                scale[scale == 0.0] = 1.0
                scaler.scale_ = scale
                joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
                print("     Global Scaler already exists — downloaded and locked.")
                return True

            # Slow path: first time — fit local, upload, wait
            print("     Fitting local scaler on raw data...")
            subprocess.run(
                ["python", str(TRAIN_SCRIPT), "--node-id", self.node_id, "--output-dir", str(MODELS_DIR), "--init-scaler-only"],
                check=True,
                capture_output=True
            )
            
            # Extract local scaler stats
            scaler = joblib.load(MODELS_DIR / "scaler.pkl")
            payload = {
                "node_id": self.node_id,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_var": scaler.var_.tolist(),
                "scaler_samples": int(scaler.n_samples_seen_)
            }
            
            # Submit to server
            print("     Submitting local scaler stats to FL server...")
            resp = requests.post(f"{self.server_url}/init_scaler", json=payload)
            if resp.status_code != 200:
                print(f"     [ERROR] Server rejected scaler stats: {resp.text}")
                return False
                
            # Wait for Global Scaler
            print("     Waiting for Global Scaler aggregation...")
            for attempt in range(1440):
                resp = requests.get(f"{self.server_url}/global_scaler")
                if resp.status_code == 200:
                    data = resp.json()
                    scaler.mean_ = np.array(data["scaler_mean"])
                    scaler.var_ = np.array(data["scaler_var"])
                    scaler.n_samples_seen_ = data["scaler_samples"]
                    scale = np.sqrt(scaler.var_)
                    scale[scale == 0.0] = 1.0
                    scaler.scale_ = scale
                    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
                    print("     Global Scaler downloaded and locked.")
                    return True
                time.sleep(5)
                
            print("     [ERROR] Timed out waiting for Global Scaler.")
            return False
            
        except Exception as e:
            print(f"     [ERROR] Failed to sync scaler: {e}")
            return False

    def _train_local_model(self) -> bool:
        """Run the training script as a subprocess."""
        print(f"  -> Step 1: Training local MLP on local dataset...")
        try:
            cmd = ["python", str(TRAIN_SCRIPT), "--node-id", self.node_id, "--output-dir", str(MODELS_DIR)]
            if self.no_kd: cmd.append("--no-kd")
            if self.no_fedprox: cmd.append("--no-fedprox")
            if self.no_fedcurv: cmd.append("--no-fedcurv")
            if self.no_freeze: cmd.append("--no-freeze")

            # We run it as a subprocess to keep the training memory separate
            # from the long-running inference process.
            subprocess.run(
                cmd,
                check=True
            )
            print("     Local training complete.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"     [ERROR] Local training failed: {e.stderr}")
            return False

    def _extract_weights(self) -> dict:
        """Load the newly trained PyTorch model and extract its weights in FL server format."""
        print("  -> Step 2: Extracting local weights...")
        try:
            model = _load_torch(MODEL_PATH)
            model.eval()

            weights = _extract_torch(model)  # {coefs: [...], intercepts: [...]}

            # Attach class counts for server-side class-aware aggregation
            class_counts_path = MODELS_DIR / "class_counts.json"
            if not self.no_class_aware and class_counts_path.exists():
                with open(class_counts_path, "r") as f:
                    counts = json.load(f)
                    weights["class_counts"] = counts
                    weights["training_samples"] = sum(counts)
            elif class_counts_path.exists():
                # Still need training_samples for simple fedavg
                with open(class_counts_path, "r") as f:
                    counts = json.load(f)
                    weights["training_samples"] = sum(counts)

            weights["node_id"]       = self.node_id
            weights["architecture"]  = "->".join(
                str(x) for x in
                [model.input_dim] + list(model.hidden_sizes) + [model.num_classes]
            )
            print(f"     Weights extracted. Architecture: {weights['architecture']}")
            return weights

        except Exception as e:
            print(f"     [ERROR] Failed to extract weights: {e}")
            return {}

    def _submit_weights(self, weights: dict) -> int:
        """POST weights to the central FL server."""
        print(f"  -> Step 3: Submitting updates to FL Server ({self.server_url})...")
        
        fisher_path = MODELS_DIR / "fisher_diagonals.json"
        if fisher_path.exists():
            with open(fisher_path, "r") as f:
                fisher = json.load(f)
                weights["fisher_coefs"] = fisher.get("fisher_coefs")
                weights["fisher_intercepts"] = fisher.get("fisher_intercepts")

        try:
            response = requests.post(f"{self.server_url}/submit_update", json=weights)
            if response.status_code == 200:
                data = response.json()
                print(f"     Server accepted update. Target Round: {data.get('round')}")
                return data.get("round")
            else:
                print(f"     [ERROR] Server rejected update: {response.text}")
                return 0
        except requests.RequestException as e:
            print(f"     [ERROR] Could not connect to FL Server: {e}")
            return 0

    def _wait_for_global_model(self, target_round: int) -> dict:
        """Poll the server every 5 seconds until the target round is ready."""
        print(f"  -> Step 4: Waiting for FedAvg aggregation (polling server)...")
        # Increased to 2 hours (1440 attempts * 5 seconds) to allow manual ESP32 swapping
        max_attempts = 1440 
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{self.server_url}/global_model")
                if response.status_code == 200:
                    data = response.json()
                    current_round = data.get("round", 0)
                    
                    if current_round >= target_round:
                        weights = data.get("weights")
                        client_states = data.get("client_states", {})
                        
                        with open(MODELS_DIR / "global_weights.json", "w") as f:
                            json.dump(weights, f)
                            
                        with open(MODELS_DIR / "client_states.json", "w") as f:
                            json.dump(client_states, f)
                            
                        print(f"     Global model for Round {current_round} received!")
                        return weights
                        
            except requests.RequestException:
                pass # Silently ignore connection errors while polling
                
            time.sleep(5)
            
        print("     [ERROR] Timed out waiting for global model.")
        return {}

    def _apply_global_weights(self, global_weights: dict, round_num: int) -> bool:
        """Overwrite the local PyTorch model with global weights, save backups, log divergence."""
        print("  -> Step 5: Applying global weights and saving backups...")
        try:
            archive_dir = MODELS_DIR / "archive"
            archive_dir.mkdir(exist_ok=True)

            coefs      = global_weights["coefs"]
            intercepts = global_weights["intercepts"]

            # Load current local model
            local_model = _load_torch(MODEL_PATH)
            local_model.eval()

            # --- Divergence Analysis ---
            local_weights  = _extract_torch(local_model)
            old_flat = np.concatenate([np.array(c).flatten() for c in local_weights["coefs"]])
            new_flat = np.concatenate([np.array(c).flatten() for c in coefs])
            divergence = np.linalg.norm(old_flat - new_flat)
            print(f"     Weight Divergence (L2 Norm): {divergence:.4f}")

            # Log divergence
            div_log_path = archive_dir / "divergence_log.csv"
            write_header = not div_log_path.exists()
            with open(div_log_path, "a") as f:
                if write_header:
                    f.write("round,divergence\n")
                f.write(f"{round_num},{divergence:.6f}\n")

            # Back up local model
            local_backup_path = archive_dir / f"fused_ids_model_local_R{round_num}.pth"
            _save_torch(local_model, local_backup_path)
            print(f"     Local model backed up to {local_backup_path.name}")
            
            if round_num == 1:
                baseline_path = MODELS_DIR / "fused_ids_model_baseline.pth"
                _save_torch(local_model, baseline_path)
                print("     [!] Initial Round 1 Baseline Model explicitly saved for future comparison.")

            # Apply global weights
            _apply_torch(local_model, coefs, intercepts)

            # Save global model
            global_backup_path = archive_dir / f"fused_ids_model_global_R{round_num}.pth"
            _save_torch(local_model, global_backup_path)
            _save_torch(local_model, MODEL_PATH)
            print("     Global weights saved to models/fused_ids_model.pth")
            print("     [Note] You must restart the EdgeNode to load the new model.")
            return True

        except Exception as e:
            print(f"     [ERROR] Failed to apply global weights: {e}")
            return False

    def _submit_validation(self, round_num: int) -> bool:
        print("  -> Step 6: Validating global model and submitting metrics...")
        eval_results_dir = BASE_DIR / "rpi" / "experiments" / "results" / "evaluation"
        if not eval_results_dir.exists():
            print("     [SKIP] No evaluation dataset found.")
            return False
            
        test_script = BASE_DIR / "rpi" / "src" / "evaluate_holdout.py"
        
        try:
            subprocess.run(
                ["python", str(test_script), 
                 "--model-dir", str(MODELS_DIR), 
                 "--output-json", str(metrics_json)],
                check=True
            )
            
            with open(metrics_json, "r") as f:
                metrics = json.load(f)
                
            payload = {
                "node_id": self.node_id,
                "round": round_num,
                "accuracy": metrics.get("accuracy", 0.0),
                "samples": metrics.get("samples", 0)
            }
            
            response = requests.post(f"{self.server_url}/submit_validation", json=payload)
            if response.status_code == 200:
                print("     Validation metrics successfully submitted to server.")
                return True
            else:
                print(f"     [ERROR] Server rejected validation metrics: {response.text}")
                return False
                
        except Exception as e:
            print(f"     [ERROR] Validation failed: {e}")
            return False



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trigger an FL Round on this Edge Node.")
    parser.add_argument("--node-id", type=str, default=NODE_ID, help="Unique ID of this node")
    parser.add_argument("--server", type=str, default=FL_SERVER_URL, help="URL of the FL Server")
    parser.add_argument("--skip-training", action="store_true", help="Skip local training and upload existing weights")
    parser.add_argument("--no-kd", action="store_true", help="Disable Knowledge Distillation")
    parser.add_argument("--no-fedprox", action="store_true", help="Disable FedProx")
    parser.add_argument("--no-fedcurv", action="store_true", help="Disable FedCurv")
    parser.add_argument("--no-freeze", action="store_true", help="Disable Output Neuron Freezing")
    parser.add_argument("--no-class-aware", action="store_true", help="Disable Class-Aware Aggregation")
    args = parser.parse_args()

    client = FederatedClient(
        server_url=args.server,
        node_id=args.node_id,
        no_kd=args.no_kd,
        no_fedprox=args.no_fedprox,
        no_fedcurv=args.no_fedcurv,
        no_freeze=args.no_freeze,
        no_class_aware=args.no_class_aware
    )
    client.trigger_round(skip_training=args.skip_training)
