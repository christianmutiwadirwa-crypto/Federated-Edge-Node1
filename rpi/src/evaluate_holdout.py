import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rpi" / "src"))
from FeatureTransformer import FeatureTransformer
from train_federated_node import GLOBAL_CLASSES
from mlp_torch import load_model

def load_unified_eval_dataset(dataset_dir: Path) -> pd.DataFrame:
    dataset_path = dataset_dir / "global_evaluation_dataset_undersampled.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    master = pd.read_csv(dataset_path)
    master["AttackLabel"] = master["AttackLabel"].replace("NormalExperiment", "Normal")
    return master

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="models", help="Directory containing the model to evaluate")
    parser.add_argument("--model-name", type=str, default="fused_ids_model.pth", help="Filename of the model checkpoint")
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON file to save metrics to")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    eval_dir = base_dir / "dataset"
    models_dir = base_dir / args.model_dir
    
    print(f"Loading holdout dataset from {eval_dir}...")
    df = load_unified_eval_dataset(eval_dir)
    print(f"Loaded {len(df)} samples across {df['AttackLabel'].nunique()} classes.")
    
    transformer = FeatureTransformer()
    featured_df = transformer.fit_transform_dataframe(df)
    
    X_raw = featured_df.drop(columns=["AttackLabel"])
    y_raw = featured_df["AttackLabel"]
    
    print(f"Loading artifacts from {models_dir}...")
    scaler = joblib.load(models_dir / "scaler.pkl")
    label_encoder = joblib.load(models_dir / "label_encoder.pkl")
    feature_columns = joblib.load(models_dir / "feature_columns.pkl")
    
    # Align features
    for col in feature_columns:
        if col not in X_raw.columns:
            X_raw[col] = 0.0
    X = X_raw[feature_columns]
    
    y_encoded = label_encoder.transform(y_raw)
    X_scaled = scaler.transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Loading PyTorch model: {args.model_name}...")
    model = load_model(models_dir / args.model_name)
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        logits = model(X_tensor)
        preds = logits.argmax(dim=1).numpy()
    
    acc = accuracy_score(y_encoded, preds)
    print(f"\nHoldout Set Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    labels_present = np.unique(np.concatenate((y_encoded, preds)))
    target_names = label_encoder.inverse_transform(labels_present)
    print(classification_report(y_encoded, preds, labels=labels_present, target_names=target_names))
    
    print("\nConfusion Matrix:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    labels = label_encoder.inverse_transform(np.unique(y_encoded))
    cm = pd.DataFrame(confusion_matrix(y_encoded, preds, labels=labels_present), index=target_names, columns=target_names)
    print(cm)
    
    if args.output_json:
        import json
        with open(args.output_json, "w") as f:
            json.dump({"accuracy": float(acc), "samples": int(len(y_encoded))}, f)
            
if __name__ == "__main__":
    main()
