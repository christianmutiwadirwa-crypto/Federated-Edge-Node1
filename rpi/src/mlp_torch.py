"""
=============================================================================
 mlp_torch.py
 PyTorch Federated MLP — shared module for Node 1, Node 2, and evaluation.
=============================================================================
 Provides:
   - FederatedMLP           : nn.Module with class-aware gradient masking
   - save_model / load_model: checkpoint I/O
   - extract_weights        : serialize to {coefs, intercepts} (FL server compat)
   - apply_weights          : deserialize from FL server response
   - federated_loss         : L_CE + λ*L_KD + (μ/2)*||W-W_g||²
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FederatedMLP(nn.Module):
    """
    MLP for federated IDS.
    Architecture: Input → 100 → ReLU → 50 → ReLU → 14 (output logits)
    Matches the existing FL server topology: 100→100→50→14 when including
    input dimension in the description.
    """

    def __init__(self, input_dim: int, hidden_sizes: tuple = (100, 50), num_classes: int = 14):
        super().__init__()
        self.input_dim    = input_dim
        self.hidden_sizes = hidden_sizes
        self.num_classes  = num_classes

        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(x))

    def register_class_freeze_hooks(self, missing_classes: set) -> None:
        """
        Prevent backpropagation from modifying output-layer parameters for
        classes that have no local training samples.

        This is the core mechanism preventing catastrophic forgetting:
        the gradient for unseen class columns is zeroed *before* the
        optimiser step, so those weights remain at the global baseline.
        """
        if not missing_classes:
            return  # All classes present — nothing to freeze

        frozen = frozenset(int(c) for c in missing_classes)

        def weight_hook(grad: torch.Tensor) -> torch.Tensor:
            g = grad.clone()
            for c in frozen:
                g[c, :] = 0.0   # shape: (num_classes, hidden) → zero row c
            return g

        def bias_hook(grad: torch.Tensor) -> torch.Tensor:
            g = grad.clone()
            for c in frozen:
                g[c] = 0.0
            return g

        self.output.weight.register_hook(weight_hook)
        self.output.bias.register_hook(bias_hook)
        print(f"  [FederatedMLP] Gradient freeze active for class indices: {sorted(frozen)}")


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_model(model: FederatedMLP, path: Path, extra_meta: dict = None) -> None:
    """Save model state dict + architecture metadata to a .pth checkpoint."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "arch": {
            "input_dim":    model.input_dim,
            "hidden_sizes": model.hidden_sizes,
            "num_classes":  model.num_classes,
        },
    }
    if extra_meta:
        checkpoint.update(extra_meta)
    torch.save(checkpoint, path)


def load_model(path: Path) -> FederatedMLP:
    """Load a FederatedMLP from a .pth checkpoint file."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = FederatedMLP(**checkpoint["arch"])
    model.load_state_dict(checkpoint["state_dict"])
    return model


# ---------------------------------------------------------------------------
# Weight Serialisation (FL Server compatibility)
# ---------------------------------------------------------------------------

def extract_weights(model: FederatedMLP) -> dict:
    """
    Serialise model weights to {coefs, intercepts} JSON-compatible format.

    IMPORTANT — Axis convention:
      sklearn stores coefs_[i] as shape (n_in, n_out).
      PyTorch stores Linear.weight as shape (n_out, n_in).
      We transpose PyTorch weights → (n_in, n_out) before sending to server
      so the existing FL server aggregation code requires zero changes.
    """
    coefs      = []
    intercepts = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            coefs.append(module.weight.detach().cpu().numpy().T.tolist())  # (n_in, n_out)
            intercepts.append(module.bias.detach().cpu().numpy().tolist())
    return {"coefs": coefs, "intercepts": intercepts}


def apply_weights(model: FederatedMLP, coefs: list, intercepts: list) -> None:
    """
    Deserialise {coefs, intercepts} received from the FL server into the model.
    Transposes each coef back from (n_in, n_out) → PyTorch (n_out, n_in).
    Modifies model in-place.
    """
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    for layer, coef, intercept in zip(linear_layers, coefs, intercepts):
        layer.weight.data = torch.tensor(np.array(coef), dtype=torch.float32).T
        layer.bias.data   = torch.tensor(np.array(intercept), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Fisher Information (FedCurv)
# ---------------------------------------------------------------------------

def compute_fisher_diagonals(model: nn.Module, dataloader: torch.utils.data.DataLoader, num_samples: int = 1000) -> list:
    """
    Computes the diagonal Fisher information matrix for the model using the local dataset.
    Returns a list of tensors aligned with model.parameters().
    """
    model.eval()
    fisher_diagonals = [torch.zeros_like(p, device=p.device) for p in model.parameters()]
    total_samples = 0
    
    # Use empirical Fisher: average of squared gradients over samples
    for x, y in dataloader:
        if total_samples >= num_samples:
            break
            
        for i in range(x.size(0)):
            model.zero_grad()
            logits = model(x[i:i+1])
            # Empirical Fisher uses ground-truth labels
            loss = F.cross_entropy(logits, y[i:i+1])
            loss.backward()
            
            with torch.no_grad():
                for f, p in zip(fisher_diagonals, model.parameters()):
                    if p.grad is not None:
                        f += p.grad.pow(2)
            total_samples += 1
            if total_samples >= num_samples:
                break

    with torch.no_grad():
        for f in fisher_diagonals:
            f /= max(total_samples, 1)
            
    return fisher_diagonals


# ---------------------------------------------------------------------------
# Composite Federated Loss
# ---------------------------------------------------------------------------

def federated_loss(
    student_logits:  torch.Tensor,
    labels:          torch.Tensor,
    teacher_logits:  torch.Tensor,
    student_params:  list,
    global_params:   list,
    missing_classes: set,
    lambda_kd: float = 0.5,
    mu:        float = 0.01,
    T:         float = 3.0,
    lambda_fedcurv: float = 1.0,
    other_clients_data: list = None,
) -> torch.Tensor:
    """
    L = L_CE  +  λ * L_KD  +  (μ/2) * ||W - W_global||²

    L_CE   : Cross-entropy on locally observed classes.
    L_KD   : KL divergence between student and teacher soft predictions.
             Encourages the student to preserve the global model's internal
             logic — critically protecting hidden-layer representations for
             unseen classes even though their output neurons are frozen.
    FedProx: Penalises excessive weight drift from the global baseline.
             Acts as a physical restoring force limiting how far local
             training can pull the model away from the global optimum.
    """
    # 1. Standard cross-entropy
    ce = F.cross_entropy(student_logits, labels)

    # 2. Masked Knowledge Distillation
    # Only penalize the student for drifting on classes it has no local data for.
    # For its own classes, it should trust its ground-truth labels (CE loss) completely.
    kd = 0.0
    if lambda_kd > 0.0:
        # KL Divergence with Temperature Scaling across all classes
        # This protects hidden-layer representations for unseen classes
        log_prob_s = F.log_softmax(student_logits / T, dim=1)
        prob_t     = F.softmax(teacher_logits.detach() / T, dim=1)
        kd = F.kl_div(log_prob_s, prob_t, reduction='batchmean') * (T * T)

    # 3. FedProx proximal term
    prox = sum(
        (w - wg.detach()).pow(2).sum()
        for w, wg in zip(student_params, global_params)
    )

    # 4. FedCurv Fisher penalty
    fedcurv_loss = 0.0
    if other_clients_data and lambda_fedcurv > 0.0:
        for client_data in other_clients_data:
            cw = client_data["weights"]
            cf = client_data["fisher"]
            for sw, w_k, f_k in zip(student_params, cw, cf):
                fedcurv_loss += (f_k * (sw - w_k.detach()).pow(2)).sum()

    return ce + lambda_kd * kd + (mu / 2) * prox + lambda_fedcurv * fedcurv_loss
