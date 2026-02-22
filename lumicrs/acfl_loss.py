# acfl_loss.py
# ACFL loss (paper-matching) = Eq.(4)-(9) + Algorithm 1
#
# Usage:
#   from acfl_loss import acfl_loss
#   loss = acfl_loss(logits, targets, class_counts=global_counts)

import torch
import torch.nn.functional as F


def acfl_loss(
    logits, targets,
    alpha=1.0, gamma=2.0,
    k=0.05,  # Top-K hard samples ratio: e.g., 0.05 -> hardest 5%
    theta_min=10, theta_max=1000,  # θ_min / θ_max are COUNT thresholds (not ratios)
    alpha_min=0.1, alpha_max=1.0,
    gamma_min=1.0, gamma_max=5.0,
    eps=1e-10,
    oversample=True, undersample=True,
    class_counts=None,   # pass global/epoch N_c if available; else use batch bincount
    reduction="mean"     # "mean" returns ACFL (normalized by Z); "sum"/"none" also supported
):
    """
    ACFL (paper-matching) = Eq.(4)-(9) + Algorithm 1.

    Args:
      logits: (B, C)
      targets: (B,)
      class_counts: (C,) global class frequencies N_c (recommended). If None -> batch statistics.
    Returns:
      ACFL (scalar) if reduction="mean"
    """

    B, C = logits.shape
    device = logits.device

    # ---- p_i: true-class probability ----
    probs = F.softmax(logits, dim=-1)                              # (B, C)
    p_i = probs.gather(1, targets.view(-1, 1)).squeeze(1)          # (B,)
    p_i = torch.clamp(p_i, min=eps, max=1.0)

    # ---- N_c: class frequency ----
    if class_counts is None:
        N_c = torch.bincount(targets, minlength=C).float().to(device) + eps
    else:
        if not torch.is_tensor(class_counts):
            N_c = torch.tensor(class_counts, dtype=torch.float32, device=device) + eps
        else:
            N_c = class_counts.to(device=device, dtype=torch.float32) + eps
        if N_c.numel() != C:
            raise ValueError(f"class_counts length must be C={C}, got {N_c.numel()}")

    # ===== Eq.(4): class weight w_c = 1 / (N_c + eps) =====
    w_c = 1.0 / (N_c + eps)                    # (C,)
    w_c_i = w_c.gather(0, targets)             # (B,)

    # ===== Eq.(8): adaptive sampling weight w_sample_c =====
    N_max = torch.max(N_c).clamp_min(eps)
    N_ci = N_c.gather(0, targets)              # (B,)

    w_sample_i = torch.ones_like(N_ci, dtype=torch.float32, device=device)

    if oversample:
        mask_over = N_ci < float(theta_min)
        w_sample_i = torch.where(mask_over, 1.0 + 1.0 / (N_ci + eps), w_sample_i)

    if undersample:
        mask_under = N_ci > float(theta_max)
        w_sample_i = torch.where(mask_under, 1.0 - (N_ci / N_max), w_sample_i)

    # ===== Eq.(5): alpha_i = clip(alpha * p_i + eps, alpha_min, alpha_max) =====
    alpha_i = torch.clamp(alpha * p_i + eps, min=alpha_min, max=alpha_max)

    # ===== Eq.(6): gamma_i = clip(gamma*(1-p_i), gamma_min, gamma_max) =====
    gamma_i = torch.clamp(gamma * (1.0 - p_i), min=gamma_min, max=gamma_max)

    # ===== Eq.(7): Top-K hard sample mask (by sample difficulty) =====
    # tau_k = quantile({p_i}, k), M_i = 1{p_i <= tau_k}
    k = float(max(0.0, min(1.0, k)))
    tau_k = torch.quantile(p_i.detach(), k)
    M_i = (p_i <= tau_k).float()               # (B,)

    # ===== Eq.(9): ACFL aggregation =====
    focal_term = (1.0 - p_i).pow(gamma_i)
    l_i = - w_c_i * w_sample_i * alpha_i * focal_term * torch.log(p_i + eps)  # (B,)

    Z = M_i.sum().clamp_min(1.0)               # Algorithm 1 normalization
    ACFL = (M_i * l_i).sum() / Z               # <- final ACFL (paper form)

    if reduction == "mean":
        return ACFL
    elif reduction == "sum":
        return (M_i * l_i).sum()
    elif reduction == "none":
        return M_i * l_i
    else:
        raise ValueError("reduction must be one of: 'mean', 'sum', 'none'")


if __name__ == "__main__":
    # Minimal sanity check (CPU)
    torch.manual_seed(0)
    B, C = 8, 5
    logits = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))
    loss = acfl_loss(logits, targets, class_counts=None)
    print("ACFL:", float(loss))
