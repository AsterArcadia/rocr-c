from typing import Dict, List

import torch


def cosine_score(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    eps = 1e-12
    return float(torch.dot(x, y) / (torch.norm(x) * torch.norm(y) + eps))


def compute_sim_band_score(f_feat, a_feat, mu: float) -> float:
    sim = cosine_score(f_feat.h_bar, a_feat.h_bar)
    return -float((sim - mu) ** 2)


def compute_strength_score(a_feat, alpha_var: float) -> float:
    return float(torch.norm(a_feat.k_bar).item() - alpha_var * a_feat.k_var)


def compute_safe_score(f_feat, a_feat, mode: str = "cheap") -> float:
    if mode != "cheap":
        raise ValueError(f"Unsupported safe mode: {mode}")

    leak_proxy = 0.5 * cosine_score(f_feat.h_bar, a_feat.h_bar) + 0.5 * cosine_score(f_feat.k_bar, a_feat.k_bar)
    return -float(leak_proxy)


def compute_compat_score(f_feat, a_feat) -> float:
    return float(0.5 * cosine_score(f_feat.h_bar, a_feat.h_bar) + 0.5 * cosine_score(f_feat.k_bar, a_feat.k_bar))


def normalize_score_dict(score_rows: List[Dict]) -> List[Dict]:
    if not score_rows:
        return score_rows

    keys = ["sim_band", "strength", "safe", "compat"]
    stats = {}
    for key in keys:
        vals = torch.tensor([float(row[key]) for row in score_rows], dtype=torch.float32)
        mean = float(vals.mean().item())
        std = float(vals.std(unbiased=False).item())
        stats[key] = (mean, std if std > 1e-12 else 1.0)

    normalized_rows = []
    for row in score_rows:
        new_row = dict(row)
        for key in keys:
            mean, std = stats[key]
            new_row[f"{key}_norm"] = (float(row[key]) - mean) / std
        normalized_rows.append(new_row)

    return normalized_rows


def compute_static_score_row(f_feat, a_feat, config: dict) -> Dict:
    weights = config.get("weights", {})
    row = {
        "forget": f_feat.name,
        "anchor": a_feat.name,
        "sim_band": compute_sim_band_score(f_feat, a_feat, mu=float(config.get("mu", 0.5))),
        "strength": compute_strength_score(a_feat, alpha_var=float(config.get("alpha_var", 0.5))),
        "safe": compute_safe_score(f_feat, a_feat, mode=config.get("safe_mode", "cheap")),
        "compat": compute_compat_score(f_feat, a_feat),
        "weight_sim_band": float(weights.get("sim_band", 1.0)),
        "weight_strength": float(weights.get("strength", 1.0)),
        "weight_safe": float(weights.get("safe", 1.0)),
        "weight_compat": float(weights.get("compat", 1.0)),
    }
    return row
