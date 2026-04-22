from typing import Dict

import torch

from .scoring import cosine_score


def filter_candidates_by_quality(
    candidate_feats: Dict,
    min_strength_quantile: float = 0.2,
    max_k_var_quantile: float = 0.8,
) -> Dict:
    if not candidate_feats:
        return {"filtered_pool": {}, "removed": [], "thresholds": {}, "log": []}

    names = list(candidate_feats.keys())
    strengths = torch.tensor([candidate_feats[name].strength for name in names], dtype=torch.float32)
    k_vars = torch.tensor([candidate_feats[name].k_var for name in names], dtype=torch.float32)

    strength_threshold = float(torch.quantile(strengths, min_strength_quantile).item())
    k_var_threshold = float(torch.quantile(k_vars, max_k_var_quantile).item())

    filtered_pool = {}
    removed = []
    logs = []
    for name in names:
        feat = candidate_feats[name]
        keep = feat.strength >= strength_threshold and feat.k_var <= k_var_threshold
        if keep:
            filtered_pool[name] = feat
        else:
            removed.append(name)
            logs.append(
                {
                    "name": name,
                    "strength": float(feat.strength),
                    "k_var": float(feat.k_var),
                    "reason": "low_strength_or_high_variance",
                }
            )

    return {
        "filtered_pool": filtered_pool,
        "removed": removed,
        "thresholds": {"min_strength": strength_threshold, "max_k_var": k_var_threshold},
        "log": logs,
    }


def greedy_maxmin_anchor_pool(
    candidate_feats: Dict,
    pool_size: int = 32,
    use_key: str = "h_bar",
) -> Dict:
    if not candidate_feats:
        return {"anchor_pool": {}, "selected_order": [], "log": []}

    items = list(candidate_feats.items())
    pool_size = min(pool_size, len(items))

    first_name, first_feat = max(items, key=lambda x: float(x[1].strength))
    selected = [first_name]
    selected_set = {first_name}
    logs = [{"step": 0, "selected": first_name, "reason": "max_strength_seed"}]

    def vec(name: str):
        return getattr(candidate_feats[name], use_key)

    while len(selected) < pool_size:
        best_name = None
        best_min_sim = None
        for name, _ in items:
            if name in selected_set:
                continue
            sims = [cosine_score(vec(name), vec(sel)) for sel in selected]
            min_sim = min(sims)
            if best_min_sim is None or min_sim < best_min_sim:
                best_min_sim = min_sim
                best_name = name

        if best_name is None:
            break
        selected.append(best_name)
        selected_set.add(best_name)
        logs.append({"step": len(selected) - 1, "selected": best_name, "min_similarity_to_pool": float(best_min_sim)})

    return {
        "anchor_pool": {name: candidate_feats[name] for name in selected},
        "selected_order": selected,
        "log": logs,
    }


def build_anchor_pool(
    candidate_feats: Dict,
    pool_size: int = 32,
    min_strength_quantile: float = 0.2,
    max_k_var_quantile: float = 0.8,
    use_key: str = "h_bar",
) -> Dict:
    filtered = filter_candidates_by_quality(
        candidate_feats,
        min_strength_quantile=min_strength_quantile,
        max_k_var_quantile=max_k_var_quantile,
    )
    diversified = greedy_maxmin_anchor_pool(filtered["filtered_pool"], pool_size=pool_size, use_key=use_key)

    return {
        "anchor_pool": diversified["anchor_pool"],
        "filtered_out": filtered["removed"],
        "thresholds": filtered["thresholds"],
        "filter_log": filtered["log"],
        "selected_order": diversified["selected_order"],
        "pool_log": diversified["log"],
    }
