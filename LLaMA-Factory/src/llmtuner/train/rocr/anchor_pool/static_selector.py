import json
from typing import Dict, List, Optional

from .scoring import compute_static_score_row, normalize_score_dict


def _compute_total(row: Dict) -> float:
    return (
        row["weight_sim_band"] * row["sim_band_norm"]
        + row["weight_strength"] * row["strength_norm"]
        + row["weight_safe"] * row["safe_norm"]
        + row["weight_compat"] * row["compat_norm"]
    )


def rank_anchors_for_forget_target(
    f_feat,
    anchor_pool: Dict,
    config: dict,
    top_m: int = 5,
) -> List[Dict]:
    rows = [compute_static_score_row(f_feat, a_feat, config) for a_feat in anchor_pool.values()]
    rows = normalize_score_dict(rows)

    for row in rows:
        row["static_total"] = _compute_total(row)

    rows.sort(key=lambda x: x["static_total"], reverse=True)
    return rows[:top_m]


def rank_all_forget_targets(
    forget_feats: Dict,
    anchor_pool: Dict,
    config: dict,
    top_m: int = 5,
    save_path: Optional[str] = None,
) -> Dict:
    result = {}
    for name, f_feat in forget_feats.items():
        result[name] = rank_anchors_for_forget_target(f_feat, anchor_pool, config, top_m=top_m)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result
