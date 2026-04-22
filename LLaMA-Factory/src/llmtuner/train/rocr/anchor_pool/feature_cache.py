import json
from pathlib import Path
from typing import Dict, Any

import torch

from .entity_feature import EntityFeature, EntityFeatureSummary


TENSOR_FILE = "feature.pt"
SUMMARY_FILE = "summary.json"


def _to_summary(feature: EntityFeature) -> EntityFeatureSummary:
    return EntityFeatureSummary(
        name=feature.name,
        h_var=float(feature.h_var),
        k_var=float(feature.k_var),
        strength=float(feature.strength),
    )


def _summary_to_dict(summary: EntityFeatureSummary, meta: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "name": summary.name,
        "h_var": summary.h_var,
        "k_var": summary.k_var,
        "strength": summary.strength,
    }
    payload["meta"] = meta or {}
    return payload


def save_entity_feature(feature: EntityFeature, save_path: str) -> None:
    base = Path(save_path)
    base.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "name": feature.name,
            "hidden_list": [x.detach().cpu() for x in feature.hidden_list],
            "act_list": [x.detach().cpu() for x in feature.act_list],
            "h_bar": feature.h_bar.detach().cpu(),
            "k_bar": feature.k_bar.detach().cpu(),
        },
        base / TENSOR_FILE,
    )

    summary = _to_summary(feature)
    with open(base / SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(_summary_to_dict(summary, feature.meta or {}), f, ensure_ascii=False, indent=2)


def load_entity_feature(load_path: str) -> EntityFeature:
    base = Path(load_path)
    tensor_payload = torch.load(base / TENSOR_FILE, map_location="cpu")

    summary_payload = {}
    summary_path = base / SUMMARY_FILE
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_payload = json.load(f)

    return EntityFeature(
        name=tensor_payload["name"],
        hidden_list=tensor_payload["hidden_list"],
        act_list=tensor_payload["act_list"],
        h_bar=tensor_payload["h_bar"],
        k_bar=tensor_payload["k_bar"],
        h_var=float(summary_payload.get("h_var", 0.0)),
        k_var=float(summary_payload.get("k_var", 0.0)),
        strength=float(summary_payload.get("strength", float(torch.norm(tensor_payload["k_bar"]).item()))),
        meta=summary_payload.get("meta", {}),
    )


def feature_exists(load_path: str) -> bool:
    base = Path(load_path)
    return (base / TENSOR_FILE).exists() and (base / SUMMARY_FILE).exists()
