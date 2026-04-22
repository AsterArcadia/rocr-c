import json
from pathlib import Path
from typing import Dict, List

from .pool_builder import build_anchor_pool
from .proxy_eval import rerank_candidates_with_proxy
from .representation_extractor import precompute_entity_features
from .static_selector import rank_all_forget_targets


def _ensure_jsonable_rankings(rankings: Dict) -> Dict:
    out = {}
    for k, rows in rankings.items():
        out[k] = [dict(row) for row in rows]
    return out


def run_anchor_selection_pipeline(
    model,
    tokenizer,
    forget_targets: list,
    candidate_targets: list,
    dataset,
    data_args,
    model_args,
    training_args,
    finetuning_args,
    hparams,
    z_layer: int,
    cache_dir: str,
    output_dir: str,
    pool_config: dict,
    static_config: dict,
    proxy_config: dict,
):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    forget_cache_dir = Path(cache_dir) / "forget_features"
    candidate_cache_dir = Path(cache_dir) / "candidate_features"

    forget_feats = precompute_entity_features(
        model,
        tokenizer,
        entity_names=forget_targets,
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        hparams=hparams,
        z_layer=z_layer,
        cache_dir=str(forget_cache_dir),
        split=getattr(data_args, "split", "train"),
        overwrite=bool(pool_config.get("overwrite_features", False)),
    )
    candidate_feats = precompute_entity_features(
        model,
        tokenizer,
        entity_names=candidate_targets,
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        hparams=hparams,
        z_layer=z_layer,
        cache_dir=str(candidate_cache_dir),
        split=getattr(data_args, "split", "train"),
        overwrite=bool(pool_config.get("overwrite_features", False)),
    )

    pool_bundle = build_anchor_pool(
        candidate_feats,
        pool_size=int(pool_config.get("pool_size", 32)),
        min_strength_quantile=float(pool_config.get("min_strength_quantile", 0.2)),
        max_k_var_quantile=float(pool_config.get("max_k_var_quantile", 0.8)),
        use_key=pool_config.get("use_key", "h_bar"),
    )
    anchor_pool = pool_bundle["anchor_pool"]

    static_rankings = rank_all_forget_targets(
        forget_feats,
        anchor_pool,
        config=static_config,
        top_m=int(static_config.get("top_m", 5)),
    )

    rocr_config = dict(proxy_config.get("rocr_config", {}))
    rocr_config["hparams"] = hparams

    proxy_rankings = {}
    final_selection = {}
    for forget_target, top_candidates in static_rankings.items():
        reranked = rerank_candidates_with_proxy(
            model,
            tokenizer,
            forget_target=forget_target,
            top_candidates=top_candidates,
            dataset=dataset,
            rocr_config=rocr_config,
            proxy_config=proxy_config,
        )
        proxy_rankings[forget_target] = reranked
        final_selection[forget_target] = reranked[0]["anchor"] if reranked else None

    with open(output_root / "anchor_pool.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_order": pool_bundle["selected_order"],
                "filtered_out": pool_bundle["filtered_out"],
                "thresholds": pool_bundle["thresholds"],
                "filter_log": pool_bundle["filter_log"],
                "pool_log": pool_bundle["pool_log"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(output_root / "static_rankings.json", "w", encoding="utf-8") as f:
        json.dump(_ensure_jsonable_rankings(static_rankings), f, ensure_ascii=False, indent=2)

    with open(output_root / "proxy_rankings.json", "w", encoding="utf-8") as f:
        json.dump(_ensure_jsonable_rankings(proxy_rankings), f, ensure_ascii=False, indent=2)

    with open(output_root / "final_anchor_selection.json", "w", encoding="utf-8") as f:
        json.dump(final_selection, f, ensure_ascii=False, indent=2)

    return {
        "forget_features": forget_feats,
        "candidate_features": candidate_feats,
        "anchor_pool": anchor_pool,
        "static_rankings": static_rankings,
        "proxy_rankings": proxy_rankings,
        "final_anchor_selection": final_selection,
    }
