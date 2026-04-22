from copy import deepcopy
from typing import Dict, List, Optional

import torch

from ..rocr.rocr_hparams import rocrHyperParams
from ..rocr.rocr_main import apply_rocr_to_model
from ..util import nethook
from .mini_eval_set import build_proxy_eval_set


def _safe_target_new(example: Dict) -> str:
    text = example.get("answer", "") if isinstance(example, dict) else ""
    text = text.strip()
    if not text:
        return "Unfortunately"
    return text.split("\n")[0][:64]


def _build_request(forget_target: str, anchor_target: str, target_new: str) -> Dict:
    return {
        "case_id": forget_target,
        "subject": anchor_target,
        "prompt": "{}.",
        "target_new": {"str": target_new},
    }


def _build_projection_mats(model, hparams: rocrHyperParams) -> torch.Tensor:
    W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
    dim = W_out.shape[1]
    num_layers = len(hparams.layers)
    eye = torch.eye(dim, dtype=torch.float32)
    return eye.unsqueeze(0).repeat(num_layers, 1, 1).cpu()


def run_lightweight_rocr_edit(
    model,
    tokenizer,
    forget_target: str,
    anchor_target: str,
    rocr_config: dict,
):
    hparams = rocr_config.get("hparams")
    if hparams is None:
        raise ValueError("rocr_config must include 'hparams'.")
    if isinstance(hparams, dict):
        hparams = rocrHyperParams(**hparams)

    requests = [
        _build_request(
            forget_target=forget_target,
            anchor_target=anchor_target,
            target_new=rocr_config.get("target_new", "Unfortunately"),
        )
    ]

    edited_model = deepcopy(model)
    edited_model.eval()

    proj = rocr_config.get("projection")
    if proj is None:
        proj = _build_projection_mats(edited_model, hparams)

    W_out = nethook.get_parameter(edited_model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
    cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")

    edited_model, _ = apply_rocr_to_model(
        model=edited_model,
        tok=tokenizer,
        requests=requests,
        hparams=hparams,
        cache_c=cache_c,
        P=proj,
    )
    return edited_model


def _avg_answer_log_prob(model, tokenizer, eval_examples: List[Dict]) -> float:
    if not eval_examples:
        return 0.0

    model.eval()
    scores = []
    with torch.no_grad():
        for ex in eval_examples:
            prompt = ex.get("prompt", "")
            answer = ex.get("answer", "")
            if not answer.strip():
                continue

            full_text = prompt + answer
            enc_full = tokenizer(full_text, return_tensors="pt").to(next(model.parameters()).device)
            enc_prompt = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

            logits = model(**enc_full).logits[:, :-1, :]
            labels = enc_full["input_ids"][:, 1:]

            prompt_len = enc_prompt["input_ids"].shape[1]
            answer_start = max(prompt_len - 1, 0)
            if answer_start >= labels.shape[1]:
                continue

            answer_logits = logits[:, answer_start:, :]
            answer_labels = labels[:, answer_start:]
            log_probs = torch.log_softmax(answer_logits, dim=-1)
            token_lp = log_probs.gather(-1, answer_labels.unsqueeze(-1)).squeeze(-1)
            scores.append(float(token_lp.mean().item()))

    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def evaluate_answer_prob_drop(
    base_model,
    edited_model,
    tokenizer,
    eval_examples: list,
) -> float:
    base_score = _avg_answer_log_prob(base_model, tokenizer, eval_examples)
    edited_score = _avg_answer_log_prob(edited_model, tokenizer, eval_examples)
    return float(base_score - edited_score)


def evaluate_forget_gain(
    base_model,
    edited_model,
    tokenizer,
    proxy_eval_set: dict,
    task_weights: dict = None,
) -> float:
    task_weights = task_weights or {
        "forget_qa": 1.0,
        "forget_fb": 1.0,
        "forget_mcp": 1.5,
        "forget_sqa": 1.5,
    }

    total_weight = 0.0
    weighted_sum = 0.0
    for task_name, weight in task_weights.items():
        drop = evaluate_answer_prob_drop(base_model, edited_model, tokenizer, proxy_eval_set.get(task_name, []))
        weighted_sum += float(weight) * drop
        total_weight += float(weight)

    return float(weighted_sum / total_weight) if total_weight > 0 else 0.0


def evaluate_retain_drop(
    base_model,
    edited_model,
    tokenizer,
    proxy_eval_set: dict,
) -> float:
    return evaluate_answer_prob_drop(base_model, edited_model, tokenizer, proxy_eval_set.get("retain", []))


def evaluate_collapse_penalty(
    edited_model,
    tokenizer,
    proxy_eval_set: dict,
    retain_drop: float,
    retain_drop_threshold: float = 0.15,
) -> float:
    if retain_drop <= retain_drop_threshold:
        return 0.0

    excess = retain_drop - retain_drop_threshold
    retain_count = len(proxy_eval_set.get("retain", []))
    scale = 1.0 + 0.1 * retain_count
    return float(excess * scale)


def compute_proxy_score(
    forget_gain: float,
    retain_drop: float,
    collapse_penalty: float,
    beta_retain: float = 1.0,
    gamma_collapse: float = 2.0,
) -> float:
    return float(forget_gain - beta_retain * retain_drop - gamma_collapse * collapse_penalty)


def rerank_candidates_with_proxy(
    model,
    tokenizer,
    forget_target: str,
    top_candidates: list,
    dataset,
    rocr_config: dict,
    proxy_config: dict,
) -> list:
    eval_cfg = proxy_config.get("eval_set", {})
    proxy_eval_set = build_proxy_eval_set(
        forget_target,
        dataset,
        num_qa=int(eval_cfg.get("num_qa", 2)),
        num_fb=int(eval_cfg.get("num_fb", 2)),
        num_mcp=int(eval_cfg.get("num_mcp", 1)),
        num_sqa=int(eval_cfg.get("num_sqa", 1)),
        num_retain=int(eval_cfg.get("num_retain", 5)),
    )

    ranked = []
    for row in top_candidates:
        anchor_target = row["anchor"]
        rocr_cfg = dict(rocr_config)
        rocr_cfg["target_new"] = _safe_target_new(proxy_eval_set.get("forget_qa", [{"answer": "Unfortunately"}])[0])

        edited_model = run_lightweight_rocr_edit(model, tokenizer, forget_target, anchor_target, rocr_cfg)
        forget_gain = evaluate_forget_gain(
            model,
            edited_model,
            tokenizer,
            proxy_eval_set,
            task_weights=proxy_config.get("task_weights"),
        )
        retain_drop = evaluate_retain_drop(model, edited_model, tokenizer, proxy_eval_set)
        collapse_penalty = evaluate_collapse_penalty(
            edited_model,
            tokenizer,
            proxy_eval_set,
            retain_drop,
            retain_drop_threshold=float(proxy_config.get("retain_drop_threshold", 0.15)),
        )
        proxy_score = compute_proxy_score(
            forget_gain,
            retain_drop,
            collapse_penalty,
            beta_retain=float(proxy_config.get("beta_retain", 1.0)),
            gamma_collapse=float(proxy_config.get("gamma_collapse", 2.0)),
        )

        final_score = float(row["static_total"] + float(proxy_config.get("lambda_proxy", 1.0)) * proxy_score)
        merged = dict(row)
        merged.update(
            {
                "proxy_score": proxy_score,
                "forget_gain": forget_gain,
                "retain_drop": retain_drop,
                "collapse_penalty": collapse_penalty,
                "final_score": final_score,
            }
        )
        ranked.append(merged)

        del edited_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked
