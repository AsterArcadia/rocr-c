#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

from transformers import AutoModelForCausalLM, AutoTokenizer


def _discover_project_root() -> Path:
    cur = Path(__file__).resolve().parent
    while True:
        if (cur / "expr_config_global.py").exists() and (cur / "src" / "llmtuner").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError("Cannot locate project root containing expr_config_global.py and src/llmtuner.")


PROJECT_ROOT = _discover_project_root()
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from expr_config_global import NAMES
from llmtuner.hparams import get_train_args
from llmtuner.train.rocr.anchor_pool import run_anchor_selection_pipeline
from llmtuner.train.rocr.rocr.rocr_hparams import rocrHyperParams

FORGET_LEVEL1 = "forget_level1.json"
FORGET_LEVEL2 = "forget_level2.json"
FORGET_LEVEL3 = "forget_level3.json"
FORGET_MCP = "forget_mcp.json"
RETAIN_MMLU = "retain_mmlu.json"


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)




def _resolve_target_dir(eval_dataset_dir: str, target: str) -> str:
    target_dir = os.path.join(eval_dataset_dir, target)
    if os.path.isdir(target_dir):
        return target_dir

    # Fallback for unicode/normalization differences in filesystem names.
    target_idx = target.split("_", 1)[0]
    candidates = sorted(
        [
            os.path.join(eval_dataset_dir, name)
            for name in os.listdir(eval_dataset_dir)
            if name.startswith(f"{target_idx}_") and os.path.isdir(os.path.join(eval_dataset_dir, name))
        ]
    )
    if candidates:
        print(f"[WARN] Target dir '{target_dir}' not found, fallback to '{candidates[0]}'")
        return candidates[0]

    return target_dir

    
def _to_examples(items: Any) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                prompt = item.get("question") or item.get("prompt") or item.get("text") or item.get("input") or ""
                answer = item.get("answer") or item.get("target") or item.get("response") or item.get("output") or ""
                examples.append({"prompt": str(prompt), "answer": str(answer)})
            elif isinstance(item, str):
                examples.append({"prompt": item, "answer": ""})
    elif isinstance(items, dict):
        prompt = items.get("question") or items.get("prompt") or items.get("text") or items.get("input") or ""
        answer = items.get("answer") or items.get("target") or items.get("response") or items.get("output") or ""
        examples.append({"prompt": str(prompt), "answer": str(answer)})
    return examples


def build_proxy_dataset(eval_dataset_dir: str, forget_targets: List[str], candidate_targets: List[str], strict_missing_files: bool = False) -> Dict[str, Any]:
    proxy_dataset: Dict[str, Any] = {"retain": []}

    for target in forget_targets:
        target_dir = _resolve_target_dir(eval_dataset_dir, target)
        level1_data = _load_json(os.path.join(target_dir, FORGET_LEVEL1))
        level2_data = _load_json(os.path.join(target_dir, FORGET_LEVEL2))
        level3_data = _load_json(os.path.join(target_dir, FORGET_LEVEL3))
        mcp_data = _load_json(os.path.join(target_dir, FORGET_MCP))

        if strict_missing_files and not os.path.exists(os.path.join(target_dir, FORGET_MCP)):
            raise FileNotFoundError(f"Missing required proxy file: {os.path.join(target_dir, FORGET_MCP)}")

        level1 = _to_examples(_load_json(os.path.join(target_dir, FORGET_LEVEL1)))
        level2 = _to_examples(_load_json(os.path.join(target_dir, FORGET_LEVEL2)))
        level3 = _to_examples(_load_json(os.path.join(target_dir, FORGET_LEVEL3)))
        mcp = _to_examples(_load_json(os.path.join(target_dir, FORGET_MCP)))

        proxy_dataset[target] = {
            "forget_qa": level1,
            "forget_fb": level2,
            "forget_sqa": level3,
            "forget_mcp": mcp,
            "retain": [],
        }

    # retain pool: sample from candidate retain_mmlu files
    for candidate in candidate_targets:
        candidate_dir = os.path.join(eval_dataset_dir, candidate)
        mmlu_path = os.path.join(candidate_dir, RETAIN_MMLU)
        if os.path.exists(mmlu_path):
            proxy_dataset["retain"].extend(_to_examples(_load_json(mmlu_path)))

    return proxy_dataset


def infer_hparams_path(model_name_or_path: str, hparams_dir: str) -> str:
    model_path = model_name_or_path.lower()
    if "llama" in model_path:
        name = "llama3-8b-instruct.json"
    elif "gpt-j" in model_path:
        name = "EleutherAI_gpt-j-6B.json"
    elif "gpt2" in model_path:
        name = "gpt2-xl.json"
    elif "phi" in model_path:
        name = "phi-1.5.json"
    elif "qwen" in model_path:
        name = "qwen2.5-7b-instruct.json"
    elif "vicuna" in model_path:
        name = "vicuna-7b-v1.5.json"
    elif "mistral" in model_path:
        name = "mistral-7b-instruct.json"
    else:
        name = "llama3-8b-instruct.json"

    return os.path.join(hparams_dir, name)


def main():
    parser = argparse.ArgumentParser(description="Run ROCR anchor selector pipeline (A/B/C)")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--template", type=str, default="llama3")
    parser.add_argument("--dataset_dir", type=str, default="./data")
    parser.add_argument("--eval_dataset_dir", type=str, default="./data/ORT/Target")
    parser.add_argument("--split", type=str, default="train")

    parser.add_argument("--forget_start", type=int, default=0)
    parser.add_argument("--forget_end", type=int, default=50)
    parser.add_argument("--candidate_start", type=int, default=50)
    parser.add_argument("--candidate_end", type=int, default=100)

    parser.add_argument("--z_layer", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default="./results/anchor_cache")
    parser.add_argument("--output_dir", type=str, default="./results/anchor_selection")

    parser.add_argument("--pool_size", type=int, default=32)
    parser.add_argument("--top_m", type=int, default=5)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--alpha_var", type=float, default=0.5)

    parser.add_argument("--lambda_proxy", type=float, default=1.0)
    parser.add_argument("--beta_retain", type=float, default=1.0)
    parser.add_argument("--gamma_collapse", type=float, default=2.0)
    parser.add_argument("--retain_drop_threshold", type=float, default=0.15)

    parser.add_argument("--hparams_path", type=str, default=None)
    parser.add_argument("--hparams_dir", type=str, default="./src/llmtuner/train/rocr/rocr_hparams")
    parser.add_argument("--strict_proxy_files", action="store_true")
    args = parser.parse_args()

    forget_targets = NAMES[args.forget_start:args.forget_end]
    candidate_targets = NAMES[args.candidate_start:args.candidate_end]

    hparams_path = args.hparams_path or infer_hparams_path(args.model_name_or_path, args.hparams_dir)
    with open(hparams_path, "r", encoding="utf-8") as f:
        hparams = rocrHyperParams(**json.load(f))

    z_layer = args.z_layer if args.z_layer is not None else hparams.layers[-1]

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_arg_dict = {
        "stage": "rocr",
        "model_name_or_path": args.model_name_or_path,
        "template": args.template,
        "dataset_dir": args.dataset_dir,
        "eval_dataset_dir": args.eval_dataset_dir,
        "split": args.split,
        "target": forget_targets[0] if forget_targets else "",
        "do_train": False,
        "output_dir": os.path.join(args.output_dir, "tmp_train_output"),
        "overwrite_output_dir": True,
        "overwrite_cache": True,
    }

    model_args, data_args, training_args, finetuning_args, _ = get_train_args(train_arg_dict)
    proxy_dataset = build_proxy_dataset(args.eval_dataset_dir, forget_targets, candidate_targets, strict_missing_files=args.strict_proxy_files,)

    pool_config = {
        "pool_size": args.pool_size,
        "min_strength_quantile": 0.2,
        "max_k_var_quantile": 0.8,
        "use_key": "h_bar",
    }
    static_config = {
        "mu": args.mu,
        "alpha_var": args.alpha_var,
        "weights": {"sim_band": 1.0, "strength": 1.0, "safe": 1.0, "compat": 1.0},
        "safe_mode": "cheap",
        "top_m": args.top_m,
    }
    proxy_config = {
        "task_weights": {"forget_qa": 1.0, "forget_fb": 1.0, "forget_mcp": 1.5, "forget_sqa": 1.5},
        "beta_retain": args.beta_retain,
        "gamma_collapse": args.gamma_collapse,
        "lambda_proxy": args.lambda_proxy,
        "retain_drop_threshold": args.retain_drop_threshold,
        "eval_set": {"num_qa": 2, "num_fb": 2, "num_mcp": 1, "num_sqa": 1, "num_retain": 5},
        "rocr_config": {},
    }

    run_anchor_selection_pipeline(
        model=model,
        tokenizer=tokenizer,
        forget_targets=forget_targets,
        candidate_targets=candidate_targets,
        dataset=proxy_dataset,
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        hparams=hparams,
        z_layer=z_layer,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        pool_config=pool_config,
        static_config=static_config,
        proxy_config=proxy_config,
    )

    print(f"Anchor selection finished. Results saved in: {args.output_dir}")
    print(f"Final mapping: {os.path.join(args.output_dir, 'final_anchor_selection.json')}")


if __name__ == "__main__":
    main()
