import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from ....data.loader import load_single_dataset
from ....data.parser import get_dataset_list
from ..rocr.compute_z import get_module_input_output_at_words
from ..rocr.compute_ks import compute_ks
from ..rocr.rocr_main import get_context_templates
from .entity_feature import EntityFeature
from .feature_cache import feature_exists, load_entity_feature, save_entity_feature


def _resolve_entity_dataset_name(data_args, entity_name: str, split: str = "train") -> str:
    dataset_info_path = Path(data_args.dataset_dir) / "dataset_info.json"
    if not dataset_info_path.exists():
        raise FileNotFoundError(f"dataset_info.json not found under {data_args.dataset_dir}")

    with open(dataset_info_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    preferred_suffix = []
    if split == "train":
        preferred_suffix = ["Positive", "Passage", "Reject", "Counterfact"]
    else:
        preferred_suffix = ["Passage", "Positive", "Reject", "Counterfact"]

    for suffix in preferred_suffix:
        key = f"{entity_name}_{suffix}"
        if key in dataset_info:
            return key

    prefix = f"{entity_name}_"
    fallback = [k for k in dataset_info if k.startswith(prefix)]
    if fallback:
        return sorted(fallback)[0]

    raise ValueError(f"Cannot resolve dataset key for entity '{entity_name}' in dataset_info.json")


def load_project_aligned_entity_samples(
    data_args,
    model_args,
    training_args,
    finetuning_args,
    entity_name: str,
    split: str = "train",
):
    aligned_data_args = copy.deepcopy(data_args)
    aligned_data_args.split = split
    aligned_data_args.dataset = _resolve_entity_dataset_name(data_args, entity_name, split=split)

    dataset_attrs = get_dataset_list(aligned_data_args)
    all_samples = []
    attr_meta = []
    for dataset_attr in dataset_attrs:
        aligned_ds = load_single_dataset(dataset_attr, model_args, aligned_data_args)
        # convert to small list for feature extraction
        samples = [row for row in aligned_ds]
        all_samples.extend(samples)
        attr_meta.append(
            {
                "dataset_name": dataset_attr.dataset_name,
                "load_from": dataset_attr.load_from,
                "subset": dataset_attr.subset,
                "folder": dataset_attr.folder,
                "formatting": dataset_attr.formatting,
            }
        )

    return all_samples, attr_meta


def _build_prompt_template_from_sample(entity_name: str, sample: Dict) -> str:
    prompt_messages = sample.get("prompt", [])
    user_contents = [x.get("content", "") for x in prompt_messages if x.get("role") == "user"]
    raw_prompt = "\n".join([x for x in user_contents if x])

    clean_name = entity_name.replace("_", " ")
    if clean_name in raw_prompt:
        return raw_prompt.replace(clean_name, "{}", 1)

    if "{}" in raw_prompt:
        return raw_prompt

    if raw_prompt.strip():
        return raw_prompt.strip() + " {}"

    return "{}"


def _build_target_new_from_sample(sample: Dict) -> str:
    response = sample.get("response", [])
    if response and isinstance(response, list):
        text = response[0].get("content", "").strip()
        if text:
            return text.split("\n")[0][:64]
    return "Unfortunately"


def build_entity_requests_from_project_samples(
    entity_name: str,
    samples,
    request_mode: str = "project_aligned",
) -> list:
    if request_mode != "project_aligned":
        raise ValueError(f"Unsupported request_mode: {request_mode}")

    subject = entity_name.replace("_", " ")
    requests = []
    for idx, sample in enumerate(samples):
        requests.append(
            {
                "case_id": f"{entity_name}_{idx}",
                "subject": subject,
                "prompt": _build_prompt_template_from_sample(entity_name, sample),
                "target_new": {"str": _build_target_new_from_sample(sample)},
            }
        )
    return requests


def get_project_context_templates(model, tok):
    return get_context_templates(model, tok)


def _reduce_expanded_reprs(expanded: torch.Tensor, context_templates: List[List[str]]) -> torch.Tensor:
    # keep the same averaging semantics as compute_ks
    context_type_lens = [len(context_type) for context_type in context_templates]
    context_len = sum(context_type_lens)

    if context_len <= 0:
        return expanded

    csum = [0]
    for x in context_type_lens:
        csum.append(csum[-1] + x)

    agg = []
    for i in range(0, expanded.size(0), context_len):
        base = expanded[i : i + context_len]
        groups = []
        for j in range(len(csum) - 1):
            start, end = csum[j], csum[j + 1]
            groups.append(base[start:end].mean(0))
        agg.append(torch.stack(groups, 0).mean(0))
    return torch.stack(agg, dim=0)


def collect_hidden_representations(
    model,
    tok,
    entity_name: str,
    requests: list,
    hparams,
    z_layer: int,
):
    context_templates = get_project_context_templates(model, tok)

    expanded_context_templates = [
        context.format(request["prompt"])
        for request in requests
        for context_type in context_templates
        for context in context_type
    ]
    expanded_words = [
        request["subject"]
        for request in requests
        for context_type in context_templates
        for _ in context_type
    ]

    _, l_output = get_module_input_output_at_words(
        model,
        tok,
        z_layer,
        context_templates=expanded_context_templates,
        words=expanded_words,
        module_template=hparams.layer_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )
    reduced = _reduce_expanded_reprs(l_output, context_templates)
    return [reduced[i].detach().cpu() for i in range(reduced.size(0))]


def collect_activation_representations(
    model,
    tok,
    entity_name: str,
    requests: list,
    hparams,
    z_layer: int,
):
    context_templates = get_project_context_templates(model, tok)
    ks = compute_ks(model, tok, requests, hparams, z_layer, context_templates)
    if torch.isnan(ks).any():
        expanded_context_templates = [
            context.format(request["prompt"])
            for request in requests
            for context_type in context_templates
            for context in context_type
        ]
        expanded_words = [
            request["subject"]
            for request in requests
            for context_type in context_templates
            for _ in context_type
        ]
        l_input, _ = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=expanded_context_templates,
            words=expanded_words,
            module_template=hparams.rewrite_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )
        ks = _reduce_expanded_reprs(l_input, context_templates)
    return [ks[i].detach().cpu() for i in range(ks.size(0))]


def _stack_mean_and_var(vectors: List[torch.Tensor]) -> Tuple[torch.Tensor, float]:
    matrix = torch.stack([v.float() for v in vectors], dim=0)
    mean_vec = matrix.mean(dim=0)
    var = float(torch.mean(torch.sum((matrix - mean_vec.unsqueeze(0)) ** 2, dim=1)).item())
    return mean_vec, var


def compute_entity_feature(
    model,
    tok,
    entity_name: str,
    data_args,
    model_args,
    training_args,
    finetuning_args,
    hparams,
    z_layer: int,
    split: str = "train",
    overwrite: bool = False,
) -> EntityFeature:
    samples, dataset_attr_meta = load_project_aligned_entity_samples(
        data_args,
        model_args,
        training_args,
        finetuning_args,
        entity_name=entity_name,
        split=split,
    )
    requests = build_entity_requests_from_project_samples(entity_name=entity_name, samples=samples)

    hidden_list = collect_hidden_representations(
        model, tok, entity_name=entity_name, requests=requests, hparams=hparams, z_layer=z_layer
    )
    act_list = collect_activation_representations(
        model, tok, entity_name=entity_name, requests=requests, hparams=hparams, z_layer=z_layer
    )

    h_bar, h_var = _stack_mean_and_var(hidden_list)
    k_bar, k_var = _stack_mean_and_var(act_list)
    strength = float(torch.norm(k_bar, p=2).item())

    feature = EntityFeature(
        name=entity_name,
        hidden_list=hidden_list,
        act_list=act_list,
        h_bar=h_bar,
        k_bar=k_bar,
        h_var=h_var,
        k_var=k_var,
        strength=strength,
        meta={
            "dataset_name": getattr(data_args, "dataset", None),
            "dataset_attrs": dataset_attr_meta,
            "split": split,
            "template_func": "rocr_main.get_context_templates",
            "z_layer": int(z_layer),
            "token_locator": "compute_z.get_module_input_output_at_words",
            "reused_paths": [
                "get_context_templates",
                "repr_tools.get_reprs_at_word_tokens",
                "compute_ks",
                "compute_z.get_module_input_output_at_words",
            ],
        },
    )
    return feature


def precompute_entity_features(
    model,
    tok,
    entity_names: list,
    data_args,
    model_args,
    training_args,
    finetuning_args,
    hparams,
    z_layer: int,
    cache_dir: str,
    split: str = "train",
    overwrite: bool = False,
) -> dict:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    feature_dict = {}
    index_payload = {}
    for entity_name in entity_names:
        entity_dir = cache_root / entity_name
        if feature_exists(str(entity_dir)) and not overwrite:
            feature = load_entity_feature(str(entity_dir))
        else:
            feature = compute_entity_feature(
                model,
                tok,
                entity_name=entity_name,
                data_args=data_args,
                model_args=model_args,
                training_args=training_args,
                finetuning_args=finetuning_args,
                hparams=hparams,
                z_layer=z_layer,
                split=split,
                overwrite=overwrite,
            )
            save_entity_feature(feature, str(entity_dir))

        feature_dict[entity_name] = feature
        index_payload[entity_name] = {
            "cache_path": str(entity_dir),
            "h_var": float(feature.h_var),
            "k_var": float(feature.k_var),
            "strength": float(feature.strength),
            "z_layer": int(z_layer),
            "split": split,
        }

    with open(cache_root / "features_index.json", "w", encoding="utf-8") as f:
        json.dump(index_payload, f, ensure_ascii=False, indent=2)

    return feature_dict
