from typing import Dict, List, Any


def _normalize_example(example: Any) -> Dict[str, str]:
    if isinstance(example, dict):
        prompt = example.get("prompt") or example.get("question") or example.get("text") or ""
        answer = example.get("answer") or example.get("response") or example.get("target") or ""
        return {"prompt": str(prompt), "answer": str(answer)}
    if isinstance(example, str):
        return {"prompt": example, "answer": ""}
    return {"prompt": str(example), "answer": ""}


def _take(examples: List[Any], n: int) -> List[Dict[str, str]]:
    return [_normalize_example(x) for x in examples[: max(0, n)]]


def build_proxy_eval_set(
    forget_target: str,
    dataset,
    num_qa: int = 2,
    num_fb: int = 2,
    num_mcp: int = 1,
    num_sqa: int = 1,
    num_retain: int = 5,
) -> dict:
    target_block = dataset.get(forget_target, {}) if isinstance(dataset, dict) else {}

    forget_qa = _take(target_block.get("forget_qa", []), num_qa)
    forget_fb = _take(target_block.get("forget_fb", []), num_fb)
    forget_mcp = _take(target_block.get("forget_mcp", []), num_mcp)
    forget_sqa = _take(target_block.get("forget_sqa", []), num_sqa)

    retain_pool = []
    if isinstance(dataset, dict):
        retain_pool.extend(dataset.get("retain", []))
        for name, block in dataset.items():
            if name in [forget_target, "retain"] or not isinstance(block, dict):
                continue
            retain_pool.extend(block.get("retain", []))
    retain = _take(retain_pool, num_retain)

    return {
        "forget_qa": forget_qa,
        "forget_fb": forget_fb,
        "forget_mcp": forget_mcp,
        "forget_sqa": forget_sqa,
        "retain": retain,
    }
