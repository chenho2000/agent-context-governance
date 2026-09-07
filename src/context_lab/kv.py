"""一个标量 causal-attention 演示；不是 Transformer 或 GPU benchmark。"""

import math


def attention(values):
    # 第一层隐状态依赖因果前缀，第二层 K/V 因而也受前文编辑影响。
    hidden = []
    for i, query in enumerate(values):
        logits = [query * k / 10 for k in values[: i + 1]]
        peak = max(logits)
        weights = [math.exp(x - peak) for x in logits]
        hidden.append(sum(w * v for w, v in zip(weights, values)) / sum(weights))
    return {"K_layer2": [h * 0.7 for h in hidden], "V_layer2": [h * 0.9 for h in hidden]}


def demo():
    original = attention([1.0, 2.0, 3.0])
    appended = attention([1.0, 2.0, 3.0, 4.0])
    edited = attention([1.0, 8.0, 3.0])
    return {
        "original": original,
        "appended": appended,
        "edited": edited,
        "append_prefix_reusable": original["K_layer2"] == appended["K_layer2"][:3],
        "edit_changes_suffix": original["K_layer2"][2] != edited["K_layer2"][2],
    }
