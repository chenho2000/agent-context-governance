"""Conservative novelty gate: abstain unless a window is explicitly low-information."""


def information_gate(objects, classifier=None, threshold=0.95):
    greetings = {"hi", "hello", "thanks", "thank you", "ok", "你好", "谢谢", "好的"}
    if any(o.metadata.get("facts") or o.metadata.get("constraint") or o.dependencies for o in objects):
        return {"keep": True, "reason": "facts, constraints or dependencies present", "tier": "rules"}
    if objects and all(o.content.strip().lower().strip(".!。！") in greetings for o in objects):
        return {"keep": False, "reason": "closed greeting vocabulary", "tier": "rules"}
    if classifier:
        result = classifier.classify(
            "\n".join(o.content for o in objects), ["new information", "social filler"]
        )
        if result["label"] == "social filler" and result["score"] >= threshold:
            return {"keep": False, "reason": "high-confidence filler suggestion", "tier": "encoder"}
    return {"keep": True, "reason": "unknown; defer to fact extractor", "tier": "abstain"}
