"""给 entailment 加 BERT 后端开关（一次性 patch，第二版：按真实源码行改）。"""

import pathlib

p = pathlib.Path("zhilv_qiling_v2/entailment.py")
text = p.read_text(encoding="utf-8")

if "def set_backend" in text:
    print("已存在 set_backend，跳过")
    raise SystemExit(0)

anchor = "def judge_pair("
assert anchor in text

block = '''_BACKEND = {"name": "rules"}
_BERT = {"loaded": False, "predictor": None}


def set_backend(name: str) -> str:
    """切换判别后端：rules（8 维弱监督，默认）/ bert（双塔 cross-attention）/ auto。

    默认留在 rules 是为了可复现：BERT 后端依赖 models/bert_entail/model.pt，
    有产物才生效；没有产物时 auto 静默退回 rules（降级留账）。
    """
    choice = (name or "rules").strip().lower()
    if choice not in ("rules", "bert", "auto"):
        choice = "rules"
    _BACKEND["name"] = choice
    return choice


def backend() -> str:
    return str(_BACKEND["name"])


def _bert_predictor():
    """惰性加载双塔判别器；不可用时返回 None。"""
    if _BERT["loaded"]:
        return _BERT["predictor"]
    _BERT["loaded"] = True
    try:
        from . import bert_entail

        if bert_entail.available():
            _BERT["predictor"] = bert_entail.load_predictor()
    except Exception:
        _BERT["predictor"] = None
    return _BERT["predictor"]


'''
text = text.replace(anchor, block + anchor, 1)

old = """    feats = clause_features(clause_text, content, risk_tags=tags, codes=codes)
    classifier = classifier or load_classifier()
    label, score, margin = classifier.predict(feats)"""
new = """    feats = clause_features(clause_text, content, risk_tags=tags, codes=codes)
    label: str | None = None
    score = 0.0
    margin = 0.0
    source = "rules"
    choice = backend()
    if choice in ("bert", "auto") and classifier is None:
        predictor = _bert_predictor()
        if predictor is not None:
            try:
                out = predictor.predict(clause_text, content)
                label = str(out["label"])
                score = float(out["score"])
                probabilities = sorted(
                    (float(v) for v in (out.get("probabilities") or {}).values()), reverse=True
                )
                margin = score - (probabilities[1] if len(probabilities) > 1 else 0.0)
                source = "bert_entail"
            except Exception:
                label = None
    if label is None:
        classifier = classifier or load_classifier()
        label, score, margin = classifier.predict(feats)"""
assert old in text, "找不到 judge_pair 特征段"
text = text.replace(old, new, 1)

old_src = "        source=classifier.source,"
assert old_src in text, "找不到 source=classifier.source"
text = text.replace(old_src, "        source=source,", 1)

p.write_text(text, encoding="utf-8")
print("entailment.py 已加后端开关")
