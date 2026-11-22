"""把 BERT-CRF 通道接进 ner.extract_elements（一次性 patch 脚本）。"""

import pathlib

p = pathlib.Path("zhilv_qiling_v2/ner.py")
text = p.read_text(encoding="utf-8")

anchor = "def extract_elements("
assert anchor in text, "找不到 extract_elements"

block = '''_BERT_CACHE: dict[str, Any] = {}


def _bert_load():
    """惰性加载 BERT-CRF；没有 GPU 产物时返回 (None, None)。"""
    if "loaded" in _BERT_CACHE:
        return _BERT_CACHE.get("model"), _BERT_CACHE.get("tokenizer")
    model = tokenizer = None
    try:
        from . import bert_ner

        if bert_ner.available():
            model, tokenizer, _meta = bert_ner.load_model()
    except Exception:
        model = tokenizer = None
    _BERT_CACHE["loaded"] = True
    _BERT_CACHE["model"] = model
    _BERT_CACHE["tokenizer"] = tokenizer
    return model, tokenizer


def bert_elements(document: DocumentParse, *, min_confidence: float = 0.0) -> list[Element]:
    """用 BERT-CRF 抽要素（节点①的模型通道，需 GPU 训练产物）。"""
    model, tokenizer = _bert_load()
    if model is None or tokenizer is None:
        return []
    from . import bert_ner

    out: list[Element] = []
    for block in document.blocks:
        if len(block.text.strip()) < 8:
            continue
        try:
            spans = bert_ner.predict(model, tokenizer, block.text)
        except Exception:
            continue
        for start, end, entity, score in spans:
            kind = _MODEL_KIND_MAP.get(entity)
            if not kind:
                continue
            surface = block.text[start:end].strip()
            if len(surface) < 2:
                continue
            normalized = _normalize_surface(kind, surface)
            if not normalized:
                continue
            confidence = 0.72 + 0.25 * max(0.0, min(1.0, float(score)))
            if confidence < min_confidence:
                continue
            if kind == "party":
                hints = _ROLE_HINTS
            elif kind == "amount":
                hints = _AMOUNT_ROLE_HINTS
            elif kind == "date":
                hints = _DATE_ROLE_HINTS
            else:
                hints = ()
            out.append(
                Element(
                    element_id=stable_id("el", "bert", kind, normalized, block.block_id),
                    kind=kind,
                    value=surface,
                    normalized=normalized,
                    role=_role_from_block(block.text, hints) if hints else "other",
                    clause_id=_find_clause_for_block(document, block.block_id),
                    span=SpanRef(block_id=block.block_id, quote=surface, page=block.page),
                    confidence=confidence,
                    source="bert_ner",
                )
            )
    return out


'''
text = text.replace(anchor, block + anchor, 1)

old_body = """    model_hits: list[Element] = []
    if config.ner_enable_model:
        model_hits = model_elements(document, tagger, min_confidence=config.ner_min_confidence)"""
new_body = """    model_hits: list[Element] = []
    bert_hits: list[Element] = []
    if config.ner_enable_model:
        if getattr(config, "ner_use_bert", True):
            bert_hits = bert_elements(document, min_confidence=config.ner_min_confidence)
        # BERT 通道有产出就不再叠加感知机（后者宏 F1 只有 0.0179，只作无 GPU 时的兜底）
        model_hits = bert_hits or model_elements(
            document, tagger, min_confidence=config.ner_min_confidence
        )"""
assert old_body in text, "找不到 model_hits 段落"
text = text.replace(old_body, new_body, 1)

old_stats = '        "model_available": tagger is not None,'
new_stats = ('        "model_available": tagger is not None or bool(bert_hits),\n'
             '        "model_channel": "bert_ner" if bert_hits else ("perceptron" if model_hits else "off"),')
assert old_stats in text, "找不到 model_available 行"
text = text.replace(old_stats, new_stats, 1)

p.write_text(text, encoding="utf-8")
print("ner.py 已接入 BERT 通道")

# config 增加开关
c = pathlib.Path("zhilv_qiling_v2/config.py")
ct = c.read_text(encoding="utf-8")
if "ner_use_bert" not in ct:
    key = "    ner_enable_model: bool = False"
    assert key in ct
    ct = ct.replace(key, key + "\n    # 有 GPU 训练产物时优先用 BERT-CRF（宏 F1 0.5348，感知机只有 0.0179）\n    ner_use_bert: bool = True", 1)
    c.write_text(ct, encoding="utf-8")
    print("config.py 已加 ner_use_bert")
else:
    print("config.py 已有 ner_use_bert")
