import pathlib, re

def patch(path, subs):
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8")
    for old, new in subs:
        if old not in text:
            print(f"[miss] {path}: {old[:60]}")
            continue
        text = text.replace(old, new, 1)
        print(f"[ok]   {path}: {old[:50]} -> ...")
    p.write_text(text, encoding="utf-8")

# bert_ner.py：改成用 pretrained.resolve() 探测，不再写死仓库名
patch("zhilv_qiling_v2/bert_ner.py", [
    ('PRETRAINED_DIR = MODEL_DIR / "pretrained" / "chinese-roberta-wwm-ext"',
     'from . import pretrained as _pretrained\n\n'
     'PRETRAINED_DIR = _pretrained.PRETRAINED_ROOT / "chinese-roberta-wwm-ext"'),
    ('            self.encoder = AutoModel.from_pretrained(str(PRETRAINED_DIR))',
     '            self.encoder = AutoModel.from_pretrained(str(_weights_dir()))'),
    ('    if not PRETRAINED_DIR.exists():\n        raise RuntimeError(f"缺少预训练权重：{PRETRAINED_DIR}")',
     '    if _weights_dir() is None:\n        raise RuntimeError(f"缺少预训练权重：{PRETRAINED_DIR}（可用 models/pretrained/ 下任一中文 BERT）")'),
    ('    tokenizer = AutoTokenizer.from_pretrained(str(PRETRAINED_DIR))',
     '    tokenizer = AutoTokenizer.from_pretrained(str(_weights_dir()))'),
    ('        "pretrained": str(PRETRAINED_DIR.name),',
     '        "pretrained": str(_weights_dir().name if _weights_dir() else ""),'),
])

# bert_entail.py
patch("zhilv_qiling_v2/bert_entail.py", [
    ('PRETRAINED_DIR = MODEL_DIR / "pretrained" / "chinese-roberta-wwm-ext"',
     'from . import pretrained as _pretrained\n\n'
     'PRETRAINED_DIR = _pretrained.PRETRAINED_ROOT / "chinese-roberta-wwm-ext"'),
    ('            self.encoder = AutoModel.from_pretrained(str(PRETRAINED_DIR))',
     '            self.encoder = AutoModel.from_pretrained(str(_weights_dir()))'),
    ('    if not PRETRAINED_DIR.exists():\n        raise RuntimeError(f"缺少预训练权重：{PRETRAINED_DIR}")',
     '    if _weights_dir() is None:\n        raise RuntimeError(f"缺少预训练权重：{PRETRAINED_DIR}（可用 models/pretrained/ 下任一中文 BERT）")'),
    ('    tokenizer = AutoTokenizer.from_pretrained(str(PRETRAINED_DIR))',
     '    tokenizer = AutoTokenizer.from_pretrained(str(_weights_dir()))'),
    ('    return (OUTPUT_DIR / "model.pt").exists() and PRETRAINED_DIR.exists()',
     '    return (OUTPUT_DIR / "model.pt").exists() and _weights_dir() is not None'),
])
print("done")
