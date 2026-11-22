import pathlib

HELPER = '''

def _weights_dir():
    """返回可用的中文 BERT 权重目录。

    优先走 `pretrained.resolve()`（按 config.json + 权重体积探测，
    兼容 chinese-roberta-wwm-ext / bert-base-chinese 等任意中文 BERT）；
    一个都没探测到时回落到默认路径，交给调用方报错。
    """
    found = _pretrained.resolve()
    if found is not None:
        return found
    return PRETRAINED_DIR if PRETRAINED_DIR.exists() else None
'''

for path, anchor in [
    ("zhilv_qiling_v2/bert_ner.py", 'PRETRAINED_DIR = _pretrained.PRETRAINED_ROOT / "chinese-roberta-wwm-ext"'),
    ("zhilv_qiling_v2/bert_entail.py", 'PRETRAINED_DIR = _pretrained.PRETRAINED_ROOT / "chinese-roberta-wwm-ext"'),
]:
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8")
    if "_weights_dir():" in text:
        print(f"[skip] {path} 已有 helper")
        continue
    if anchor not in text:
        print(f"[miss] {path}")
        continue
    text = text.replace(anchor, anchor + "\n" + HELPER, 1)
    p.write_text(text, encoding="utf-8")
    print(f"[ok] {path} 插入 _weights_dir()")

# BertCrfNer / 双塔模型里直接调用处加保底
for path, old, new in [
    ("zhilv_qiling_v2/bert_ner.py",
     '            self.encoder = AutoModel.from_pretrained(str(_weights_dir()))',
     '            weights = _weights_dir() or PRETRAINED_DIR\n            self.encoder = AutoModel.from_pretrained(str(weights))'),
    ("zhilv_qiling_v2/bert_entail.py",
     '            self.encoder = AutoModel.from_pretrained(str(_weights_dir()))',
     '            weights = _weights_dir() or PRETRAINED_DIR\n            self.encoder = AutoModel.from_pretrained(str(weights))'),
]:
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8")
    if old in text:
        text = text.replace(old, new, 1)
        p.write_text(text, encoding="utf-8")
        print(f"[ok] {path} 模型构造加保底")
    else:
        print(f"[skip] {path} 已处理")

# bert_embed：允许 bge-m3 或任意 bge-* 目录
p = pathlib.Path("zhilv_qiling_v2/bert_embed.py")
text = p.read_text(encoding="utf-8")
old = 'WEIGHTS_DIR = MODEL_DIR / "pretrained" / "bge-m3"'
if old in text and "_embed_dir" not in text:
    new = (old + '''

def _embed_dir():
    """BGE 权重目录：优先 bge-m3，其次任意 bge-* （小模型也能跑，只是维度不同）。"""
    exact = _pretrained.resolve("bge-m3")
    if exact is not None:
        return exact
    for path in sorted(_pretrained.candidates()):
        if path.name.startswith("bge") and _pretrained._has_weights(path):
            return path
    return None
''')
    text = text.replace(old, new, 1)
    if "from . import pretrained as _pretrained" not in text:
        text = text.replace(old, "from . import pretrained as _pretrained\n\n" + new, 1)
    p.write_text(text, encoding="utf-8")
    print("[ok] bert_embed.py 插入 _embed_dir()")
else:
    print("[skip] bert_embed.py")
print("done")
