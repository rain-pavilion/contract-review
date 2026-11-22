import pathlib

p = pathlib.Path("zhilv_qiling_v2/bert_ner.py")
text = p.read_text(encoding="utf-8")

# 1) _scores 与 _partition 必须用同一张转移矩阵（都含 illegal），否则 gold 分与 logZ 不同尺度，
#    训练会朝"把 logZ 压小"的方向跑，loss 变成负数。
old = '                trans = self.transitions[tags[:, step], tags[:, step - 1]]'
new = '                trans = (self.transitions + self.illegal)[tags[:, step], tags[:, step - 1]]'
assert old in text, "scores 未命中"
text = text.replace(old, new, 1)

# 文档字符串同步修正
old_doc = """            注意：这里**不加** `illegal` 惩罚。金标里 subword 续接位被标成 O（-100→0），
            真实实体的 I-X 因而会"跨过"一个 O 位，加上 -1e4 会让 gold 分数直接崩到 -10⁴，
            loss 变成一万多。非法转移只在 `_partition` / `viterbi` 里禁止（推理时才需要）。
"""
new_doc = """            与 `_partition` 用**同一张** `transitions + illegal` 矩阵：两侧尺度必须一致，
            否则 gold 分与 log Z 不可比（曾出现 loss 变成负数的退化）。
            金标链由 `_encode` 保证不会出现非法转移（重叠 span 已剔除）。
"""
text = text.replace(old_doc, new_doc, 1)

# 2) char_labels 构造：剔除重叠 span，保证 I-X 的前一位一定是 B-X/I-X
old_block = """    chars = list(piece)
    char_labels = ["O"] * len(chars)
    for start, end, kind in spans:
        if start >= len(char_labels):
            continue
        char_labels[start] = f"B-{kind}"
        for index in range(start + 1, min(end, len(char_labels))):
            char_labels[index] = f"I-{kind}"
"""
new_block = """    chars = list(piece)
    char_labels = ["O"] * len(chars)
    # 按起点排序并剔除重叠 span：重叠会让后一个 span 的 B- 覆盖前一个的 I-，
    # 从而造出 "O -> I-X" 这种非法转移（实测 8241 个转移里有 18 个）。
    occupied_until = -1
    for start, end, kind in sorted(spans, key=lambda item: (item[0], -item[1])):
        if start >= len(char_labels) or start < occupied_until:
            continue
        end = min(end, len(char_labels))
        if end <= start:
            continue
        char_labels[start] = f"B-{kind}"
        for index in range(start + 1, end):
            char_labels[index] = f"I-{kind}"
        occupied_until = end
"""
assert old_block in text, "encode 块未命中"
text = text.replace(old_block, new_block, 1)
p.write_text(text, encoding="utf-8")
print("patched bert_ner.py")
