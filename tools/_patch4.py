import pathlib, re
p = pathlib.Path("zhilv_qiling_v2/bert_ner.py")
text = p.read_text(encoding="utf-8")
before = text.count("(self.transitions + self.illegal)")
text = text.replace("(self.transitions + self.illegal)", "self.transitions")
print(f"[ok] 替换 {before} 处 transitions+illegal → transitions")

# 更新说明：illegal 仅作诊断用的先验提示，不参与训练与解码
text = text.replace(
"""            与 `_partition` 用**同一张** `transitions + illegal` 矩阵：两侧尺度必须一致，
            否则 gold 分与 log Z 不可比（曾出现 loss 变成负数的退化）。
            金标链由 `_encode` 保证不会出现非法转移（重叠 span 已剔除）。
""",
"""            说明：这里与 `_partition` / `viterbi` 都用**纯 transitions**，不叠 `illegal`。
            原因有实测教训：一旦把 -1e4 的硬惩罚混进来，只要金标里存在极少量非法转移
            （重叠 span 造成 18/8241），gold 分与 log Z 就不再可比，loss 会变成负数并退化。
            非法转移本来在数据里从不出现，学到的转移权重自然很低，不需要手工惩罚。
""",
1)
text = text.replace(
'''    """返回 (transitions, start, end, mask)，mask 里禁止 I-X 从非 B-X/I-X 转入。"""''',
'''    """构建 CRF；`illegal` 仅作为先验提示保留（诊断/分析用），训练与解码不用它。"""''',
1)
p.write_text(text, encoding="utf-8")
print("done")
