import pathlib, re

p = pathlib.Path("zhilv_qiling_v2/bert_ner.py")
text = p.read_text(encoding="utf-8")

# 1) build_crf 换成 pytorch-crf（自己写的那版 logZ 与穷举对不上，不再冒险）
start = text.index("def build_crf(")
end = text.index("def build_model(")
new_crf = '''def build_crf(num_tags: int):
    """返回 torchcrf.CRF。

    自己手写过一版前向算法，log Z 与穷举结果对不上（0.41 的偏差），
    训练 loss 一路走成负数、F1 掉到 0。改用 `pytorch-crf`（0.7.2，成熟实现），
    它内部用同一套 `transitions` 一致地算 gold 分与 log Z，不会再出现尺度不一致。
    """
    torch = _torch()
    try:
        from torchcrf import CRF as _CRF
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("需要 pytorch-crf：python -m pip install pytorch-crf") from exc
    return _CRF(num_tags, batch_first=True)


def illegal_transition_mask(num_tags: int, *, illegal_penalty: float = -1e4):
    """把"BIO 语法上不允许的转移"列成矩阵，只用于诊断与测试，不参与训练。"""
    torch = _torch()
    mask = torch.zeros(num_tags, num_tags)
    for cur in range(num_tags):
        label_cur = ID2LABEL[cur]
        for prev in range(num_tags):
            label_prev = ID2LABEL[prev]
            if label_cur.startswith("I-"):
                kind = label_cur[2:]
                if label_prev not in (f"B-{kind}", f"I-{kind}"):
                    mask[cur, prev] = illegal_penalty
    return mask


'''
text = text[:start] + new_crf + text[end:]

# 2) 模型层级：去掉自研 CRF 的 nll/viterbi 调用
text = text.replace("            loss = model.crf.nll(emissions, tags, label_mask)",
                    "            # torchcrf 返回 log-likelihood，取负即 NLL\n"
                    "            loss = -model.crf(emissions, tags, label_mask.bool(), reduction=\"mean\")")
text = text.replace("            paths = model.crf.viterbi(emissions, mask)",
                    "            paths = [model.crf.decode(emissions, mask.bool())[0]]")
text = text.replace('''            paths = [model.crf.decode(emissions, mask.bool())[0]]''',
                    '''            decoded = model.crf.decode(emissions, mask.bool())
            paths = [decoded[0]]''')
p.write_text(text, encoding="utf-8")
print("patched")
