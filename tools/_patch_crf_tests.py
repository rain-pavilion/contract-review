"""把 test_gpu_models 的 CRF 三个用例改到 pytorch-crf 的契约上（一次性 patch）。"""

import pathlib

p = pathlib.Path("zhilv_qiling_v2/tests_v2/test_gpu_models.py")
text = p.read_text(encoding="utf-8")

start = text.find("class CrfMathTest")
end = text.find("class EntailmentPairTest")
assert start > 0 and end > start

new_block = '''class CrfMathTest(unittest.TestCase):
    """CRF 层：非法转移掩码只作诊断，训练/解码走 pytorch-crf 的成熟实现。

    早期版本是手写 CRF，gold 分与 log Z 用了两套转移矩阵，loss 会走成负数、模型塌成
    全 O。改为 pytorch-crf 后这两条一致性由库保证，因此测试只校验"约束被正确表达"
    与"调用契约正确"，不再校验内部前向算法。
    """

    @unittest.skipUnless(HAS_TORCH, "需要 torch")
    def test_illegal_transition_mask_blocks_orphan_inside(self) -> None:
        from zhilv_qiling_v2 import learned

        labels = list(learned.NER_LABELS)
        mask = bert_ner.illegal_transition_mask(len(labels))
        for i, src in enumerate(labels):
            for j, dst in enumerate(labels):
                if dst.startswith("I-") and not (src == "B" + dst[1:] or src == dst):
                    # 掩码是 [当前标签, 前一个标签]
                    self.assertEqual(float(mask[i, j]), -1e4, msg=f"{src}->{dst} 应被禁止")

    @unittest.skipUnless(HAS_TORCH, "需要 torch")
    def test_crf_nll_is_finite_and_positive(self) -> None:
        n_tags, seq_len, batch = 5, 7, 2
        crf = bert_ner.build_crf(n_tags)
        emissions = torch.randn(batch, seq_len, n_tags)
        tags = torch.randint(0, n_tags, (batch, seq_len))
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        # torchcrf 返回 log-likelihood，取负即 NLL
        loss = -crf(emissions, tags, mask, reduction="mean")
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)

    @unittest.skipUnless(HAS_TORCH, "需要 torch")
    def test_decode_returns_one_tag_per_token(self) -> None:
        n_tags, seq_len, batch = 5, 6, 3
        crf = bert_ner.build_crf(n_tags)
        emissions = torch.randn(batch, seq_len, n_tags)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        paths = crf.decode(emissions, mask)
        self.assertEqual(len(paths), batch)
        self.assertEqual(len(paths[0]), seq_len)
        for path in paths:
            self.assertTrue(all(0 <= int(tag) < n_tags for tag in path))


'''
text = text[:start] + new_block + text[end:]
p.write_text(text, encoding="utf-8")
print("CRF 用例已改到 pytorch-crf 契约")
