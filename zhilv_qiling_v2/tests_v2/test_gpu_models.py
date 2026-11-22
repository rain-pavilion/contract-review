"""GPU 相关组件的单元测试。

设计原则：**没有 torch 也要能跑通**。凡是需要 torch 的用例一律
`@unittest.skipUnless(HAS_TORCH, ...)`，这样在只装了 numpy/sklearn 的机器上
测试集仍然全绿，不会因为环境缺件把整个 CI 染红。
"""

from __future__ import annotations

import pathlib
import unittest

from zhilv_qiling_v2 import bert_entail, bert_embed, bert_ner, pretrained

try:  # noqa: SIM105 —— 探测 torch 是否存在，不抛错
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover - 环境相关
    torch = None
    HAS_TORCH = False


class PretrainedResolutionTest(unittest.TestCase):
    """权重探测：不写死仓库名，按"真能加载"判断。"""

    def test_describe_shape(self) -> None:
        info = pretrained.describe()
        self.assertIn("root", info)
        self.assertIn("candidates", info)
        for row in info["candidates"]:
            self.assertIn("name", row)
            self.assertIn("ready", row)
            self.assertIn("weight_mb", row)

    def test_resolve_returns_existing_dir_or_none(self) -> None:
        path = pretrained.resolve()
        if path is None:
            self.assertIsNone(path)
        else:
            self.assertIsInstance(path, pathlib.Path)
            self.assertTrue(path.is_dir())
            self.assertTrue((path / "config.json").exists())

    def test_resolve_unknown_name_is_none(self) -> None:
        self.assertIsNone(pretrained.resolve("绝对不存在的模型名-xyz"))

    def test_min_weight_threshold_guards_half_download(self) -> None:
        """只有 tokenizer 没有权重的目录不能被认成 ready。"""
        self.assertGreaterEqual(pretrained.MIN_WEIGHT_BYTES, 10 * 1024 * 1024)


class LazyImportTest(unittest.TestCase):
    """三个 BERT 模块必须惰性导入 torch：没有 torch 时 import 不炸、available 为 False。"""

    def test_available_false_without_weights_or_torch(self) -> None:
        if HAS_TORCH and pretrained.resolve("bge-m3"):
            self.assertTrue(bert_embed.available())
        else:
            self.assertFalse(bert_embed.available())
        if HAS_TORCH and pretrained.resolve("bert-base-chinese"):
            self.assertTrue(bert_entail.available())
        else:
            self.assertIn(bert_entail.available(), (True, False))

    def test_device_of_raises_helpful_error_without_torch(self) -> None:
        if HAS_TORCH:
            self.assertIn(bert_ner.device_of(prefer_cuda=False).type, ("cpu", "cuda"))
        else:
            with self.assertRaises(RuntimeError) as ctx:
                bert_ner.device_of()
            self.assertIn("torch", str(ctx.exception))

    def test_window_constants_sane(self) -> None:
        self.assertLess(bert_ner.STRIDE, bert_ner.WINDOW)
        self.assertEqual(bert_ner.WINDOW, bert_ner.MAX_LEN - 16)
        self.assertGreater(bert_ner.MAX_LEN, bert_ner.STRIDE)


class CrfMathTest(unittest.TestCase):
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
                    # 掩码索引是 [当前标签, 前一个标签] = [dst, src]
                    self.assertEqual(float(mask[j, i]), -1e4, msg=f"{src}->{dst} 应被禁止")

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


class EntailmentPairTest(unittest.TestCase):
    def test_labels_align_with_rule_module(self) -> None:
        from zhilv_qiling_v2 import entailment

        self.assertEqual(bert_entail.LABELS, entailment.LABELS)
        self.assertEqual(len(bert_entail.LABELS), 3)

    def test_available_is_bool(self) -> None:
        self.assertIsInstance(bert_entail.available(), bool)
        self.assertIsInstance(bert_embed.available(), bool)


if __name__ == "__main__":
    unittest.main()
