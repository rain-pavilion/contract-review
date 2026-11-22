"""给 bert_entail 的双塔加"规则残差旁路"。

动机（实测）：BERT 后端对「均不承担任何赔偿责任」判 neutral 0.9995，
而 8 维规则判 contradict 0.9983 —— 训练出来的模型反而不如规则。
根因是监督信号本来就是规则的蒸馏：模型要从 token 里重新学一遍规则，
258 条样本学不出来，还丢掉了规则本来算得对的那部分。

做法（残差式集成，数学上保证不劣于规则）：
  logits = zero_init(head(features)) + rule_head(rule_feats)
  - rule_head 用 entailment.DEFAULT_WEIGHTS 精确初始化 → 起点就是规则本身
  - head 最后一层零初始化 → 起点时 BERT 分支贡献恰好为 0
  - 训练只在规则之上学残差，学不到东西就退化成规则，不会更差

中文文件必须用 Python 读写（PowerShell 的 Get-Content/Set-Content 会按 GBK
解码 UTF-8 中文，把整个文件写成双重编码乱码）。
"""

from __future__ import annotations

import pathlib
import sys

TARGET = pathlib.Path(__file__).resolve().parents[1] / "zhilv_qiling_v2" / "bert_entail.py"


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    original = text

    # A1 签名加 rule_dim
    old = "def build_model(*, dropout: float = 0.15):"
    new = "def build_model(*, dropout: float = 0.15, rule_dim: int = 8):"
    assert old in text, "A1 未命中"
    text = text.replace(old, new, 1)

    # A2 head 之后挂 rule_head，并把 head 最后一层零初始化
    old = """            self.head = nn.Sequential(
                nn.Linear(hidden * 5 + 1, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, len(LABELS)),
            )
"""
    new = """            self.head = nn.Sequential(
                nn.Linear(hidden * 5 + 1, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, len(LABELS)),
            )
            # 规则残差旁路：把 entailment 的 8 维特征接一条线性层，与 BERT 头相加。
            # 权重用 DEFAULT_WEIGHTS 精确初始化，且 head 最后一层零初始化 ——
            # 训练起点恰好等于规则后端的判断，模型只在其上学残差。
            # 这样即使样本少、学不动，效果也不会低于 rules（实测旧版会掉到 neutral 0.9995）。
            self.rule_head = nn.Linear(rule_dim, len(LABELS))
            with torch.no_grad():
                for index, label in enumerate(LABELS):
                    row = E.DEFAULT_WEIGHTS.get(label, {})
                    for slot, name in enumerate(E.FEATURES):
                        self.rule_head.weight[index, slot] = float(row.get(name, 0.0))
                    self.rule_head.bias[index] = float(row.get("_bias", 0.0))
                for param in self.head[-1].parameters():
                    nn.init.zeros_(param)
"""
    assert old in text, "A2 未命中"
    text = text.replace(old, new, 1)

    # A3 forward 签名
    old = "        def forward(self, clause_ids, clause_mask, article_ids, article_mask):"
    new = "        def forward(self, clause_ids, clause_mask, article_ids, article_mask, rule_feats=None):"
    assert old in text, "A3 未命中"
    text = text.replace(old, new, 1)

    # A4 forward 返回
    old = "            return self.head(features), cosine.squeeze(1), weights"
    new = """            logits = self.head(features)
            if rule_feats is not None:
                logits = logits + self.rule_head(rule_feats)
            return logits, cosine.squeeze(1), weights"""
    assert old in text, "A4 未命中"
    text = text.replace(old, new, 1)

    # C encode 产出规则特征
    old = """        labels = torch.tensor([LABEL2ID[row["label"]] for row in batch])
        return clauses, articles, labels"""
    new = """        labels = torch.tensor([LABEL2ID[row["label"]] for row in batch])
        feats = torch.tensor(
            [[float(row["features"].get(name, 0.0)) for name in E.FEATURES] for row in batch],
            dtype=torch.float32,
        )
        return clauses, articles, labels, feats"""
    assert old in text, "C 未命中"
    text = text.replace(old, new, 1)

    # D train 前向
    old = """            clauses, articles, labels = encode(batch)
            logits, cosine, _ = model(
                clauses["input_ids"].to(device), clauses["attention_mask"].to(device),
                articles["input_ids"].to(device), articles["attention_mask"].to(device),
            )"""
    new = """            clauses, articles, labels, feats = encode(batch)
            logits, cosine, _ = model(
                clauses["input_ids"].to(device), clauses["attention_mask"].to(device),
                articles["input_ids"].to(device), articles["attention_mask"].to(device),
                rule_feats=feats.to(device),
            )"""
    assert old in text, "D 未命中"
    text = text.replace(old, new, 1)

    # E evaluate 前向
    old = """            logits, _, _ = model(
                clauses["input_ids"].to(device), clauses["attention_mask"].to(device),
                articles["input_ids"].to(device), articles["attention_mask"].to(device),
            )"""
    new = """            feats = torch.tensor(
                [[float(row["features"].get(name, 0.0)) for name in E.FEATURES] for row in batch],
                dtype=torch.float32,
            )
            logits, _, _ = model(
                clauses["input_ids"].to(device), clauses["attention_mask"].to(device),
                articles["input_ids"].to(device), articles["attention_mask"].to(device),
                rule_feats=feats.to(device),
            )"""
    assert old in text, "E 未命中"
    text = text.replace(old, new, 1)

    # F Predictor.predict 前向
    old = """        with torch.no_grad():
            logits, cosine, weights = self.model(
                clause_encoded["input_ids"].to(self.device),
                clause_encoded["attention_mask"].to(self.device),
                article_encoded["input_ids"].to(self.device),
                article_encoded["attention_mask"].to(self.device),
            )"""
    new = """        features = E.clause_features(clause, article, risk_tags=(), codes=())
        rule_feats = torch.tensor(
            [[float(features.get(name, 0.0)) for name in E.FEATURES]], dtype=torch.float32
        ).to(self.device)
        with torch.no_grad():
            logits, cosine, weights = self.model(
                clause_encoded["input_ids"].to(self.device),
                clause_encoded["attention_mask"].to(self.device),
                article_encoded["input_ids"].to(self.device),
                article_encoded["attention_mask"].to(self.device),
                rule_feats=rule_feats,
            )"""
    assert old in text, "F 未命中"
    text = text.replace(old, new, 1)

    # G 旧权重兼容（没有 rule_head 的 checkpoint）
    old = "    model.load_state_dict(torch.load(weights, map_location=\"cpu\"))"
    new = "    model.load_state_dict(torch.load(weights, map_location=\"cpu\"), strict=False)"
    assert old in text, "G 未命中"
    text = text.replace(old, new, 1)

    if text == original:
        print("没有改动")
        return 1
    TARGET.write_text(text, encoding="utf-8")
    print(f"已改写 {TARGET}（{len(original.splitlines())} → {len(text.splitlines())} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
