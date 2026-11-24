"""Safe Markdown projection of persisted, bounded context-review evidence."""
from __future__ import annotations

import html
import re


def _rows(value):
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _text(value):
    if value is None:
        return ""
    text = html.escape(str(value), quote=False)
    return re.sub(r"([\\`*_{}\[\]()#+.!|>\-])", r"\\\1", text)


def context_review_markdown(context: dict) -> list[str]:
    """Export original quotations, derivation steps and explicit unknowns."""
    summary = context.get("summary") if isinstance(context.get("summary"), dict) else {}
    completeness = context.get("completeness") if isinstance(context.get("completeness"), dict) else {}
    constraints = context.get("constraints") if isinstance(context.get("constraints"), dict) else {}
    labels = {"not_evaluated": "尚未执行", "completed": "已执行", "needs_manual": "待人工核对", "error": "执行异常"}
    lines = ["## 跨条款与跨附件上下文审查", "",
        "组合发现属于限定规则的文本审查线索，需核对适用范围；不计入既有风险数量与辅助评分。", "",
        f"- 信息齐全性：{labels.get(completeness.get('status'), '状态未记录')}",
        f"- 本次条款：{_text(summary.get('clause_count', '未记录'))}；来源：{_text(summary.get('source_count', '未记录'))}",
        f"- 图约束问题：{_text(summary.get('constraint_violation_count', '未记录'))}；组合发现：{_text(summary.get('finding_count', '未记录'))}", ""]
    for issue in _rows(constraints.get("violations")):
        lines += ["- 图约束：" + _text(issue.get("message") or issue.get("kind") or issue.get("constraint_id"))]
    findings = _rows(context.get("findings"))
    if not findings:
        lines += ["本次未形成组合发现；不表示所有跨条款问题均已排除。", ""]
    for finding in findings:
        lines += ["### " + _text(finding.get("title") or finding.get("kind") or "组合审查事项"), "",
            "- 结果：" + ("发现问题线索" if finding.get("status") == "issue_found" else "待人工核对"),
            "- 具体问题：" + _text(finding.get("actual_issue")),
            "- 可能影响：" + _text(finding.get("impact")),
            "- 修改建议：" + _text(finding.get("recommendation")), "", "**原文证据与来源**", ""]
        for evidence in _rows(finding.get("evidence")):
            title = evidence.get("source_title") or evidence.get("document_id") or evidence.get("source_id") or "所给合同文本"
            page = evidence.get("source_page_start", evidence.get("source_page"))
            location = f"，原件页码 {_text(page)}" if page is not None else ""
            identity = "证据编号：" + _text(evidence["evidence_id"]) + "；" if evidence.get("evidence_id") else ""
            lines += [f"- {identity}{_text(title)}，条款 {_text(evidence.get('clause_id'))}{location}，字符 {_text(evidence.get('start'))}—{_text(evidence.get('end'))}：{_text(evidence.get('quote'))}",
                "  原文指纹：" + _text(evidence.get("source_sha256") or "未记录")]
        lines += ["", "**推导步骤**", ""]
        for index, step in enumerate(_rows(finding.get("reasoning_steps")), 1):
            premises = step.get("premise_ids") if isinstance(step.get("premise_ids"), list) else []
            suffix = "；依据：" + "、".join(_text(item) for item in premises) if premises else ""
            lines += [f"{index}. {_text(step.get('claim') or step.get('description'))}{suffix}"]
        for key, label in (("assumptions", "适用前提"), ("pending_reasons", "待核事项")):
            values = finding.get(key)
            if isinstance(values, list) and values:
                lines += ["", f"**{label}**", "", *("- " + _text(item) for item in values)]
        lines.append("")
    if completeness.get("message") or completeness.get("reason"):
        lines += ["信息齐全性说明：" + _text(completeness.get("message") or completeness.get("reason")), ""]
    return lines
