"""Keep provisional model claims out of scoring and authoritative summaries.

This is a presentation/aggregation gate. It never deletes a candidate, verifies
legal truth, or changes an upstream pending/unknown result into a confirmation.
"""
from __future__ import annotations

from typing import Any, Sequence

VERSION = 'judge-evidence-bound-v1'


def risk_basis(risk: Any) -> dict[str, Any]:
    records = [getattr(ev, 'extra', {}) or {} for ev in getattr(risk, 'evidence', [])]
    statuses = [str(row.get('claim_grounding_status')) for row in records
                if row.get('claim_grounding_status') is not None]
    model_evidence = any(getattr(ev, 'source', '') in ('node2_llm', 'reviewer_strong')
                         for ev in getattr(risk, 'evidence', []))
    if 'extraction_gap' in statuses:
        status = 'extraction_gap'
    elif 'unsupported' in statuses:
        status = 'unsupported'
    elif any(value != 'supported' for value in statuses):
        status = 'unknown'
    elif statuses:
        status = 'supported'
    elif model_evidence:
        status = 'not_assessed'
    else:
        status = 'deterministic_candidate'
    substantive = getattr(risk, 'code', '') != 'no_substantive_risk'
    eligible = substantive and status in ('supported', 'deterministic_candidate')
    reasons = list(dict.fromkeys(str(reason) for row in records
                   for reason in (row.get('reasons') or []) if isinstance(reason, str)))
    return {'risk_id': str(getattr(risk, 'risk_id', '')), 'code': str(getattr(risk, 'code', '')),
            'text_support_status': status, 'score_eligible': eligible,
            'review_required': bool(getattr(risk, 'review_required', True)),
            'legal_truth_established': False, 'support_reasons': reasons,
            'reason': '' if eligible else (
                'extraction_coverage_notice_not_a_contract_defect' if status == 'extraction_gap' else
                'execution_notice' if not substantive else
                'model_claim_text_support_not_confirmed')}


def summary_basis(risks: Sequence[Any]) -> dict[str, Any]:
    rows = [risk_basis(risk) for risk in risks if getattr(risk, 'code', '') != 'no_substantive_risk']
    return {'version': VERSION, 'candidate_count': len(rows),
            'score_eligible_count': sum(row['score_eligible'] for row in rows),
            'unsubstantiated_count': sum(not row['score_eligible'] for row in rows),
            'review_required_count': sum(row['review_required'] for row in rows),
            'not_a_clearance_or_accuracy': True, 'items': rows}


def render_summary(basis: dict[str, Any]) -> str:
    count = basis['candidate_count']
    if not count:
        return '本次未生成具体风险候选；不代表审查通过，缺失事实和未完成检查仍需核对。'
    return (f'本次保留{count}项风险候选，其中{basis["unsubstantiated_count"]}项模型主张的文本支持性未确认，'
            f'未计入辅助扣分；{basis["review_required_count"]}项标记人工复核。'
            '辅助评分不是准确率、完成率或法律结论。')


def render_focus(risks: Sequence[Any], requested_ids: Sequence[Any] = ()) -> list[str]:
    by_id = {str(r.risk_id): r for r in risks if r.code != 'no_substantive_risk'}
    ids = list(dict.fromkeys(v for v in requested_ids if isinstance(v, str) and v in by_id))[:3]
    if not ids:
        ids = list(by_id)[:3]
    result = []
    for key in ids:
        risk = by_id[key]
        basis = risk_basis(risk)
        prefix = '主张待核实' if not basis['score_eligible'] else ('风险待复核' if risk.review_required else '文本风险候选')
        result.append(f'{prefix}：{risk.message}')
    return result
