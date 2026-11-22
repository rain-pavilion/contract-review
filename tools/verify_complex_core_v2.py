"""Run a versioned synthetic integration diagnostic with explicit value scopes.

The existing public v1 execution tool remains unchanged. This wrapper replaces
only its in-process fixture and provenance functions, never serving code. The
contract below is an engineering fixture, not a transaction or legal gold.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import verify_complex_core_workflow as base


def scoped_contract() -> str:
    rows = [
        ('第一条 服务范围', '本合同总价人民币100000元。本项目为技术服务项目，双方约定交付源代码和文档。'),
        ('第二条 付款安排', '本合同尾款人民币70000元；甲方依据第九条完成最终验收后支付该尾款。'),
        ('第三条 交付安排', '本项目最终交付日期为2026年12月1日。'),
        ('第四条 其他付款阶段', '本合同预付款人民币30000元。本合同不含税质保金人民币5000元。'),
        ('第五条 信息保密', '双方应对履约获取的技术资料保密，经书面同意方可披露。'),
        ('第六条 不同阶段和标的', '本项目初步验收日期为2026年11月25日；设备B最终交付日期为2027年2月1日。'),
        ('第七条 补充付款约定', '本合同尾款人民币80000元；该笔为同一合同的尾款约定，未写明替代前款。'),
        ('第八条 补充交付安排', '本项目最终交付日期为2027年1月15日。'),
        ('第九条 验收安排', '验收以第二条所列付款安排为前提，甲方支付全部合同价款后乙方才启动最终验收。'),
        ('第十条 责任及附件', '甲方无论因何原因违约，均不承担任何赔偿责任。附录末尾标识：' + base.TAIL + '。'),
    ]
    result = [
        '合成诊断合同 复杂跨章节技术服务合同（非真实交易、非人工法律金标）',
        '甲方：' + base.SUBJECT, '甲方统一社会信用代码：' + base.BUYER_ID,
        '乙方：' + base.SUBJECT, '乙方统一社会信用代码：' + base.SELLER_ID,
    ]
    for number, (heading, core) in enumerate(rows, 1):
        note_count = 34 if number in (3, 7, 8) else 9
        notes = ''.join(
            f'材料{number}-{i}应记录版本号、经办人、归档目录和完整性；双方以书面收件记录确认资料清单。'
            for i in range(1, note_count + 1)
        )
        # Put the numeric/date source at a real clause tail, beyond a small
        # planner segment. Administrative detail is data, not model instruction.
        result.extend((heading, notes + core))
    return '\n'.join(result) + '\n'


def main(argv=None):
    original_prepare = base.prepare_case
    original_provenance = base.source_provenance
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def provenance():
        return {**original_provenance(), 'diagnostic_runner': Path(__file__).name,
                'diagnostic_runner_sha256': runner_hash,
                'fixture_version': 'scoped_value_tail_v2',
                'fixture_is_independent_legal_gold': False}

    def prepare(*args, **kwargs):
        if kwargs.get('public_contract_dir') is not None:
            raise ValueError('v2 wrapper is only for its versioned synthetic fixture')
        result = original_prepare(*args, **kwargs)
        plan = result['plan']
        candidates = plan.get('obligation_value_candidates', [])
        slots = {item['slot'] for item in candidates}
        if not any(slot.startswith('payment:tail:') for slot in slots):
            raise AssertionError('scoped tail-payment candidate is absent')
        if 'date:delivery:final' not in slots:
            raise AssertionError('scoped final-delivery-date candidate is absent')
        initial_ids = {row['window_id'] for row in result['windows']}
        for prefix in ('payment:tail:', 'date:delivery:final'):
            expected = [item for item in candidates if item['slot'].startswith(prefix)]
            if not any(item['window_id'] in initial_ids for item in expected):
                raise AssertionError('expected implicit value candidate is outside explicit run budget')
        result['source_manifest'].update(fixture_version='scoped_value_tail_v2',
            runner_sha256=runner_hash, source_has_explicit_local_scope=True,
            expectation_scope='source-value differences and dependency-cycle diagnostics; no legal truth label')
        result['case_expectations'] = [
            {'id': 'explicit_reference_cycle', 'kind': 'diagnostic_text_relationship',
             'description': '第二条等待第九条验收，第九条等待全部付款；非人工法律金标'},
            {'id': 'implicit_payment_value_difference', 'kind': 'diagnostic_text_relationship',
             'slot': 'payment:tail', 'values': ['70000元', '80000元']},
            {'id': 'implicit_final_delivery_date_difference', 'kind': 'diagnostic_text_relationship',
             'slot': 'date:delivery:final', 'values': ['2026年12月1日', '2027年1月15日']},
            {'id': 'different_obligation_negative_controls', 'kind': 'engineering_control',
             'description': '预付款、不同税价、初步验收、设备B不得拼成同槽潜在差异'},
            {'id': 'tail_preserved', 'kind': 'exact_text_invariant', 'marker': base.TAIL},
            {'id': 'same_name_expired_unknown_evidence', 'kind': 'protocol_invariant'},
        ]
        return result

    base.synthetic_contract = scoped_contract
    base.source_provenance = provenance
    base.prepare_case = prepare
    return base.main(argv)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
