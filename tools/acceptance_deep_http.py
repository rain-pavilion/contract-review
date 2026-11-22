"""Real BERT plus rule disagreement through the complete HTTP archive path.

No LLM calls or human labels are made. A successful integration check is not a
legal accuracy score. The text is an explicitly synthetic diagnostic example.
"""
import argparse
import json
from pathlib import Path

from acceptance_common import ROOT, digest, multipart, provenance, request, require, write_json

SAMPLE = ('合成验收用技术服务合同\n甲方：合成甲科技有限公司\n乙方：合成乙技术有限公司\n'
          '第一条 标的\n乙方向甲方提供软件维护服务。\n'
          '第二条 价款\n服务费人民币10000元，验收后支付。\n'
          '第三条 责任\n甲方无论因何原因违约，均不承担任何赔偿责任。\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8610')
    parser.add_argument('--out', type=Path, default=ROOT/'tools/acceptance_results/http_deep_bert.json')
    args = parser.parse_args()
    result = {'status': 'running', 'provenance': provenance(), 'synthetic_input': True,
              'accuracy_evaluated': False, 'llm_requested': False, 'errors': []}
    try:
        health = require(*request(args.base, '/api/health', timeout=30)[:2])
        result['runtime'] = health.get('runtime')
        assert health.get('runtime', {}).get('active_backend') == 'bert', 'actual BERT not initialized'
        raw, content_type = multipart('acceptance-synthetic-deep-bert.txt', SAMPLE.encode(),
                                      {'engine': 'multi', 'llm': '0', 'store': 'true'})
        status, report, seconds = request(args.base, '/api/review', raw=raw,
                                         content_type=content_type, timeout=180)
        require(status, report)
        assert report.get('saved') and report.get('report_id'), report
        archived = require(*request(args.base, '/api/workspace/reports/'+report['report_id'])[:2])
        assert archived['stats'] == report['stats']
        calls = report.get('stats', {}).get('model_calls', [])
        assert any(c.get('component') == 'entailment' and c.get('operation') == 'predict_pair'
                   and c.get('status') == 'succeeded' for c in calls), 'no real deep pair event'
        assert not any(c.get('component') == 'llm' and c.get('status') == 'started' for c in calls)
        checks = report.get('stats', {}).get('statute_checks', [])
        disagreements = [c for c in checks if c.get('disagreement')]
        manual = [r for r in report.get('risks', []) if r.get('review_required')]
        assert disagreements, 'diagnostic text did not expose model/rule disagreement'
        assert manual, 'disagreement was not preserved for manual review'
        result.update(status='passed', client_upload_to_archive_response_seconds=round(seconds, 6),
                      report_id=report['report_id'], text_sha256=digest(SAMPLE),
                      server_timings=report['stats'].get('timings'),
                      actual_model_calls=calls, disagreements=disagreements,
                      manual_review_risks=[{'risk_id': r['risk_id'], 'rationale': r.get('rationale')}
                                           for r in manual],
                      scope='one real BERT/rule HTTP integration example; LLM disabled; no accuracy conclusion')
    except Exception as exc:
        result['status'] = 'failed'
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    write_json(args.out, result)
    print(json.dumps({'status': result['status'], 'out': str(args.out), 'errors': result['errors']}, ensure_ascii=False))
    return result['status'] != 'passed'


if __name__ == '__main__':
    raise SystemExit(main())
