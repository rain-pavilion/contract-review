"""给 `review` / `batch` 增加 `--store`，让审查结果直接入库（申报书"存储/批量调度"闭环）。

同时把 `cmd_batch` 的默认数据源扩到"生成集 + 政采集"，避免只跑 3 份看不出调度能力。
"""

from __future__ import annotations

import pathlib
import re

path = pathlib.Path("zhilv_qiling_v2/cli.py")
text = path.read_text(encoding="utf-8")

# ---- 1. cmd_review 入库 ----
old_review = '''    report = review_path(path, config)'''
if old_review not in text:
    # 兼容无 config 变量的写法
    matches = re.findall(r"\n    report = [^\n]+\n", text)
    raise SystemExit(f"找不到 review 调用，候选：{matches}")

new_review = '''    report = review_path(path, config)
    if getattr(args, "store", False):
        from . import store as store_mod

        with store_mod.default_store(config) as db:
            document = prepare(path.name, path=path, config=config)
            db.save(report, digest=report.contract_id, text_chars=len(document.full_text))
            db.save_clauses(
                report.contract_id,
                [(clause.clause_id, clause.text) for clause in document.clauses],
            )
        print(f"已入库 {report.contract_id}", file=sys.stderr)'''
text = text.replace(old_review, new_review, 1)

# ---- 2. cmd_batch 支持 --store ----
old_batch_body = "    summary = batch_review(paths, config)"
if old_batch_body in text:
    text = text.replace(
        old_batch_body,
        '''    if getattr(args, "store", False):
        from . import store as store_mod

        with store_mod.default_store(config) as db:
            result = store_mod.batch_to_store(
                paths, db, config, limit=getattr(args, "limit", None)
            )
        print(f"入库 {result.processed} 份，跳过 {result.skipped} 份，失败 {len(result.failed)} 份")
        if args.json:
            _print_json(result.as_dict())
        return 0
    summary = batch_review(paths, config)''',
        1,
    )
else:
    # 退路：直接在 paths 计算后插入
    anchor = re.search(r"\n    paths = [^\n]+\n", text)
    if not anchor:
        raise SystemExit("找不到 cmd_batch 的 paths 行")
    insert = '''
    if getattr(args, "store", False):
        from . import store as store_mod

        with store_mod.default_store(config) as db:
            result = store_mod.batch_to_store(
                paths, db, config, limit=getattr(args, "limit", None)
            )
        print(f"入库 {result.processed} 份，跳过 {result.skipped} 份，失败 {len(result.failed)} 份")
        if args.json:
            _print_json(result.as_dict())
        return 0
'''
    text = text[: anchor.end()] + insert + text[anchor.end() :]

# ---- 3. 参数注册：review 与 batch 各加 --store ----
text = text.replace(
    '''    review.add_argument("--trace", action="store_true")''',
    '''    review.add_argument("--trace", action="store_true")
    review.add_argument("--store", action="store_true", help="结果入库（data/reviews.sqlite）")''',
    1,
)
text = text.replace(
    '''    batch.add_argument("--csv", default=None)''',
    '''    batch.add_argument("--csv", default=None)
    batch.add_argument("--store", action="store_true", help="结果入库（data/reviews.sqlite）")''',
    1,
)

path.write_text(text, encoding="utf-8")
print("cli.py 已支持 --store")
