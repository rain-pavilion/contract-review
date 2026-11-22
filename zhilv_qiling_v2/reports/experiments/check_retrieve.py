import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zhilv_qiling_v2 import retrieve as R, config as C

ROOT = Path(__file__).resolve().parent.parent
cfg = C.config_from_env()
t0 = time.time()
idx = R.build_index(ROOT / "data" / "法条结构化数据.jsonl", cfg)
print("build %.2fs" % (time.time() - t0), idx.stats)
ret = R.Retriever(idx, cfg, R.registry().rerank() or R.default_ranker())

cases = [
    ("甲方应当于验收后十五个工作日内支付价款，逾期按日万分之五支付违约金", ("价款支付", "违约责任")),
    ("租赁期限二十年以上，押金为三个月租金", ("租赁期限", "押金")),
    ("出租人可将房屋转让给第三人，承租人不得主张任何权利", ("权利转让",)),
    ("受托人超越权限处理委托事务，造成损失的由委托人自行承担", ("代理权限", "责任减免")),
]

for text, expect in cases:
    plan = R.rewrite_query(text, contract_type="housing_rent")
    t1 = time.time()
    out = ret.search(plan, top_k=5)
    dt = (time.time() - t1) * 1000
    print("=" * 70)
    print("Q:", text[:40], "| %.1fms" % dt)
    print("REWRITE:", plan.rewritten[:90])
    print("STATS:", out.stats["channels"], "cand=", out.stats["candidates"])
    for h in out.hits:
        print("  %.4f %s %s | %s" % (h.score, h.law_name[:12], h.article_no, h.content[:52].replace("\n", " ")))
