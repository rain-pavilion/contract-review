"""修 test_inbox_filters_by_kind：这条只测收件箱的 kind 过滤，不该受拓扑约束。"""

from pathlib import Path

p = Path(__file__).resolve().parents[1] / "zhilv_qiling_v2" / "tests_v2" / "test_multi_agent.py"
s = p.read_text(encoding="utf-8")
old = """    def test_inbox_filters_by_kind(self):
        bus = M.MessageBus(strict=True)"""
new = """    def test_inbox_filters_by_kind(self):
        # 只测收件箱的 kind 过滤，不测拓扑：logic->ner 在真实拓扑里是会被拒的边
        # （logic 只能发给 subject/statute/judge），所以这里放开 strict。
        bus = M.MessageBus(strict=False)"""
assert old in s, "锚点没找到"
p.write_text(s.replace(old, new), encoding="utf-8")
print("test_inbox_filters_by_kind 已改为 strict=False")
