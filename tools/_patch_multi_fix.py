"""把多智能体两处收尾补上：dump_trace 带 messages、engine 的 notes 兼容 dict。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

p = ROOT / "zhilv_qiling_v2" / "agents_multi_run.py"
s = p.read_text(encoding="utf-8")
old = """    def dump_trace(self) -> str:
        return json.dumps(self.trace().as_dict(), ensure_ascii=False, indent=2)"""
new = '''    def dump_trace(self) -> str:
        """完整通信轨迹（JSON 字符串）：给答辩演示与问题复现用。

        除了 steps，还带 messages —— 只有边计数看不出“谁给谁发了什么”，
        而拓扑是否真的约束住通信，恰恰要看消息本身。
        """
        payload = self.trace().as_dict()
        payload["messages"] = self.bus.dump() if self.bus else []
        return json.dumps(payload, ensure_ascii=False, indent=2)'''
assert old in s, "dump_trace 锚点没找到"
p.write_text(s.replace(old, new), encoding="utf-8")
print("agents_multi_run.py: dump_trace 已带 messages")

p2 = ROOT / "zhilv_qiling_v2" / "engine.py"
s2 = p2.read_text(encoding="utf-8")
old2 = '                note="; ".join(step.get("notes") or []),'
new2 = '''                note="; ".join(
                    n if isinstance(n, str) else json.dumps(n, ensure_ascii=False, default=str)
                    for n in (step.get("notes") or [])
                ),'''
assert old2 in s2, "engine notes 锚点没找到"
s2 = s2.replace(old2, new2)
if "\nimport json\n" not in s2:
    s2 = s2.replace("\nimport time\n", "\nimport json\nimport time\n", 1)
p2.write_text(s2, encoding="utf-8")
print("engine.py: notes 兼容 dict，json 已导入")
