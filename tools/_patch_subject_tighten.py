"""收紧节点③的两条易误报规则（need_license / need_authorization）。

对抗集实测：接入节点③后召回仍 1.000，但 issue-code 层 FP 从 24 涨到 43，
其中 need_license fp=10、need_authorization fp=8 —— 都是"每份合同都命中"的宽规则。
本脚本按两条口径收紧：
  1. 资质：只有命中**强行业词**才算 requires 边；弱词（信息/数据/系统/平台）不再单独触发。
  2. 权限：全文必须真的出现"代理/授权/委托/签署"语境，才检查是否缺法定代表人表述。
"""

import pathlib

p = pathlib.Path("zhilv_qiling_v2/subject_graph.py")
text = p.read_text(encoding="utf-8")

old_soft = '    (("软件", "信息", "数据", "系统", "平台", "运维"), "信息系统相关资质（如涉密则另需保密资质）", "采购需求文件"),'
new_soft = ('    # 只保留强线索：泛化到"信息/数据/系统/平台"会让每份政采数据类合同都命中（实测 fp=10）\n'
            '    (("涉密", "等级保护", "信息系统集成", "信息安全服务", "CMMI", "网络安全"), '
            '"信息系统集成或信息安全服务资质", "采购需求文件"),')
assert old_soft in text, "找不到软件信息资质组"
text = text.replace(old_soft, new_soft, 1)

old_auth = """    # 2) 权限：有机构主体、但正文没有任何法定代表人/授权委托线索
    if org_nodes and not has_auth_token:"""
new_auth = """    # 2) 权限：只在正文确实涉及代理/授权/签署语境时，才检查法定代表人线索是否缺失。
    # 原实现"有机构主体就查"，对每份模板合同都报（实测 fp=8），改为先看语境。
    auth_context = any(token in full_text for token in AUTH_CONTEXT)
    if org_nodes and auth_context and not has_auth_token:"""
assert old_auth in text, "找不到权限检查块"
text = text.replace(old_auth, new_auth, 1)

anchor = "#: 签约权限线索（出现任意一个即认为权限链条有交代）"
assert anchor in text
text = text.replace(
    anchor,
    '#: 签约权限的**语境**词：出现才值得检查法定代表人是否交代（避免"有主体就报"）\nAUTH_CONTEXT: tuple[str, ...] = (\n    "代理", "授权", "委托", "代表人", "签署", "签章", "签字盖章",\n)\n\n' + anchor,
    1,
)

p.write_text(text, encoding="utf-8")
print("已收紧 need_license / need_authorization")

# 同步把 need_license 置信度略降（强词命中仍在 0.6 附近，不抢其他风险的位次）
text = p.read_text(encoding="utf-8")
if 'confidence=0.62,' in text:
    text = text.replace('confidence=0.62,', 'confidence=0.58,', 1)
    p.write_text(text, encoding="utf-8")
    print("need_license 置信度 0.62 → 0.58")
